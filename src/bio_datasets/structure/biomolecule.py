import io
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from biotite import structure as bs
from biotite.structure.io.pdb import PDBFile
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices

from .chemical import Molecule, T
from .residue import (
    ResidueDictionary,
    get_residue_starts_mask,
    tile_residue_annotation_to_atoms,
)

ALL_EXTRA_FIELDS = ["occupancy", "b_factor", "atom_id", "charge"]


def create_complete_atom_array_from_restype_index(
    restype_index: np.ndarray,
    residue_dictionary: ResidueDictionary,
    chain_id: str,
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
):
    """
    Populate annotations from restype_index, assuming all atoms are present.

    Assumes single chain for now
    """
    if backbone_only:
        residue_sizes = len(residue_dictionary.backbone_atoms) * len(restype_index)
    else:
        residue_sizes = residue_dictionary.residue_sizes[
            restype_index
        ]  # (n_residues,) NOT (n_atoms,) -- add 1 to account for OXT

    residue_starts = np.concatenate(
        [[0], np.cumsum(residue_sizes)[:-1]]
    )  # (n_residues,)
    new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
    residue_index = (
        np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
    )

    full_annot_names = []
    new_atom_array.set_annotation(
        "chain_id",
        np.full(len(new_atom_array), chain_id, dtype=new_atom_array.chain_id.dtype),
    )
    full_annot_names.append("chain_id")

    relative_atom_index = np.arange(len(new_atom_array)) - residue_starts[residue_index]
    atom_names = new_atom_array.atom_name
    new_atom_array.set_annotation("restype_index", restype_index[residue_index])
    atom_names = residue_dictionary.standard_atoms_by_residue[
        new_atom_array.restype_index,
        relative_atom_index,
    ]
    new_atom_array.set_annotation("atom_name", atom_names)
    new_atom_array.set_annotation(
        "res_name",
        np.array(residue_dictionary.residue_names)[new_atom_array.restype_index],
    )
    new_atom_array.set_annotation("residue_index", residue_index)
    new_atom_array.set_annotation("res_id", residue_index + 1)
    new_atom_array.set_annotation(
        "element", np.char.array(new_atom_array.atom_name).astype("U1")
    )
    full_annot_names += [
        "atom_name",
        "restype_index",
        "res_name",
        "residue_index",
        "res_id",
    ]
    if extra_fields is not None:
        for f in extra_fields:
            new_atom_array.add_annotation(
                f, dtype=float if f in ["occupancy", "b_factor"] else int
            )
    return new_atom_array, residue_starts, full_annot_names


class Biomolecule(Molecule):
    """Base class for biomolecule objects.

    Biomolecules (DNA, RNA and Proteins) are chains of residues.

    Biomolecule and modality-specific subclasses provide convenience
    methods for interacting with residue-level properties.
    """

    resname_converter: Callable
    backbone_atoms: List[str]
    residue_dictionary: ResidueDictionary

    def __init__(
        self,
        atoms: bs.AtomArray,
        verbose: bool = False,
        backbone_only: bool = False,
        exclude_hydrogens: bool = True,
        standardisation_kwargs: Optional[Dict] = None,
    ):
        self.backbone_only = backbone_only
        self._residue_starts = get_residue_starts(atoms)
        atoms = self.convert_residues(atoms)
        atoms = self.filter_atoms(atoms)
        atoms = self.set_atom_annotations(
            atoms,
            residue_dictionary=self.residue_dictionary,
            residue_starts=self._residue_starts,
        )
        self.atoms, self._residue_starts = self.standardise_atoms(
            atoms,
            residue_dictionary=self.residue_dictionary,
            residue_starts=self._residue_starts,
            verbose=verbose,
            backbone_only=self.backbone_only,
            exclude_hydrogens=exclude_hydrogens,
            **standardisation_kwargs,
        )
        self._standardised = True

    @staticmethod
    def convert_residues(atoms: bs.AtomArray, residue_dictionary: ResidueDictionary):
        for conversion_dict in residue_dictionary.conversions or []:
            atom_swaps = conversion_dict["atom_swaps"]
            from_mask = atoms[atoms.res_name == conversion_dict["residue"]]
            for swap in atom_swaps:
                atoms[from_mask & atoms.atom_name == swap[0]].atom_name = swap[1]
            atoms[from_mask].res_name = conversion_dict["to_residue"]
        return atoms

    @staticmethod
    def set_atom_annotations(
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        residue_starts: np.ndarray,
    ):
        atoms.set_annotation(
            "atom_type_index",
            map_categories_to_indices(atoms.atom_name, residue_dictionary.atom_types),
        )
        atoms.set_annotation(
            "res_type_index", residue_dictionary.resname_to_index(atoms.res_name)
        )
        atoms.set_annotation(
            "res_index",
            np.cumsum(get_residue_starts_mask(atoms, residue_starts)) - 1,
        )
        return atoms

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_dictionary,
        residue_starts: Optional[np.ndarray] = None,
        verbose: bool = False,
        backbone_only: bool = False,
        exclude_hydrogens: bool = True,
        expected_relative_atom_indices: Optional[
            np.ndarray
        ] = None,  # pass custom for oxt
    ):
        if exclude_hydrogens:
            assert (
                "element" in atoms._annot
            ), "Elements must be present to exclude hydrogens"
            atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
        else:
            raise ValueError("Hydrogens are not supported in standardisation")
        if residue_starts is None:
            residue_starts = get_residue_starts(atoms)
        # first we get an array of atom indices for each residue (i.e. a mapping from atom type index to expected index
        # then we index into this array to get the expected index for each atom
        if (
            not isinstance(expected_relative_atom_indices, np.ndarray)
            and expected_relative_atom_indices is None
        ):
            expected_relative_atom_indices = (
                residue_dictionary.relative_atom_indices_mapping[
                    atoms.aa_index, atoms.atom_type_index
                ]
            )

        unexpected_atom_mask = expected_relative_atom_indices == -100
        if np.any(
            unexpected_atom_mask
            & (atoms.res_name != residue_dictionary.unknown_residue_name)
        ):
            unexpected_atoms = atoms.atom_name[unexpected_atom_mask]
            unexpected_residues = atoms.res_name[unexpected_atom_mask]
            unexpected_str = "\n".join(
                [
                    f"{res_name} {res_id} {atom_name}"
                    for res_name, res_id, atom_name in zip(
                        unexpected_residues,
                        atoms.res_id[unexpected_atom_mask],
                        unexpected_atoms,
                    )
                ]
            )
            raise ValueError(
                f"At least one unexpected atom detected in a residue: {unexpected_str}.\n"
                f"HETATMs are not supported."
            )

        # for unk residues, we just drop any e.g. sidechain atoms without raising an exception
        unexpected_unk_atom_mask = unexpected_atom_mask & (
            atoms.res_name == residue_dictionary.unknown_residue_name
        )
        atoms = atoms[~unexpected_unk_atom_mask]
        expected_relative_atom_indices = expected_relative_atom_indices[
            ~unexpected_unk_atom_mask
        ]
        residue_starts = get_residue_starts(atoms)

        (
            new_atom_array,
            full_residue_starts,
            full_annot_names,
        ) = create_complete_atom_array_from_restype_index(
            atoms.restype_index[residue_starts],
            atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
        )
        assert len(full_residue_starts) == len(
            residue_starts
        ), f"Full residue starts: {full_residue_starts} and residue starts: {residue_starts} do not match"
        existing_atom_indices_in_full_array = (
            full_residue_starts[atoms.res_index] + expected_relative_atom_indices
        ).astype(int)

        for annot_name, annot in atoms._annot.items():
            if (
                annot_name in ["atomtype_index", "mask"]
                or annot_name in full_annot_names
            ):
                continue
            getattr(new_atom_array, annot_name)[
                existing_atom_indices_in_full_array
            ] = annot.astype(new_atom_array._annot[annot_name].dtype)

        # set_annotation vs setattr: set_annotation adds to annot and verifies size
        new_atom_array.coord[existing_atom_indices_in_full_array] = atoms.coord
        # if we can create a res start index for each atom, we can assign the value based on that...

        new_atom_array.set_annotation(
            "res_id",
            atoms.res_id[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.res_id.dtype
            ),
        )  # override with auth res id
        new_atom_array.set_annotation(
            "chain_id",
            atoms.chain_id[residue_starts][new_atom_array.residue_index].astype(
                new_atom_array.chain_id.dtype
            ),
        )
        new_atom_array.set_annotation(
            "ins_code",
            atoms.ins_code[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.ins_code.dtype
            ),
        )

        new_atom_array.set_annotation(
            "atom_type_index",
            map_categories_to_indices(
                new_atom_array.atom_name, residue_dictionary.atom_types
            ),
        )
        assert np.all(
            new_atom_array.atom_name != ""
        ), "All atoms must be assigned a name"
        mask = np.zeros(len(new_atom_array), dtype=bool)
        mask[existing_atom_indices_in_full_array] = True
        missing_atoms_strings = [
            f"{res_name} {res_id} {atom_name}"
            for res_name, res_id, atom_name in zip(
                new_atom_array.res_name[~mask],
                new_atom_array.res_id[~mask],
                new_atom_array.atom_name[~mask],
            )
        ]
        if verbose:
            print("Filled in missing atoms:\n", "\n".join(missing_atoms_strings))
        new_atom_array.set_annotation("mask", mask)
        if backbone_only:
            # TODO: more efficient backbone only
            new_atom_array = new_atom_array[
                np, isin(new_atom_array.atom_name, residue_dictionary.backbone_atomss)
            ]
            full_residue_starts = get_residue_starts(new_atom_array)
        return new_atom_array, full_residue_starts

    @classmethod
    def from_pdb(cls, pdb_path: str):
        pdbf = PDBFile.read(pdb_path)
        atoms = pdbf.get_structure()
        return cls(atoms)

    def to_pdb(self, pdb_path: str):
        # to write to pdb file, we have to drop nan coords
        atoms = self.atoms[~self.nan_mask]
        pdbf = PDBFile()
        pdbf.set_structure(atoms)
        pdbf.write(pdb_path)

    def to_pdb_string(self):
        with io.StringIO() as f:
            self.to_pdb(f)
            return f.getvalue()

    @property
    def nan_mask(self):
        return np.isnan(self.atoms.coord).any(axis=-1)

    @property
    def residue_index(self):
        return self.atoms["residue_index"][self._residue_starts]

    @property
    def restype_index(self):
        # TODO: parameterise this via a name e.g. 'aa'
        return self.atoms["restype_index"][self._residue_starts]

    @property
    def sequence(self) -> str:
        return "".join(
            self.residue_dictionary.residue_types[
                self.atoms.restype_index[self._residue_starts]
            ]
        )

    @property
    def num_residues(self) -> int:
        return len(self._residue_starts)

    @property
    def backbone_mask(self):
        return np.isin(self.atoms.atom_name, self.backbone_atoms)

    def __len__(self):
        return self.num_residues  # n.b. -- not equal to len(self.atoms)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.atoms[key]
        else:
            return self.__class__(self.atoms[key])

    def filter_atoms(self, atoms):
        raise NotImplementedError("Must be implemented on child class")

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        assert all(
            [atom in self.backbone_atoms for atom in atom_names]
        ), f"Invalid entries in atom names: {atom_names}"
        assert self._standardised, "Atoms must be in standard order"
        backbone_coords = self.atoms.coord[self.backbone_mask].reshape(
            -1, len(self.backbone_atoms), 3
        )
        if atom_names is None:
            return backbone_coords
        else:
            backbone_atom_indices = [
                self.backbone_atoms.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords = np.zeros(
                (len(backbone_coords), len(atom_names), 3), dtype=np.float32
            )
            selected_backbone_indices = [
                atom_names.index(atom) for atom in atom_names if atom != "CB"
            ]
            selected_coords[:, selected_backbone_indices] = backbone_coords[
                :, backbone_atom_indices
            ]
            return selected_coords

    def distances(
        self,
        atom_names: Union[str, List[str]],
        residue_mask_from: Optional[np.ndarray] = None,
        residue_mask_to: Optional[np.ndarray] = None,
        nan_fill=None,
        multi_atom_calc_type: str = "min",
    ) -> np.ndarray:
        # TODO: handle nans
        # TODO: allow non-backbone atoms
        if residue_mask_from is None:
            residue_mask_from = np.ones(self.num_residues, dtype=bool)
        if residue_mask_to is None:
            residue_mask_to = np.ones(self.num_residues, dtype=bool)
        backbone_coords = self.backbone_coords()
        if isinstance(atom_names, str) and atom_names in self.backbone_atoms:
            at_index = self.backbone_atoms.index(atom_names)
            dists = np.sqrt(
                np.sum(
                    (
                        backbone_coords[None, residue_mask_from, at_index, :]
                        - backbone_coords[residue_mask_to, None, at_index, :]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        else:
            raise NotImplementedError(
                "Muliple atom distance calculations not yet supported"
            )
        if nan_fill is not None:
            if isinstance(nan_fill, float) or isinstance(nan_fill, int):
                dists = np.nan_to_num(dists, nan=nan_fill)
            elif nan_fill == "max":
                max_dist = np.nanmax(dists, axis=-1)
                dists = np.nan_to_num(dists, nan=max_dist)
            else:
                raise ValueError(
                    f"Invalid nan_fill: {nan_fill}. Please specify a float or int."
                )
        return dists

    def contacts(self, atom_name: str, threshold: float) -> np.ndarray:
        return self.distances(atom_name, nan_fill="max") < threshold

    def backbone(self) -> T:
        return self.__class__(self.atoms[self.backbone_mask])

    def get_chain(self) -> "BiomoleculeChain":
        raise NotImplementedError()


class BiomoleculeChain:
    pass


class BiomoleculeComplex(Biomolecule):
    def __init__(self, chains: List[BiomoleculeChain]):
        self._chain_ids = [mol.chain_id for mol in chains]
        self._chains_lookup = {mol.chain_id: mol for mol in chains}

    @property
    def atoms(self):
        return sum(
            [prot.atoms for prot in self._proteins_lookup.values()],
            bs.AtomArray(length=0),
        )

    def get_chain(self, chain_id: str) -> "BiomoleculeChain":
        return self._chains_lookup[chain_id]

    def interface(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        threshold: float = 10.0,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> T:
        distances = self.interface_distances(
            atom_names=atom_names, chain_pair=chain_pair, nan_fill=nan_fill
        )
        interface_mask = distances < threshold
        return self.__class__.from_atoms(self.atoms[interface_mask])

    def interface_distances(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> np.ndarray:
        if chain_pair is None:
            if len(self._chain_ids) != 2:
                raise ValueError(
                    "chain_pair must be specified for non-binary complexes"
                )
            chain_pair = (self._chain_ids[0], self._chain_ids[1])
        residue_mask_from = self.atoms.chain_id == chain_pair[0]
        residue_mask_to = self.atoms.chain_id == chain_pair[1]
        return self.distances(
            atom_names=atom_names,
            residue_mask_from=residue_mask_from,
            residue_mask_to=residue_mask_to,
            nan_fill=nan_fill,
        )
