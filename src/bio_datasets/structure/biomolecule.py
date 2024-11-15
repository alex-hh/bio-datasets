import io
from typing import Generic, List, Optional, TypeVar, Union

import numpy as np
from biotite import structure as bs
from biotite.structure.io.pdb import PDBFile
from biotite.structure.residues import get_residue_starts

from bio_datasets.np_utils import map_categories_to_indices
from bio_datasets.structure.parsing import load_structure

from .residue import (
    ResidueDictionary,
    create_complete_atom_array_from_restype_index,
    get_residue_starts_mask,
)

# from biotite.structure.filter import filter_highest_occupancy_altloc  performed automatically by biotite


ALL_EXTRA_FIELDS = ["occupancy", "b_factor", "atom_id", "charge"]


T = TypeVar("T", bound="Biomolecule")


class Biomolecule(Generic[T]):
    """Base class for biomolecule objects.

    Biomolecules (DNA, RNA and Proteins) are chains of residues.

    Biomolecule and modality-specific subclasses provide convenience
    methods for interacting with residue-level properties.

    n.b. as well as proteins, dna and rna, the PDB also contains hybrid dna/rna molecules.
    other classes of biopolymers are polysaccharides and peptidoglycans.
    """

    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
        keep_hydrogens: bool = False,  # TODO: handle with residue dictionary
        raise_error_on_unexpected: bool = False,
        replace_unexpected_with_unknown: bool = False,
    ):
        self.residue_dictionary = residue_dictionary
        self.backbone_only = backbone_only
        self.raise_error_on_unexpected = raise_error_on_unexpected
        self.replace_unexpected_with_unknown = replace_unexpected_with_unknown
        self.keep_hydrogens = keep_hydrogens
        atoms = self.convert_residues(
            atoms,
            self.residue_dictionary,
            replace_unexpected_with_unknown=self.replace_unexpected_with_unknown,
        )
        atoms = self.filter_atoms(
            atoms,
            self.residue_dictionary,
            raise_error_on_unexpected=self.raise_error_on_unexpected,
            keep_hydrogens=self.keep_hydrogens,
        )  # e.g. check for standard residues.
        self.atoms = self.standardise_atoms(
            atoms,
            residue_dictionary=self.residue_dictionary,
            verbose=verbose,
            backbone_only=self.backbone_only,
        )
        self._standardised = True

    @property
    def backbone_atoms(self):
        return self.residue_dictionary.backbone_atoms

    @classmethod
    def from_file(
        cls,
        file_path,
        file_type: Optional[str] = None,
        residue_dictionary: Optional[ResidueDictionary] = None,
        extra_fields: Optional[List[str]] = None,
        **kwargs,
    ):
        if residue_dictionary is None:
            residue_dictionary = (
                ResidueDictionary.from_ccd_dict()
            )  # TODO: better default?
        atoms = load_structure(
            file_path, file_type=file_type, extra_fields=extra_fields
        )
        return cls(atoms, residue_dictionary, **kwargs)

    @property
    def is_standardised(self):
        return self._standardised

    @staticmethod
    def convert_residues(
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        replace_unexpected_with_unknown: bool = False,
    ):
        if replace_unexpected_with_unknown:
            raise NotImplementedError("Not implemented")
        for conversion_dict in residue_dictionary.conversions or []:
            atom_swaps = conversion_dict["atom_swaps"]
            element_swaps = conversion_dict["element_swaps"]
            from_mask = (atoms.res_name == conversion_dict["residue"]).astype(bool)
            for swap in atom_swaps:
                atoms.atom_name[
                    from_mask & (atoms.atom_name == swap[0]).astype(bool)
                ] = swap[1]
            for swap in element_swaps:
                atoms.element[
                    from_mask & (atoms.element == swap[0]).astype(bool)
                ] = swap[1]
            atoms.res_name[from_mask] = conversion_dict["to_residue"]
        return atoms

    @staticmethod
    def filter_atoms(
        atoms,
        residue_dictionary: Optional[ResidueDictionary] = None,
        raise_error_on_unexpected: bool = False,
        keep_hydrogens: bool = False,
    ):
        # drop water
        atoms = atoms[atoms.res_name != "HOH"]
        if not keep_hydrogens:
            assert (
                "element" in atoms._annot
            ), "Elements must be present to exclude hydrogens"
            atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
        if residue_dictionary is None or not getattr(
            residue_dictionary, "keep_oxt", False
        ):
            # oxt complicates things for residue dictionary.
            atoms = atoms[atoms.atom_name != "OXT"]
        # TODO: we actually want to use residue_dictionary.residue_atoms
        if residue_dictionary is not None:
            expected_residue_mask = np.isin(
                atoms.res_name, residue_dictionary.residue_names
            )
            if raise_error_on_unexpected and ~expected_residue_mask.any():
                unexpected_residues = np.unique(atoms[~expected_residue_mask].res_name)
                raise ValueError(
                    f"Found unexpected residues: {unexpected_residues} in atom array"
                )
            return atoms[expected_residue_mask]
        return atoms

    @staticmethod
    def reorder_chains(atoms):
        chain_ids = np.unique(atoms.chain_id)
        atom_arrs = []
        for chain_id in chain_ids:
            chain_mask = atoms.chain_id == chain_id
            atom_arrs.append(atoms[chain_mask])
        new_atoms = sum(atom_arrs, bs.AtomArray(length=0))
        for annot_name in atoms._annot:
            if annot_name not in new_atoms._annot:
                new_atoms.set_annotation(annot_name, atoms._annot[annot_name])
        return new_atoms

    @staticmethod
    def set_index_annotations(atoms, residue_dictionary, residue_starts):
        """Set numeric annotations for discrete categories to help with standardisation."""
        if (
            "atomtype_index" not in atoms._annot
            and residue_dictionary.atom_types is not None
        ):
            atoms.set_annotation(
                "atomtype_index",
                map_categories_to_indices(
                    atoms.atom_name, residue_dictionary.atom_types
                ),
            )
        if (
            "elemtype_index" not in atoms._annot
            and residue_dictionary.element_types is not None
        ):
            atoms.set_annotation(
                "elemtype_index",
                map_categories_to_indices(
                    atoms.element, residue_dictionary.element_types
                ),
            )
        if "restype_index" not in atoms._annot:
            atoms.set_annotation(
                "restype_index", residue_dictionary.res_name_to_index(atoms.res_name)
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
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        atoms = Biomolecule.reorder_chains(atoms)
        residue_starts = get_residue_starts(atoms)
        atoms = Biomolecule.set_index_annotations(
            atoms, residue_dictionary, residue_starts
        )

        (
            new_atom_array,
            full_residue_starts,
            full_annot_names,
        ) = create_complete_atom_array_from_restype_index(
            atoms.restype_index[residue_starts],
            residue_dictionary=residue_dictionary,
            chain_id=atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
            backbone_only=backbone_only,
        )

        # first we get an array of atom indices for each residue (i.e. a mapping from atom type index to expected index
        # then we index into this array to get the expected relative index for each atom
        expected_relative_atom_indices = (
            residue_dictionary.get_expected_relative_atom_indices(
                atoms.restype_index, atoms.atomtype_index
            )
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

        assert len(full_residue_starts) == len(
            residue_starts
        ), f"Full residue starts: {full_residue_starts} and residue starts: {residue_starts} do not match"

        existing_atom_indices_in_full_array = (
            full_residue_starts[atoms.res_index] + expected_relative_atom_indices
        )

        for annot_name, annot in atoms._annot.items():
            if (
                annot_name in ["atomtype_index", "mask"]
                or annot_name in full_annot_names
            ):
                continue

            if annot_name not in new_atom_array._annot:
                new_atom_array.set_annotation(
                    annot_name, np.zeros(len(new_atom_array), annot.dtype)
                )
            getattr(new_atom_array, annot_name)[
                existing_atom_indices_in_full_array
            ] = annot.astype(new_atom_array._annot[annot_name].dtype)

        # set_annotation vs setattr: set_annotation adds to annot and verifies size
        new_atom_array.coord[existing_atom_indices_in_full_array] = atoms.coord
        # if we can create a res start index for each atom, we can assign the value based on that...
        assert (
            np.unique(new_atom_array.res_index) == np.unique(atoms.res_index)
        ).all(), f"New res index contains unexpected values (not matching old res index) {np.unique(new_atom_array.res_index)} {np.unique(atoms.res_index)}"
        new_atom_array.set_annotation(
            "res_id",
            atoms.res_id[residue_starts][new_atom_array.res_index].astype(
                new_atom_array.res_id.dtype
            ),
        )  # override with auth res id
        new_atom_array.set_annotation(
            "chain_id",
            atoms.chain_id[residue_starts][new_atom_array.res_index].astype(
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
            "atomtype_index",
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
            assert residue_dictionary.backbone_atoms is not None
            # TODO: more efficient backbone only
            new_atom_array = new_atom_array[
                np.isin(new_atom_array.atom_name, residue_dictionary.backbone_atoms)
            ]
            full_residue_starts = get_residue_starts(new_atom_array)
        return new_atom_array

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
        return self.atoms["res_type_index"][self._residue_starts]

    @property
    def sequence(self) -> str:
        return "".join(
            self.residue_dictionary.residue_letters[
                self.atoms.res_type_index[self._residue_starts]
            ]
        )

    @property
    def num_residues(self) -> int:
        return len(self._residue_starts)

    @property
    def backbone_mask(self):
        assert self.residue_dictionary.backbone_atoms is not None
        return np.isin(self.atoms.atom_name, self.residue_dictionary.backbone_atoms)

    @property
    def chains(self):
        chain_ids = np.unique(self.atoms.chain_id)
        return [self.get_chain(chain_id) for chain_id in chain_ids]

    def __len__(self):
        return self.num_residues  # n.b. -- not equal to len(self.atoms)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.atoms[key]
        else:
            return self.__class__(self.atoms[key])

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        if atom_names is None:
            atom_names = self.backbone_atoms
        # requires self.backbone_atoms to be in correct order
        assert all(
            atom in self.backbone_atoms for atom in atom_names
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

    def residue_all_atom_coords(self):
        assert self.residue_dictionary.atom_types is not None
        all_atom_coords = np.full(
            (len(self.num_residues), len(self.residue_dictionary.atom_types), 3), np.nan
        )
        for ix, at in enumerate(self.residue_dictionary.atom_types):
            # must be at most one atom per atom type per residue
            at_mask = self.atoms.atom_name == at
            residue_indices = self.atoms.residue_index[at_mask]
            assert len(np.unique(residue_indices)) == len(
                residue_indices
            ), "Multiple atoms with same atom type in residue"
            all_atom_coords[residue_indices, ix] = self.atoms.coord[at_mask]
        return all_atom_coords

    def residue_distances(
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

    def residue_contacts(
        self,
        atom_names: Union[str, List[str]],
        threshold: float,
        multi_atom_calc_type: str = "min",
    ) -> np.ndarray:
        return (
            self.residue_distances(
                atom_names, nan_fill="max", multi_atom_calc_type=multi_atom_calc_type
            )
            < threshold
        )

    def backbone(self) -> T:
        # TODO: might need to also modify residue dictionary to avoid explicit nan atom coords
        return self.__class__(self.atoms[self.backbone_mask], self.residue_dictionary)

    def get_chain(self, chain_id: str):
        chain_filter = self.atoms.chain_id == chain_id
        return self.__class__(self.atoms[chain_filter].copy(), self.residue_dictionary)


class BiomoleculeChain(Biomolecule):
    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: ResidueDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
        keep_hydrogens: bool = False,
        raise_error_on_unexpected: bool = False,
        replace_unexpected_with_unknown: bool = False,
    ):
        assert (
            len(np.unique(atoms.chain_id)) == 1
        ), f"Expected single chain, found chain ids {np.unique(atoms.chain_id)}"
        super().__init__(
            atoms=atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            keep_hydrogens=keep_hydrogens,
            raise_error_on_unexpected=raise_error_on_unexpected,
            replace_unexpected_with_unknown=replace_unexpected_with_unknown,
        )

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]


class BaseBiomoleculeComplex(Biomolecule):
    def __init__(self, chains: List[BiomoleculeChain]):
        self._chain_ids = [mol.chain_id for mol in chains]
        self._chains_lookup = {mol.chain_id: mol for mol in chains}
        self.residue_dictionary = (
            None  # residue dictionary can be different for different chains
        )
        self._standardised = True

    def __str__(self):
        return str(self._chains_lookup)

    @staticmethod
    def default_residue_dictionary():
        return ResidueDictionary.from_ccd_dict()

    @classmethod
    def from_atoms(
        cls,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        **kwargs,
    ) -> "BaseBiomoleculeComplex":
        # basically ensures that chains are in alphabetical order and all constituents are single-chain.
        atoms = Biomolecule.filter_atoms(atoms, residue_dictionary, **kwargs)
        chain_ids = sorted(np.unique(atoms.chain_id))
        assert len(chain_ids) > 0, "No chains found"
        if residue_dictionary is None:
            residue_dictionary = cls.default_residue_dictionary()
        return cls(
            [
                BiomoleculeChain(
                    atoms[atoms.chain_id == chain_id], residue_dictionary, **kwargs
                )
                for chain_id in chain_ids
            ]
        )

    def get_chain(self, chain_id: str):
        return self._chains_lookup[chain_id]

    def __getitem__(self, key):
        return self.get_chain(key)

    @property
    def atoms(self):
        chain_atoms = [chain.atoms for chain in self._chains_lookup.values()]
        atoms = sum(chain_atoms, bs.AtomArray(length=0))
        for annot_name in chain_atoms[0]._annot:
            if annot_name not in atoms._annot:
                atoms.set_annotation(
                    annot_name,
                    np.concatenate([at._annot[annot_name] for at in chain_atoms]),
                )
        return atoms

    @classmethod
    def from_file(
        cls,
        file_path: str,
        file_type: Optional[str] = None,
        extra_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> T:
        return cls.from_atoms(
            load_structure(file_path, file_type=file_type, extra_fields=extra_fields),
            **kwargs,
        )
