"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""
import copy
from dataclasses import dataclass
from typing import List, Optional, Union

import biotite.structure as bs
import numpy as np

from bio_datasets.np_utils import map_categories_to_indices
from bio_datasets.protein import constants as protein_constants
from bio_datasets.structure.biomolecule import Alphabet, Biomolecule
from bio_datasets.structure.residue import ResidueDictionary

from .constants import (
    RESTYPE_ATOM37_TO_ATOM14,
    atom_types,
    residue_atoms_ordered,
    resnames,
    restype_name_to_atom14_names,
)

# from biotite.structure.filter import filter_amino_acids  includes non-standard


@dataclass
class StandardProteinDictionary(ResidueDictionary):
    """Just the 20 standard amino acids"""

    residue_names = copy.deepcopy(protein_constants.resnames)
    residue_types = copy.deepcopy(protein_constants.restypes_with_x)
    atom_types = copy.deepcopy(protein_constants.atom_types)
    residue_atoms = copy.deepcopy(protein_constants.residue_atoms_ordered)


BACKBONE_ATOMS = ["N", "CA", "C", "O"]
ALL_EXTRA_FIELDS = ["occupancy", "b_factor", "atom_id", "charge"]


ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING = get_relative_atom_indices_mapping(
    BACKBONE_ATOMS
)  # (21, 37)
STANDARD_ATOMS_BY_RESIDUE = np.asarray(
    [restype_name_to_atom14_names[resname] for resname in resnames]
)  # (21, 14) indexable atom name strings


def filter_standard_amino_acids(array):
    return np.isin(array.res_name, resnames)


def filter_backbone(array):
    """
    Filter all peptide backbone atoms of one array.

    N, CA, C and O

    Parameters
    ----------
    array : AtomArray or AtomArrayStack
        The array to be filtered.

    Returns
    -------
    filter : ndarray, dtype=bool
        This array is `True` for all indices in `array`, where an atom
        is a part of the peptide backbone.
    """

    return np.isin(array.atom_name, BACKBONE_ATOMS) & filter_standard_amino_acids(array)


def set_annotation_at_masked_atoms(
    atoms: bs.AtomArray, annot_name: str, new_annot: np.ndarray
):
    assert "mask" in atoms._annot
    atoms.add_annotation(annot_name, dtype=new_annot.dtype)
    if len(new_annot) != len(atoms):
        assert len(new_annot) == np.sum(atoms.mask)
        getattr(atoms, annot_name)[atoms.mask] = new_annot
    else:
        getattr(atoms, annot_name)[atoms.mask] = new_annot[atoms.mask]


def create_complete_atom_array_from_aa_index(
    aa_index: np.ndarray,
    chain_id: Union[str, np.ndarray],
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
    add_oxt: bool = False,
):
    """
    Populate annotations from aa_index, assuming all atoms are present.
    """
    if backbone_only:
        residue_sizes = [4] * len(aa_index)
    else:
        residue_sizes = RESIDUE_SIZES[
            aa_index
        ]  # (n_residues,) NOT (n_atoms,) -- add 1 to account for OXT
    if isinstance(chain_id, str) and not backbone_only and add_oxt:
        residue_sizes[-1] += 1  # final OXT
    else:
        if not backbone_only:
            assert len(chain_id) == len(residue_sizes)
            final_residue_in_chain = chain_id != np.concatenate(
                [chain_id[1:], ["ZZZZ"]]
            )
            if add_oxt:
                residue_sizes[final_residue_in_chain] += 1
    residue_starts = np.concatenate(
        [[0], np.cumsum(residue_sizes)[:-1]]
    )  # (n_residues,)
    new_atom_array = bs.AtomArray(length=np.sum(residue_sizes))
    residue_index = (
        np.cumsum(get_residue_starts_mask(new_atom_array, residue_starts)) - 1
    )

    full_annot_names = []
    if isinstance(chain_id, str):
        new_atom_array.set_annotation(
            "chain_id",
            np.full(len(new_atom_array), chain_id, dtype=new_atom_array.chain_id.dtype),
        )
    else:
        new_atom_array.set_annotation(
            "chain_id",
            chain_id[residue_index].astype(new_atom_array.chain_id.dtype),
        )
    full_annot_names.append("chain_id")

    # final atom in chain is OXT
    relative_atom_index = np.arange(len(new_atom_array)) - residue_starts[residue_index]
    atom_names = new_atom_array.atom_name
    if not backbone_only and add_oxt:
        oxt_mask = new_atom_array.chain_id != np.concatenate(
            [new_atom_array.chain_id[1:], ["ZZZZ"]]
        )
        atom_names[oxt_mask] = "OXT"
    else:
        oxt_mask = np.zeros(len(new_atom_array), dtype=bool)
    new_atom_array.set_annotation("aa_index", aa_index[residue_index])
    atom_names[~oxt_mask] = STANDARD_ATOMS_BY_RESIDUE[
        new_atom_array.aa_index[~oxt_mask],
        relative_atom_index[~oxt_mask],
    ]
    new_atom_array.set_annotation("atom_name", atom_names)
    new_atom_array.set_annotation(
        "res_name", np.array(resnames)[new_atom_array.aa_index]
    )
    new_atom_array.set_annotation("residue_index", residue_index)
    new_atom_array.set_annotation("res_id", residue_index + 1)
    new_atom_array.set_annotation(
        "element", np.char.array(new_atom_array.atom_name).astype("U1")
    )
    full_annot_names += ["atom_name", "aa_index", "res_name", "residue_index", "res_id"]
    if extra_fields is not None:
        for f in extra_fields:
            new_atom_array.add_annotation(
                f, dtype=float if f in ["occupancy", "b_factor"] else int
            )
    return new_atom_array, residue_starts, full_annot_names


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class ProteinMixin:

    backbone_atoms = BACKBONE_ATOMS
    alphabet: Alphabet

    def to_complex(self):
        return ProteinComplex.from_atoms(self.atoms)

    @staticmethod
    def set_atom_annotations(atoms, residue_starts):
        # convert selenium to sulphur
        mse_selenium_mask = (atoms.res_name == "MSE") & (atoms.atom_name == "SE")
        sec_selenium_mask = (atoms.res_name == "SEC") & (atoms.atom_name == "SE")
        atoms.atom_name[mse_selenium_mask] = "SD"
        atoms.atom_name[sec_selenium_mask] = "SG"
        atoms.res_name[atoms.res_name == "MSE"] = "MET"
        atoms.res_name[atoms.res_name == "SEC"] = "CYS"

        atoms.set_annotation(
            "atom_type_index", map_categories_to_indices(atoms.atom_name, atom_types)
        )
        return Biomolecule.set_atom_annotations(atoms)

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_starts: Optional[np.ndarray] = None,
        verbose: bool = False,
        backbone_only: bool = False,
        drop_oxt: bool = False,
        exclude_hydrogens: bool = True,
    ):
        """We want all atoms to be present, with nan coords if any are missing.

        We also want to ensure that atoms are in the correct order.

        We can do this in a vectorised way by calculating the expected index of each atom,
        creating a new atom array with number of atoms equal to the expected number of atoms,
        and then filling in the present atoms in the new array according to the expected index.

        This standardisation ensures that methods like `backbone_positions`,`to_atom14`,
        and `to_atom37` can be applied safely downstream.
        """
        if exclude_hydrogens:
            assert (
                "element" in atoms._annot
            ), "Elements must be present to exclude hydrogens"
            atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
        else:
            raise ValueError("Hydrogens are not supported in standardisation")
        if residue_starts is None:
            residue_starts = get_residue_starts(atoms)

        # TODO: order chains alphabetically.
        atoms = Protein.set_atom_annotations(atoms, residue_starts)
        # first we get an array of atom indices for each residue (i.e. a mapping from atom37 index to expected index
        # then we index into this array to get the expected index for each atom
        expected_relative_atom_indices = ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING[
            atoms.aa_index, atoms.atom37_index
        ]
        final_residue_in_chain = atoms.chain_id[residue_starts] != np.concatenate(
            [atoms.chain_id[residue_starts][1:], ["ZZZZ"]]
        )
        final_residue_in_chain = tile_residue_annotation_to_atoms(
            atoms, final_residue_in_chain, residue_starts
        )
        oxt_mask = (atoms.atom_name == "OXT") & final_residue_in_chain
        if np.any(oxt_mask) and not drop_oxt:
            expected_relative_atom_indices[oxt_mask] = (
                ATOM37_TO_RELATIVE_ATOM_INDEX_MAPPING[atoms.aa_index[oxt_mask]].max()
                + 1
            )
        elif drop_oxt:
            atoms = atoms[~oxt_mask]
            expected_relative_atom_indices = expected_relative_atom_indices[~oxt_mask]
            oxt_mask = np.zeros(len(atoms), dtype=bool)
        unexpected_atom_mask = expected_relative_atom_indices == -100
        if np.any(unexpected_atom_mask & (atoms.res_name != "UNK")):
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
        unexpected_unk_atom_mask = unexpected_atom_mask & (atoms.res_name == "UNK")
        atoms = atoms[~unexpected_unk_atom_mask]
        expected_relative_atom_indices = expected_relative_atom_indices[
            ~unexpected_unk_atom_mask
        ]
        oxt_mask = oxt_mask[~unexpected_unk_atom_mask]
        residue_starts = get_residue_starts(atoms)

        (
            new_atom_array,
            full_residue_starts,
            full_annot_names,
        ) = create_complete_atom_array_from_aa_index(
            atoms.aa_index[residue_starts],
            atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
            add_oxt=np.any(oxt_mask),
        )
        existing_atom_indices_in_full_array = (
            full_residue_starts[atoms.residue_index] + expected_relative_atom_indices
        ).astype(int)

        for annot_name, annot in atoms._annot.items():
            if annot_name in ["atom37_index", "mask"] or annot_name in full_annot_names:
                continue
            getattr(new_atom_array, annot_name)[
                existing_atom_indices_in_full_array
            ] = annot.astype(new_atom_array._annot[annot_name].dtype)

        # set_annotation vs setattr: set_annotation adds to annot and verifies size
        new_atom_array.coord[existing_atom_indices_in_full_array] = atoms.coord
        # if we can create a res start index for each atom, we can assign the value based on that...
        assert len(full_residue_starts) == len(
            residue_starts
        ), f"Full residue starts: {full_residue_starts} and residue starts: {residue_starts} do not match"
        new_atom_array.set_annotation(
            "res_id",
            atoms.res_id[residue_starts][new_atom_array.residue_index].astype(
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
            atoms.ins_code[residue_starts][new_atom_array.residue_index].astype(
                new_atom_array.ins_code.dtype
            ),
        )

        new_atom_array.set_annotation(
            "atom37_index",
            map_categories_to_indices(new_atom_array.atom_name, atom_types),
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
            new_atom_array = new_atom_array[filter_backbone(new_atom_array)]
            full_residue_starts = get_residue_starts(new_atom_array)
        return new_atom_array, full_residue_starts

    @property
    def aa_index(self):
        return self.atoms["aa_index"][self._residue_starts]

    def beta_carbon_coords(self) -> np.ndarray:
        has_beta_carbon = self.atoms.res_name != "GLY"
        beta_carbon_coords = np.zeros((self.num_residues, 3), dtype=np.float32)
        beta_carbon_coords[has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[has_beta_carbon] + 4
        ]
        beta_carbon_coords[~has_beta_carbon[self._residue_starts]] = self.atoms.coord[
            self._residue_starts[~has_beta_carbon] + 1
        ]  # ca for gly
        return beta_carbon_coords

    def backbone_coords(self, atom_names: Optional[List[str]] = None) -> np.ndarray:
        assert all(
            [atom in BACKBONE_ATOMS + ["CB"] for atom in atom_names]
        ), f"Invalid entries in atom names: {atom_names}"
        coords = super().backbone_coords([at for at in atom_names if at != "CB"])
        if "CB" in atom_names:
            cb_index = atom_names.index("CB")
            coords_with_cb = np.zeros(
                (len(coords), len(atom_names), 3), dtype=np.float32
            )
            coords_with_cb[:, cb_index] = self.beta_carbon_coords()
            non_cb_indices = [atom_names.index(at) for at in atom_names if at != "CB"]
            coords_with_cb[:, non_cb_indices] = coords
            return coords_with_cb
        return coords

    def contacts(self, atom_name: str = "CA", threshold: float = 8.0) -> np.ndarray:
        return super().contacts(atom_name=atom_name, threshold=threshold)

    def atom14_coords(self) -> np.ndarray:
        atom14_coords = np.full((len(self.num_residues), 14, 3), np.nan)
        atom14_index = RESTYPE_ATOM37_TO_ATOM14[
            self.atoms.residue_index, self.atoms.atom37_index
        ]
        atom14_coords[self.atoms.residue_index, atom14_index] = self.atoms.coord
        return atom14_coords

    def atom37_coords(self) -> np.ndarray:
        # since we have standardised the atoms we can just return standardised atom37 indices for each residue
        atom37_coords = np.full((len(self.num_residues), len(atom_types), 3), np.nan)
        atom37_coords[
            self.atoms.residue_index, self.atoms.atom37_index
        ] = self.atoms.coord
        return atom37_coords

    def get_chain(self, chain_id: str):
        chain_filter = self.atoms.chain_id == chain_id
        return ProteinChain(self.atoms[chain_filter].copy())

    def sequence(self) -> str:
        return np.array()[self.atoms.aa_index[self._residue_starts]]


class ProteinChain(Protein):

    """A single protein chain."""

    def __init__(
        self,
        atoms: bs.AtomArray,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        """
        Parameters
        ----------
        atoms : AtomArray
            The atoms of the protein.
        """
        super().__init__(atoms, verbose=verbose, backbone_only=backbone_only)
        assert (
            np.unique(atoms.chain_id).size == 1
        ), "Only a single chain is supported by `Protein` objects. Consider using a different feature type."

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]


class ProteinComplex:
    """A protein complex."""

    def __init__(self, proteins: List[ProteinChain]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
