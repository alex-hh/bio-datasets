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

from bio_datasets.structure.biomolecule import (
    BaseBiomoleculeComplex,
    Biomolecule,
    BiomoleculeChain,
)
from bio_datasets.structure.protein import constants as protein_constants
from bio_datasets.structure.residue import ResidueDictionary, register_preset_res_dict

from .constants import RESTYPE_ATOM37_TO_ATOM14, atom_types

# TODO: RESTYPE ATOM37 TO ATOM14 can be derived from ResidueDictionary (atom14_coords)


register_preset_res_dict(
    "protein",
    residue_names=copy.deepcopy(protein_constants.resnames),
    residue_types=copy.deepcopy(protein_constants.restypes_with_x),
    atom_types=copy.deepcopy(protein_constants.atom_types),
    residue_atoms=copy.deepcopy(protein_constants.residue_atoms),
    residue_elements=copy.deepcopy(protein_constants.residue_elements),
    backbone_atoms=["N", "CA", "C", "O"],
    unknown_residue_name="UNK",
    conversions=[
        {"residue": "MSE", "to_residue": "MET", "atom_swaps": [("SE", "SD")]},
        {"residue": "SEC", "to_residue": "CYS", "atom_swaps": [("SE", "SG")]},
    ],
)


@dataclass
class ProteinDictionary(ResidueDictionary):
    """Defaults configure a dictionary with just the 20 standard amino acids"""

    drop_oxt: bool = False

    def _check_atom14_compatible(self):
        return all(len(res_ats) <= 14 for res_ats in self.residue_atoms.values())

    def _check_atom37_compatible(self):
        assert self.atom_types is not None
        return all(
            at in protein_constants.atom_types
            for res_ats in self.residue_atoms.values()
            for at in res_ats
        )

    def __post_init__(self):
        self._atom37_compatible = self._check_atom37_compatible()
        self._atom14_compatible = self._check_atom14_compatible()
        return super().__post_init__()

    @property
    def atom37_compatible(self):
        return self._atom37_compatible

    @property
    def atom14_compatible(self):
        return self._atom14_compatible

    def get_residue_sizes(self, restype_index, chain_id: Union[str, np.ndarray]):
        # should only be called with single chain
        if isinstance(chain_id, np.ndarray):
            assert len(np.unique(chain_id)) == 1
        residue_sizes = self.residue_sizes[restype_index].copy()
        if not self.drop_oxt:
            residue_sizes[-1] += 1  # add oxt
        return residue_sizes

    def get_expected_relative_atom_indices(self, restype_index, atomtype_index):
        assert self.atom_types is not None
        expected_relative_atom_indices = np.zeros(restype_index.shape[0]).astype(int)
        oxt_id = self.atom_types.index("OXT")
        oxt_mask = atomtype_index == oxt_id
        residues_with_oxt_sizes = self.residue_sizes[restype_index[oxt_mask]]
        expected_relative_atom_indices[
            ~oxt_mask
        ] = super().get_expected_relative_atom_indices(
            restype_index[~oxt_mask], atomtype_index[~oxt_mask]
        )
        expected_relative_atom_indices[oxt_mask] = residues_with_oxt_sizes
        return expected_relative_atom_indices

    def get_atom_names(
        self,
        restype_index: np.ndarray,
        relative_atom_index: np.ndarray,
        chain_id: np.ndarray,
    ):
        assert len(np.unique(chain_id)) == 1
        final_residue_mask = restype_index == restype_index[-1]
        oxt_mask = final_residue_mask & (
            relative_atom_index == self.residue_sizes[restype_index]
        )
        atom_names = np.full((len(restype_index)), "", dtype="U6")
        atom_names[~oxt_mask] = self.standard_atoms_by_residue[
            restype_index[~oxt_mask],
            relative_atom_index[~oxt_mask],
        ]
        atom_names[oxt_mask] = "OXT"
        return atom_names


def filter_backbone(array, residue_dictionary):
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

    return np.isin(array.atom_name, residue_dictionary.backbone_atoms) & np.isin(
        array.res_name, residue_dictionary.residue_names
    )


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


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class ProteinMixin:
    def to_complex(self):
        return ProteinComplex.from_atoms(self.atoms)

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_dictionary: ProteinDictionary,
        verbose: bool = False,
        backbone_only: bool = False,
    ):
        """We want all atoms to be present, with nan coords if any are missing.

        We also want to ensure that atoms are in the correct order.

        We can do this in a vectorised way by calculating the expected index of each atom,
        creating a new atom array with number of atoms equal to the expected number of atoms,
        and then filling in the present atoms in the new array according to the expected index.

        This standardisation ensures that methods like `backbone_positions`,`to_atom14`,
        and `to_atom37` can be applied safely downstream.
        """
        if residue_dictionary.drop_oxt:
            atoms = atoms[atoms.atom_name != "OXT"]
        return Biomolecule.standardise_atoms(
            atoms,
            residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
        )

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
            [
                atom in self.residue_dictionary.backbone_atoms + ["CB"]
                for atom in atom_names
            ]
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
        assert (
            self.residue_dictionary.atom14_compatible
            and self.residue_dictionary.atom37_compatible
        ), "Atom14 representation assumes use of standard amino acid dictionary"
        atom14_coords = np.full((len(self.num_residues), 14, 3), np.nan)
        atom14_index = RESTYPE_ATOM37_TO_ATOM14[
            self.atoms.residue_index, self.atoms.atom37_index
        ]
        atom14_coords[self.atoms.residue_index, atom14_index] = self.atoms.coord
        return atom14_coords

    def atom37_coords(self) -> np.ndarray:
        assert (
            self.residue_dictionary.atom37_compatible
        ), "Atom37 representation assumes use of standard amino acid dictionary"
        # since we have standardised the atoms we can just return standardised atom37 indices for each residue
        atom37_coords = np.full((len(self.num_residues), len(atom_types), 3), np.nan)
        atom37_coords[
            self.atoms.residue_index, self.atoms.atom37_index
        ] = self.atoms.coord
        return atom37_coords


class ProteinChain(ProteinMixin, BiomoleculeChain):

    """A single protein chain."""

    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        verbose: bool = False,
        backbone_only: bool = False,
        keep_hydrogens: bool = False,
        keep_oxt: bool = False,
        replace_unexpected_with_unknown: bool = False,
        raise_error_on_unexpected: bool = False,
    ):
        if residue_dictionary is None:
            residue_dictionary = ProteinDictionary(drop_oxt=not keep_oxt)
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            keep_hydrogens=keep_hydrogens,
            keep_oxt=keep_oxt,
            replace_unexpected_with_unknown=replace_unexpected_with_unknown,
            raise_error_on_unexpected=raise_error_on_unexpected,
        )

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]


class ProteinComplex(ProteinMixin, BaseBiomoleculeComplex):
    """A protein complex."""

    @classmethod
    def from_atoms(cls, atoms: bs.AtomArray, **kwargs) -> "ProteinComplex":
        # basically ensures that chains are in alphabetical order and all constituents are single-chain.
        chain_ids = sorted(np.unique(atoms.chain_id))
        return cls(
            [
                ProteinChain(atoms[atoms.chain_id == chain_id], **kwargs)
                for chain_id in chain_ids
            ]
        )
