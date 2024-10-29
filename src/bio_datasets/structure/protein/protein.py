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

# from biotite.structure.filter import filter_amino_acids  includes non-standard
from biotite.structure.residues import get_residue_starts

from bio_datasets.structure.biomolecule import (
    ALL_EXTRA_FIELDS,
    BiomoleculeChain,
    BiomoleculeComplex,
)
from bio_datasets.structure.protein import constants as protein_constants
from bio_datasets.structure.residue import (
    ResidueDictionary,
    tile_residue_annotation_to_atoms,
)

from .constants import RESTYPE_ATOM37_TO_ATOM14, atom_types


@dataclass
class StandardProteinDictionary(ResidueDictionary):
    """Just the 20 standard amino acids"""

    residue_names = copy.deepcopy(protein_constants.resnames)
    residue_types = copy.deepcopy(protein_constants.restypes_with_x)
    atom_types = copy.deepcopy(protein_constants.atom_types)
    residue_atoms = copy.deepcopy(protein_constants.residue_atoms_ordered)
    backbone_atoms = ["N", "CA", "C", "O"]
    unknown_residue_name = "UNK"
    # convert selenium to sulphur
    conversions = [
        {"residue": "MSE", "to_residue": "MET", "atom_swaps": [("SE", "SD")]},
        {"residue": "SEC", "to_residue": "CYS", "atom_swaps": [("SE", "SG")]},
    ]


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


def create_complete_atom_array_from_restype_index(
    restype_index: np.ndarray,
    residue_dictionary: ResidueDictionary,
    chain_id: Union[str, np.ndarray],
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
    add_oxt: bool = False,
):
    """
    Populate annotations from aa_index, assuming all atoms are present.
    """

    (
        new_atom_array,
        residue_starts,
        full_annot_names,
    ) = create_complete_atom_array_from_restype_index(
        restype_index=restype_index,
        residue_dictionary=residue_dictionary,
        chain_id=chain_id,
        extra_fields=extra_fields,
        backbone_only=backbone_only,
    )
    if add_oxt:
        new_atom = new_atom_array[-1].copy()
        # TODO: test this
        if new_atom.res_name != "UNK":
            new_atom.atom_name = "OXT"  # other annotations shared with final atom
            new_atom_array.append(new_atom)

    return new_atom_array, residue_starts, full_annot_names


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class ProteinMixin:

    residue_dictionary = (
        StandardProteinDictionary  # override by subclassing if necessary
    )

    def to_complex(self):
        return ProteinComplex.from_atoms(self.atoms)

    @staticmethod
    def standardise_atoms(
        atoms,
        residue_dictionary: ResidueDictionary,
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
        # first we get an array of atom indices for each residue (i.e. a mapping from atom37 index to expected index
        # then we index into this array to get the expected index for each atom
        expected_relative_atom_indices = (
            residue_dictionary.relative_atom_indices_mapping[
                atoms.aa_index, atoms.atom37_index
            ]
        )
        final_residue_in_chain = atoms.chain_id[residue_starts] != np.concatenate(
            [atoms.chain_id[residue_starts][1:], ["ZZZZ"]]
        )
        final_residue_in_chain = tile_residue_annotation_to_atoms(
            atoms, final_residue_in_chain, residue_starts
        )
        oxt_mask = (atoms.atom_name == "OXT") & final_residue_in_chain
        if np.any(oxt_mask) and not drop_oxt:
            expected_relative_atom_indices[oxt_mask] = (
                residue_dictionary.relative_atom_indices_mapping[
                    atoms.aa_index[oxt_mask]
                ].max()
                + 1
            )
        elif drop_oxt:
            atoms = atoms[~oxt_mask]
            expected_relative_atom_indices = expected_relative_atom_indices[~oxt_mask]
            oxt_mask = np.zeros(len(atoms), dtype=bool)

        unexpected_atom_mask = expected_relative_atom_indices == -100
        # for unk residues, we just drop any e.g. sidechain atoms without raising an exception
        unexpected_unk_atom_mask = unexpected_atom_mask & (
            atoms.res_name == residue_dictionary.unknown_residue_name
        )
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
        ) = create_complete_atom_array_from_restype_index(
            atoms.restype_index[residue_starts],
            atoms.chain_id[residue_starts],
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
            add_oxt=np.any(oxt_mask),
        )
        return self.standardise_new_atom_array()

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
        # TODO: replace this with standard residue dictionary methods
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


class ProteinChain(ProteinMixin, BiomoleculeChain):

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


class ProteinComplex(ProteinMixin, BiomoleculeComplex):
    """A protein complex."""

    def __init__(self, proteins: List[ProteinChain]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
