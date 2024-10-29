"""Defines protein objects that are lightweight wrappers around Biotite's AtomArray and AtomArrayStack.

This library is not intended to be a general-purpose library for protein structure analysis.
We simply wrap Biotite's AtomArray and AtomArrayStack to offer a few convenience methods
for dealing with protein structures in an ML context.
"""
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

import biotite.structure as bs
import numpy as np

# from biotite.structure.filter import filter_amino_acids  includes non-standard
from biotite.structure.residues import get_residue_starts

from bio_datasets.structure.biomolecule import (
    ALL_EXTRA_FIELDS,
    Biomolecule,
    BiomoleculeChain,
    BiomoleculeComplex,
    create_complete_atom_array_from_restype_index,
)
from bio_datasets.structure.protein import constants as protein_constants
from bio_datasets.structure.residue import (
    ResidueDictionary,
    tile_residue_annotation_to_atoms,
)

from .constants import RESTYPE_ATOM37_TO_ATOM14, atom_types

# TODO: RESTYPE ATOM37 TO ATOM14 can be derived from ResidueDictionary


@dataclass
class ProteinDictionary(ResidueDictionary):
    """Defaults configure a dictionary with just the 20 standard amino acids"""

    # TODO: these are actually all constants
    residue_names = copy.deepcopy(protein_constants.resnames)
    residue_types = copy.deepcopy(protein_constants.restypes_with_x)
    atom_types = copy.deepcopy(protein_constants.atom_types)
    residue_atoms = copy.deepcopy(protein_constants.residue_atoms_ordered)
    backbone_atoms = ["N", "CA", "C", "O"]
    unknown_residue_name = "UNK"
    conversions = [
        {"residue": "MSE", "to_residue": "MET", "atom_swaps": [("SE", "SD")]},
        {"residue": "SEC", "to_residue": "CYS", "atom_swaps": [("SE", "SG")]},
    ]

    def _check_atom14_compatible(self):
        return all(len(res_ats) <= 14 for res_ats in self.residue_atoms.values())

    def _check_atom37_compatible(self):
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

    def to_terminal_dictionary(self):
        """for standardising C-terminal residues, we need to add OXT to the list of atoms"""
        return ResidueDictionary(
            residue_names=self.residue_names,
            residue_types=self.residue_types,
            atom_types=self.atom_types,
            residue_atoms={k: v + ["OXT"] for k, v in self.residue_atoms.items()},
            backbone_atoms=self.backbone_atoms,
            unknown_residue_name=self.unknown_residue_name,
            conversions=self.conversions,
        )


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


def create_protein_atom_array_from_restype_index(
    restype_index: np.ndarray,
    residue_dictionary: ResidueDictionary,
    chain_id: str,
    extra_fields: Optional[List[str]] = None,
    backbone_only: bool = False,
    add_oxt: bool = False,
):
    """
    Populate annotations from aa_index, assuming all atoms are present, optionally adding OXT atoms.
    """
    if not add_oxt:
        return create_complete_atom_array_from_restype_index(
            restype_index=restype_index,
            residue_dictionary=residue_dictionary,
            chain_id=chain_id,
            extra_fields=extra_fields,
            backbone_only=backbone_only,
        )
    if isinstance(chain_id, np.ndarray):
        unique_chain_ids = np.unique(chain_id)
        chain_atom_arrays = []
        chain_residue_starts = []
        residue_starts_offset = 0
        for chain_id in unique_chain_ids:
            (
                atom_array,
                residue_starts,
                full_annot_names,
            ) = create_protein_atom_array_from_restype_index(
                restype_index=restype_index,
                residue_dictionary=residue_dictionary,
                chain_id=chain_id,
                extra_fields=extra_fields,
                backbone_only=backbone_only,
                add_oxt=True,
            )
            residue_starts_offset += len(atom_array)
            chain_atom_arrays.append(atom_array)
            chain_residue_starts.append(residue_starts + residue_starts_offset)
        return (
            sum(chain_atom_arrays, bs.AtomArray(length=0)),
            np.concatenate(chain_residue_starts),
            full_annot_names,
        )
    else:
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
        new_atom = new_atom_array[-1].copy()
        # TODO: test this
        if new_atom.res_name != "UNK":
            new_atom.atom_name = "OXT"  # other annotations shared with final atom
            new_atom_array.append(new_atom)

        return new_atom_array, residue_starts, full_annot_names


# TODO: add support for batched application of these functions (i.e. to multiple proteins at once)
class ProteinMixin:
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
    ):
        """We want all atoms to be present, with nan coords if any are missing.

        We also want to ensure that atoms are in the correct order.

        We can do this in a vectorised way by calculating the expected index of each atom,
        creating a new atom array with number of atoms equal to the expected number of atoms,
        and then filling in the present atoms in the new array according to the expected index.

        This standardisation ensures that methods like `backbone_positions`,`to_atom14`,
        and `to_atom37` can be applied safely downstream.
        """
        if drop_oxt:
            atoms = atoms[atoms.atom_name != "OXT"]
            return Biomolecule.standardise_atoms(
                atoms,
                residue_dictionary,
                residue_starts=residue_starts,
                verbose=verbose,
                backbone_only=backbone_only,
            )
        atoms = atoms[~np.isin(atoms.element, ["H", "D"])]
        residue_starts = get_residue_starts(atoms)
        # create a new atom array with oxts added - specific to protein standardisation
        new_atom_array = create_protein_atom_array_from_restype_index(
            atoms.restype_index[residue_starts],
            residue_dictionary,
            chain_id=atoms.chain_id,
            extra_fields=[f for f in ALL_EXTRA_FIELDS if f in atoms._annot],
            add_oxt=True,
        )
        atoms = Biomolecule.standardise_atoms(
            atoms,
            new_atom_array=new_atom_array,
            residue_dictionary=residue_dictionary.to_terminal_dictionary(),  # doesn't affect the size of the array - but allows oxts to be present
            verbose=verbose,
            backbone_only=backbone_only,
        )
        # check that oxts are always at final residue in chain
        final_residue_in_chain = atoms.chain_id[residue_starts] != np.concatenate(
            [atoms.chain_id[residue_starts][1:], ["ZZZ"]]  # ZZZ an arbitrary chain ID
        )
        final_residue_in_chain = tile_residue_annotation_to_atoms(
            atoms, final_residue_in_chain, residue_starts
        )
        oxt_mask = atoms.atom_name == "OXT"
        assert not np.any(
            oxt_mask[~final_residue_in_chain]
        ), "OXTs must be at final residue in chain"
        return atoms

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

    def get_chain(self, chain_id: str):
        chain_filter = self.atoms.chain_id == chain_id
        return ProteinChain(self.atoms[chain_filter].copy())


class ProteinChain(ProteinMixin, BiomoleculeChain):

    """A single protein chain."""

    def __init__(
        self,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        verbose: bool = False,
        backbone_only: bool = False,
        exclude_hydrogens: bool = True,
        standardisation_kwargs: Optional[Dict] = None,
    ):
        if residue_dictionary is None:
            residue_dictionary = ProteinDictionary()
        super().__init__(
            atoms,
            residue_dictionary=residue_dictionary,
            verbose=verbose,
            backbone_only=backbone_only,
            exclude_hydrogens=exclude_hydrogens,
            standardisation_kwargs=standardisation_kwargs,
        )

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]


class ProteinComplex(ProteinMixin, BiomoleculeComplex):
    """A protein complex."""

    def __init__(self, proteins: List[ProteinChain]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
