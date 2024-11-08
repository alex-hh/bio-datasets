from typing import List

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_from_ccd

from bio_datasets.structure.biomolecule import Biomolecule
from bio_datasets.structure.residue import ResidueDictionary, get_residue_starts_mask


def get_smiles_from_ccd(res_name: str, program: str = "CACTVS"):
    descriptor_cat = get_from_ccd("pdbx_chem_comp_descriptor", res_name)
    canonical_smiles_mask = (
        descriptor_cat["type"].as_array(str) == "SMILES_CANONICAL"
    ) & (descriptor_cat["program"].as_array(str) == program)
    assert (
        canonical_smiles_mask.sum() == 1
    ), "Expected exactly one canonical smiles entry"
    smiles = descriptor_cat["descriptor"].as_array()[canonical_smiles_mask][0]
    return smiles


class SmallMolecule:
    """A small molecule. This is treated as a single residue in the PDB.

    Small molecules in the PDB are cross-referenced with the chemical component dictionary (CCD).
    The three letter 'res_name' is a unique identifier for a chemical component dictionary entry.
    The CCD maps to SMILES and InChI strings, as well as idealised 3D coordinates.

    The true 3D coordinates are still the best representation - and already implicitly contain all bond information.

    Refs:
    CCD: https://www.wwpdb.org/data/ccd
    General info on small molecules in the PDB: https://www.rcsb.org/docs/general-help/ligand-structure-quality-in-pdb-structures
    """

    def __init__(
        self,
        atoms: bs.AtomArray,
        keep_hydrogens: bool = False,
        verbose: bool = False,
        keep_non_hetero: bool = False,
    ):
        atoms = self.filter_atoms(
            atoms, keep_non_hetero=keep_non_hetero, keep_hydrogens=keep_hydrogens
        )
        atoms = self.standardise_atoms(atoms, verbose=verbose)
        self.atoms = atoms
        self._standardised = True
        assert (
            len(np.unique(atoms.res_id)) == 1
        ), "Small molecules must be a single residue"

    def __repr__(self):
        return super().__repr__() + f": {self.res_name}"

    @property
    def res_name(self):
        return self.atoms.res_name[0]

    @property
    def chain_id(self):
        return self.atoms.chain_id[0]

    @property
    def bonds(self):
        return self.atoms.bonds

    @staticmethod
    def standardise_atoms(atoms, verbose: bool = False):
        assert np.all(atoms.res_name[0] == atoms.res_name), "Expected single residue"
        residue_dictionary = ResidueDictionary.from_ccd_dict(
            residue_names=[atoms.res_name[0]]
        )
        atoms = Biomolecule.standardise_atoms(
            atoms,
            residue_dictionary=residue_dictionary,
            backbone_only=False,
            verbose=verbose,
        )
        atoms.set_annotation(
            "hetero", np.ones(len(atoms), dtype=bool)
        )  # N.B. this may sometimes be misleading - e.g. if we convert a protein to a small molecule
        atoms.bonds = bs.connect_via_residue_names(atoms, inter_residue=False)
        return atoms

    def filter_atoms(
        self, atoms, keep_non_hetero: bool = False, keep_hydrogens: bool = False
    ):
        if not keep_hydrogens:
            atoms = atoms[(atoms.element != "H") & (atoms.element != "D")]
        if not keep_non_hetero:
            atoms = atoms[atoms.hetero]
        return atoms

    def adjacency_matrix(self):
        return self.bonds.adjacency_matrix()

    def bond_type_matrix(self):
        # convert sparse bond types to dense matrix; missing values are -1 (no bond)
        return self.bonds.bond_type_matrix()

    def to_smiles(self):
        return get_smiles_from_ccd(self.res_name)

    def to_rdkit(self):
        raise NotImplementedError()


def to_small_molecules(atoms: bs.AtomArray) -> List[SmallMolecule]:
    """Break all polymer bonds, converting to small molecules ('residue gas').

    TODO: check whether bond lookups are efficient / cached.
    """
    molecules = []
    if "res_index" in atoms._annot:
        res_indices = np.unique(atoms.res_index)
    else:
        atoms.set_annotation(
            "res_index",
            np.cumsum(get_residue_starts_mask(atoms)) - 1,
        )
        res_indices = np.unique(atoms.res_index)
    for res_idx in res_indices:  # residue index should be sorted; res_id may not be
        res_atoms = atoms[atoms.res_index == res_idx]
        molecules.append(SmallMolecule(res_atoms, keep_non_hetero=True))
    return molecules
