from typing import Generic, TypeVar

import numpy as np
from biotite import structure as bs
from biotite.structure.info.ccd import get_from_ccd

T = TypeVar("T", bound="Molecule")


class Molecule(Generic[T]):
    pass


class SmallMolecule(Molecule):
    """A small molecule. This is treated as a single residue in the PDB.

    Small molecules in the PDB are cross-referenced with the chemical component dictionary (CCD).
    The three letter 'res_name' is a unique identifier for a chemical component dictionary entry.
    The CCD maps to SMILES and InChI strings, as well as idealised 3D coordinates.

    The true 3D coordinates are still the best representation - and already implicitly contain all bond information.

    Refs:
    CCD: https://www.wwpdb.org/data/ccd
    General info on small molecules in the PDB: https://www.rcsb.org/docs/general-help/ligand-structure-quality-in-pdb-structures

    We want to provide the option to load this information into rdkit format, e.g. for graph representations.

    TODO: decide whether to load residue dictionary from CCD...
    """

    def __init__(
        self,
        atoms: bs.AtomArray,
        verbose: bool = False,
    ):
        atoms = self.filter_atoms(atoms)
        self.atoms = self.standardise_atoms(
            atoms,
            verbose=verbose,
        )
        self._standardised = True
        assert np.unique(atoms.res_id) == 1, "Small molecules must be a single residue"

    @property
    def res_name(self):
        return self.atoms.res_name[0]

    def filter_atoms(self, atoms):
        return atoms[atoms.hetero]

    def standardise_atoms(self, atoms, verbose: bool = False):
        # TODO: use CCD to standardise atoms
        standard = get_from_ccd("chem_comp_atom", self.res_name)
        raise NotImplementedError()
        return atoms, None

    def adjacency_matrix(self):
        raise NotImplementedError()

    def dense_bond_types(self):
        # convert sparse bond types to dense matrix; missing values are 0 (no bond)
        # add_bonds? biotite...
        raise NotImplementedError()

    def to_smiles(self):
        smiles = get_from_ccd("pdbx_chem_comp_descriptor", self.res_name)
        raise NotImplementedError()

    def to_rdkit(self):
        raise NotImplementedError()
