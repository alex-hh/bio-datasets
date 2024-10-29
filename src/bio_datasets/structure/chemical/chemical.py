from typing import Dict, Generic, Optional, TypeVar

from biotite import structure as bs

T = TypeVar("T", bound="Molecule")


class Molecule(Generic[T]):
    def __init__(
        self,
        atoms: bs.AtomArray,
        verbose: bool = False,
        standardisation_kwargs: Optional[Dict] = None,
        backbone_only: bool = False,
    ):
        self.backbone_only = backbone_only
        atoms = self.filter_atoms(atoms)
        self.atoms, self._residue_starts = self.standardise_atoms(
            atoms, verbose=verbose, **standardisation_kwargs
        )
        self._standardised = True

    def filter_atoms(self, atoms: bs.AtomArray) -> bs.AtomArray:
        raise NotImplementedError()

    def standardise_atoms(self, atoms: bs.AtomArray) -> bs.AtomArray:
        raise NotImplementedError()


class SmallMolecule:
    """A small molecule. This is treated as a single residue in the PDB.

    Small molecules in the PDB are cross-referenced with the chemical component dictionary (CCD).
    The three letter 'res_name' is a unique identifier for a chemical component dictionary entry.
    The CCD maps to SMILES and InChI strings, as well as idealised 3D coordinates.

    https://www.wwpdb.org/data/ccd

    We want to provide the option to load this information into rdkit format, e.g. for graph representations.

    TODO: decide whether to load residue dictionary from CCD...
    """

    def filter_atoms(self, atoms):
        return atoms[atoms.hetero]

    def standardise_atoms(self, atoms):
        return atoms, None
