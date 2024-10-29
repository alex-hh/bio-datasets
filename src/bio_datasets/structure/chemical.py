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
