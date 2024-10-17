from typing import List, Tuple

import biotite.structure as bs
import numpy as np

from .protein import Protein


class ProteinComplex(Protein):
    """A protein complex."""

    def __init__(self, proteins: List[Protein]):
        self._chain_ids = [prot.chain_id for prot in proteins]
        self._proteins_lookup = {prot.chain_id: prot for prot in proteins}
        self.atoms = sum([prot.atoms for prot in proteins], bs.AtomArray())

    @classmethod
    def from_atoms(cls, atoms: bs.AtomArray) -> "ProteinComplex":
        chain_ids = np.unique(atoms.chain_id)
        return cls(
            [
                Protein.from_atoms(atoms[atoms.chain_id == chain_id])
                for chain_id in chain_ids
            ]
        )

    @property
    def get_chain(self, chain_id: str) -> Protein:
        return self._proteins_lookup[chain_id]

    def interface(
        self,
        atom_name: str = "CA",
        chain_pair: Tuple[str, str] = None,
        threshold: float = 8.0,
    ) -> "ProteinComplex":
        if chain_pair is None:
            if len(self._chain_ids) != 2:
                raise ValueError(
                    "chain_pair must be specified for non-binary complexes"
                )
            chain_pair = (self._chain_ids[0], self._chain_ids[1])
        raise NotImplementedError("Not implemented yet")
