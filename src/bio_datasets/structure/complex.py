import numpy as np
from biotite import structure as bs
from typing import List, Optional, Tuple, Union
from .biomolecule import Biomolecule, BiomoleculeChain, T
from .protein import ProteinChain, ProteinDictionary, ProteinComplex
from .residue import ResidueDictionary, CHEM_COMPONENT_CATEGORIES


class BiomoleculeComplex(Biomolecule):
    def __init__(self, chains: List[BiomoleculeChain]):
        self._chain_ids = [mol.chain_id for mol in chains]
        self._chains_lookup = {mol.chain_id: mol for mol in chains}

    @classmethod
    def from_atoms(cls, atoms: bs.AtomArray, residue_dictionary: Optional[ResidueDictionary] = None):
        # TODO: check chain-based stuff works with pdb files as well as cif files - are chain ids null sometimes for hetatms?
        chains = []
        chain_categories = [CHEM_COMPONENT_CATEGORIES[r] for r in atoms.res_name[atoms.chain_id == chain_id]]
        if len(set(chain_categories)) == 1:
            category = chain_categories[0]
            if category == "protein":
                return ProteinComplex.from_atoms(
                    atoms,
                    residue_dictionary=ProteinDictionary.from_ccd() if residue_dictionary is None else residue_dictionary
                )
            else:
                raise ValueError(f"Unsupported chain category: {category}")
        else:
            print(f"Warning: found multiple categories in chain {chain_id}: {chain_categories}")
        for chain_id, chain_category in zip(np.unique(atoms.chain_id), chain_categories):
            # TODO: auto-ccd residue dictionary
            if len(np.unique(chain_category)) > 1:
                # raise ValueError(f"Found multiple categories in chain {chain_id}: {chain_categories}")
                print(f"Warning: found multiple categories in chain {chain_id}: {chain_categories}")
                chains.append(
                    BiomoleculeChain(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_ccd() if residue_dictionary is None else residue_dictionary
                    )
                )
            elif chain_category[0] == "protein":
                chains.append(
                    ProteinChain(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ProteinDictionary.from_ccd() if residue_dictionary is None else residue_dictionary
                    )
                )
            elif chain_category[0] in ["dna", "rna", "saccharide", "chemical"]:
                chains.append(
                    BiomoleculeChain(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_ccd() if residue_dictionary is None else residue_dictionary
                    )
                )
            else:
                raise ValueError(f"Unsupported chain category: {chain_categories[0]}")
        return cls(chains)

    @property
    def atoms(self):
        return sum(
            [chain.atoms for chain in self._chains_lookup.values()],
            bs.AtomArray(length=0),
        )

    @property
    def chain_ids(self):
        return self._chain_ids

    @property
    def chains(self):
        return [(chain_id, self.get_chain(chain_id)) for chain_id in self.chain_ids]

    def get_chain(self, chain_id: str) -> "BiomoleculeChain":
        return self._chains_lookup[chain_id]

    def interface(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        threshold: float = 10.0,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> T:
        distances = self.interface_distances(
            atom_names=atom_names, chain_pair=chain_pair, nan_fill=nan_fill
        )
        interface_mask = distances < threshold
        return self.__class__.from_atoms(self.atoms[interface_mask])

    def interface_distances(
        self,
        atom_names: Union[str, List[str]] = "CA",
        chain_pair: Optional[Tuple[str, str]] = None,
        nan_fill: Optional[Union[float, str]] = None,
    ) -> np.ndarray:
        if chain_pair is None:
            if len(self._chain_ids) != 2:
                raise ValueError(
                    "chain_pair must be specified for non-binary complexes"
                )
            chain_pair = (self._chain_ids[0], self._chain_ids[1])
        residue_mask_from = self.atoms.chain_id == chain_pair[0]
        residue_mask_to = self.atoms.chain_id == chain_pair[1]
        return self.distances(
            atom_names=atom_names,
            residue_mask_from=residue_mask_from,
            residue_mask_to=residue_mask_to,
            nan_fill=nan_fill,
        )
