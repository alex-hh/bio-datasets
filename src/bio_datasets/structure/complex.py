from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from biotite import structure as bs

from .biomolecule import BaseBiomoleculeComplex, BiomoleculeChain, T
from .chemical import SmallMolecule
from .nucleic import DNAChain, RNAChain
from .protein import ProteinChain, ProteinDictionary
from .residue import CHEM_COMPONENT_CATEGORIES, ResidueDictionary


class BiomoleculeComplex(BaseBiomoleculeComplex):
    """A collection of biomolecules, each represented by its own class type.

    As well as providing access to molecule type-specific methods, this supports
    use of different residue dictionaries for different molecule types.

    To enable this, the `category_to_res_dict_preset_name` argument should be used to
    specify the preset name of the residue dictionary to use for each chain category
    (i.e. "protein", "dna", "rna"), or `use_canonical_presets` can be set to `True`
    to use the canonical presets for protein, dna, and rna.
    """

    @classmethod
    def from_atoms(
        cls,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        category_to_res_dict_preset_name: Optional[Dict[str, str]] = None,
        chain_mapping: Optional[Dict[str, Type[BiomoleculeChain]]] = None,
        use_canonical_presets: bool = False,
    ):
        """N.B. default residue dictionaries exclude non-canonical residues."""
        # TODO: check chain-based stuff works with pdb files as well as cif files - are chain ids null sometimes for hetatms?
        # and are chains often mixed - e..g ions in the same chain...
        chains = []
        if use_canonical_presets:
            assert (
                category_to_res_dict_preset_name is None
            ), "Cannot specify both category_to_res_dict_preset_name and use_canonical_presets"
            category_to_res_dict_preset_name = {
                "protein": "protein",
                "dna": "dna",
                "rna": "rna",
            }
        if category_to_res_dict_preset_name is None:
            chain_mapping = chain_mapping or {
                "protein": BiomoleculeChain,
                "dna": BiomoleculeChain,
                "rna": BiomoleculeChain,
                "chemical": BiomoleculeChain,
            }
            residue_dictionary = residue_dictionary or ResidueDictionary.from_ccd_dict()

        else:
            chain_mapping = chain_mapping or {
                "protein": ProteinChain,
                "dna": DNAChain,
                "rna": RNAChain,
                "chemical": SmallMolecule,
            }

        chain_categories = [
            CHEM_COMPONENT_CATEGORIES[r]
            for r in atoms.res_name[atoms.chain_id == chain_id]
        ]
        if len(set(chain_categories)) == 1:
            category = chain_categories[0]
            if category == "protein":
                return chain_mapping["protein"].from_atoms(
                    atoms,
                    residue_dictionary=ProteinDictionary.from_preset(
                        category_to_res_dict_preset_name["protein"]
                    )
                    if residue_dictionary is None
                    else residue_dictionary,
                )
            else:
                raise ValueError(f"Unsupported chain category: {category}")
        else:
            print(
                f"Warning: found multiple categories in chain {chain_id}: {chain_categories}"
            )
        for chain_id, chain_category in zip(
            np.unique(atoms.chain_id), chain_categories
        ):
            # TODO: auto-ccd residue dictionary
            if len(np.unique(chain_category)) > 1:
                # raise ValueError(f"Found multiple categories in chain {chain_id}: {chain_categories}")
                print(
                    f"Warning: found multiple categories in chain {chain_id}: {chain_categories}"
                )
                chains.append(
                    BiomoleculeChain(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_ccd_dict()
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category[0] == "protein":
                chains.append(
                    chain_mapping["protein"].from_atoms(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ProteinDictionary.from_preset(
                            category_to_res_dict_preset_name["protein"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category[0] == "dna":
                chains.append(
                    chain_mapping["dna"].from_atoms(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_preset(
                            category_to_res_dict_preset_name["dna"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category[0] == "rna":
                chains.append(
                    chain_mapping["rna"].from_atoms(
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_preset(
                            category_to_res_dict_preset_name["rna"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category[0] == "chemical":
                chains.append(
                    chain_mapping["chemical"].from_atoms(
                        atoms[atoms.chain_id == chain_id],
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
