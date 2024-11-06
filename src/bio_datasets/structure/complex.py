import string
from itertools import product
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
from biotite import structure as bs

from .biomolecule import BaseBiomoleculeComplex, BiomoleculeChain, T
from .chemical import SmallMolecule
from .nucleic import DNAChain, RNAChain
from .protein import ProteinChain, ProteinDictionary
from .residue import ResidueDictionary, get_res_categories


def chain_name_generator():
    # Single letter chains from A to Z
    single_letters = string.ascii_uppercase
    for letter in single_letters:
        yield letter

    # Double letter chains AA, AB, ..., ZZ, then AAA, AAB, etc.
    for length in range(2, 1e3):  # Extend as needed (e.g., up to "ZZZZ")
        for combo in product(single_letters, repeat=length):
            yield "".join(combo)


class BiomoleculeComplex(BaseBiomoleculeComplex):
    """A collection of biomolecules, each represented by its own class type.

    As well as providing access to molecule type-specific methods, this supports
    use of different residue dictionaries for different molecule types.

    To enable this, the `category_to_res_dict_preset_name` argument should be used to
    specify the preset name of the residue dictionary to use for each chain category
    (i.e. "protein", "dna", "rna"), or `use_canonical_presets` can be set to `True`
    to use the canonical presets for protein, dna, and rna.
    """

    @staticmethod
    def split_relabel_chains(atoms: bs.AtomArray):
        chain_ids = np.unique(atoms.chain_id)
        new_arrs = []
        chain_name_gen = chain_name_generator()
        for chain_id in chain_ids:
            chain_atoms = atoms[atoms.chain_id == chain_id]
            chain_atoms.set_annotation(
                "molecule_type", get_res_categories(chain_atoms.res_name)
            )
            molecule_types = chain_atoms.molecule_type
            if len(np.unique(molecule_types)) == 1:
                chain_atoms.set_annotation("auth_chain_id", chain_atoms.chain_id)
                chain_atoms.set_annotation(
                    "chain_id", np.full_like(chain_atoms.chain_id, next(chain_name_gen))
                )
                new_arrs.append(chain_atoms)
            else:
                for molecule_type in np.unique(molecule_types):
                    if molecule_type == "small_molecule":
                        # each residue is a separate small molecule
                        all_small_molecule_atoms = chain_atoms[
                            chain_atoms.molecule_type == "small_molecule"
                        ]
                        res_ids = np.unique(all_small_molecule_atoms.res_id)
                        for res_id in res_ids:
                            molecule_atoms = all_small_molecule_atoms[
                                all_small_molecule_atoms.res_id == res_id
                            ]
                            molecule_atoms.set_annotation(
                                "auth_chain_id", molecule_atoms.chain_id
                            )
                            molecule_atoms.set_annotation(
                                "chain_id",
                                np.full_like(
                                    molecule_atoms.chain_id, next(chain_name_gen)
                                ),
                            )
                            new_arrs.append(molecule_atoms)
                    else:
                        molecule_atoms = chain_atoms[
                            chain_atoms.molecule_type == molecule_type
                        ]
                        molecule_atoms.set_annotation(
                            "auth_chain_id", molecule_atoms.chain_id
                        )
                        molecule_atoms.set_annotation(
                            "chain_id",
                            np.full_like(molecule_atoms.chain_id, next(chain_name_gen)),
                        )
                        new_arrs.append(molecule_atoms)
        # TODO: add annotations also.
        relabelled_atoms = sum(new_arrs, bs.AtomArray(length=0))
        for key in atoms._annot.keys():
            if key not in relabelled_atoms._annot:
                relabelled_atoms.set_annotation(
                    key,
                    np.concatenate(
                        [molecule_atoms._annot[key] for molecule_atoms in new_arrs]
                    ),
                )
        relabelled_atoms.set_annotation(
            "molecule_type", get_res_categories(relabelled_atoms.res_name)
        )
        return relabelled_atoms

    @classmethod
    def from_atoms(
        cls,
        atoms: bs.AtomArray,
        residue_dictionary: Optional[ResidueDictionary] = None,
        category_to_res_dict_preset_name: Optional[Dict[str, str]] = None,
        molecule_type_objects: Optional[Dict[str, Type[BiomoleculeChain]]] = None,
        use_canonical_presets: bool = True,
    ):
        """N.B. default residue dictionaries exclude non-canonical residues."""
        # N.B. doing like this, each chain will be individually standardised. Alternatively standardise once at start then split and prevent re-standardisation.
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
            molecule_type_objects = molecule_type_objects or {
                "protein": BiomoleculeChain,
                "dna": BiomoleculeChain,
                "rna": BiomoleculeChain,
                "small_molecule": BiomoleculeChain,
            }
            residue_dictionary = residue_dictionary or ResidueDictionary.from_ccd_dict()

        else:
            molecule_type_objects = molecule_type_objects or {
                "protein": ProteinChain,
                "dna": DNAChain,
                "rna": RNAChain,
                "small_molecule": SmallMolecule,
            }

        atoms = BiomoleculeComplex.filter_atoms(atoms, keep_hydrogens=False)
        atoms = BiomoleculeComplex.split_relabel_chains(atoms)

        for chain_id in np.unique(atoms.chain_id):
            chain_categories = atoms.molecule_type[atoms.chain_id == chain_id]
            assert len(np.unique(chain_categories)) == 1
            chain_category = chain_categories[0]
            if chain_category == "protein":
                chains.append(
                    molecule_type_objects["protein"](
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ProteinDictionary.from_preset(
                            category_to_res_dict_preset_name["protein"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category == "dna":
                chains.append(
                    molecule_type_objects["dna"](
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_preset(
                            category_to_res_dict_preset_name["dna"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category == "rna":
                chains.append(
                    molecule_type_objects["rna"](
                        atoms[atoms.chain_id == chain_id],
                        residue_dictionary=ResidueDictionary.from_preset(
                            category_to_res_dict_preset_name["rna"]
                        )
                        if residue_dictionary is None
                        else residue_dictionary,
                    )
                )
            elif chain_category == "small_molecule":
                chains.append(
                    molecule_type_objects["small_molecule"](
                        atoms[atoms.chain_id == chain_id],
                    )
                )
            else:
                raise ValueError(f"Unsupported chain category: {chain_category}")
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

    def __str__(self):
        return str(self._chains_lookup)

    def __repr__(self):
        return f"{self.__class__.__name__}: {str(self._chains_lookup)}"
