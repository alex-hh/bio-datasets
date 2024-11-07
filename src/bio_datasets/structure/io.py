import gzip
from io import StringIO
from os import PathLike

import numpy as np

from bio_datasets import config as bio_config

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx.convert import _get_block

from .residue import ResidueDictionary, create_complete_atom_array_from_restype_index


def permute_residues(atoms, permutation):
    """
    Permute the residues in the given AtomArray according to the provided permutation.

    Args:
        atoms (bs.AtomArray): The AtomArray containing the residues to be permuted.
        permutation (List[int]): A list of indices representing the new order of residues.

    Returns:
        bs.AtomArray: A new AtomArray with residues permuted according to the provided permutation.
    """
    residue_starts = get_residue_starts(atoms)
    residue_lengths = np.diff(np.append(residue_starts, len(atoms)))

    # Calculate the relative atom indices within each residue
    relative_atom_indices = np.arange(len(atoms)) - np.repeat(
        residue_starts, residue_lengths
    )

    # Permute the residue starts according to the permutation
    permuted_residue_starts = residue_starts[permutation]

    # Calculate the new atom indices using the permuted residue starts and relative atom indices
    permuted_atom_indices = np.add.outer(
        permuted_residue_starts, relative_atom_indices
    ).flatten()

    # Return the permuted AtomArray
    return atoms[permuted_atom_indices]


FILE_TYPE_TO_EXT = {
    "pdb": "pdb",
    "PDB": "pdb",
    "CIF": "cif",
    "cif": "cif",
    "bcif": "bcif",
    "FCZ": "fcz",
    "fcz": "fcz",
    "foldcomp": "fcz",
}


def is_open_compatible(file):
    return isinstance(file, (str, PathLike))


def load_structure(
    fpath_or_handler,
    format="pdb",
    model: int = 1,
    extra_fields=None,
    fill_missing_residues=False,
):
    """
    TODO: support foldcomp format, binary cif format
    TODO: support model choice / multiple models (multiple conformations)
    Args:
        fpath: filepath to either pdb or cif file
        chain: the chain id or list of chain ids to load
    Returns:
        biotite.structure.AtomArray
    """
    format = FILE_TYPE_TO_EXT[format]
    if isinstance(fpath_or_handler, str) and fpath_or_handler.endswith(".gz"):
        # https://github.com/biotite-dev/biotite/issues/193
        with gzip.open(fpath_or_handler, "rt") as f:
            return load_structure(
                f,
                format=format,
                model=model,
                extra_fields=extra_fields,
                fill_missing_residues=fill_missing_residues,
            )
    if fill_missing_residues:
        assert format in [
            "cif",
            "bcif",
        ], "Fill missing residues only supported for cif files"

    if format in ["cif", "bcif"]:
        if format == "cif":
            pdbxf = pdbx.CIFFile.read(fpath_or_handler)
        else:
            pdbxf = pdbx.BCIFFile.read(fpath_or_handler)
        structure = pdbx.get_structure(
            pdbxf,
            model=model,
            extra_fields=extra_fields,
            use_author_fields=False,  # be careful with this...
        )
        auth_structure = pdbx.get_structure(
            pdbxf,
            model=model,
            extra_fields=extra_fields,
            use_author_fields=True,  # be careful with this...
        )
        structure.set_annotation("auth_chain_id", auth_structure.chain_id)
        structure.set_annotation("auth_res_id", auth_structure.res_id)
        structure.set_annotation("auth_res_name", auth_structure.res_name)
        structure.set_annotation("auth_atom_id", auth_structure.atom_id)
        if fill_missing_residues:
            block = _get_block(pdbxf, None)
            entity_poly = block["entity_poly"]  # n.b. there's also pdbx_entity_nonpoly
            entity_poly_seq = block["entity_poly_seq"]
            chain_ids = entity_poly["pdbx_strand_id"].as_array(str)
            entity_ids = entity_poly["entity_id"].as_array(int)
            residue_dict = ResidueDictionary.from_ccd_dict()
            # entity_types = entity_poly["type"].as_array("str")
            for entity_chain_ids, entity_id in zip(chain_ids, entity_ids):
                poly_seq_entity_mask = entity_poly_seq["entity_id"] == entity_id
                if not poly_seq_entity_mask.any():
                    continue
                entity_res_name = entity_poly_seq["mon_id"].as_array(str)[
                    poly_seq_entity_mask
                ]
                entity_restype_index = residue_dict.res_name_to_index(entity_res_name)
                # TODO: set res_id correctly
                for chain_id in entity_chain_ids.split(","):
                    chain_atoms = structure[structure.auth_chain_id == chain_id]
                    complete_res_id = entity_poly_seq["num"].as_array(int, -1)
                    missing_res_mask = ~np.isin(
                        complete_res_id,
                        np.unique(chain_atoms.res_id),
                        chain_id=chain_id,
                    )
                    missing_atoms = create_complete_atom_array_from_restype_index(
                        entity_restype_index[missing_res_mask],
                        residue_dict,
                        chain_id,
                        res_id=complete_res_id[missing_res_mask],
                    )  # TODO: backbone only?
                    complete_atoms = chain_atoms + missing_atoms
                    residue_perm = np.concatenate(
                        complete_res_id[~missing_res_mask],
                        complete_res_id[missing_res_mask],
                    )
                    # now if we compute relative atom indices, we can apply res perm to residue sizes to get permuted
                    # res starts then add relative atom indices to get atom perm.
                    raise NotImplementedError()  # infer permutation. TODO write a permute residues helper function.

    elif format == "pdb":
        if bio_config.FASTPDB_AVAILABLE:
            pdbf = fastpdb.PDBFile.read(fpath_or_handler)
        else:
            pdbf = PDBFile.read(fpath_or_handler)
        structure = pdbf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    elif format == "fcz":
        if not bio_config.FOLDCOMP_AVAILABLE:
            raise ImportError(
                "Foldcomp is not installed. Please install it with `pip install foldcomp`"
            )

        if is_open_compatible(fpath_or_handler):
            with open(fpath_or_handler, "rb") as fcz:
                fcz_binary = fcz.read()
        else:
            raise ValueError(f"Unsupported file type: expected path or bytes handler")
        (_, pdb_str) = foldcomp.decompress(fcz_binary)
        lines = pdb_str.splitlines()
        pdbf = PDBFile()
        pdbf.lines = lines
        structure = pdbf.get_structure(
            model=model,
            extra_fields=extra_fields,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")

    return structure
