import gzip
import os
from os import PathLike
from typing import Optional, Union

import numpy as np

from bio_datasets import config as bio_config

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from biotite import structure as bs
from biotite.structure.filter import (
    filter_first_altloc,
    filter_highest_occupancy_altloc,
)
from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx.convert import (
    _filter_model,
    _get_block,
    _get_model_starts,
)
from biotite.structure.residues import get_residue_starts, get_residue_starts_mask

from .residue import ResidueDictionary, create_complete_atom_array_from_restype_index

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


def _load_cif_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
    fill_missing_residues=False,
    altloc="first",
):
    """Load a structure from cif or binary cif format.

    Optionally fill in missing residues with nan coordinates and standard atom names,
    by cross-referencing the entity_poly_seq header with the atom_site information and
    the CCD dictionary.
    """
    # we use filter_altloc all to make it easier to get the chain id mapping
    if format == "cif":
        pdbxf = pdbx.CIFFile.read(fpath_or_handler)
    else:
        pdbxf = pdbx.BCIFFile.read(fpath_or_handler)
    extra_fields = extra_fields or (["occupancy"] if altloc == "occupancy" else [])
    if "occupancy" not in extra_fields and altloc == "occupancy":
        extra_fields.append("occupancy")
    structure = pdbx.get_structure(
        pdbxf,
        model=model,
        extra_fields=extra_fields,
        use_author_fields=False,  # be careful with this...
        altloc="all",  # handle later so that atom site lines up
    )
    # atom_site
    # auth_chain_id -> chain_id mapping from atom_site
    block = _get_block(pdbxf, None)
    atom_site = block["atom_site"]
    models = atom_site["pdbx_PDB_model_num"].as_array(np.int32)
    model_starts = _get_model_starts(models)
    atom_site = _filter_model(atom_site, model_starts, model)
    chain_id = atom_site["label_asym_id"].as_array(str)
    auth_chain_id = atom_site["auth_asym_id"].as_array(str)

    structure.set_annotation("auth_chain_id", auth_chain_id)
    processed_chain_atoms = []
    if fill_missing_residues:
        entity_poly = block["entity_poly"]  # n.b. there's also pdbx_entity_nonpoly
        entity_poly_seq = block["entity_poly_seq"]
        chain_ids = entity_poly["pdbx_strand_id"].as_array(str)
        entity_ids = entity_poly["entity_id"].as_array(int)
        residue_dict = ResidueDictionary.from_ccd_dict()
        # entity_types = entity_poly["type"].as_array("str")
        for entity_chain_ids, entity_id in zip(chain_ids, entity_ids):
            poly_seq_entity_mask = entity_poly_seq["entity_id"] == entity_id
            if not poly_seq_entity_mask.any():
                # TODO: we still need to track the entity chains somehow.
                for chain_id in entity_chain_ids.split(","):
                    processed_chain_atoms.append(
                        structure[structure.auth_chain_id == chain_id]
                    )
            else:
                entity_res_name = entity_poly_seq["mon_id"].as_array(str)[
                    poly_seq_entity_mask
                ]
                entity_restype_index = residue_dict.res_name_to_index(entity_res_name)

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
                    )  # TODO: backbone only? requires modality-specific residue dictionary
                    complete_atoms = chain_atoms + missing_atoms
                    residue_starts = get_residue_starts(complete_atoms)

                    # infer permutation of residues that would sort complete_res_id
                    res_perm = np.argsort(complete_res_id)
                    residue_sizes = np.diff(
                        np.append(residue_starts, len(complete_atoms))
                    )
                    permuted_residue_starts = residue_starts[res_perm]
                    _res_index = (
                        np.cumsum(
                            get_residue_starts_mask(
                                complete_atoms, permuted_residue_starts
                            )
                        )
                        - 1
                    )
                    # infer atom permutation
                    permuted_relative_atom_index = (
                        np.arange(len(complete_atoms))
                        - permuted_residue_starts[_res_index]
                    )
                    atom_perm = (
                        np.repeat(permuted_residue_starts, residue_sizes[res_perm])
                        + permuted_relative_atom_index
                    )

                    # Apply the permutation to the complete atoms
                    complete_atoms = complete_atoms[atom_perm]

                    processed_chain_atoms.append(complete_atoms)

    filled_structure = sum(processed_chain_atoms, bs.AtomArray(length=0))
    for key in structure._annot.keys():
        if key not in filled_structure._annot:
            filled_structure.set_annotation(
                key,
                np.concatenate(
                    [structure._annot[key] for structure in processed_chain_atoms]
                ),
            )

    if altloc == "occupancy":
        return filled_structure[
            filter_highest_occupancy_altloc(
                filled_structure, filled_structure.altloc_ids
            )
        ]
    elif altloc == "first":
        return filled_structure[
            filter_first_altloc(filled_structure, filled_structure.altloc_ids)
        ]
    elif altloc == "all":
        return filled_structure
    else:
        raise ValueError(f"'{altloc}' is not a valid 'altloc' option")


def _load_pdb_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
):
    if bio_config.FASTPDB_AVAILABLE:
        pdbf = fastpdb.PDBFile.read(fpath_or_handler)
    else:
        pdbf = PDBFile.read(fpath_or_handler)
    structure = pdbf.get_structure(
        model=model,
        extra_fields=extra_fields,
    )
    return structure


def _load_foldcomp_structure(
    fpath_or_handler,
    model=1,
    extra_fields=None,
):
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
    return structure


def infer_format_from_path(fpath: Union[str, PathLike]):
    return os.path.splitext(os.path.splitext(os.path.basename(fpath))[0])[1]


def load_structure(
    fpath_or_handler,
    format: Optional[str] = None,
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
    if isinstance(fpath_or_handler, (str, PathLike)) and fpath_or_handler.endswith(
        ".gz"
    ):
        format = infer_format_from_path(fpath_or_handler)
        # https://github.com/biotite-dev/biotite/issues/193
        with gzip.open(fpath_or_handler, "rt") as f:
            return load_structure(
                f,
                format=format,
                model=model,
                extra_fields=extra_fields,
                fill_missing_residues=fill_missing_residues,
            )

    if format is None and isinstance(fpath_or_handler, str):
        format = infer_format_from_path(fpath_or_handler)
    assert (
        format is not None
    ), "Format must be specified if fpath_or_handler is not a path"

    format = FILE_TYPE_TO_EXT[format]
    if fill_missing_residues:
        assert format in [
            "cif",
            "bcif",
        ], "Fill missing residues only supported for cif files"

    if format in ["cif", "bcif"]:
        return _load_cif_structure(
            fpath_or_handler, model, extra_fields, fill_missing_residues
        )

    elif format == "pdb":
        return _load_pdb_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
            fill_missing_residues=fill_missing_residues,
        )
    elif format == "fcz":
        return _load_foldcomp_structure(
            fpath_or_handler,
            model=model,
            extra_fields=extra_fields,
            fill_missing_residues=fill_missing_residues,
        )
    else:
        raise ValueError(f"Unsupported file format: {format}")
