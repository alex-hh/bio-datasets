import gzip
from io import StringIO
from os import PathLike

from bio_datasets import config as bio_config

if bio_config.FASTPDB_AVAILABLE:
    import fastpdb

if bio_config.FOLDCOMP_AVAILABLE:
    import foldcomp

from biotite.structure.io import pdbx
from biotite.structure.io.pdb import PDBFile
from biotite.structure.io.pdbx.convert import _get_block

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
            entity_poly_seq = _get_block(pdbxf)["entity_poly_seq"]
            res_name = entity_poly_seq["mon_id"].as_array(str)
            res_id = entity_poly_seq["num"].as_array(int, -1)
            raise NotImplementedError("Fill missing residues not implemented yet")

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
