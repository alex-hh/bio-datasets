"""
Upload a foldcomp database to the hub.
"""
import argparse
import io
import itertools
import os
from typing import Optional

import foldcomp
from datasets import Dataset, Features, Value, NamedSplit

from bio_datasets.features import ProteinAtomArrayFeature, ProteinStructureFeature
from bio_datasets.features.atom_array import load_structure
from bio_datasets.structure import ProteinChain


def examples_generator(
    db_file, max_examples: Optional[int] = None, as_array: bool = False
):
    assert os.path.exists(db_file)
    with foldcomp.open(db_file, decompress=True) as db:
        for (name, pdb_str) in itertools.islice(db, max_examples):
            # if we opened with decompress False, we wouldn't get name
            if as_array:
                atoms = load_structure(io.StringIO(pdb_str))
                example = {
                    "name": name,
                    "structure": ProteinChain(atoms),
                }
                yield example
            else:
                pdb_bytes = foldcomp.compress(name, pdb_str)
                # pdb_bytes = bytes(pdb_str)
                example = {
                    "name": name,
                    "structure": {"bytes": pdb_bytes, "path": None, "type": "fcz"},
                }
                yield example


def main(
    repo_id: str,
    db_file: str,
    as_array: bool,
    config_name: Optional[str] = None,
    max_examples: Optional[int] = None,
    backbone_only: bool = False,
):
    # from_generator calls GeneratorBasedBuilder.download_and_prepare and as_dataset
    features = Features(
        name=Value("string"),
        structure=ProteinAtomArrayFeature.from_preset(
            "afdb", drop_sidechains=backbone_only
        )
        if as_array
        else ProteinStructureFeature(with_b_factor=True),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.from_generator(
            examples_generator,
            gen_kwargs={
                "db_file": db_file,
                "max_examples": max_examples,
                "as_array": as_array,
            },
            features=features,
            cache_dir=temp_dir,
            split=NamedSplit("train"),
        )
        ds.push_to_hub(repo_id, config_name=config_name or "default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id", type=str)
    parser.add_argument("--foldcomp_db_name", type=str)
    parser.add_argument("--foldcomp_db_path", type=str)
    parser.add_argument("--as_array", action="store_true")
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--backbone_only", action="store_true")
    args = parser.parse_args()
    if args.foldcomp_db_name is None and args.foldcomp_db_path is None:
        raise ValueError("Either foldcomp_db_name or foldcomp_db_path must be provided")
    if args.foldcomp_db_name is not None:
        try:
            os.chdir("data")
            foldcomp.setup(args.foldcomp_db_name)
        except KeyError as e:
            # https://github.com/steineggerlab/foldcomp/issues/60
            print("Ignoring foldcomp setup error: ", e)
        os.chdir("..")
        args.foldcomp_db_path = os.path.join("data", args.foldcomp_db_name)
    main(
        args.repo_id,
        args.foldcomp_db_path,
        args.as_array,
        config_name=args.config_name,
        max_examples=args.max_examples,
        backbone_only=args.backbone_only,
    )
