"""We upload assemblies, which contain atoms for PDB entries in correct oligomeric state, and no unnecessary experimental info.

https://pdbsnapshots.s3.us-west-2.amazonaws.com/index.html#20240101/pub/pdb/data/assemblies/mmCIF/divided/aq/
"""
import argparse
import os
import subprocess
import tempfile

from bio_datasets import Dataset, Features, NamedSplit, Value
from bio_datasets.features import AtomArrayFeature, StructureFeature


def get_pdb_id(assembly_file):
    return assembly_file.split("-")[0]


def examples_generator(pair_codes):
    if pair_codes is None:
        raise NotImplementedError("No pair codes provided")
    else:
        for pair_code in pair_codes:
            os.makedirs(f"data/pdb/{pair_code}", exist_ok=True)
            # download from s3
            # TODO use boto3
            subprocess.run(
                [
                    "aws",
                    "s3",
                    "cp",
                    "--recursive",
                    "--no-sign-request",
                    f"s3://pdbsnapshots/20240101/pub/pdb/data/assemblies/mmCIF/divided/{pair_code}",
                    f"data/pdb/{pair_code}",
                ],
                check=True,
            )

            downloaded_assemblies = os.listdir(f"data/pdb/{pair_code}")
            for assembly_file in downloaded_assemblies:
                # TODO: add extra metadata perhaps?
                yield {
                    "id": get_pdb_id(assembly_file),
                    "structure": {
                        "path": f"data/pdb/{pair_code}/{assembly_file}",
                        "type": "cif",
                    },
                }


def main(args):
    features = Features(
        id=Value("string"),
        structure=AtomArrayFeature() if args.as_array else StructureFeature(),
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        ds = Dataset.from_generator(
            examples_generator,
            gen_kwargs={
                "pair_codes": args.pair_codes,
            },
            features=features,
            cache_dir=temp_dir,
            split=NamedSplit("train"),
        )
        ds.push_to_hub("biodatasets/pdb", config_name=args.config_name or "default")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default=None)
    parser.add_argument(
        "--pair_codes", nargs="+", help="PDB 2-letter codes", default=None
    )
    parser.add_argument(
        "--backbone_only", action="store_true", help="Whether to drop sidechains"
    )
    parser.add_argument(
        "--as_array", action="store_true", help="Whether to return an array"
    )
    args = parser.parse_args()

    main(args)
