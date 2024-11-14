# flake8: noqa: E402, F401
import os
from pathlib import Path

# Change cache location - n.b. this will also affect the datasets cache in same session
DEFAULT_XDG_CACHE_HOME = "~/.cache"
XDG_CACHE_HOME = os.getenv("XDG_CACHE_HOME", DEFAULT_XDG_CACHE_HOME)
DEFAULT_HF_CACHE_HOME = os.path.join(XDG_CACHE_HOME, "huggingface")
HF_CACHE_HOME = os.path.expanduser(os.getenv("HF_HOME", DEFAULT_HF_CACHE_HOME))
os.environ["HF_DATASETS_CACHE"] = os.path.join(HF_CACHE_HOME, "bio_datasets")

import importlib
import inspect
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import biotite
from packaging import version

logger = logging.getLogger(__name__)


if version.parse(biotite.__version__) > version.parse("1.0.2"):
    ccd_path = Path(__file__).parent / "structure" / "library" / "components.bcif"
    from biotite.structure.info import set_ccd_path

    if ccd_path.exists():
        set_ccd_path(ccd_path)
    else:
        logger.warning(
            f"CCD file not found at {ccd_path}, SMILES support may not be available"
        )
else:
    logger.warning(
        "Biotite version is less than 1.0.2, SMILES support may not be available"
    )

# imports required for overrides
from .features import Features
from .info import DatasetInfo


def override_features():

    SPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None
    import datasets
    import datasets.io
    import datasets.io.abc
    import datasets.io.csv
    import datasets.io.generator
    import datasets.io.json
    import datasets.io.parquet
    import datasets.io.sql
    import datasets.io.text

    def cast(self, target_schema, *args, **kwargs):
        """
        Cast table values to another schema.
        Only overridden because of Features import.

        Args:
            target_schema (`Schema`):
                Schema to cast to, the names and order of fields must match.
            safe (`bool`, defaults to `True`):
                Check for overflows or other unsafe conversions.

        Returns:
            `datasets.table.Table`
        """

        table = datasets.table.table_cast(self.table, target_schema, *args, **kwargs)
        target_features = Features.from_arrow_schema(target_schema)
        blocks = []
        for subtables in self.blocks:
            new_tables = []
            fields = list(target_schema)
            for subtable in subtables:
                subfields = []
                for name in subtable.column_names:
                    subfields.append(
                        fields.pop(
                            next(
                                i
                                for i, field in enumerate(fields)
                                if field.name == name
                            )
                        )
                    )
                subfeatures = Features(
                    {
                        subfield.name: target_features[subfield.name]
                        for subfield in subfields
                    }
                )
                subschema = subfeatures.arrow_schema
                new_tables.append(subtable.cast(subschema, *args, **kwargs))
            blocks.append(new_tables)
        return datasets.table.ConcatenationTable(table, blocks)

    @staticmethod
    def _build_metadata(
        info: DatasetInfo, fingerprint: Optional[str] = None
    ) -> Dict[str, str]:
        info_keys = [
            "features"
        ]  # we can add support for more DatasetInfo keys in the future
        info_as_dict = info.to_dict()
        metadata = {}
        metadata["info"] = {key: info_as_dict[key] for key in info_keys}
        if fingerprint is not None:
            metadata["fingerprint"] = fingerprint
        return {"huggingface": json.dumps(metadata)}

    datasets.table.Table.cast = cast
    datasets.arrow_writer.ArrowWriter._build_metadata = _build_metadata

    datasets.info.DatasetInfo = DatasetInfo
    datasets.DatasetInfo = DatasetInfo
    datasets.arrow_writer.DatasetInfo = DatasetInfo
    datasets.arrow_dataset.DatasetInfo = DatasetInfo
    datasets.builder.DatasetInfo = DatasetInfo
    datasets.combine.DatasetInfo = DatasetInfo
    datasets.dataset_dict.DatasetInfo = DatasetInfo
    datasets.inspect.DatasetInfo = DatasetInfo
    datasets.iterable_dataset.DatasetInfo = DatasetInfo
    datasets.load.DatasetInfo = DatasetInfo

    datasets.Features = Features
    datasets.features.Features = Features
    datasets.features.features.Features = Features
    datasets.arrow_writer.Features = Features
    datasets.arrow_dataset.Features = Features
    datasets.iterable_dataset.Features = Features
    datasets.builder.Features = Features
    datasets.info.Features = Features
    datasets.io.abc.Features = Features
    datasets.io.csv.Features = Features
    datasets.io.generator.Features = Features
    datasets.io.json.Features = Features
    datasets.io.parquet.Features = Features
    datasets.io.text.Features = Features
    datasets.io.sql.Features = Features
    datasets.utils.metadata.Features = Features
    datasets.dataset_dict.Features = Features
    datasets.load.Features = Features
    datasets.formatting.formatting.Features = Features
    # datasets.formatting.polars_formatter.Features = BioFeatures

    if SPARK_AVAILABLE:
        import datasets.io.spark

        datasets.io.spark.Features = Features


override_features()


# safe references to datasets objects to avoid import order errors due to monkey patching
# otherwise just import bio_datasets before importing anything from datasets
import datasets
from datasets import *

from .features import *
from .packaged_modules import *
from .structure import *


def load_dataset(*args, use_feature_type: str = "bio", **kwargs):
    """
    Load a dataset with custom features. Wrapper around `datasets.load_dataset`.

    Args:
        use_feature_type:
            "bio" uses the (possibly bio-specific) features defined in the dataset info,
            "fallback" uses only features compatible with the standard Datasets library,
            "none" removes the features types, providing only the raw data.
    """
    assert use_feature_type in ["bio", "fallback", "none"]
    ds = datasets.load_dataset(*args, **kwargs)
    if use_feature_type == "fallback":
        ds.info.features = ds.info.features.to_fallback()
    elif use_feature_type == "none":
        ds.info.features = None
    return ds
