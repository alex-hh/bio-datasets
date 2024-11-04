import importlib
import inspect
from typing import Optional

import datasets

from .features import *
from .info import DatasetInfo


def override_features():

    SPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None

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

    datasets.table.Table.cast = cast

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


from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES, _hash_python_lines

from .packaged_modules.structurefolder import structurefolder
from .structure import *

_PACKAGED_BIO_MODULES = {
    "structurefolder": (
        structurefolder.__name__,
        _hash_python_lines(inspect.getsource(structurefolder).splitlines()),
    )
}

_PACKAGED_DATASETS_MODULES.update(_PACKAGED_BIO_MODULES)


from datasets import Dataset, load_dataset

# safe references to datasets objects to avoid import order errors due to monkey patching
from datasets.features import *
