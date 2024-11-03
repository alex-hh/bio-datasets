import copy
import dataclasses
import inspect

import datasets

from .features import *


def monkey_patch_features():

    SPARK_AVAILABLE = importlib.util.find_spec("pyspark") is not None

    import datasets.io
    import datasets.io.abc
    import datasets.io.csv
    import datasets.io.generator
    import datasets.io.json
    import datasets.io.parquet
    import datasets.io.sql
    import datasets.io.text
    from datasets.splits import SplitDict

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

    def _to_yaml_dict(self) -> dict:
        yaml_dict = {}
        fallback_features = self.features.to_fallback()
        dataset_info_dict = dataclasses.asdict(self)
        for key in dataset_info_dict:
            if key == "features":
                yaml_dict["bio_features"] = dataset_info_dict[
                    "features"
                ]._to_yaml_list()
                yaml_dict["features"] = fallback_features._to_yaml_list()
            elif key in self._INCLUDED_INFO_IN_YAML:
                value = getattr(self, key)
                if hasattr(value, "_to_yaml_list"):  # Features, SplitDict
                    yaml_dict[key] = value._to_yaml_list()
                elif hasattr(value, "_to_yaml_string"):  # Version
                    yaml_dict[key] = value._to_yaml_string()
                else:
                    yaml_dict[key] = value
        return yaml_dict

    @classmethod
    def _from_yaml_dict(cls, yaml_data: dict) -> "DatasetInfo":
        yaml_data = copy.deepcopy(yaml_data)
        if yaml_data.get("bio_features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["bio_features"])
        elif yaml_data.get("features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            yaml_data["splits"] = SplitDict._from_yaml_list(yaml_data["splits"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})

    datasets.info.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.info.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.arrow_writer.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.arrow_writer.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.arrow_dataset.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.arrow_dataset.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.builder.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.builder.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.combine.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.combine.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.dataset_dict.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.dataset_dict.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.inspect.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.inspect.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.iterable_dataset.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.iterable_dataset.DatasetInfo._to_yaml_dict = _to_yaml_dict
    datasets.load.DatasetInfo._from_yaml_dict = _from_yaml_dict
    datasets.load.DatasetInfo._to_yaml_dict = _to_yaml_dict

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
        datasets.io.spark.Features = Features


monkey_patch_features()


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
