import copy
import dataclasses
import json
from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, List

from datasets.info import DatasetInfo
from datasets.splits import SplitDict

from bio_datasets.features import Features


@dataclass
class DatasetInfo(DatasetInfo):
    """Information about a dataset.

    Written to ensure compatibility with standard Datasets library when user-defined
    features not available.

    DatasetInfo.features needs to be the features we actually want to use - i.e. bio_features if available -
    but during serialisation, features needs to be fallback features (compatible with standard Datasets lib).
    """

    _INCLUDED_INFO_IN_YAML: ClassVar[List[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "features",
        "splits",
    ]

    def _to_yaml_dict(self) -> dict:
        # sometimes features are None
        datasets_features = self.features.to_fallback()
        ret = super()._to_yaml_dict()
        ret["bio_features"] = ret["features"]
        ret["features"] = datasets_features._to_yaml_list()
        return ret

    @classmethod
    def from_dict(cls, info_dict: Dict):
        return super().from_dict(info_dict)

    def to_dict(self):
        new_info = self.copy()
        new_info.features = self.features.to_fallback()
        return asdict(new_info)

    def _dump_info(self, file, pretty_print=False):
        """Dump info in `file` file-like object open in bytes mode (to support remote files)"""
        file.write(
            json.dumps(self.to_dict(), indent=4 if pretty_print else None).encode(
                "utf-8"
            )
        )

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
