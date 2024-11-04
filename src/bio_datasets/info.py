import copy
import dataclasses
import json
from dataclasses import asdict, dataclass
from typing import ClassVar, List, Optional

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

    bio_features: Optional[Features] = None

    _INCLUDED_INFO_IN_YAML: ClassVar[List[str]] = [
        "config_name",
        "download_size",
        "dataset_size",
        "features",
        "bio_features",
        "splits",
    ]

    def __post_init__(self):
        super().__post_init__()
        if self.bio_features is None and self.features is not None:
            self.bio_features = self.features
        if self.bio_features is not None and not isinstance(
            self.bio_features, Features
        ):
            self.bio_features = Features.from_dict(self.bio_features)
        if self.bio_features is not None:
            self.features = self.bio_features

    def _to_yaml_dict(self) -> dict:
        # sometimes features are None
        if self.bio_features is not None:
            self.features = self.bio_features.to_fallback()
        ret = super()._to_yaml_dict()
        if self.bio_features is not None:
            self.features = self.bio_features
        return ret

    @classmethod
    def from_dict(cls, dict):
        return super().from_dict(dict)

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
            yaml_data["bio_features"] = Features._from_yaml_list(
                yaml_data["bio_features"]
            )
        if yaml_data.get("features") is not None:
            yaml_data["features"] = Features._from_yaml_list(yaml_data["features"])
        if yaml_data.get("splits") is not None:
            yaml_data["splits"] = SplitDict._from_yaml_list(yaml_data["splits"])
        field_names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in yaml_data.items() if k in field_names})
