"""
Custom features for bio datasets.

Written to ensure compatibility with datasets loading / uploading when bio datasets not available.
"""
import copy
from dataclasses import asdict
from typing import ClassVar, Dict, Optional, Union

from datasets import Features, LargeList, Sequence, Value
from datasets.features.features import (
    FeatureType,
    cast_to_python_objects,
    decode_nested_example,
    encode_nested_example,
)
from datasets.naming import camelcase_to_snakecase, snakecase_to_camelcase
from datasets.utils.py_utils import zip_dict


class CustomFeature:
    """
    Base class for feature types like Audio, Image, ClassLabel, etc that require special treatment (encoding/decoding).
    """

    requires_encoding: ClassVar[bool] = False
    requires_decoding: ClassVar[bool] = False

    def encode_example(self, example):
        if self.requires_encoding:
            return self._encode_example(example)
        return example

    def _encode_example(self, example):
        raise NotImplementedError(
            "Should be implemented by child class if `requires_encoding` is True"
        )

    def decode_example(self, example):
        if self.requires_decoding:
            return self._decode_example(example)
        return example

    def _decode_example(self, example):
        raise NotImplementedError(
            "Should be implemented by child class if `requires_decoding` is True"
        )


def encode_nested_example(schema, obj, is_nested: bool = False):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    if isinstance(schema, CustomFeature) and schema.requires_encoding:
        return schema.encode_example(obj) if obj is not None else None
    else:
        return encode_nested_example(schema, obj, is_nested)


def decode_nested_example(
    schema, obj, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None
):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be decoded.
    If the first element needs to be decoded, then all the elements of the list will be decoded, otherwise they'll stay the same.
    """
    if isinstance(schema, CustomFeature) and schema.requires_decoding:
        # we pass the token to read and decode files from private repositories in streaming mode
        if obj is not None and schema.decode:
            return schema.decode_example(obj, token_per_repo_id=token_per_repo_id)
    else:
        return decode_nested_example(schema, obj, token_per_repo_id)


def is_bio_feature(feature: FeatureType) -> bool:
    if isinstance(feature, CustomFeature):
        return True
    elif isinstance(feature, dict):
        return any(is_bio_feature(v) for v in feature.values())
    elif isinstance(feature, list):
        return any(is_bio_feature(v) for v in feature)
    elif isinstance(feature, tuple):
        return any(is_bio_feature(v) for v in feature)
    elif isinstance(feature, (Sequence, LargeList)):
        return is_bio_feature(feature.feature)
    else:
        return False


# assumption is that we basically just need;
# yaml_data["features"] = Features._from_yaml_list(yaml_data["features"]) to work as expected
class Features(Features):

    # TODO: do we need to modify from_arrow_schema / arrow_schema ?

    def feature_is_bio(self, feature_name: str) -> bool:
        return is_bio_feature(self[feature_name])

    def _to_yaml_list(self) -> list:
        # we compute the YAML list from the dict representation that is used for JSON dump
        yaml_data = self.to_dict()

        def simplify(feature: dict, type_prefix: str = "") -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                # These might be difficult to handle with bio datasets
                # I guess we do bio_list, bio_sequence, bio_large_list
                # or we don't simplify, which also seems to be supported.
                #
                # list_type:                ->              list_type: int32
                #   dtype: int32            ->
                #
                if isinstance(feature.get(list_type), dict) and list(
                    feature[list_type]
                ) == ["dtype"]:
                    feature[type_prefix + list_type] = feature.pop(list_type)["dtype"]

                #
                # list_type:                ->              list_type:
                #   struct:                 ->              - name: foo
                #   - name: foo             ->                dtype: int32
                #     dtype: int32          ->
                #
                if isinstance(feature.get(list_type), dict) and list(
                    feature[list_type]
                ) == ["struct"]:
                    feature[type_prefix + list_type] = feature.pop(list_type)["struct"]

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #   - negative              ->                  '0': negative
            #   - positive              ->                  '1': positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(
                feature["class_label"].get("names"), list
            ):
                # server-side requirement: keys must be strings
                feature["class_label"]["names"] = {
                    str(label_id): label_name
                    for label_id, label_name in enumerate(
                        feature["class_label"]["names"]
                    )
                }
            return feature

        def to_yaml_inner(obj: Union[dict, list]) -> dict:
            if isinstance(obj, dict):
                _type = obj.pop("_type", None)
                if _type == "LargeList":
                    _feature = obj.pop("feature")
                    return simplify({"large_list": to_yaml_inner(_feature), **obj})
                elif _type == "Sequence":
                    _feature = obj.pop("feature")
                    return simplify({"sequence": to_yaml_inner(_feature), **obj})
                elif _type == "Value":
                    return obj
                elif _type and not obj:
                    # base type
                    return {f"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    # nested type -- TODO: get example
                    raise NotImplementedError(f"Support for {_type} is not implemented")
                    return {
                        f"{ret_prefix}dtype": simplify(
                            {camelcase_to_snakecase(_type): obj}
                        )
                    }
                else:

                    def get_feature_dict(feature_name: str, feature_dict: dict):
                        if self.feature_is_bio(feature_name):
                            d = {
                                "name": feature_name,
                                "bio": to_yaml_inner(feature_dict, type_prefix="_bio")
                                ** to_yaml_inner(
                                    asdict(self[feature_name].fallback_feature())
                                ),
                            }
                        else:
                            d = {"name": feature_name, **to_yaml_inner(feature_dict)}
                        return d

                    return {
                        "struct": [
                            get_feature_dict(name, _feature)
                            for name, _feature in obj.items()
                        ]
                    }
            elif isinstance(obj, list):
                return simplify({"list": simplify(to_yaml_inner(obj[0]))})
            elif isinstance(obj, tuple):
                return to_yaml_inner(list(obj))
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        def to_yaml_types(obj: dict) -> dict:
            if isinstance(obj, dict):
                return {k: to_yaml_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [to_yaml_types(v) for v in obj]
            elif isinstance(obj, tuple):
                return to_yaml_types(list(obj))
            else:
                return obj

        return to_yaml_types(to_yaml_inner(yaml_data)["struct"])

    @classmethod
    def _from_yaml_list(cls, yaml_data: list) -> "Features":
        yaml_data = copy.deepcopy(yaml_data)

        # we convert the list obtained from YAML data into the dict representation that is used for JSON dump

        def unsimplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type: int32          ->              list_type:
                #                           ->                dtype: int32
                #
                if isinstance(feature.get(list_type), str):
                    feature[list_type] = {"dtype": feature[list_type]}

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #     '0': negative              ->               - negative
            #     '1': positive              ->               - positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(
                feature["class_label"].get("names"), dict
            ):
                label_ids = sorted(feature["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(
                    range(int(label_ids[-1]) + 1)
                ):
                    raise ValueError(
                        f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                    )
                feature["class_label"]["names"] = [
                    feature["class_label"]["names"][label_id] for label_id in label_ids
                ]
            return feature

        def from_yaml_inner(obj: Union[dict, list]) -> Union[dict, list]:
            if isinstance(obj, dict):
                if not obj:
                    return {}
                _type = next(iter(obj))
                if _type == "large_list":
                    _feature = unsimplify(obj).pop(_type)
                    return {
                        "feature": from_yaml_inner(_feature),
                        **obj,
                        "_type": "LargeList",
                    }
                if _type == "sequence":
                    _feature = unsimplify(obj).pop(_type)
                    return {
                        "feature": from_yaml_inner(_feature),
                        **obj,
                        "_type": "Sequence",
                    }
                if _type == "list":
                    return [from_yaml_inner(unsimplify(obj)[_type])]
                if _type == "struct":
                    return from_yaml_inner(obj["struct"])
                elif _type == "dtype":
                    # we can just add bio_dtype as well
                    if isinstance(obj["dtype"], str):
                        # e.g. int32, float64, string, audio, image
                        try:
                            Value(obj["dtype"])
                            return {**obj, "_type": "Value"}
                        except ValueError:
                            # e.g. Audio, Image, ArrayXD
                            return {"_type": snakecase_to_camelcase(obj["dtype"])}
                    else:
                        return from_yaml_inner(obj["dtype"])
                else:
                    return {
                        "_type": snakecase_to_camelcase(_type),
                        **unsimplify(obj)[_type],
                    }
            elif isinstance(obj, list):
                names = [_feature.pop("name") for _feature in obj]
                return {
                    name: from_yaml_inner(
                        _feature["bio"] if "bio" in _feature else _feature
                    )
                    for name, _feature in zip(names, obj)
                }
            else:
                raise TypeError(f"Expected a dict or a list but got {type(obj)}: {obj}")

        return cls.from_dict(from_yaml_inner(yaml_data))

    def encode_example(self, example):
        """
        Encode example into a format for Arrow.

        Args:
            example (`dict[str, Any]`):
                Data in a Dataset row.

        Returns:
            `dict[str, Any]`
        """
        example = cast_to_python_objects(example)
        return encode_nested_example(self, example)

    def encode_column(self, column, column_name: str):
        """
        Encode column into a format for Arrow.

        Args:
            column (`list[Any]`):
                Data in a Dataset column.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        column = cast_to_python_objects(column)
        return [
            encode_nested_example(self[column_name], obj, level=1) for obj in column
        ]

    def encode_batch(self, batch):
        """
        Encode batch into a format for Arrow.

        Args:
            batch (`dict[str, list[Any]]`):
                Data in a Dataset batch.

        Returns:
            `dict[str, list[Any]]`
        """
        encoded_batch = {}
        if set(batch) != set(self):
            raise ValueError(
                f"Column mismatch between batch {set(batch)} and features {set(self)}"
            )
        for key, column in batch.items():
            column = cast_to_python_objects(column)
            encoded_batch[key] = [
                encode_nested_example(self[key], obj, level=1) for obj in column
            ]
        return encoded_batch

    def decode_example(
        self,
        example: dict,
        token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None,
    ):
        """Decode example with custom feature decoding.

        Args:
            example (`dict[str, Any]`):
                Dataset row data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary `repo_id (str) -> token (bool or str)`.

        Returns:
            `dict[str, Any]`
        """

        return {
            column_name: decode_nested_example(
                feature, value, token_per_repo_id=token_per_repo_id
            )
            if self._column_requires_decoding[column_name]
            else value
            for column_name, (feature, value) in zip_dict(
                {key: value for key, value in self.items() if key in example}, example
            )
        }

    def decode_column(self, column: list, column_name: str):
        """Decode column with custom feature decoding.

        Args:
            column (`list[Any]`):
                Dataset column data.
            column_name (`str`):
                Dataset column name.

        Returns:
            `list[Any]`
        """
        return (
            [
                decode_nested_example(self[column_name], value)
                if value is not None
                else None
                for value in column
            ]
            if self._column_requires_decoding[column_name]
            else column
        )

    def decode_batch(
        self,
        batch: dict,
        token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None,
    ):
        """Decode batch with custom feature decoding.

        Args:
            batch (`dict[str, list[Any]]`):
                Dataset batch data.
            token_per_repo_id (`dict`, *optional*):
                To access and decode audio or image files from private repositories on the Hub, you can pass
                a dictionary repo_id (str) -> token (bool or str)

        Returns:
            `dict[str, list[Any]]`
        """
        decoded_batch = {}
        for column_name, column in batch.items():
            decoded_batch[column_name] = (
                [
                    decode_nested_example(
                        self[column_name], value, token_per_repo_id=token_per_repo_id
                    )
                    if value is not None
                    else None
                    for value in column
                ]
                if self._column_requires_decoding[column_name]
                else column
            )
        return decoded_batch
