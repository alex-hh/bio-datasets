"""
Custom features for bio datasets.

Written to ensure compatibility with datasets loading / uploading when bio datasets not available.
"""
from typing import ClassVar, Dict, Optional, Union

from datasets import Features, LargeList, Sequence
from datasets.features.features import (
    FeatureType,
    cast_to_python_objects,
    decode_nested_example,
    encode_nested_example,
    register_feature,
    require_decoding,
)
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

    def fallback_feature(self):
        # TODO: automatically infer fallback feature?
        raise NotImplementedError(
            "Should be implemented by child class if `fallback_feature` is True"
        )


def bio_encode_nested_example(schema, obj, is_nested: bool = False):
    """Encode a nested example.
    This is used since some features (in particular ClassLabel) have some logic during encoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be encoded.
    If the first element needs to be encoded, then all the elements of the list will be encoded, otherwise they'll stay the same.
    """
    if isinstance(schema, CustomFeature) and schema.requires_encoding:
        return schema.encode_example(obj) if obj is not None else None
    else:
        return encode_nested_example(schema, obj, is_nested)


def bio_decode_nested_example(
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


def is_custom_feature(feature: FeatureType) -> bool:
    # TODO: check feature is registered
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


_BIO_FEATURE_TYPES: Dict[str, FeatureType] = {}


def register_bio_feature(feature_cls):
    assert issubclass(
        feature_cls, CustomFeature
    ), f"Expected a subclass of CustomFeature but got {feature_cls}"
    _BIO_FEATURE_TYPES[feature_cls.__name__] = feature_cls
    register_feature(feature_cls, feature_cls.__name__)


def is_bio_feature(class_name: str) -> bool:
    return class_name in _BIO_FEATURE_TYPES


# assumption is that we basically just need;
# yaml_data["features"] = Features._from_yaml_list(yaml_data["features"]) to work as expected
class Features(Features, dict):

    """We have things like

    {'name': feature_name, 'feature_type_name': feature_type_dict}
    feature_type_name can be e.g. 'class_label' or 'sequence' or 'struct'
    when we load from yaml, we need to convert this somehow
    _type = next(iter(obj))
    if _type == "struct":
        return from_yaml_inner(obj["struct"])
    if _type == "sequence":
        _feature = unsimplify(obj).pop(_type)
    obj['struct']
    """

    # TODO: do we need to modify from_arrow_schema / arrow_schema ?
    def __init__(*args, **kwargs):
        # init method overridden to avoid infinite recursion
        # self not in the signature to allow passing self as a kwarg
        if not args:
            raise TypeError(
                "descriptor '__init__' of 'Features' object needs an argument"
            )
        self, *args = args
        dict.__init__(self, *args, **kwargs)
        self._column_requires_decoding: Dict[str, bool] = {
            col: require_decoding(feature) for col, feature in self.items()
        }

    # TODO: is arrow schema stuff necessary?

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
        return bio_encode_nested_example(self, example)

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
            bio_encode_nested_example(self[column_name], obj, level=1) for obj in column
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
                bio_encode_nested_example(self[key], obj, level=1) for obj in column
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
            column_name: bio_decode_nested_example(
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
                bio_decode_nested_example(self[column_name], value)
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
                    bio_decode_nested_example(
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

    def to_fallback(self):
        return Features(
            **{
                col: feature.fallback_feature()
                if isinstance(feature, CustomFeature)
                else feature
                for col, feature in self.items()
            }
        )
