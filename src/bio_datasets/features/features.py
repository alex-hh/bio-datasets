"""
Custom features for bio datasets.

Written to ensure compatibility with datasets loading / uploading when bio datasets not available.
"""
import copy
import json
from typing import ClassVar, Dict, Optional, Union

import numpy as np
import pyarrow as pa
from datasets.features.features import (
    Audio,
    ClassLabel,
    Features,
    Image,
    LargeList,
    Sequence,
    TranslationVariableLanguages,
    Value,
    Video,
    _ArrayXD,
)
from datasets.features.features import (
    FeatureType,
    _check_non_null_non_empty_recursive,
    cast_to_python_objects,
    generate_from_arrow_type,
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


def encode_nested_example(schema, obj, level: int = 0):
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        if level == 0 and obj is None:
            raise ValueError("Got None but expected a dictionary instead")
        return (
            {
                k: encode_nested_example(schema[k], obj.get(k), level=level + 1)
                for k in schema
            }
            if obj is not None
            else None
        )

    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        elif isinstance(obj, np.ndarray):
            return encode_nested_example(schema, obj.tolist())
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if (
                    encode_nested_example(sub_schema, first_elmt, level=level + 1)
                    != first_elmt
                ):
                    return [
                        encode_nested_example(sub_schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                sub_schema = schema.feature
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if (
                    encode_nested_example(sub_schema, first_elmt, level=level + 1)
                    != first_elmt
                ):
                    return [
                        encode_nested_example(sub_schema, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    elif isinstance(schema, Sequence):
        if obj is None:
            return None
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            # dict of list to fill
            list_dict = {}
            if isinstance(obj, (list, tuple)):
                # obj is a list of dict
                for k in schema.feature:
                    list_dict[k] = [
                        encode_nested_example(
                            schema.feature[k], o.get(k), level=level + 1
                        )
                        for o in obj
                    ]
                return list_dict
            else:
                # obj is a single dict
                for k in schema.feature:
                    list_dict[k] = (
                        [
                            encode_nested_example(schema.feature[k], o, level=level + 1)
                            for o in obj[k]
                        ]
                        if k in obj
                        else None
                    )
                return list_dict
        # schema.feature is not a dict
        if isinstance(obj, str):  # don't interpret a string as a list
            raise ValueError(f"Got a string but expected a list instead: '{obj}'")
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, schema.feature):
                        break
                # be careful when comparing tensors here
                if (
                    not isinstance(first_elmt, list)
                    or encode_nested_example(
                        schema.feature, first_elmt, level=level + 1
                    )
                    != first_elmt
                ):
                    return [
                        encode_nested_example(schema.feature, o, level=level + 1)
                        for o in obj
                    ]
            return list(obj)
    # Object with special encoding:
    # ClassLabel will convert from string to int, TranslationVariableLanguages does some checks
    elif isinstance(
        schema,
        (
            Audio,
            Image,
            ClassLabel,
            TranslationVariableLanguages,
            Value,
            _ArrayXD,
            Video,
        ),
    ):
        return schema.encode_example(obj) if obj is not None else None
    elif isinstance(schema, CustomFeature) and schema.requires_encoding:
        return schema.encode_example(obj) if obj is not None else None
    # Other object should be directly convertible to a native Arrow type (like Translation and Translation)
    return obj


def decode_nested_example(
    schema, obj, token_per_repo_id: Optional[Dict[str, Union[str, bool, None]]] = None
):
    """Decode a nested example.
    This is used since some features (in particular Audio and Image) have some logic during decoding.

    To avoid iterating over possibly long lists, it first checks (recursively) if the first element that is not None or empty (if it is a sequence) has to be decoded.
    If the first element needs to be decoded, then all the elements of the list will be decoded, otherwise they'll stay the same.
    """
    # Nested structures: we allow dict, list/tuples, sequences
    if isinstance(schema, dict):
        return (
            {
                k: decode_nested_example(sub_schema, sub_obj)
                for k, (sub_schema, sub_obj) in zip_dict(schema, obj)
            }
            if obj is not None
            else None
        )
    elif isinstance(schema, (list, tuple)):
        sub_schema = schema[0]
        if obj is None:
            return None
        else:
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, LargeList):
        if obj is None:
            return None
        else:
            sub_schema = schema.feature
            if len(obj) > 0:
                for first_elmt in obj:
                    if _check_non_null_non_empty_recursive(first_elmt, sub_schema):
                        break
                if decode_nested_example(sub_schema, first_elmt) != first_elmt:
                    return [decode_nested_example(sub_schema, o) for o in obj]
            return list(obj)
    elif isinstance(schema, Sequence):
        # We allow to reverse list of dict => dict of list for compatibility with tfds
        if isinstance(schema.feature, dict):
            return {
                k: decode_nested_example([schema.feature[k]], obj[k])
                for k in schema.feature
            }
        else:
            return decode_nested_example([schema.feature], obj)
    # Object with special decoding:
    elif isinstance(schema, (Audio, Image, Video)):
        # we pass the token to read and decode files from private repositories in streaming mode
        if obj is not None and schema.decode:
            return schema.decode_example(obj, token_per_repo_id=token_per_repo_id)
    elif isinstance(schema, CustomFeature) and schema.requires_decoding:
        return schema.decode_example(obj, token_per_repo_id=token_per_repo_id)
    return obj


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
    @property
    def arrow_schema(self):
        """
        Features schema.

        Returns:
            :obj:`pyarrow.Schema`
        """
        hf_metadata = {
            "info": {
                "features": self.to_fallback().to_dict(),
                "bio_features": self.to_dict(),
            }
        }
        return pa.schema(self.type).with_metadata(
            {"huggingface": json.dumps(hf_metadata)}
        )

    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema) -> "Features":
        """
        Construct [`Features`] from Arrow Schema.
        It also checks the schema metadata for Hugging Face Datasets features.
        Non-nullable fields are not supported and set to nullable.

        Also, pa.dictionary is not supported and it uses its underlying type instead.
        Therefore datasets convert DictionaryArray objects to their actual values.

        Args:
            pa_schema (`pyarrow.Schema`):
                Arrow Schema.

        Returns:
            [`Features`]
        """
        # try to load features from the arrow schema metadata
        metadata_features = Features()
        if (
            pa_schema.metadata is not None
            and "huggingface".encode("utf-8") in pa_schema.metadata
        ):
            metadata = json.loads(
                pa_schema.metadata["huggingface".encode("utf-8")].decode()
            )
            if (
                "info" in metadata
                and "bio_features" in metadata["info"]
                and metadata["info"]["bio_features"] is not None
            ):
                metadata_features = Features.from_dict(metadata["info"]["bio_features"])
            elif (
                "info" in metadata
                and "features" in metadata["info"]
                and metadata["info"]["features"] is not None
            ):
                metadata_features = Features.from_dict(metadata["info"]["features"])
        metadata_features_schema = metadata_features.arrow_schema
        obj = {
            field.name: (
                metadata_features[field.name]
                if field.name in metadata_features
                and metadata_features_schema.field(field.name) == field
                else generate_from_arrow_type(field.type)
            )
            for field in pa_schema
        }
        return cls(**obj)

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
