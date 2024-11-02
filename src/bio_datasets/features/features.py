import json
from collections import OrderedDict

import pyarrow as pa
from datasets.features.features import (
    Feature,
    Features,
    generate_from_arrow_type,
    get_nested_type,
)


# N.B. Image and Audio features could inherit from this.
class StructFeature(Feature, OrderedDict):
    """
    A feature that is a dictionary of features. It will be converted to a pyarrow struct.

    Initialise with a list of (key, Feature) tuples.
    """

    def __call__(self):
        pa_type = get_nested_type(self.features)
        return pa_type


# worry is whether just modifying Features is robust enough to changes to the datasets library.
# (we also need to modify ArrowWriter to build metadata correctly; and possibly methods like
# update_metadata_with_features). 
class Features(Features):

    @property
    def arrow_schema(self):
        """
        Features schema.

        Returns:
            :obj:`pyarrow.Schema`
        """
        hf_metadata = {"info": {"features": self.to_dict()}}
        return pa.schema(self.type).with_metadata({"huggingface": json.dumps(hf_metadata)})

    # TODO: this is where we need to load the bio features.
    @classmethod
    def from_arrow_schema(cls, pa_schema: pa.Schema, force_hf_features: bool = False):
        if (
            pa_schema.metadata is not None
            and "biodatasets".encode("utf-8") in pa_schema.metadata
            and not force_hf_features
        ):
            metadata = json.loads(pa_schema.metadata["biodatasets"].decode("utf-8"))
            if "features" in metadata and metadata["features"] is not None:
                metadata_features = cls.from_dict(metadata["info"]["features"])
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
        else:
            return super().from_arrow_schema(pa_schema)

    def _to_yaml_list(self) -> list:
        # we compute the YAML list from the dict representation that is used for JSON dump
        yaml_data = self.to_dict()

        def simplify(feature: dict) -> dict:
            if not isinstance(feature, dict):
                raise TypeError(f"Expected a dict but got a {type(feature)}: {feature}")

            for list_type in ["large_list", "list", "sequence"]:
                #
                # list_type:                ->              list_type: int32
                #   dtype: int32            ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["dtype"]:
                    feature[list_type] = feature[list_type]["dtype"]

                #
                # list_type:                ->              list_type:
                #   struct:                 ->              - name: foo
                #   - name: foo             ->                dtype: int32
                #     dtype: int32          ->
                #
                if isinstance(feature.get(list_type), dict) and list(feature[list_type]) == ["struct"]:
                    feature[list_type] = feature[list_type]["struct"]

            #
            # class_label:              ->              class_label:
            #   names:                  ->                names:
            #   - negative              ->                  '0': negative
            #   - positive              ->                  '1': positive
            #
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), list):
                # server-side requirement: keys must be strings
                feature["class_label"]["names"] = {
                    str(label_id): label_name for label_id, label_name in enumerate(feature["class_label"]["names"])
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
                    return {"dtype": camelcase_to_snakecase(_type)}
                elif _type:
                    return {"dtype": simplify({camelcase_to_snakecase(_type): obj})}
                else:
                    return {"struct": [{"name": name, **to_yaml_inner(_feature)} for name, _feature in obj.items()]}
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
            if isinstance(feature.get("class_label"), dict) and isinstance(feature["class_label"].get("names"), dict):
                label_ids = sorted(feature["class_label"]["names"], key=int)
                if label_ids and [int(label_id) for label_id in label_ids] != list(range(int(label_ids[-1]) + 1)):
                    raise ValueError(
                        f"ClassLabel expected a value for all label ids [0:{int(label_ids[-1]) + 1}] but some ids are missing."
                    )
                feature["class_label"]["names"] = [feature["class_label"]["names"][label_id] for label_id in label_ids]
            return feature

    def encode_example(self, example):
        raise NotImplementedError("TODO.")

    def decode_example(self, example):
        raise NotImplementedError("TODO.")
