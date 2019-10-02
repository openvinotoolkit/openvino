"""
 Copyright (c) 2018-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import logging as log


def cast_to_string(descriptor, value):
    if descriptor.type != descriptor.TYPE_BOOL:
        return str(value)
    return str(int(value))


def append_unique(attrs, new_attr, value):
    if new_attr in attrs:
        log.error('The parameter {} overwrites already existing value. '.format(new_attr) +
                  'This happens due to flattening nested parameters. ' +
                  'Use enable_flattening_nested_params to flatten nesting')
    return {new_attr: value}


def append_unique_enum(attrs: dict, descriptor, value):
    enum_name = '{}.{}'.format(
        descriptor.enum_type.full_name.rsplit('.', 1)[0],  # remove enum name Z from X.Y.Z name
        descriptor.enum_type.values[value].name)
    return append_unique(attrs, descriptor.name, str(enum_name))


def unrolled_name(descriptor_name: str, enable_flattening_nested_params: bool = False, prefix: str = '') -> str:
    if not enable_flattening_nested_params:
        return descriptor_name
    elif prefix:
        return '{}__{}'.format(prefix, descriptor_name)
    return descriptor_name


def collect_optional_attributes(obj, prefix: str = '', disable_omitting_optional: bool = False,
                                enable_flattening_nested_params: bool = False):
    """
    Collect all optional attributes from protobuf message
    Args:
        attrs: dictionary with attributes
        obj: protobuf message
        prefix: prefix for this protobuf.message
        disable_omitting_optional: disable omitting optional flag
        enable_flattening_nested_params: disable flattening optional params flag
    """
    attrs = {}
    fields = [field[0].name for field in obj.ListFields()]
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        name = unrolled_name(descriptor.name, enable_flattening_nested_params, prefix)
        if descriptor.label != descriptor.LABEL_OPTIONAL:
            continue
        if (descriptor.has_default_value or disable_omitting_optional) or descriptor.name in fields:
            if descriptor.type == descriptor.TYPE_MESSAGE:
                attrs.update(collect_optional_attributes(value,
                                                         prefix=name,
                                                         disable_omitting_optional=disable_omitting_optional,
                                                         enable_flattening_nested_params=enable_flattening_nested_params))
            elif descriptor.type == descriptor.TYPE_ENUM:
                attrs.update(append_unique_enum(attrs, descriptor, value))
            else:
                attrs.update(append_unique(attrs, name, cast_to_string(descriptor, value)))
    return attrs


def collect_attributes(obj, prefix: str = '', disable_omitting_optional: bool = False,
                       enable_flattening_nested_params: bool = False):
    """
    Collect all attributes from protobuf message
    Args:
        attrs: dictionary with attributes
        obj: protobuf message
        prefix: prefix for this protobuf.message
        disable_omitting_optional: disable omitting optional flag
        enable_flattening_nested_params: disable flattening optional params flag
    """
    attrs = collect_optional_attributes(obj, prefix, disable_omitting_optional, enable_flattening_nested_params)
    fields = [field[0].name for field in obj.ListFields()]
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        name = unrolled_name(descriptor.name, enable_flattening_nested_params, prefix)
        if descriptor.label == descriptor.LABEL_REPEATED:
            if descriptor.name not in fields:
                log.warning('Field {} was ignored'.format(descriptor.name))
                continue
            if descriptor.type == descriptor.TYPE_MESSAGE:
                for x in value:
                    attrs.update(collect_attributes(x, prefix=name))
            else:
                attrs.update(append_unique(attrs, name, ",".join([str(v) for v in value])))
        elif descriptor.label == descriptor.LABEL_REQUIRED:
            if descriptor.type == descriptor.TYPE_MESSAGE:
                for x in value:
                    attrs.update(collect_attributes(x, prefix=name))
            else:
                attrs.update(append_unique(attrs, name, cast_to_string(descriptor, value)))
    return attrs


def merge_attrs(param, update_attrs: dict):
    all_attrs = collect_attributes(param)
    mandatory_attrs = set(all_attrs.keys()).intersection(set(update_attrs.keys()))
    return {value: update_attrs[value] for value in mandatory_attrs}
