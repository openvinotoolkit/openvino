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
from builtins import AttributeError
from defusedxml import ElementTree

from mo.front.caffe.collect_attributes import collect_attributes
from mo.front.caffe.extractor import node_pb_arg
from mo.front.common.register_custom_ops import check_for_duplicates, add_or_override_extractor


def expected_attribs(layer_attrs: list, attrs: list, fileName: str):
    missing = [attr for attr in attrs if attr not in layer_attrs]
    if len(missing):
        layer = "layer {}".format(layer_attrs['NativeType']) if 'NativeType' in layer_attrs else "one of the layers"
        log.error('Missing required attribute(s) {} for {} in {}. Skipped.'.format(', '.join(missing), layer, fileName))
        return False
    return True


def load_layers_xml(fileName: str):
    try:
        xml = ElementTree.parse(fileName).getroot()
    except:
        return {}

    layers_map = {}
    for child in xml:
        if child.tag == 'CustomLayer':
            if expected_attribs(child.attrib, ['NativeType', 'hasParam'], fileName):
                layer = child.attrib['NativeType']
                if layer in layers_map:
                    log.error('Duplicated layer definition in {} for NativeType = {}. Skipped.'.format(fileName, layer))
                else:
                    has_param = child.attrib['hasParam'].lower()
                    if has_param == 'true' and expected_attribs(child.attrib, ['protoParamName'],
                                                                fileName) or has_param == 'false':
                        layers_map[layer] = child.attrib
                    else:
                        log.error(
                            'Cannot recognize {} value for hasParam for layer {}. Should be true or false. Skipped.'.format(
                                child.attrib['hasParam'], layer))

        else:
            log.error('Unexpected "{}" tag in {}. Should be CustomLayer. Skipped.'.format(child.tag, fileName))
    return layers_map


special_keys = ['id', 'name', 'precision', 'type', 'layer', 'value', 'shape', 'op', 'kind', 'infer']

obfuscation_counter = 0


def new_obfuscated_key(attrs: dict, key: str):
    global obfuscation_counter
    while True:
        new_key = key + str(obfuscation_counter)
        obfuscation_counter += 1
        if new_key not in attrs and new_key not in special_keys:
            return new_key


def obfuscate_attr_key(attrs: dict, key: str, keys: list):
    """
    Replace attribute with key by another key that is not in
    special_keys list and do not match other attributes.
    """
    if key not in attrs or key not in special_keys:
        return

    new_key = new_obfuscated_key(attrs, key)
    assert new_key not in attrs
    assert new_key not in keys
    attrs[new_key] = attrs[key]
    del attrs[key]
    key_index = keys.index(key)
    keys[key_index] = (key, new_key)
    log.debug('Obfuscated attribute name {} to {}'.format(key, new_key))


def obfuscate_special_attrs(attrs: dict, keys: list):
    for key in special_keys:
        obfuscate_attr_key(attrs, key, keys)


def proto_extractor(pb, model_pb, mapping, disable_omitting_optional, enable_flattening_nested_params):
    log.info("Custom extractor for layer {} with mapping {}".format(pb.type, mapping))
    log.debug('Found custom layer {}. Params are processed'.format(pb.name))
    if mapping['hasParam'].lower() != 'true':
        return {}
    try:
        native_attr = collect_attributes(getattr(pb, mapping['protoParamName']),
                                         disable_omitting_optional=disable_omitting_optional,
                                         enable_flattening_nested_params=enable_flattening_nested_params)
    except AttributeError as e:
        error_message = 'Layer {} has no attribute {}'.format(pb.type, str(e).split(' ')[-1])
        log.error(error_message)
        raise ValueError(error_message)
    keys = list(native_attr.keys())
    obfuscate_special_attrs(native_attr, keys)
    # avoid 'mo_caffe' appearing in param
    for attr in native_attr:
        if 'mo_caffe' in native_attr[attr]:
            native_attr[attr] = native_attr[attr].replace('mo_caffe', 'caffe')
    log.debug(str(keys))
    log.debug(str(native_attr))

    attrs = {
        'IE': [(
            'layer',
            [('id', lambda node: node.id), 'name', 'precision', 'type'],
            [
                ('data', keys, []),
                '@ports',
                '@consts'])]}
    attrs.update(native_attr)
    return attrs


def update_extractors(extractors, layers_map, disable_omitting_optional, enable_flattening_nested_params):
    keys = check_for_duplicates(extractors)
    for layer, attrs in layers_map.items():
        add_or_override_extractor(
            extractors,
            keys,
            layer,
            (
                lambda l: node_pb_arg(
                    lambda pb, model_pb: proto_extractor(
                        pb, model_pb, l, disable_omitting_optional, enable_flattening_nested_params
                    )
                )
            )(layers_map[layer]),
            'custom layer {} from custom layers mapping xml file'.format(layer)
        )
    check_for_duplicates(extractors)
