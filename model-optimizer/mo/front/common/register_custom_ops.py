"""
 Copyright (c) 2017-2019 Intel Corporation

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
from collections import defaultdict

from mo.front.extractor import FrontExtractorOp
from mo.ops.op import Op
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


def extension_extractor(node, ex_cls, disable_omitting_optional: bool = False,
                        enable_flattening_optional_params: bool = False):
    ex = ex_cls()
    supported = ex.extract(node)
    return node.graph.node[node.id] if supported else None


def extension_op_extractor(node, op_cls):
    op_cls.update_node_stat(node)
    # TODO Need to differentiate truly supported ops extractors and ops extractors generated here
    return node.graph.node[node.id]


def find_case_insensitive_duplicates(extractors_collection: dict):
    """
    Searches for case-insensitive duplicates among extractors_collection keys.
    Returns a list of groups, where each group is a list of case-insensitive duplicates.
    Also returns a dictionary with lowered keys.
    """
    keys = defaultdict(list)
    for k in extractors_collection.keys():
        keys[k.lower()].append(k)
    return [duplicates for duplicates in keys.values() if len(duplicates) > 1], keys


def check_for_duplicates(extractors_collection: dict):
    """
    Check if extractors_collection has case-insensitive duplicates, if it does,
    raise exception with information about duplicates
    """
    # Check if extractors_collection is a normal form, that is it doesn't have case-insensitive duplicates
    duplicates, keys = find_case_insensitive_duplicates(extractors_collection)
    if len(duplicates) > 0:
        raise Error('Extractors collection have case insensitive duplicates {}. ' +
                    refer_to_faq_msg(47), duplicates)
    return {k: v[0] for k, v in keys.items()}


def add_or_override_extractor(extractors: dict, keys: dict, name, extractor, extractor_desc):
    name_lower = name.lower()
    if name_lower in keys:
        old_name = keys[name_lower]
        assert old_name in extractors
        del extractors[old_name]
        log.debug('Overridden extractor entry {} by {}.'.format(old_name, extractor_desc))
        if old_name != name:
            log.debug('Extractor entry {} was changed to {}.'.format(old_name, name))
    else:
        log.debug('Added a new entry {} to extractors with {}.'.format(name, extractor_desc))
    # keep extractor name in case-sensitive form for better diagnostics for the user
    # but we will continue processing of extractor keys in case-insensitive way
    extractors[name] = extractor
    keys[name_lower] = name


def update_extractors_with_extensions(extractors_collection: dict = None,
                                      disable_omitting_optional: bool = False,
                                      enable_flattening_optional_params: bool = False):
    """
    Update tf_op_extractors based on mnemonics registered in Op and FrontExtractorOp.
    FrontExtractorOp extends and overrides default extractors.
    Op extends but doesn't override extractors.
    """
    keys = check_for_duplicates(extractors_collection)
    for op, ex_cls in FrontExtractorOp.registered_ops.items():
        add_or_override_extractor(
            extractors_collection,
            keys,
            op,
            lambda node, cls=ex_cls: extension_extractor(
                node, cls, disable_omitting_optional, enable_flattening_optional_params),
            'custom extractor class {}'.format(ex_cls)
        )

    for op, op_cls in Op.registered_ops.items():
        op_lower = op.lower()
        if op_lower not in keys:
            extractors_collection[op] = (lambda c: lambda node: extension_op_extractor(node, c))(op_cls)
            log.debug('Added a new entry {} to extractors with custom op class {}.'.format(op, op_cls))
            keys[op_lower] = op
    check_for_duplicates(extractors_collection)
