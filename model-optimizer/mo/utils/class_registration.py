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
import os
from enum import Enum

import networkx as nx

from mo.graph.graph import Graph
from mo.middle.passes.eliminate import graph_clean_up_tf, graph_clean_up_onnx, graph_clean_up
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg

_registered_classes_dict = {}


def _check_unique_ids():
    """
    Check that idxs is unique for all registered replacements.
    """
    unique_idxs = set()
    for class_type, classes_set in _registered_classes_dict.items():
        for cls in classes_set:
            replacers = [c for c in cls.registered_cls if not hasattr(c, 'op')] + \
                        [c for op, c in cls.registered_ops.items() if c]
            for replacer_cls in replacers:
                if hasattr(replacer_cls, 'id'):
                    id_cls = getattr(replacer_cls, 'id')

                    if id_cls in unique_idxs:
                        raise Error('Found replacer {} with not unique id!'.format(replacer_cls))
                    unique_idxs.add(id_cls)
    log.debug("All replacers has unique idxs.")


def get_enabled_and_disabled_transforms():
    """
    :return: tuple of lists with force enabled and disabled id of transformations.
    """
    disabled_transforms = os.environ['MO_DISABLED_TRANSFORMS'] if 'MO_DISABLED_TRANSFORMS' in os.environ else ''
    enabled_transforms = os.environ['MO_ENABLED_TRANSFORMS'] if 'MO_ENABLED_TRANSFORMS' in os.environ else ''

    assert isinstance(enabled_transforms, str)
    assert isinstance(disabled_transforms, str)

    disabled_transforms = disabled_transforms.split(',')
    enabled_transforms = enabled_transforms.split(',')

    return enabled_transforms, disabled_transforms


class ClassType(Enum):
    EXTRACTOR = 0
    OP = 1
    FRONT_REPLACER = 2
    MIDDLE_REPLACER = 3
    BACK_REPLACER = 4


def _update(cls, registered_list: list, registered_dict: dict, key: str, enabled_transforms: list, disabled_transforms: list):
    new_keys = {}  # maps a custom name to class
    new_keys_lower = {}  # translates lowered custom name to its original form
    # print('Registering new subclasses for', cls)

    for c in cls.__subclasses__():
        # Force enabling operations
        if hasattr(c, 'id') and c.id in enabled_transforms:
            setattr(c, 'enabled', True)

        # Force disabling operations
        if hasattr(c, 'id') and c.id in disabled_transforms:
            setattr(c, 'enabled', False)

        if c not in registered_list and (not hasattr(c, 'enabled') or c.enabled):
            if hasattr(cls, 'excluded_classes') and c in cls.excluded_classes:
                continue
            registered_list.append(c)
            log.info('New subclass: {}'.format(c))
            if hasattr(c, key):
                k = getattr(c, key)
                if k.lower() in new_keys_lower:
                    raise Error(
                        'Attempt to register of custom name {} for the second time as class {}. ' \
                        'Note that custom names are case-insensitive. ' +
                        refer_to_faq_msg(55), k, c)
                else:
                    new_keys_lower[k.lower()] = k
                    new_keys[k] = c
                    log.info('Registered a new subclass with key: {}'.format(k))
        else:
            log.warning('Skipped {} registration because it was already registered or it was disabled. '.format(c))
    registered_dict.update(new_keys)


def update_registration(classes: list, enabled_transforms: list, disabled_transforms: list):
    for cls in classes:
        _update(cls, cls.registered_cls, cls.registered_ops, 'op', enabled_transforms, disabled_transforms)
        _registered_classes_dict.setdefault(cls.class_type(), set()).add(cls)


def apply_replacements(graph: Graph, replacements_type):
    """
    Apply all patterns that do not have 'op' first, then apply patterns from registered_ops.
    If two or more classes replaces the same op (both have op class attribute and values match), such
    pattern is not applied (while registration it will warn user that we have a conflict).
    """
    dependency_graph = Graph()
    for class_type, classes_set in _registered_classes_dict.items():
        if class_type == replacements_type:
            for cls in classes_set:
                replacers = [c for c in cls.registered_cls if not hasattr(c, 'op')] + \
                            [c for op, c in cls.registered_ops.items() if c]
                for replacer_cls in replacers:
                    if replacer_cls in cls.excluded_replacers:
                        # skip infrastructure classes
                        continue

                    dependency_graph.add_node(replacer_cls)
                    for cls_after in replacer_cls().run_before():
                        log.debug("Replacer {} will be run before {}".format(replacer_cls, cls_after))
                        dependency_graph.add_edge(replacer_cls, cls_after)
                    for cls_before in replacer_cls().run_after():
                        log.debug("Replacer {} will be run after {}".format(replacer_cls, cls_before))
                        dependency_graph.add_edge(cls_before, replacer_cls)

    try:
        replacers_order = list(nx.topological_sort(dependency_graph))
    except nx.NetworkXUnfeasible as exception:
        cycles = nx.simple_cycles(dependency_graph)
        raise Error('There is(are) cyclic dependency(ies) between replacers. One of the cycles is the following: {}',
                    ' -> '.join([str(node) for node in list(cycles)[0]])) from exception

    for replacer_cls in replacers_order:
        replacer = replacer_cls()

        replacement_id = 'REPLACEMENT_ID'
        if hasattr(replacer, 'replacement_id'):
            replacement_id = replacer.replacement_id

        if hasattr(replacer, 'enabled') and not replacer.enabled:
            log.info("Skip replacer {} (enabled = False)".format(replacer_cls))
            continue

        if hasattr(replacer, 'graph_condition') and \
                not all([condition(graph) for condition in replacer.graph_condition]):
            log.info("Skip replacer {} (graph_condition not satisfied)".format(replacer_cls))
            continue

        log.debug("Run replacer {}".format(replacer_cls))

        try:
            replacer.find_and_replace_pattern(graph)

            if hasattr(replacer, 'force_clean_up') and replacer.force_clean_up:
                for_graph_and_each_sub_graph_recursively(
                    graph,
                    graph_clean_up_tf if graph.graph['fw'] == 'tf' else
                    graph_clean_up_onnx if graph.graph['fw'] == 'onnx' else
                    graph_clean_up)

            for_graph_and_each_sub_graph_recursively(graph, lambda _: graph.check_empty_graph(replacer_cls))
            for_graph_and_each_sub_graph_recursively(graph, lambda _: graph.check_shapes_consistency())

        except Error as err:
            raise Error('Exception occurred during running replacer "{}" ({}): {}'.format(
                replacement_id,
                replacer_cls,
                str(err).replace('[REPLACEMENT_ID]', replacement_id),
            )) from err
        except Exception as err:
            raise Exception('Exception occurred during running replacer "{} ({})": {}'.format(
                replacement_id,
                replacer_cls,
                str(err).replace('[REPLACEMENT_ID]', replacement_id),
            )) from err
