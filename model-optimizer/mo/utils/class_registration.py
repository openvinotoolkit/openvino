"""
 Copyright (c) 2018 Intel Corporation

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
from enum import Enum

import networkx as nx

from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg
from mo.graph.graph import check_empty_graph

_registered_classes_dict = {}


class ClassType(Enum):
    EXTRACTOR = 0
    OP = 1
    FRONT_REPLACER = 2
    MIDDLE_REPLACER = 3
    BACK_REPLACER = 4


def _update(cls, registered_list: list, registered_dict: dict, key: str):
    new_keys = {}  # maps a custom name to class
    new_keys_lower = {}  # translates lowered custom name to its original form
    # print('Registering new subclasses for', cls)
    for c in cls.__subclasses__():
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


def update_registration(classes: list):
    for cls in classes:
        _update(cls, cls.registered_cls, cls.registered_ops, 'op')
        _registered_classes_dict.setdefault(cls.class_type(), set()).add(cls)


def apply_replacements(graph: nx.MultiDiGraph, replacements_type):
    """
    Apply all patterns that do not have 'op' first, then apply patterns from registered_ops.
    If two or more classes replaces the same op (both have op class attribute and values match), such
    pattern is not applied (while registration it will warn user that we have a conflict).
    """
    dependency_graph = nx.DiGraph()
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
        replacers_order = nx.topological_sort(dependency_graph)
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

        log.debug("Run replacer {}".format(replacer_cls))

        try:
            replacer.find_and_replace_pattern(graph)
            check_empty_graph(graph, replacer_cls)
        except Error as err:
            raise Error('Exception occurred during running replacer "{}": {}'.format(replacement_id, str(err).replace(
                '[REPLACEMENT_ID]', replacement_id))) from err
        except Exception as err:
            raise Exception('Exception occurred during running replacer "{}": {}'.format(
                replacement_id, str(err).replace('[REPLACEMENT_ID]', replacement_id))) from err
