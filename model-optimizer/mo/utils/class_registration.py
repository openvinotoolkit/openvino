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
from mo.middle.passes.eliminate import graph_clean_up_tf, graph_clean_up_onnx, graph_clean_up, shape_inference
from mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from mo.utils.error import Error, InternalError
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

        if c not in registered_list:
            if hasattr(cls, 'excluded_classes') and c in cls.excluded_classes:
                continue
            registered_list.append(c)
            log.info('New subclass: {}'.format(c))
            if hasattr(c, key) and getattr(c, key) is not None:
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


class DependencyGraph(Graph):
    def __init__(self, data=None, **attr):
        super().__init__(data, **attr)

    def dump_graph_for_graphviz(self, node_attrs: list = [], edge_attrs: list = [], nodes_to_dump: list = None,
                                save_to_svg=False, highlight_nodes: list = None):
        log.debug("---- GRAPHVIZ OUTPUT STARTS ----")
        if nodes_to_dump is None:
            nodes_to_dump = self.nodes()
        string = '\ndigraph {\n'
        string += 'node [color=lightblue2, style=filled];\n'

        for node in nodes_to_dump:
            attrs = ""
            if hasattr(node, 'enabled') and not node.enabled:
                attrs += "color=gray70,"
            string += '"{}" [{}];\n'.format(node, attrs)

        visited_nodes = set()
        for src_node_name, dst_node_name, attrs in self.edges(data=True):
            visited_nodes.add(src_node_name)
            visited_nodes.add(dst_node_name)
            if src_node_name not in nodes_to_dump or dst_node_name not in nodes_to_dump:
                continue
            src_node = self.node[src_node_name]
            dst_node = self.node[dst_node_name]
            src_node_string = str(src_node_name) + '\\n'.join(
                [str(key) + '=' + str(src_node.get(key, 'None')) for key in node_attrs if key in src_node])
            dst_node_string = str(dst_node_name) + '\\n'.join(
                [str(key) + '=' + str(dst_node.get(key, 'None')) for key in node_attrs if key in dst_node])
            edge_string = ' '.join([str(key) + '=' + str(attrs.get(key, 'None')) for key in edge_attrs if key in attrs])
            string += '"{}" -> "{}" [label = "{}"];\n'.format(src_node_string, dst_node_string, edge_string)

        for node in nodes_to_dump:
            if node not in visited_nodes:
                string += '"{}";\n'.format(node)
                visited_nodes.add(node)

        string += '}'
        log.debug(string)
        log.debug("---- GRAPHVIZ OUTPUT ENDS ----")

        if save_to_svg:
            try:
                import graphviz
                import os
                file_name = "{}_{}.txt".format(self.name.replace('/', '_'), 0)
                id = 1
                while os.path.exists(file_name):
                    file_name = "{}_{}.txt".format(self.name.replace('/', '_'), id)
                    id += 1
                with open(file_name, "w") as f:
                    f.write(string)
                graphviz.render('dot', 'svg', file_name)
                print('Graph was saved to {}.{}'.format(file_name, 'svg'))
            except ImportError:
                raise ImportError('Can\'t import graphviz')
            except Exception as e:
                raise Error('Can\'t save graph to svg') from e

        return string

    def cycle_check(self):
        try:
            list(nx.topological_sort(self))
        except nx.NetworkXUnfeasible as exception:
            cycles = nx.simple_cycles(self)
            raise Error(
                'There is(are) cyclic dependency(ies) between replacers. One of the cycles is the following: {}',
                ' -> '.join([str(node) for node in list(cycles)[0]])) from exception

    def repeated_cls_names_check(self):
        name_to_class_map = {}
        for transform_class in self.node:
            transform_name = transform_class.__name__
            assert transform_name not in name_to_class_map, \
                'Transform name `{}` is not unique: at least {} and {} exist' \
                ''.format(transform_name, transform_class, name_to_class_map[transform_name])
            name_to_class_map[transform_name] = transform_class

    def sort_util(self, v, visited, stack):
        visited.append(v)
        for i in sorted([child for _, child in self.out_edges(v)], key=lambda x: x.__name__):
            if i not in visited:
                self.sort_util(i, visited, stack)
        stack.insert(0, v)

    def determined_sort(self):
        self.cycle_check()
        self.repeated_cls_names_check()
        transforms = sorted([cls for cls in self.nodes() if len(self.in_edges(cls)) == 0], key=lambda x: x.__name__)
        order, visited = [], []
        for transform in transforms:
            self.sort_util(transform, visited, order)

        graph_copy = self.copy()
        for i in range(len(order) - 1):
            graph_copy.add_edge(order[i], order[i + 1])
        try:
            nx_order = list(nx.topological_sort(graph_copy))
        except Exception as e:
            raise InternalError(
                "Internal DependencyGraph determined_sort function behaves unexpectedly: cycle found") from e
        assert nx_order == order, \
            "Internal DependencyGraph determined_sort function behaves unexpectedly: nx_order != order"
        return order


def apply_replacements(graph: Graph, replacements_type):
    """
    Apply all patterns that do not have 'op' first, then apply patterns from registered_ops.
    If two or more classes replaces the same op (both have op class attribute and values match), such
    pattern is not applied (while registration it will warn user that we have a conflict).
    """
    dependency_graph = DependencyGraph(name=ClassType(replacements_type).name)
    for class_type, classes_set in _registered_classes_dict.items():
        if class_type == replacements_type:
            replacers = []
            for cls in classes_set:
                cur_cls_replacers = [c for c in cls.registered_cls if not hasattr(c, 'op')] + \
                                    [c for op, c in cls.registered_ops.items() if c]
                replacers.extend([replacer for replacer in cur_cls_replacers if replacer not in cls.excluded_replacers])

            for replacer_cls in replacers:
                dependency_graph.add_node(replacer_cls)

            for replacer_cls in replacers:
                for cls_after in replacer_cls().run_before():
                    log.debug("Replacer {} will be run before {}".format(replacer_cls, cls_after))
                    dependency_graph.add_edge(replacer_cls, cls_after)
                for cls_before in replacer_cls().run_after():
                    log.debug("Replacer {} will be run after {}".format(replacer_cls, cls_before))
                    dependency_graph.add_edge(cls_before, replacer_cls)

    replacers_order = dependency_graph.determined_sort()
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
            if hasattr(replacer, 'run_not_recursively'):
                replacer.find_and_replace_pattern(graph)
            else:
                for_graph_and_each_sub_graph_recursively(graph, replacer.find_and_replace_pattern)

            if hasattr(replacer, 'force_clean_up') and replacer.force_clean_up:
                for_graph_and_each_sub_graph_recursively(
                    graph,
                    graph_clean_up_tf if graph.graph['fw'] == 'tf' else
                    graph_clean_up_onnx if graph.graph['fw'] == 'onnx' else
                    graph_clean_up)

            if hasattr(replacer, 'force_shape_inference') and replacer.force_shape_inference:
                shape_inference(graph)

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
