# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os
from enum import Enum

import networkx as nx

from openvino.tools.mo.front.common.custom_replacement_registry import CustomReplacementRegistry
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.middle.passes.eliminate import shape_inference
from openvino.tools.mo.middle.pattern_match import for_graph_and_each_sub_graph_recursively
from openvino.tools.mo.utils.error import Error, InternalError, FrameworkError
from openvino.tools.mo.utils.logger import progress_bar  # pylint: disable=no-name-in-module,import-error

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
    IR_READER_EXTENDER = 5
    LOADER = 6


def _update(cls, registered_list: list, registered_dict: dict, key: str, enabled_transforms: list,
            disabled_transforms: list, exclude_modules: set):
    new_keys = {}  # maps a custom name to class
    new_keys_lower = {}  # translates lowered custom name to its original form
    # print('Registering new subclasses for', cls)

    for c in cls.__subclasses__():
        if need_exclude_class(c, exclude_modules):
            continue
        # Force enabling operations
        if hasattr(c, 'id') and c.id in enabled_transforms or \
                ".".join([c.__module__, c.__name__]) in enabled_transforms:
            setattr(c, 'enabled', True)

        # Force disabling operations
        if hasattr(c, 'id') and c.id in disabled_transforms or \
                ".".join([c.__module__, c.__name__]) in disabled_transforms:
            setattr(c, 'enabled', False)

        if c not in registered_list:
            if hasattr(cls, 'excluded_classes') and c in cls.excluded_classes:
                continue
            registered_list.append(c)
            log.info('New subclass: {}'.format(c))
            if hasattr(c, key) and getattr(c, key) is not None:
                k = getattr(c, key)
                if k.lower() in new_keys_lower:
                    # log.warning('Attempt to register of custom name {} for the second time as class {}. '
                    #             'Note that custom names are case-insensitive. ' + refer_to_faq_msg(55), k, c)
                    continue
                else:
                    new_keys_lower[k.lower()] = k
                    new_keys[k] = c
                    log.info('Registered a new subclass with key: {}'.format(k))
        else:
            log.warning('Skipped {} registration because it was already registered or it was disabled. '.format(c))
    registered_dict.update(new_keys)


def update_registration(classes: list, enabled_transforms: list, disabled_transforms: list, exclude_modules: set):
    for cls in classes:
        _update(cls, cls.registered_cls, cls.registered_ops, 'op', enabled_transforms, disabled_transforms, exclude_modules)
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
        string += 'node [color=lightblue2, style=filled, shape=box];\n'

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


def need_exclude_class(class_type, excluded_frameworks):
    for framework in excluded_frameworks:
        if "." + framework + "." in str(class_type):
            return True
    return False


def get_replacers_order(transform_types: list):
    """
    Gets all transforms that do not have 'op'.
    If two or more classes replaces the same op (both have op class attribute and values match), such
    pattern is not applied (while registration it will warn user that we have a conflict).
    """
    dependency_graph = DependencyGraph(name="UnifiedPipeline" if len(transform_types) != 1 else transform_types[0].name)

    replacers = []
    for class_type, classes_set in _registered_classes_dict.items():
        if class_type in transform_types:
            for cls in classes_set:
                cur_cls_replacers = [c for c in cls.registered_cls if not hasattr(c, 'op')] + \
                                    [c for op, c in cls.registered_ops.items() if c]
                replacers.extend(
                    [replacer for replacer in cur_cls_replacers if replacer not in cls.excluded_replacers])

    for replacer_cls in replacers:
        dependency_graph.add_node(replacer_cls)

    for i, replacer_cls in enumerate(replacers):
        for cls_after in replacer_cls().run_before():
            if cls_after in replacers:
                dependency_graph.add_edge(replacer_cls, cls_after)
        for cls_before in replacer_cls().run_after():
            if cls_before in replacers:
                dependency_graph.add_edge(cls_before, replacer_cls)

    replacers_order = dependency_graph.determined_sort()

    debug_msg_list = ['|  id  | enabled | class ']
    for i, replacer_cls in enumerate(replacers_order):
        debug_msg_list.append('|{:5} |{:^9}| {}'.format(i, str(getattr(replacer_cls, 'enabled', None)), replacer_cls))
    log.debug('Replacers execution order: \n{}'.format('\n'.join(debug_msg_list)))

    return replacers_order


@progress_bar
def apply_transform(graph: Graph, replacer_cls, **kwargs):
    """
    Safely executes transform if it should be and validates graph after transform execution
    """
    replacer = replacer_cls()
    replacement_id = 'REPLACEMENT_ID'
    if hasattr(replacer, 'replacement_id'):
        replacement_id = replacer.replacement_id

    if hasattr(replacer, 'enabled') and not replacer.enabled:
        log.info("Skip replacer {} (enabled = False)".format(replacer_cls))
        return

    if hasattr(replacer, 'graph_condition') and \
            not all([condition(graph) for condition in replacer.graph_condition]):
        log.info("Skip replacer {} (graph_condition not satisfied)".format(replacer_cls))
        return

    log.debug("Run replacer {}".format(replacer_cls))

    try:
        if hasattr(replacer, 'run_not_recursively') and replacer.run_not_recursively:
            replacer.find_and_replace_pattern(graph)
        else:
            for_graph_and_each_sub_graph_recursively(graph, replacer.find_and_replace_pattern)

        if hasattr(replacer, 'force_clean_up') and replacer.force_clean_up:
            for_graph_and_each_sub_graph_recursively(graph, lambda G: G.clean_up())

        if hasattr(replacer, 'force_shape_inference') and replacer.force_shape_inference:
            shape_inference(graph)

        if hasattr(replacer, 'run_not_recursively') and replacer.run_not_recursively:
            graph.check_empty_graph(replacer_cls)
            graph.check_shapes_consistency()
        else:
            for_graph_and_each_sub_graph_recursively(graph, lambda _: graph.check_empty_graph(replacer_cls))
            for_graph_and_each_sub_graph_recursively(graph, lambda _: graph.check_shapes_consistency())

    except Error as err:
        raise Error('Exception occurred during running replacer "{}" ({}): {}'.format(
            replacement_id,
            replacer_cls,
            str(err).replace('[REPLACEMENT_ID]', replacement_id),
        )) from err
    except FrameworkError as err:
        raise FrameworkError('{}'.format(str(err))) from err
    except Exception as err:
        raise Exception('Exception occurred during running replacer "{} ({})": {}'.format(
            replacement_id,
            replacer_cls,
            str(err).replace('[REPLACEMENT_ID]', replacement_id),
        )) from err


def apply_replacements_list(graph: Graph, replacers_order: list):
    """
    Apply all transformations from replacers_order
    """
    for i, replacer_cls in enumerate(replacers_order):
        apply_transform(
            graph=graph,
            replacer_cls=replacer_cls,
            curr_transform_num=i,
            num_transforms=len(replacers_order))


def apply_replacements(graph: Graph, replacements_type: list):
    """
    Apply all patterns that do not have 'op' first, then apply patterns from registered_ops.
    If two or more classes replaces the same op (both have op class attribute and values match), such
    pattern is not applied (while registration it will warn user that we have a conflict).
    """
    replacers_order = get_replacers_order(replacements_type)
    apply_replacements_list(graph, replacers_order)


def clear_registered_classes_dict():
    CustomReplacementRegistry.registry = {}
    _registered_classes_dict.clear()
