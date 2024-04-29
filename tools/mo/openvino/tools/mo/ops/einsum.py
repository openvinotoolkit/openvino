# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import shape_array
from openvino.tools.mo.graph.graph import Node, Graph
from openvino.tools.mo.ops.op import Op
from openvino.tools.mo.utils.broadcasting import bi_directional_shape_broadcasting


class Einsum(Op):
    op = 'Einsum'
    enabled = False

    def __init__(self, graph: Graph, attrs: dict):
        mandatory_props = {
            'type': self.op,
            'op': self.op,
            'version': 'opset7',
            'infer': self.infer,
            'out_ports_count': 1,
        }
        super().__init__(graph, mandatory_props, attrs)

    def backend_attrs(self):
        return ['equation']

    @staticmethod
    def is_label_elsewhere(input_subscripts: list, label_to_check: str, excluded_subscript_inds: list) -> bool:
        """
        Check if the given label is met in input subscripts excluding ones specified by a list of indices
        excluded_subscript_inds

        :param input_subscripts: input subscripts among which to check if the label is met
        :param label_to_check: a label to check
        :param excluded_subscript_inds: indices of input subscripts to be excluded for this check
        :return: True - met, False - otherwise
        """
        for ind, input_subscript in enumerate(input_subscripts):
            if ind not in excluded_subscript_inds and label_to_check in input_subscript:
                return True
        return False

    @staticmethod
    def parse_equation(node_name: str, equation: str) -> (list, str):
        """
        Parse Einsum equation and check that its format is correct to make sure that
        all input subscripts consists of only alphabetic letters or alphabetic letters with one ellipsis.
        In case of implicit mode the method recovers the right-hand part.

        :param node_name: Einsum node name for which to parse an equation
        :param equation: Equation to be parsed and checked
        :return: A tuple of a list of input subscripts and output subscript
        """
        # normalize equation by removing white-spaces
        equation = equation.strip()

        # split equation into the left and right hands
        splitted_equation = equation.split('->')
        assert len(splitted_equation) <= 2, "Einsum node {} has `equation` of incorrect format".format(node_name)

        # split left-hand side of the equation and check a format of input subscripts
        input_subscripts = splitted_equation[0]
        input_subscripts_list = input_subscripts.split(',')

        # prepare pattern to check a format of subscripts
        subscript_pattern = re.compile("^[a-zA-Z]*(\\.\\.\\.){0,1}[a-zA-Z]*$")
        ellipsis_pattern = re.compile("\\.\\.\\.")

        is_ellipsis_met = False
        for input_subscript in input_subscripts_list:
            assert re.match(subscript_pattern, input_subscript) is not None, \
                "Einsum node {} has `equation` with incorrect input subscript: {}".format(node_name, input_subscript)
            is_ellipsis_met = is_ellipsis_met or re.search(ellipsis_pattern, input_subscript)

        if len(splitted_equation) == 2:
            output_subscript = splitted_equation[1]
            assert re.match(subscript_pattern, output_subscript), \
                "Einsum node {} has `equation` with incorrect output subscript: {}".format(node_name, output_subscript)
            # if ellipsis is met, the output subscript must contain it as well
            if is_ellipsis_met:
                assert re.search(ellipsis_pattern, output_subscript), \
                    "The output subscript of Einsum node {} must contain ellipsis".format(node_name)
        elif len(splitted_equation) == 1:
            # recover output subscript in case implicit mode
            output_subscript = ""
            for ind, input_subscript in enumerate(input_subscripts_list):
                labels = Einsum.extract_subscript_labels(node_name, input_subscript)
                for label in labels:
                    if Einsum.is_label_elsewhere(input_subscripts_list, label, [ind]) is False:
                        output_subscript += label
            output_subscript = ''.join(sorted(list(set(output_subscript) - {'.'})))
            if is_ellipsis_met:
                output_subscript = "..." + output_subscript
        else:
            assert False, "Einsum node {} equation has incorrect format. " \
                          "It must be in either explicit or implicit mode.".format(node_name)

        return input_subscripts_list, output_subscript

    @staticmethod
    def normalize_equation(node_name: str, equation: str) -> str:
        """
        Recover explicit mode of equation.

        :param node_name: Einsum node name for which to recover explicit mode
        :param equation: Einsum equation to recover explicit mode
        :return: Recovered equation in explicit mode
        """
        input_subscripts_list, output_subscript = Einsum.parse_equation(node_name, equation)
        return ','.join(input_subscripts_list) + "->" + output_subscript

    @staticmethod
    def extract_subscript_labels(node_name: str, subscript: str) -> list:
        """
        Extract labels for given subscript. Each label can be either alphabetic letter or ellipsis

        :param node_name: Einsum node name
        :param subscript: Given subscript
        :return: A list of labels
        """
        labels = []
        len_subscript = len(subscript)
        label_ind = 0
        while label_ind < len_subscript:
            if subscript[label_ind].isalpha():
                labels.append(subscript[label_ind])
                label_ind += 1
            elif len_subscript - label_ind > 2 and subscript[label_ind:label_ind + 3] == "...":
                labels.append("...")
                label_ind += 3
            else:
                assert False, "Einsum node {} has `equation` with incorrect subscript: {}".format(node_name, subscript)
        return labels

    @staticmethod
    def adjust_equation_with_NCHW_layout(node_name: str, equation: str, input_ranks: list, output_rank: int,
                                         input_correct_layout_mask: list, output_correct_layout_mask: bool) -> (
            str, list, bool):
        """
        In order to satisfy NCHW layout, subscripts for tensors with rank greater than three must be adjusted by moving labels
        of the last dimension to the second position in the subscript. There is an exception for such tensors when
        the label is ellipsis and it covers multiple tail dimensions. The method returns equation with adjusted subscripts
        to NCHW layout along with a boolean mask to indicate which subscripts are adjusted.

        :param node_name: Einsum node name for which equation is adjusted
        :param equation: Equation to be adjusted
        :param input_ranks: a list of input ranks
        :param output_rank: output rank
        :return: adjusted equation, boolean mask for inputs, and boolean flag if output subscript is adjusted
        """
        is_inputs_adjusted = []
        input_subscripts, output_subscript = Einsum.parse_equation(node_name, equation)
        num_inputs = len(input_ranks)
        assert len(input_subscripts) == num_inputs, "The number of inputs must match a number " \
                                                    "of input subscripts"
        assert len(input_correct_layout_mask) == num_inputs, "The number of inputs must match a number " \
                                                             "elements in input_correct_layout_mask list"

        # permute labels in input subscripts and mark inputs for which inference in NCHW layout is acceptable
        # in case ellipsis covering multiple dimensions in the end, the permutation is impossible
        # so the corresponding input must be in the original format (NHWC)
        permuted_input_subscripts = []
        for input_ind in range(num_inputs):
            input_subscript = input_subscripts[input_ind]
            input_rank = input_ranks[input_ind]
            labels = Einsum.extract_subscript_labels(node_name, input_subscript)
            num_broadcasted_dims = input_rank - len(labels) + 1
            if input_correct_layout_mask[input_ind]:
                is_inputs_adjusted.append(True)
            elif input_rank > 3 and (labels[-1] != "..." or labels[-1] == "..." and num_broadcasted_dims == 1):
                is_inputs_adjusted.append(True)
                labels.insert(1, labels[-1])
                del labels[-1]
            else:
                is_inputs_adjusted.append(False)
            permuted_input_subscript = ''.join(labels)
            permuted_input_subscripts.append(permuted_input_subscript)

        # perform the same procedure for the output subscript as for the inputs subscripts
        labels = Einsum.extract_subscript_labels(node_name, output_subscript)
        num_broadcasted_dims = output_rank - len(labels) + 1
        if output_correct_layout_mask:
            is_output_adjusted = True
        elif output_rank > 3 and (labels[-1] != "..." or labels[-1] == "..." and num_broadcasted_dims == 1):
            is_output_adjusted = True
            labels.insert(1, labels[-1])
            del labels[-1]
        else:
            is_output_adjusted = False
        permuted_output_subscript = ''.join(labels)

        # concatenate the left and right hands of the resulted equation
        left_hand = ','.join(permuted_input_subscripts)
        right_hand = permuted_output_subscript
        permuted_equation = left_hand + "->" + right_hand
        return permuted_equation, is_inputs_adjusted, is_output_adjusted

    @staticmethod
    def infer(node: Node):
        node_name = node.soft_get('name', node.id)
        connected_in_ports = [port for port in node.in_ports().values() if not port.disconnected()]
        num_inputs = len(connected_in_ports)
        assert node.has_valid('equation'), "Einsum node {} must contain `equation` attribute".format(node_name)
        equation = node.equation

        # parse the equation and extract input and output subscripts
        input_subscripts, output_subscript = Einsum.parse_equation(node_name, equation)

        # check that each operand has the corresponding input subscript
        assert len(input_subscripts) == num_inputs, "The number of input operands of Einsum node {} " \
                                                    "must match the number of input subscripts " \
                                                    "in `equation`".format(node_name)

        # check compatibility of dimension sizes with the same label and generate a dictionary of shapes for labels
        label_to_shape = {}
        for input_ind in range(num_inputs):
            input_shape = node.in_port(input_ind).data.get_shape()
            input_subscript = input_subscripts[input_ind]
            labels = Einsum.extract_subscript_labels(node_name, input_subscript)
            num_dims = len(input_shape)
            num_labels = len(labels)
            num_broadcasted_dims = num_dims - num_labels + 1
            dim_ind = 0
            label_ind = 0
            while label_ind < num_labels and dim_ind < num_dims:
                label = labels[label_ind]
                if label == "...":
                    sub_shape = input_shape[dim_ind:dim_ind + num_broadcasted_dims]
                    if label in label_to_shape.keys():
                        common_shape = bi_directional_shape_broadcasting(sub_shape, label_to_shape[label])
                        assert common_shape is not None, "The dimensions labeled of ellipsis must be broadcastable " \
                                                         "for Einsum node {}".format(node_name)
                        label_to_shape[label] = common_shape
                    else:
                        label_to_shape[label] = sub_shape
                    dim_ind += num_broadcasted_dims
                else:
                    dim_size = input_shape[dim_ind]
                    sub_shape = shape_array([dim_size])
                    assert label not in label_to_shape.keys() or np.array_equal(label_to_shape[label], sub_shape), \
                        "Sizes of dimensions with the same label of Einsum node {} " \
                        "must be compatible".format(node_name)
                    label_to_shape[label] = sub_shape
                    dim_ind += 1
                label_ind += 1

        # generate output shape based on the output subscript
        output_shape = shape_array([])
        labels = Einsum.extract_subscript_labels(node_name, output_subscript)
        for label in labels:
            assert label in label_to_shape.keys(), "The label in the output subscript must appear" \
                                                   " in input subscripts in equation {} " \
                                                   "of Einsum node {}".format(equation, node_name)
            output_shape = np.ma.concatenate((output_shape, label_to_shape[label]))

        node.out_port(0).data.set_shape(output_shape)
