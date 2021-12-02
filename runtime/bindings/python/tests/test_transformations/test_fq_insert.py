# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import numpy as np

from openvino import Function, PartialShape, opset8
import openvino as ov

from openvino.impl.passes import Manager, Matcher, MatcherPass, WrapType, Serialize


def find_operation_matches(ops, op):
    return True


def is_ignored(ops, op):
    return False


def create_fq(output):
    shape = [x for x in output.get_shape()]
    shape[1] = output.get_shape()[1]

    return opset8.fake_quantize(
        output,
        opset8.constant(np.ones((shape), dtype=np.float32)),
        opset8.constant(np.ones((shape), dtype=np.float32)),
        opset8.constant(np.ones((shape), dtype=np.float32)),
        opset8.constant(np.ones((shape), dtype=np.float32)),
        255,
    )


def insert_fake_quantize(input_ports):
    for port in input_ports:
        fq = create_fq(port.get_source_output())
        port.replace_source_output(fq.output(0))


class InsertFakeQuantize(MatcherPass):
    @property
    def quantize_operations(self):
        return getattr(self, '_quantize_operations', [])

    @quantize_operations.setter
    def quantize_operations(self, value):
        setattr(self, '_quantize_operations', value)

    @property
    def ignored_params(self):
        return getattr(self, '_ignored_params', {'skip_model': False, 'scope': [], 'operations': []})

    @ignored_params.setter
    def ignored_params(self, value):
        setattr(self, '_ignored_params', value)

    @staticmethod
    def is_const(output):
        return output.get_node().get_type_name() == "Constant"

    @staticmethod
    def quantize_only_input(node):
        type_name = node.get_type_name()
        if type_name in ['Interpolate', 'Power', 'ReduceMean', 'NormalizeL2',
                         'Assign', 'PReLU', 'ReLU', 'Sigmoid', 'Tanh', 'Clamp']:
            return True
        # ScaleSift case, FQ only for input
        if type_name == 'Multiply' and InsertFakeQuantize.is_const(node.input_value(1)):
            output_node = node.output(0).get_target_inputs()[0]
            if output_node.get_type_name() == 'Add' and InsertFakeQuantize.is_const(output_node.input_value(1)):
                # logger.debug('Scaleshift found at {}->{}'.format(node.name, output_node.name))
                return True
        return False

    def register_pattern(self):
        pattern = WrapType([op_type for op_type in self.quantize_operations])

        def callback(m: Matcher):
            m_op = m.get_match_root()
            if not find_operation_matches(self.quantize_operations, m_op) \
                    or is_ignored(self.ignored_params, m_op):
                return False

            if m_op.get_type_name() in ['Convolution', 'GroupConvolution', 'ConvolutionBackpropData',
                                        'GroupConvolutionBackpropData', 'MatMul']:
                insert_fake_quantize([m_op.input(0), m_op.input(1)])
            elif self.quantize_only_input(m_op):
                insert_fake_quantize([m_op.input(0)])
            else:
                insert_fake_quantize(m_op.inputs())
            print("DONE!")
            return True

        self.register_matcher(Matcher(pattern, "InsertFakeQuantize"), callback)


def get_test_function():
    param = opset8.parameter(PartialShape([1, 3, 22, 22]), name="parameter")
    relu = opset8.relu(param.output(0))
    res = opset8.result(relu.output(0), name="result")
    return Function([res], [param], "test")


def pot_pipeline():
    m = Manager()
    m.register_pass(Serialize("/tmp/before.xml", "/tmp/before.bin"))

    fq_insert = m.register_pass(InsertFakeQuantize())
    fq_insert.quantize_operations = ['opset8.Relu', 'opset8.Convolution']  # Specify pattern operation types
    fq_insert.register_pattern()

    m.register_pass(Serialize("/tmp/after.xml", "/tmp/after.bin"))
    m.run_passes(get_test_function())


pot_pipeline()
