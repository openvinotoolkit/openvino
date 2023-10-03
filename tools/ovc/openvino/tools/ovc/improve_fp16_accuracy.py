from typing import Any, List, Dict, Union, Tuple

from openvino.runtime import Model  # pylint: disable=no-name-in-module,import-error
from openvino.tools.ovc.logger import get_logger_state, restore_logger_state
import numpy as np
from openvino.runtime import Core, serialize
from openvino.runtime.op import Parameter, Constant
from openvino.runtime.opset12 import add, multiply, matmul, convert
import openvino as ov
from openvino.runtime.utils.types import get_element_type
from openvino._pyopenvino import Node


def improve_fp16_accuracy(orig_model: Model, example_input: Union[List, Dict]) -> Model:
    nodes_to_track, outs_to_track, _ = insert_results_for_tracked_ops(orig_model)
    fp16_infer_values = try_first_f16_infer(orig_model, example_input, nodes_to_track)
    fp32_infer_values = compare_fp16_fp32_inference(nodes_to_track, fp16_infer_values)
    nodes_to_fix = analyse_diff(nodes_to_track, fp16_infer_values, fp32_infer_values)
    upcast_nodes(nodes_to_fix)

    # insert rt_info for them
    # check that save_model preserves this rt_info

    # add unit-test for this mechanism
    pass


def insert_results_for_tracked_ops(model) -> (Dict, List, List):
    # todo: write only limited list of excluded ops like Slice, Reshape, Transpose
    ops_to_track = [
        'MatMul',
        # 'SoftMaX',
        # 'ReduceSum',
        # 'ReduceMean',
        # 'Exp',
        # 'Interpolate'
    ]

    # additional outputs to track inputs and output values of operations of interest
    nodes_to_track = []
    outputs = []
    for i, op in enumerate(model.get_ordered_ops()):
        if op.get_type_name() not in ops_to_track:
            continue
        outputs.append(op.output(0).get_any_name())
        nodes_to_track.append(op)

        node_0 = op.input(0).get_source_output().get_node()
        node_1 = op.input(1).get_source_output().get_node()
        for node in [node_0, node_1]:
            if node.get_type_name() not in ['Constant', 'Convert']:  # for Consts we can take inputs from ov::Model
                outputs.append(node.output(0).get_any_name())

    orig_outputs = model.outputs.copy()
    model.add_outputs(outputs)
    return nodes_to_track, outputs, orig_outputs


def get_const_value_from_ovmodel(node: Union[Constant, Node]) -> np.ndarray:
    node_type = node.get_type_name()
    if node_type == 'Constant':
        assert node.get_element_type() == ov.Type.f32
        return node.get_data()
    elif node_type == 'Convert':
        # if model is compressed and constant values flow through decompression convert
        const_node = node.input_value(0).get_node()
        assert const_node.get_type_name() == 'Constant'
        assert const_node.get_element_type().is_real()
        return np.array(node.input_value(0).get_node().get_data(), dtype=np.float32)
    else:
        raise Exception(
            f'Cannot get const values from ov.Model for node {node.get_friendly_name()} with type {node.get_type_name()}')


def try_first_f16_infer(orig_model: ov.Model, example_inputs: List, nodes_to_track: List[Node]) -> List[Tuple]:
    ie = Core()
    exec_net = ie.compile_model(orig_model, 'GPU', config={"INFERENCE_PRECISION_HINT": "f16"})
    request = exec_net.create_infer_request()
    request.infer(example_inputs)

    node_data_values = []  # each item contains a tuple with node output values and all input values
    for node in nodes_to_track:
        res_item = [request.get_tensor(node.output(0).get_any_name()).data]
        for input in node.input_values():
            if input.get_node().get_type_name() not in ['Constant', 'Convert']:
                res_item.append(request.get_tensor(input.get_any_name()).data)
            else:
                res_item.append(get_const_value_from_ovmodel(input.get_node()))
        node_data_values.append(tuple(res_item))
    del request, exec_net
    return node_data_values


def compare_fp16_fp32_inference(nodes_to_track: List[Node], node_data_values: List[Tuple]) -> List:
    results = []
    for node, value in zip(nodes_to_track, node_data_values):
        results.append(infer_tracked_op(node, value[1:]))
    return results


def infer_tracked_op(op: Node, input_vals: Union[List, Dict, Tuple]) -> np.ndarray:
    parameters = []
    # todo: if inputs is a dict add names to parameters
    for input_val in input_vals:
        parameters.append(Parameter(get_element_type(input_val.dtype), ov.PartialShape(input_val.shape)))

    if op.get_type_name() != 'MatMul':
        # todo: implement for other ops
        raise NotImplementedError(f"inference track for operations {op.get_type_name()} are not implemented yet")

    trans = op.get_attributes()['transpose_a'], op.get_attributes()['transpose_b']
    new_op = matmul(*parameters, *trans)
    ov_model = ov.Model([new_op], parameters)

    ie = Core()
    exec_net = ie.compile_model(ov_model, 'GPU', config={"INFERENCE_PRECISION_HINT": "f32"})
    request = exec_net.create_infer_request()
    result = request.infer(input_vals)[0]
    del request, exec_net, ov_model
    return result


def analyse_diff(nodes: List[Node], fp16_infer_vals: List, fp32_infer_vals: List) -> List[Node]:
    nodes_with_errors = []
    for node, fp16_val, fp32_val in zip(nodes, fp16_infer_vals, fp32_infer_vals):
        if compare_tensors(fp16_val, fp32_val):
            nodes_with_errors.append(node)


def compare_tensors(a, b) -> bool:
    """
    If values differ more than a certain metric then function returns True
    """
    assert np.array_equal(a.shape, b.shape), f'Shapes differ {a.shape} and {b.shape}'
    out_size = int(np.prod(a.shape))
    a_, b_ = np.reshape(a, out_size), np.reshape(b, out_size)

    rel_diff_a = np.abs((a_ - b_) / a_)
    rel_diff_b = np.abs((a_ - b_) / b_)
    rel_error = np.min([rel_diff_a, rel_diff_b], axis=0)
    mean_rel_error = np.mean(rel_error)
    rel_tol = 0.1
    if mean_rel_error < rel_tol:
        return False

    rel_threshold = 0.1  # if more than 10% of values differ more than
    # number of values that have rel diff more rel_threshold
    num_rel_diff = np.size(np.where(rel_error >= rel_threshold))
    if num_rel_diff / out_size > 0.25:
        return True

def upcast_nodes(nodes: List[Node]) -> None:
    for node in nodes:
        pass