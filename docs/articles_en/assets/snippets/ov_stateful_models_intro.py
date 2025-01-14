# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np

import openvino  as ov
from openvino.runtime import opset13 as ops
from openvino.runtime.op.util import VariableInfo, Variable
from openvino.runtime.passes import LowLatency2, MakeStateful, Manager
from openvino.runtime.utils import replace_node


def state_model_example():
    #! [ov:stateful_model]
    input = ops.parameter([1, 1], dtype=np.float32, name="data")
    init_const = ops.constant([[0]], dtype=np.float32)

    # Typically ReadValue/Assign operations are presented as pairs in models.
    # ReadValue operation reads information from an internal memory buffer, Assign operation writes data to this buffer.
    # For each pair, its own Variable object must be created.
    # Variable defines name, shape and type of the buffer.
    var_info = VariableInfo()
    var_info.data_shape = init_const.get_shape()
    var_info.data_type = init_const.get_element_type()
    var_info.variable_id = "variable0"
    variable = Variable(var_info)

    # Creating Model
    read = ops.read_value(init_const, variable)
    add = ops.add(input, read)
    assign = ops.assign(add, variable)
    result = ops.result(add)
    model = ov.Model(results=[result], sinks=[assign], parameters=[input], name="model")
    #! [ov:stateful_model]

    return model


def low_latency_2_example():
    #! [ov:low_latency_2]
    # Precondition for Model.
    # TensorIterator and Parameter are created in body of TensorIterator with names
    tensor_iterator_name = "TI_name"
    body_parameter_name = "body_parameter_name"
    idx = "0" # this is a first variable in the model

    # The State will be named "TI_name/param_name/variable_0"
    state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx

    #! [ov:get_ov_model]
    core = ov.Core()
    ov_model = core.read_model("path_to_the_model")
    #! [ov:get_ov_model]

    # reshape input if needed

    #! [ov:reshape_ov_model]
    ov_model.reshape({"X": ov.PartialShape([1, 1, 16])})
    #! [ov:reshape_ov_model]

    #! [ov:apply_low_latency_2]
    manager = Manager()
    manager.register_pass(LowLatency2())
    manager.run_passes(ov_model)
    #! [ov:apply_low_latency_2]

    compied_model = core.compile_model(ov_model)
    # Try to find the Variable by name
    infer_request = compied_model.create_infer_request()
    states = infer_request.query_state()
    for state in states:
        name = state.get_name()
        if (name == state_name):
            # some actions
    #! [ov:low_latency_2]
            pass

    #! [ov:low_latency_2_use_parameters]
    manager.register_pass(LowLatency2(False))
    #! [ov:low_latency_2_use_parameters]


def replace_non_reshapable_const():
    #! [ov:replace_const]
    # OpenVINO example. How to replace a Constant with hardcoded values of shapes in the model with another one with the new values.
    # Assume we know which Constant (const_with_hardcoded_shape) prevents the reshape from being applied.
    # Then we can find this Constant by name in the model and replace it with a new one with the correct shape.
    core = ov.Core()
    model = core.read_model("path_to_model");
    # Creating the new Constant with a correct shape.
    # For the example shown in the picture above, the new values of the Constant should be 1, 1, 10 instead of 1, 49, 10
    new_const = ops.constant( """value_with_correct_shape, type""")
    for node in model.get_ops():
        # Trying to find the problematic Constant by name.
        if node.get_friendly_name() != "name_of_non_reshapable_const":
            continue
        # Replacing the problematic Constant with a new one. Do this for all the problematic Constants in the model, then
        # you can apply the reshape feature.
        replace_node(node, new_const)

    #! [ov:replace_const]


def apply_make_stateful_tensor_names():
    #! [ov:make_stateful_tensor_names]
    core = ov.Core()
    ov_model = core.read_model("path_to_the_model")
    tensor_names = {"tensor_name_1": "tensor_name_4",
                    "tensor_name_3": "tensor_name_6"}
    manager = Manager()
    manager.register_pass(MakeStateful(tensor_names))
    manager.run_passes(ov_model)
    #! [ov:make_stateful_tensor_names]


def apply_make_stateful_ov_nodes():
    #! [ov:make_stateful_ov_nodes]
    core = ov.Core()
    ov_model = core.read_model("path_to_the_model")
    # Parameter_1, Result_1, Parameter_3, Result_3 are 
    # ops.parameter/ops.result in the ov_model
    pairs = ["""(Parameter_1, Result_1), (Parameter_3, Result_3)"""]
    manager = Manager()
    manager.register_pass(MakeStateful(pairs))
    manager.run_passes(ov_model)
    #! [ov:make_stateful_ov_nodes]


def main():

    #! [ov:state_api_usage]
    # 1. Load inference engine
    log.info("Loading OpenVINO")
    core = ov.Core()

    # 2. Read a model
    log.info("Loading model files")
    model = core.read_model("path_to_ir_xml_from_the_previous_section");
    model.get_parameters()[0].set_layout("NC");
    ov.set_batch(model, 1);

    # 3. Load the model to CPU
    compiled_model = core.compile_model(model, "CPU")

    # 4. Create Infer Request
    infer_request = compiled_model.create_infer_request()

    # 5. Reset memory states before starting
    states = infer_request.query_state()
        
    if len(states) != 1:
        log.error(f"Invalid queried state number. Expected 1, but got {str(states.size())}")
        return -1

    infer_request.reset_state()

    # 6. Inference
    input_data = np.arange(start=1, stop=12, dtype=np.float32)

    # This example demonstrates how to work with OpenVINO State API.
    # Input_data: some array with 12 float numbers

    # Part1: read the first four elements of the input_data array sequentially.
    # Expected output for the first utterance:
    # sum of the previously processed elements [ 1, 3, 6, 10]

    # Part2: reset state value (set to 0) and read the next four elements.
    # Expected output for the second utterance:
    # sum of the previously processed elements [ 5, 11, 18, 26]

    # Part3: set state value to 5 and read the next four elements.
    # Expected output for the third utterance:
    # sum of the previously processed elements + 5 [ 14, 24, 35, 47]
    target_state = states[0]
    
    # Part 1
    log.info("Infer the first utterance")
    for next_input in range(len(input_data)/3):
        infer_request.infer({0 : input_data[next_input]})
        state_buf = target_state.state.data
        log.info(state_buf[0])

    # Part 2
    log.info("\nReset state between utterances...\n")
    target_state.reset()

    log.info("Infer the second utterance")

    for next_input in range(len(input_data)/3, (len(input_data)/3 * 2)):
        infer_request.infer({0 : input_data[next_input]})
        state_buf = target_state.state.data
        log.info(state_buf[0])

    # Part 3
    log.info("\nSet state value between utterances to 5...\n")
    data = np.asarray([5], dtype=np.float32)
    tensor = ov.Tensor(data, shared_memory=True)
    target_state.state = tensor

    log.info("Infer the third utterance")
    for next_input in range((len(input_data)/3 * 2), len(input_data)):
        infer_request.infer({0 : input_data[next_input]})

        state_buf = target_state.state.data
        log.info(state_buf[0])

    log.info("Execution successful")
    #! [ov:state_api_usage]
    return 0
