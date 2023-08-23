# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np
import sys

import openvino as ov
from openvino.runtime.passes import LowLatency2, MakeStateful, Manager


def state_network_example():
    #! [ov:state_network]
    input = ov.opset10.parameter([1, 1], dtype=np.float32)
    read = ov.opset10.read_value(input, "variable0")
    add = ov.opset10.add(read, input)
    save = ov.opset10.assign(add, "variable0")
    result = ov.opset10.result(add)
    model = ov.Model(results=[result], sinks=[save], parameters=[input])
    #! [ov:state_network]


def low_latency_2_example():
    #! [ov:low_latency_2]
    # Precondition for Model.
    # TensorIterator and Parameter are created in body of TensorIterator with names
    tensor_iterator_name = "TI_name"
    body_parameter_name = "body_parameter_name"
    idx = "0" # this is a first variable in the network

    # The State will be named "TI_name/param_name/variable_0"
    state_name = tensor_iterator_name + "//" + body_parameter_name + "//" + "variable_" + idx # todo

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
    hd_specific_model = ov.compile_model(ov_model)
    # Try to find the Variable by name
    infer_request = hd_specific_model.create_infer_request()
    states = infer_request.query_state()
    for state in states:
        name = state.get_name()
        if (name == state_name):
            # some actions
    #! [ov:low_latency_2]

    #! [ov:low_latency_2_use_parameters]
    manager.register_pass(LowLatency2(False))
    #! [ov:low_latency_2_use_parameters]


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
    # ov.opset10.parameter/ov.opset10.result in the ov_model
    pairs = ["""(Parameter_1, Result_1), (Parameter_3, Result_3)"""]
    manager = Manager()
    manager.register_pass(MakeStateful(pairs))
    manager.run_passes(ov_model)
    #! [ov:make_stateful_ov_nodes]


def main():
    #! [ov:state_api_usage]
    # 1. Load inference engine
    log.info("Loading Inference Engine")
    core = ov.Core()

    # 2. Read a model
    log.info("Loading network files")
    model = core.read_model("path_to_the_model")
    

    # 3. Load network to CPU
    hw_specific_model = ov.compile_model(model, "CPU")

    # 4. Create Infer Request
    infer_request = hw_specific_model.create_infer_request()

    # 5. Reset memory states before starting
    states = infer_request.query_state()
    if (states.size() != 1):
        log.error(f"Invalid queried state number. Expected 1, but got {str(states.size())}")
        return -1
    
    for state in states:
        state.reset()

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
    v = np.asarray([5], dtype=np.float32)
    tensor = ov.Tensor(v, shared_memory=True)
    target_state.state = tensor

    log.info("Infer the third utterance")
    for next_input in range((input_data.size()/3 * 2), input_data.size()):
        infer_request.infer({0 : input_data[next_input]})

        state_buf = target_state.state.data
        log.info(state_buf[0])

    log.info("Execution successful")
    #! [ov:state_api_usage]
    return 0


if __name__ == '__main__':
    sys.exit(main())
