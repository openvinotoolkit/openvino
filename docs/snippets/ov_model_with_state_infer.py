# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import numpy as np
import sys

import openvino  as ov
from openvino.runtime import opset12 as ops


def main():
    # 1. Load inference engine
    log.info("Loading OpenVINO")
    core = ov.Core()

    #! [ov:model_create]
    # 2. Creating ov.Model
    input = ops.parameter([1, 1], dtype=np.float32, name="data")
    init_const = ops.constant([0], dtype=np.float32)
    read = ops.read_value(init_const, "variable0")
    add = ops.add(input, read)
    assign = ops.assign(add, "variable0")
    add2 = ops.add(add, read)
    result = ops.result(add2)
    model = ov.Model(results=[result], sinks=[assign], parameters=[input], name="model")
     #! [ov:model_create]

    log.info("Loading network files")

    # 3. Load network to CPU
    compiled_model = core.compile_model(model, "CPU")
    # 4. Create Infer Request
    infer_request = compiled_model.create_infer_request()

    # 5. Prepare inputs
    input_tensors = []
    for input in compiled_model.inputs:
        input_tensors.append(infer_request.get_tensor(input))

    # 6. Prepare outputs
    output_tensors = []
    for output in compiled_model.outputs:
        output_tensors.append(infer_request.get_tensor(output))

    # 7. Initialize memory state before starting
    for state in infer_request.query_state():
        state.reset()

    #! [ov:part1]
    # input data
    input_data = np.arange(start=1, stop=7, dtype=np.float32)
    log.info("Infer the first utterance")
    for next_input in range(int(len(input_data)/2)):
        infer_request.infer({"data" : np.asarray([input_data[next_input]]).reshape([1,1])})
        # check states
        states = infer_request.query_state()
        if len(states) == 0:
            log.error("Queried states are empty")
            return -1
        mstate = states[0].state
        if not mstate:
            log.error("Can't cast state to MemoryBlob")
            return -1
        state_buf = mstate.data
        log.info(state_buf[0])

    log.info("\nReset state between utterances...\n")
    for state in infer_request.query_state():
        state.reset()
        
    log.info("Infer the second utterance")
    for next_input in range(int(len(input_data)/2), len(input_data)):
        infer_request.infer({0 : np.asarray([input_data[next_input]]).reshape([1,1])})
        # check states
        states = infer_request.query_state()
        mstate = states[0].state
        state_buf = mstate.data
        log.info(state_buf[0])
    #! [ov:part1]

    log.info("Execution successful")

    return 0
