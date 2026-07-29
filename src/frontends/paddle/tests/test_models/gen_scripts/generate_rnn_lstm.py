# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
from save_model import saveModel
import sys


def _get_framework_pb2():
    try:
        from paddle.fluid.proto import framework_pb2
        return framework_pb2
    except Exception:
        pass
    try:
        from paddle.base.proto import framework_pb2
        return framework_pb2
    except Exception:
        from paddle.framework.proto import framework_pb2
        return framework_pb2


def _corrupt_weightlist_inputs(model_path: str):
    framework_pb2 = _get_framework_pb2()
    prog = framework_pb2.ProgramDesc()
    with open(model_path, "rb") as f:
        prog.ParseFromString(f.read())

    modified = False
    for block in prog.blocks:
        for op in block.ops:
            for inp in op.inputs:
                if inp.parameter == "WeightList" and len(inp.arguments) > 0:
                    if len(inp.arguments) >= 2:
                        del inp.arguments[-2:]
                    else:
                        del inp.arguments[:]
                    modified = True

    if not modified:
        raise RuntimeError("Failed to modify WeightList inputs in model")

    with open(model_path, "wb") as f:
        f.write(prog.SerializeToString())


def paddle_rnn_lstm(input_size, hidden_size, layers, direction, seq_len, name_override=None):
    import paddle
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    num_of_directions = 1 if direction == 'forward' else 2
    with paddle.static.program_guard(main_program, startup_program):

        rnn = paddle.nn.LSTM(input_size, hidden_size, layers, direction)

        data = paddle.static.data(
            name='x', shape=[4, 3, input_size], dtype='float32')
        prev_h = paddle.ones(
            shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32)
        prev_c = paddle.ones(
            shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32)

        if seq_len:
            seq_lengths = paddle.static.data(name='sl', shape=[4], dtype='int32')
            y, (h, c) = rnn(data, (prev_h, prev_c), seq_lengths)
        else:
            y, (h, c) = rnn(data, (prev_h, prev_c))

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(startup_program)

        if seq_len:
            outs = exe.run(
                feed={'x': np.ones([4, 3, input_size]).astype(
                    np.float32), 'sl': np.array(seq_len).astype(np.int32)},
                fetch_list=[y, h, c],
                program=main_program)
            model_name = name_override or ("rnn_lstm_layer_" + str(layers) + '_' + str(direction) + '_seq_len_' + str(len(seq_len)))
            saveModel(model_name, exe, feed_vars=[data, seq_lengths],
                      fetchlist=[y, h, c], inputs=[np.ones([4, 3, input_size]).astype(np.float32), np.array(seq_len).astype(np.int32)], outputs=outs, target_dir=sys.argv[1])
        else:
            outs = exe.run(
                feed={'x': np.ones([4, 3, input_size]).astype(
                    np.float32)},
                fetch_list=[y, h, c],
                program=main_program)
            model_name = name_override or ("rnn_lstm_layer_" + str(layers) + '_' + str(direction))
            saveModel(model_name, exe, feed_vars=[data],
                      fetchlist=[y, h, c], inputs=[np.ones([4, 3, input_size]).astype(np.float32)], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


if __name__ == "__main__":

    testCases = [
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'forward',
            'seq_len': [],
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'bidirectional',
            'seq_len': [],
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 2,
            'direction': 'forward',
            'seq_len': [],
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 2,
            'direction': 'bidirectional',
            'seq_len': [],
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'forward',
            'seq_len': [1, 2, 3, 3],
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 2,
            'direction': 'bidirectional',
            'seq_len': [2, 2, 3, 3],
        }
    ]

    for test in testCases:
        paddle_rnn_lstm(test['input_size'], test['hidden_size'],
                      test['layers'], test['direction'], test['seq_len'])

    paddle_rnn_lstm(2, 2, 1, 'forward', [], name_override='rnn_lstm_weightlist_oob')
    model_path = os.path.join(sys.argv[1], 'rnn_lstm_weightlist_oob', 'rnn_lstm_weightlist_oob.pdmodel')
    _corrupt_weightlist_inputs(model_path)
