# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from save_model import saveModel
import sys


def paddle_rnn_lstm(input_size, hidden_size, layers, direction):
    import paddle
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()

    num_of_directions = 1 if direction == 'forward' else 2
    with paddle.static.program_guard(main_program, startup_program):

        rnn = paddle.nn.LSTM(input_size, hidden_size, layers, direction, name="lstm")

        data = paddle.static.data(name='x', shape=[4, 3, input_size], dtype='float32')
        prev_h = paddle.ones(shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32, name="const_1")
        prev_c = paddle.ones(shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32, name="const_2")

        y, (h, c) = rnn(data, (prev_h, prev_c))
        relu_1 = paddle.nn.functional.relu(c, name="relu_1")
        relu_2 = paddle.nn.functional.relu(c, name="relu_2")
        relu_3 = paddle.nn.functional.relu(c, name="relu_3")

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(startup_program)

        outs = exe.run(
            feed={'x': np.ones([4, 3, input_size]).astype(np.float32)},
            fetch_list=[y, h, c],
            program=main_program)
        saveModel("place_test_model", exe, feedkeys=['x'],
                  fetchlist=[y, h, c, relu_1, relu_2, relu_3],
                  inputs=[np.ones([4, 3, input_size]).astype(np.float32)],
                  outputs=[outs[0], outs[1], outs[2]], target_dir=sys.argv[1])
    return outs[0]


if __name__ == "__main__":

    testCases = [
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'forward',
        },
    ]

    for test in testCases:
        paddle_rnn_lstm(test['input_size'], test['hidden_size'], test['layers'], test['direction'])