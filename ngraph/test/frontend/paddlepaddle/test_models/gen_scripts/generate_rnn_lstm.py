import numpy as np
from save_model import saveModel
import sys


def pdpd_rnn_lstm(input_size, hidden_size, layers, direction):
    import paddle as pdpd
    pdpd.enable_static()
    main_program = pdpd.static.Program()
    startup_program = pdpd.static.Program()

    num_of_directions = 1 if direction == 'forward' else 2
    with pdpd.static.program_guard(main_program, startup_program):

        rnn = pdpd.nn.LSTM(input_size, hidden_size, layers, direction)

        data = pdpd.static.data(name='x', shape=[4, 3, input_size], dtype='float32')
        prev_h = pdpd.ones(shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32)
        prev_c = pdpd.ones(shape=[layers * num_of_directions, 4, hidden_size], dtype=np.float32)

        y, (h, c) = rnn(data, (prev_h, prev_c))

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        exe.run(startup_program)

        outs = exe.run(
            feed={'x': np.ones([4, 3, input_size]).astype(np.float32)},
            fetch_list=[y, h, c],
            program=main_program)
        saveModel("rnn_lstm_layer_" + str(layers) + '_' + str(direction), exe, feedkeys=['x'],
                  fetchlist=[y, h, c], inputs=[np.ones([4, 3, input_size]).astype(np.float32)], outputs=[outs[0], outs[1], outs[2]], target_dir=sys.argv[1])
    return outs[0]


if __name__ == "__main__":

    testCases = [
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'forward',
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 1,
            'direction': 'bidirectional',
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 2,
            'direction': 'forward',
        },
        {
            'input_size': 2,
            'hidden_size': 2,
            'layers': 2,
            'direction': 'bidirectional',
        }
    ]

    for test in testCases:
        pdpd_rnn_lstm(test['input_size'], test['hidden_size'], test['layers'], test['direction'])