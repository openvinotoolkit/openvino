#
# swish paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def swish(name: str, x, data_type):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(
            name='input_x', shape=x.shape, dtype=data_type)
        if paddle.__version__ >= '2.0.0':
            out = paddle.nn.functional.swish(x=node_x, name='swish')
        else:
            out = paddle.fluid.layers.swish(x=node_x, name='swish')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'input_x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_type = 'float32'
    input_beta = 1.0
    x = np.random.randn(2, 3).astype(data_type)
    swish("swish_default_params", x, data_type)

    input_beta = 2.0
    x = np.random.randn(2, 3).astype(data_type)
    swish("swish_beta", x, data_type)


if __name__ == "__main__":
    main()
