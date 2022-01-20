#
# expand paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import paddle.fluid as fluid
import sys

data_type = 'float32'


def expand(name: str, x, expand_times: list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        out = fluid.layers.expand(
            node_x, expand_times=expand_times, name='expand')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def expand_tensor(name: str, x, expand_times, use_tensor_in_list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            expand_times[0] = paddle.assign(
                np.array((expand_times[0],)).astype('int32'))
            out = fluid.layers.expand(
                node_x, expand_times=expand_times, name='expand')
        else:
            expand_times = np.array(expand_times).astype('int32')
            node_shape = paddle.assign(expand_times, output=None)
            out = fluid.layers.expand(
                node_x, expand_times=node_shape, name='expand')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.random.rand(1, 1, 6).astype(data_type)

    expand("expand", data, [2, 3, 1])
    expand_tensor("expand_tensor", data, [2, 3, 1], False)
    expand_tensor("expand_tensor_list", data, [2, 3, 1], True)


if __name__ == "__main__":
    main()
