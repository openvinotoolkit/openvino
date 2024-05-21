#
# flip paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def flip(name: str, x, axis, is_dynamic=False):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if is_dynamic:
            data = paddle.static.data(name='x', shape=(-1, ) * len(x.shape), dtype=x.dtype)
        else:
            data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.flip(data, axis=axis)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[data], fetchlist=[out], inputs=[
                  x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_type = 'int32'
    axis = [2, 3]
    x = np.random.randint(0, 5, (2, 3, 4, 5)).astype(data_type)
    flip("flip_1", x, axis)

    data_type = 'float32'
    axis = [-1, -3]
    # axis = [3, 1]
    x = np.random.randn(3, 2, 1, 5).astype(data_type)
    flip("flip_2", x, axis)

    data_type = 'float32'
    axis = [0, 1]
    x = np.random.randn(1, 1, 1, 1).astype(data_type)
    flip("flip_3", x, axis)

    data_type = 'int64'
    axis = 1
    x = np.random.randint(-1, 3, (5, 3, 1, 1)).astype(data_type)
    flip("flip_4", x, axis)

    data_type = 'float32'
    axis = -1
    x = np.random.randn(1).astype(data_type)
    flip("flip_5", x, axis)

    data_type = 'int64'
    axis = 3
    x = np.random.randint(-5, 5, (1, 1, 4, 1)).astype(data_type)
    flip("flip_dynamic_1", x, axis, True)

    data_type = 'float32'
    axis = [-1, -2]
    x = np.random.randn(1, 4, 1).astype(data_type)
    flip("flip_dynamic_2", x, axis, True)

if __name__ == "__main__":
    main()
