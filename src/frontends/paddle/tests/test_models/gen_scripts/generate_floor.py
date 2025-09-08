#
# floor paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def floor(name: str, x):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = paddle.floor(data)

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
    data_type = 'float32'
    x = np.array([-0.4, -0.2, 2.1, 0.3]).astype(data_type)

    floor("floor_float32", x)


if __name__ == "__main__":
    main()
