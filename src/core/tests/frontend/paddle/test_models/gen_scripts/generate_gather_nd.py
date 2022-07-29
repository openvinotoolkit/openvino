#
# gather_nd paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def gather_nd(name: str, x, y):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        index = paddle.static.data(name="index", shape=y.shape, dtype=y.dtype)
        out = paddle.gather_nd(data, index)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={"x": x, "index": y}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feedkeys=["x", "index"],
            fetchlist=[out],
            inputs=[x, y],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    tests_cases = [
        "float32",
        "int32",
        "int64",
    ]
    x_shape = (10, 20)

    for test in tests_cases:
        x = np.random.rand(*x_shape).astype(test)
        y = np.array([[0, 1], [1, 1]]).astype("int32")
        gather_nd("gather_nd_" + test, x, y)


if __name__ == "__main__":
    main()
