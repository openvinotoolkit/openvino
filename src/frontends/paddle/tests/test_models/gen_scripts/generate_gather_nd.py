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
            feed_vars=[data, index],
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

    x = np.random.rand(*x_shape).astype("float32")
    y = np.array([[], []]).astype("int32")
    gather_nd("gather_nd_empty", x, y)

    y = np.array([[1], [2]]).astype("int32")
    gather_nd("gather_nd_low_index", x, y)

    x_shape = (5, 2, 3, 1, 10)
    x = np.random.rand(*x_shape).astype("float32")
    y = np.array(
        [
            [np.random.randint(0, s) for s in x_shape],
            [np.random.randint(0, s) for s in x_shape],
        ]
    ).astype("int32")
    gather_nd("gather_nd_high_rank1", x, y)

    index = (
        np.array([np.random.randint(0, s, size=100) for s in x_shape]).astype("int32").T
    )
    y = index.reshape([10, 5, 2, 5])
    gather_nd("gather_nd_high_rank2", x, y)


if __name__ == "__main__":
    main()
