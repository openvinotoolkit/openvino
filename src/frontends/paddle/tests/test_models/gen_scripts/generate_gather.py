#
# gather paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys


def gather(name: str, x, y, z):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        index = paddle.static.data(name='index', shape=y.shape, dtype=y.dtype)
        if (z == None):
            out = paddle.gather(data, index)
        else:
            axis = paddle.static.data(
                name='axis', shape=z.shape, dtype=z.dtype)
            out = paddle.gather(data, index, axis)
        if x.dtype == "int64":
            out = paddle.cast(out, "float32")

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        if (z == None):
            outs = exe.run(
                feed={'x': x, 'index': y},
                fetch_list=[out])

            saveModel(name, exe, feed_vars=[data, index], fetchlist=[out], inputs=[
                x, y], outputs=[outs[0]], target_dir=sys.argv[1])
        else:
            outs = exe.run(
                feed={'x': x, 'index': y, 'axis': z},
                fetch_list=[out])

            saveModel(name, exe, feed_vars=[data, index, axis], fetchlist=[
                      out], inputs=[x, y, z], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    # For multi-dimension input
    x_shape = (10, 20)
    x_type = "float32"
    index = [1, 3, 5]
    index_type = "int32"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)
    axis_np = None

    gather("gather_multi_dimension", xnp, index_np, axis_np)

    # For one_dimension input
    x_shape = (100)
    x_type = "int64"
    index = [1, 3, 5]
    index_type = "int64"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)
    axis_np = None

    gather("gather_one_dimension", xnp, index_np, axis_np)

    # For one_dimension input2
    x_shape = (100)
    x_type = "int64"
    index = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    index_type = "int64"

    xnp = np.random.random(x_shape).astype(x_type)
    index_np = np.array(index).astype(index_type)
    axis_np = None

    gather("gather_one_dimension2", xnp, index_np, axis_np)

    # For axis as input
    x_shape = (6, 88, 3)
    x_type = "float32"
    index = [1, 3, 5]
    index_type = "int32"
    axis = [0]
    axis_type = "int32"

    xnp = np.random.random(x_shape).astype(x_type)
    axis_np = np.array(axis).astype(axis_type)
    index_np = np.array(index).astype(index_type)

    gather("gather_axis_input", xnp, index_np, axis_np)


if __name__ == "__main__":
    main()
