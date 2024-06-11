import sys
import paddle
import numpy as np
from save_model import saveModel


def rsqrt(name: str, x):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        out = paddle.rsqrt(node_x)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # Initialize the variables with the provided inputs
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[node_x],
            fetchlist=[out],
            inputs=[x],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )


def main():
    # Test case 1: float32, 1D array
    x1 = np.array([0.25, 1, 4, 16], dtype=np.float32)
    rsqrt("rsqrt_float32_1D", x1)

    # Test case 2: float32, 2D array
    x2 = np.array([[0.25, 1], [4, 16]], dtype=np.float32)
    rsqrt("rsqrt_float32_2D", x2)
    
    x3 = np.random.rand(2, 3, 4, 5).astype(np.float32)
    rsqrt("rsqrt_float32_4D", x3)


if __name__ == "__main__":
    main()
