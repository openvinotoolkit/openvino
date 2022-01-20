#
# top_k_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def top_k_v2(name: str, x, k: int, axis=None, largest=True, sorted=True):

    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        value, indices = paddle.topk(
            node_x, k=k, axis=axis, largest=largest, sorted=sorted, name="top_k")
        indices = paddle.cast(indices, np.float32)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[value, indices])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[value, indices], inputs=[
                  x], outputs=outs, target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.random.random([8, 9, 10]).astype("float32")
    # sorted must be true
    top_k_v2("top_k_v2_test_1", data, k=5, axis=-2, largest=True, sorted=True)
    top_k_v2("top_k_v2_test_2", data, k=6, axis=-1, largest=True, sorted=True)
    top_k_v2("top_k_v2_test_3", data, k=4, axis=0, largest=False, sorted=True)
    top_k_v2("top_k_v2_test_4", data, k=7,
             axis=None, largest=True, sorted=True)
    top_k_v2("top_k_v2_test_5", data, k=6, axis=2, largest=False, sorted=True)


if __name__ == "__main__":
    main()
