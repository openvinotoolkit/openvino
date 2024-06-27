# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# index_select paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = "float32"


def index_select(name: str, x, index, axis):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        data = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        tensor_index = paddle.static.data(
            name="index", shape=index.shape, dtype="int32"
        )
        out = paddle.index_select(
            data,
            index=tensor_index,
            axis=axis,
        )

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "index": index}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[data, tensor_index],
            fetchlist=[out],
            inputs=[x, index],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    x = np.random.rand(8, 24, 32).astype(data_type)
    index = np.random.randint(0, 7, (5)).astype("int32")
    index_select("index_select_axis_0", x, index, axis=0)
    index_select("index_select_axis_1", x, index, axis=1)
    index_select("index_select_axis_native_-1", x, index, axis=-1)
    index_select("index_select_axis_native_-2", x, index, axis=-2)


if __name__ == "__main__":
    main()
