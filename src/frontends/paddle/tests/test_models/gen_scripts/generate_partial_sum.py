# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# partial_sum paddle model generator
#
import numpy as np
import os
from save_model import saveModel
import paddle
import sys


def _get_framework_pb2():
    try:
        from paddle.fluid.proto import framework_pb2
        return framework_pb2
    except Exception:
        pass
    try:
        from paddle.base.proto import framework_pb2
        return framework_pb2
    except Exception:
        from paddle.framework.proto import framework_pb2
        return framework_pb2


def _corrupt_partial_sum_inputs(model_path: str):
    framework_pb2 = _get_framework_pb2()
    prog = framework_pb2.ProgramDesc()
    with open(model_path, "rb") as f:
        prog.ParseFromString(f.read())

    modified = False
    for block in prog.blocks:
        for op in block.ops:
            if op.type == "partial_sum":
                for inp in op.inputs:
                    if inp.parameter == "X" and len(inp.arguments) > 1:
                        del inp.arguments[1:]
                        modified = True

    if not modified:
        raise RuntimeError("Failed to modify partial_sum X inputs in model")

    with open(model_path, "wb") as f:
        f.write(prog.SerializeToString())


def partial_sum(name: str, x, y, start_index=0, length=-1):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        x_data = paddle.static.data(name="x", shape=x.shape, dtype=x.dtype)
        y_data = paddle.static.data(name="y", shape=x.shape, dtype=y.dtype)

        if paddle.__version__ >= '2.5.1':
            out = paddle.incubate.layers.nn.partial_sum(
                [x_data, y_data], start_index=start_index, length=length
            )
        else:
            out = paddle.fluid.contrib.layers.partial_sum(
            [x_data, y_data], start_index=start_index, length=length
            )


        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x, "y": y}, fetch_list=[out])

        saveModel(
            name,
            exe,
            feed_vars=[x_data, y_data],
            fetchlist=[out],
            inputs=[x, y],
            outputs=[outs[0]],
            target_dir=sys.argv[1],
        )

    return outs[0]


def main():
    dtype = 'float32'
    x = np.random.randn(6, 4).astype(dtype)
    y = np.random.randn(6, 4).astype(dtype)
    partial_sum("partial_sum_1", x, y, start_index=2, length=2)


    dtype = 'int32'
    x = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    y = np.random.randint(-10, 10, [5, 3]).astype(dtype)
    partial_sum("partial_sum_2", x, y, start_index=1, length=-1)

    dtype = 'int64'
    x = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    y = np.random.randint(-10, 10, [8, 10]).astype(dtype)
    partial_sum("partial_sum_3", x, y, start_index=1, length=5)

    dtype = 'float32'
    x = np.random.randn(4, 8).astype(dtype)
    y = np.random.randn(4, 8).astype(dtype)
    partial_sum("partial_sum_oob", x, y, start_index=0, length=4)
    model_path = os.path.join(sys.argv[1], "partial_sum_oob", "partial_sum_oob.pdmodel")
    _corrupt_partial_sum_inputs(model_path)

if __name__ == "__main__":
    main()
