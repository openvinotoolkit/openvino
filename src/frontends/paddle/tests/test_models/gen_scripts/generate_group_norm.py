# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

#
# group norm paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle

data_type = "float32"

def group_norm(name: str, x, groups, epsilon, scale, bias, data_layout):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name="x", shape=x.shape, dtype=data_type)
        if scale is False:
            scale_attr = scale
        else:
            scale_attr = paddle.ParamAttr(name="scale1", initializer=paddle.nn.initializer.Assign(scale))
        if bias is False:
            bias_attr = bias
        else:
            bias_attr = paddle.ParamAttr(name="bias1", initializer=paddle.nn.initializer.Assign(bias))

        out = paddle.static.nn.group_norm(node_x, groups=groups,
                                          epsilon=epsilon,
                                          param_attr=scale_attr,
                                          bias_attr=bias_attr,
                                          data_layout=data_layout)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={"x": x}, fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def _set_data_layout(model_path, data_layout):
    # Paddle only accepts NCHW and NHWC, so the unsupported layout is patched into the
    # generated program to emulate a model produced by another framework or tool
    try:
        from paddle.fluid.proto import framework_pb2
    except Exception:
        from paddle.base.proto import framework_pb2

    program_desc = framework_pb2.ProgramDesc()
    with open(model_path, "rb") as model_file:
        program_desc.ParseFromString(model_file.read())

    updated = False
    for block in program_desc.blocks:
        for op in block.ops:
            if op.type != "group_norm":
                continue
            for attr in op.attrs:
                if attr.name == "data_layout":
                    attr.type = framework_pb2.AttrType.STRING
                    attr.s = data_layout
                    updated = True

    if not updated:
        raise RuntimeError("group_norm op not found when mutating pdmodel")

    with open(model_path, "wb") as model_file:
        model_file.write(program_desc.SerializeToString())


def main():
    # data layout is NCHW
    data = np.random.random((2, 4, 3, 4)).astype(np.float32)
    groups = 2
    epsilon = 1e-05
    scale = np.random.random(4).astype(np.float32)
    bias = np.random.random(4).astype(np.float32)
    group_norm("group_norm_1", data, groups, epsilon, scale, bias, "NCHW")

    # data layout is NHWC
    data = np.random.random((2, 4, 3, 4)).astype(np.float32)
    groups = 2
    epsilon = 1e-05
    scale = np.random.random(4).astype(np.float32)
    bias = np.random.random(4).astype(np.float32)
    group_norm("group_norm_2", data, groups, epsilon, scale, bias, "NHWC")

    # scale and bias are None
    scale = False
    bias = False
    group_norm("group_norm_3", data, groups, epsilon, scale, bias, "NHWC")

    # unsupported data layout, used by the negative conversion test
    data = np.random.random((2, 4, 3, 4)).astype(np.float32)
    scale = np.random.random(4).astype(np.float32)
    bias = np.random.random(4).astype(np.float32)
    group_norm("group_norm_unsupported_layout", data, groups, epsilon, scale, bias, "NCHW")
    _set_data_layout(os.path.join(sys.argv[1], "group_norm_unsupported_layout", "group_norm_unsupported_layout.pdmodel"),
                     "NCDHW")


if __name__ == "__main__":
    main()
