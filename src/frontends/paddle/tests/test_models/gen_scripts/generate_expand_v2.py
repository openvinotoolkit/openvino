# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# expand_v2 paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle
import sys

data_type = 'float32'


def expand_v2(name: str, x, shape: list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(),
                                     paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        out = paddle.expand(node_x, shape=shape, name='expand_v2')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={'x': x}, fetch_list=[out])

        saveModel(name,
                  exe,
                  feed_vars=[node_x],
                  fetchlist=[out],
                  inputs=[x],
                  outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def expand_v2_tensor(name: str, x, out_shape, use_tensor_in_list):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(),
                                     paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            out_shape[0] = paddle.assign(
                np.array((out_shape[0], )).astype('int32'))
            out = paddle.expand(node_x, shape=out_shape, name='expand_v2')
        else:
            out_shape = np.array(out_shape).astype('int32')
            node_shape = paddle.assign(out_shape, output=None)
            out = paddle.expand(node_x, shape=node_shape, name='expand_v2')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={'x': x}, fetch_list=[out])

        saveModel(name,
                  exe,
                  feed_vars=[node_x],
                  fetchlist=[out],
                  inputs=[x],
                  outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def expand_as_v2(name: str, x, y):
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(),
                                     paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        node_y = paddle.static.data(name='y', shape=y.shape, dtype=data_type)
        out = paddle.expand_as(node_x, node_y, name='expand_as_v2')

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(feed={'x': x, 'y': y}, fetch_list=[out])

        saveModel(name,
                  exe,
                  feed_vars=[node_x, node_y],
                  fetchlist=[out],
                  inputs=[x, y],
                  outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def expand_v2_conditional(name: str, x, rows):
    # expand_v2 nested inside a conditional_block, mirroring the control-flow
    # structure of the FastRCNN model where the expand target shape is assembled
    # at runtime. Exercises the expand_v2 translator path that pins the Broadcast
    # output rank to the statically-known op output rank.
    paddle.enable_static()

    with paddle.static.program_guard(paddle.static.Program(),
                                     paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=data_type)
        node_im = paddle.static.data(name='im',
                                     shape=[-1, x.shape[-1]],
                                     dtype=data_type)

        def true_fn():
            n = paddle.shape(node_im)[0].reshape([1]).astype('int32')
            tgt = paddle.concat(
                [n, paddle.assign(np.array([x.shape[-1]]).astype('int32'))])
            return paddle.expand(node_x, shape=tgt, name='expand_v2')

        def false_fn():
            return paddle.expand(node_x, shape=list(x.shape), name='expand_v2')

        out = paddle.static.nn.cond(
            paddle.shape(node_im)[0] > 0, true_fn, false_fn)

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        im = np.random.rand(rows, x.shape[-1]).astype(data_type)
        outs = exe.run(feed={'x': x, 'im': im}, fetch_list=[out])

        saveModel(name,
                  exe,
                  feed_vars=[node_x, node_im],
                  fetchlist=[out],
                  inputs=[x, im],
                  outputs=[outs[0]],
                  target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.random.rand(1, 1, 6).astype(data_type)

    expand_v2("expand_v2", data, [2, 3, -1])
    expand_v2_tensor("expand_v2_tensor", data, [2, 3, -1], False)
    expand_v2_tensor("expand_v2_tensor_list", data, [2, 3, -1], True)
    expand_v2_tensor("expand_v2_tensor_list2", data, [2, 2, 2, 3, -1], True)

    # expand_as_v2
    data_x = np.random.rand(1, 1, 6).astype(data_type)
    data_y1 = np.random.rand(2, 3, 6).astype(data_type)
    data_y2 = np.random.rand(4, 2, 3, 6).astype(data_type)
    expand_as_v2("expand_as_v2_1", data_x, data_y1)
    expand_as_v2("expand_as_v2_2", data_x, data_y2)

    # expand_v2 inside control flow (FastRCNN-like)
    expand_v2_conditional("expand_v2_conditional",
                          np.random.rand(1, 4).astype(data_type), 3)


if __name__ == "__main__":
    main()
