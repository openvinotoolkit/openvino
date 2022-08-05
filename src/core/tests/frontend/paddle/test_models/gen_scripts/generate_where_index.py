# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# where_index paddle model generator
#
import numpy as np
from save_model import saveModel
import sys
import paddle
from paddle.fluid.layer_helper import LayerHelper

paddle.enable_static()


def where_index_ref(x, name=None):
    helper = LayerHelper('p_norm', **locals())
    out = helper.create_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(type='where_index',
                     inputs={'Condition': x},
                     outputs={'Out': out})
    return out


def where_index(name: str, x):
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)

        out = where_index_ref(node_x)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    datatype = "int32"
    condition = np.random.randint(0, 5, size=[5, 8, 2], dtype=datatype)
    paddle_out = where_index("where_index_1", condition)

    datatype = "float32"
    condition = (np.random.randint(0, 5, size=[8, 3, 2]) * 1.1).astype(datatype)
    paddle_out = where_index("where_index_2", condition)

if __name__ == "__main__":
    main()
