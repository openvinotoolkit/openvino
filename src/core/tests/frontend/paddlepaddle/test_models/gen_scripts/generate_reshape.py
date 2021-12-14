#
# reshape paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

data_type = 'float32'


def reshape(name : str, x, out_shape):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = pdpd.fluid.layers.reshape(x=node_x, shape=out_shape)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def reshape_tensor(name : str, x, out_shape, use_tensor_in_list):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        if use_tensor_in_list:
            out_shape[0] = pdpd.assign(np.array((out_shape[0],)).astype('int32'))
            out = pdpd.fluid.layers.reshape(x=node_x, shape=out_shape)
        else:
            out_shape = np.array(out_shape).astype('int32')
            node_shape = pdpd.assign(out_shape)
            out = pdpd.fluid.layers.reshape(x=node_x, shape=node_shape)

        out = pdpd.pow(out, 1)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]]], dtype=np.float32)
    out_shape = [1, 1, 2, 8]
    reshape("reshape", data, out_shape)
    reshape_tensor("reshape_tensor", data, out_shape, False)
    reshape_tensor("reshape_tensor_list", data, out_shape, True)


if __name__ == "__main__":
    main()
