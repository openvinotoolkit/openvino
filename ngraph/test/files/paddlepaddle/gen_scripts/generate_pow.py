#
# pow paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def pdpd_pow(name : str, x, y, data_type):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.pow(node_x, y, name = 'pow')
        out = pdpd.cast(out, "float32")
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def pdpd_pow_tensor(name : str, x, y, data_type):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        node_y = pdpd.static.data(name='y', shape=y.shape, dtype=data_type)
        out = pdpd.fluid.layers.pow(node_x, node_y, name='pow')
        out = pdpd.cast(out, "float32")

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'y': y},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out],
                  inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    for test in test_cases:
        x = np.array([0, 1, 2, -10]).astype(test)
        y = np.array([2]).astype(test)
        pdpd_pow("pow_" + test, x, y, test)

    x = np.array([0, 1, 2, -10]).astype("float32")
    y = np.array([2.0]).astype("float32")
    pdpd_pow_tensor("pow_y_tensor", x, y, 'float32')


if __name__ == "__main__":
    main()
