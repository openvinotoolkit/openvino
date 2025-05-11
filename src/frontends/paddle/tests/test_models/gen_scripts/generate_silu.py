#
# silu paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def silu(name: str, x, data_type, use_static=True):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        if use_static:
            node_x = pdpd.static.data(
                name='input_x', shape=x.shape, dtype=data_type)
        else:
            node_x = pdpd.static.data(
                name='input_x', shape=[1, 1, -1, -1], dtype=data_type)
        out = pdpd.nn.functional.silu(x=node_x, name='silu')

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'input_x': x},
            fetch_list=[out])

        saveModel(name, exe, feed_vars=[node_x], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    x1 = np.random.randn(2,).astype('float32')
    silu("silu_static_test1", x1, 'float32', True)

    x2 = np.random.randn(2, 3).astype('float32')
    silu("silu_static_test2", x2, 'float32', True)

    x3 = np.random.randn(2, 3, 4).astype('float32')
    silu("silu_static_test3", x3, 'float32', True)

    x4 = np.random.randn(2, 3, 4, 5).astype('float32')
    silu("silu_static_test4", x4, 'float32', True)

    x5 = np.random.randn(1, 1, 32, 32).astype('float32')
    silu("silu_dynamic_test1", x5, 'float32', False)

    x6 = np.random.randn(1, 1, 64, 64).astype('float32')
    silu("silu_dynamic_test2", x6, 'float32', False)

    x7 = np.random.randn(1, 1, 128, 128).astype('float32')
    silu("silu_dynamic_test3", x7, 'float32', False)

    x8 = np.random.randn(1, 1, 256, 256).astype('float32')
    silu("silu_dynamic_test4", x8, 'float32', False)


if __name__ == "__main__":
    main()
