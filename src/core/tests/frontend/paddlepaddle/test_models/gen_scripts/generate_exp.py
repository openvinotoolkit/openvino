#
# exp paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def exp(name: str, x):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.fluid.layers.exp(x=node_x)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[
                  x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    input_shape = (1, 2, 3)
    input_data = np.random.rand(*input_shape).astype(np.float32)
    exp("exp_test_float32", input_data)


if __name__ == "__main__":
    main()
