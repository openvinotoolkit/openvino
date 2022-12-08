#
# swish paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def swish(name: str, x, data_type, input_beta):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(
            name='input_x', shape=x.shape, dtype=data_type)
        out = pdpd.fluid.layers.swish(x=node_x, beta=input_beta, name='swish')

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'input_x': x},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['input_x'], fetchlist=[out],
                  inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data_type = 'float32'
    input_beta = 1.0
    x = np.random.randn(2, 3).astype(data_type)
    swish("swish_default_params", x, data_type, input_beta)

    input_beta = 2.0
    x = np.random.randn(2, 3).astype(data_type)
    swish("swish_beta", x, data_type, input_beta)


if __name__ == "__main__":
    main()
