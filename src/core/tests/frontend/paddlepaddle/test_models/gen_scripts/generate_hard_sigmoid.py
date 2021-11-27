#
# hard_sigmoid paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def hard_sigmoid(name: str, x, slope: float = 0.2, offset: float = 0.5, data_type='float32'):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.hard_sigmoid(node_x, slope=slope, offset=offset, name='hard_sigmoid')

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
    data_type = 'float32'
    data = np.array([0, 1, 2, 3, 4, 5, 6, -10]).astype(data_type)

    hard_sigmoid("hard_sigmoid", data, 0.1, 0.6, data_type)


if __name__ == "__main__":
    main()
