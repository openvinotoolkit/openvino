#
# sigmoid paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def hard_swish(name: str, x, threshold=6.0, scale=6.0, offset=3.0, data_type='float32'):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = pdpd.fluid.layers.hard_swish(node_x, threshold=threshold, scale=scale, offset=offset, name='hard_swish')

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
    data = np.array([-6, 1, 6]).astype(data_type)
    hard_swish("hard_swish", data, data_type='float32')


if __name__ == "__main__":
    main()
