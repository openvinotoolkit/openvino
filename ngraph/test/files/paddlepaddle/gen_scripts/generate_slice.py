#
# slice paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def slice(name : str, x, axes : list, start : list, end : list):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.slice(node_x, axes = axes, starts = start, ends = end)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]])

    return outs[0]

def main():
    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(4, 3, 5).astype(data_type)
    slice("slice", x, axes=[1, 2], start=(0, 1), end=(-1, 3))

    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(2, 30).astype(data_type)
    slice("slice_1d", x, axes=[0], start=[0], end=[1])

if __name__ == "__main__":
    main()