#
# unsqueeze paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def unsqueeze(name : str, x, axes : list):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.unsqueeze(node_x, axes = axes, name = 'unsqueeze')

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
    data = np.random.rand(5, 10).astype(data_type)

    unsqueeze("unsqueeze", data, [1])

if __name__ == "__main__":
    main()