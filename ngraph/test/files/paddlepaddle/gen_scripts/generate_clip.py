#
# clip paddle model generator
#
import numpy as np
from save_model import saveModel


def clip(name: str, x, min, max):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        out = pdpd.fluid.layers.clip(node_x, min=min, max=max)

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
    data = np.random.random([2, 3, 4]).astype('float32')
    min = 0
    max = 0.8

    clip("clip", data, min, max)


if __name__ == "__main__":
    main()