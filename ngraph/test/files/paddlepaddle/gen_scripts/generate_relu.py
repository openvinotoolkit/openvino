#
# relu paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def relu(name: str, x):
    import paddle as pdpd
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    out = pdpd.nn.functional.relu(node_x)

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
    data = np.array([-2, 0, 1]).astype('float32')

    relu("relu", data)


if __name__ == "__main__":
    main()