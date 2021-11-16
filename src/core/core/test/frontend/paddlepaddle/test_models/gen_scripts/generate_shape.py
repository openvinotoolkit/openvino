#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def pdpd_shape(name : str, x):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        out = pdpd.shape(node_x)
        out = pdpd.cast(out, np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():

    data = np.random.random(size=(2, 3)).astype('float32')
    pdpd_shape("shape", data)


if __name__ == "__main__":
    main()     