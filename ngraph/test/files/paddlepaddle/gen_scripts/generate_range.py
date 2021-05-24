#
# range paddle model generator
#
import numpy as np
from save_model import saveModel
import sys


def pdpd_range(name : str, x, start, end, step, out_type):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        # Range op only support fill_constant input, since dynamic op is not supported in ov
        out = pdpd.fluid.layers.range(start, end, step, out_type)
        out = pdpd.cast(out, np.float32)
        out = pdpd.add(node_x, out)
        #out = pdpd.cast(out, np.float32)
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
    start = 1.5
    end = 10.5
    step = 2
    data = np.random.random([1, 5]).astype("float32")
    out_type = ["float32", "int32", "int64"]
    for i, dtype in enumerate(out_type):
        pdpd_range("range"+str(i), data, start, end, step, dtype)


if __name__ == "__main__":
    main()     
