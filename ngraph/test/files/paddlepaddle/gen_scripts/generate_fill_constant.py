#
# fill_const paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys


def fill_constant(name : str, shape : list, dtype, value):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        x1 = pdpd.fluid.layers.fill_constant(shape=shape, value=value, dtype=dtype, name='fill_constant')
        x2 = pdpd.fluid.layers.fill_constant(shape=shape, value=value, dtype=dtype, name='fill_constant')
        out = pdpd.add(pdpd.cast(x1, np.float32), pdpd.cast(x2, np.float32))
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    fill_constant("fill_constant", [2, 3, 4], 'float32', 0.03)
    fill_constant("fill_constant_int32", [2, 3, 4], "int32", 2)
    fill_constant("fill_constant_int64", [2, 3, 4], "int64", 4)


if __name__ == "__main__":
    main()