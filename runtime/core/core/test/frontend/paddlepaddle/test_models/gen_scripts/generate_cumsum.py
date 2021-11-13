#
# cumsum paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys

data_type = 'float32'

def cumsum(name:str, x, axis, dtype=None):
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.cumsum(data, axis, dtype=dtype)
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
    x = np.linspace(1, 12, 12, dtype=data_type)
    x = np.reshape(x, (3, 4))

    cumsum("cumsum", x, axis=None)
    cumsum("cumsum_f32", x, axis=-1, dtype='float32')
    cumsum("cumsum_f64", x, axis=0, dtype='float64')
    cumsum("cumsum_i32", x, axis=0, dtype='int32')
    cumsum("cumsum_i64", x, axis=0, dtype='int64')

if __name__ == "__main__":
    main()