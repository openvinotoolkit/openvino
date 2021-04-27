#
# fill_constant_batch_size_like paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd

data_type = 'float32'

def fill_constant_batch_size_like(name : str, x, shape, dtype, value, input_dim_idx=0, output_dim_idx=0):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        like = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.fill_constant_batch_size_like(input=like, shape=shape, \
            value=value, dtype=dtype, \
            output_dim_idx=output_dim_idx, input_dim_idx=input_dim_idx)

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
    x = np.random.rand(4, 3, 2).astype(data_type)
    fill_constant_batch_size_like("fill_constant_batch_size_like", \
        x, [1, -1, 3], data_type, 0.03, 2, 1)

if __name__ == "__main__":
    main()