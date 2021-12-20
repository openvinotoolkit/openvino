#
# layer_norm paddle model generator
#
import numpy as np
from paddle.fluid import param_attr
from save_model import saveModel
import paddle as pdpd
import sys

data_type = 'float32'

def layer_norm(name:str, x, begin_norm_axis, scale=True, shift=True, param_attr=None, bias_attr=None):
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.static.nn.layer_norm(input=data, scale=scale, shift=shift,\
            begin_norm_axis=begin_norm_axis, param_attr=param_attr, bias_attr=bias_attr)

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
    x = np.random.rand(8, 24, 32).astype(data_type)
    random_data = np.random.rand(24 * 32).astype(data_type)
    attr = pdpd.ParamAttr(
              initializer=pdpd.fluid.initializer.NumpyArrayInitializer(random_data))
    layer_norm("layer_norm", x, begin_norm_axis=1, param_attr=attr, bias_attr=attr)
    layer_norm("layer_norm_noscale", x, scale=False, begin_norm_axis=2)
    layer_norm("layer_norm_noshift", x, shift=False, begin_norm_axis=1)
    layer_norm("layer_norm_noall", x, scale=False, shift=False, begin_norm_axis=1)

if __name__ == "__main__":
    main()