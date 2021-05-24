#
# pool2d paddle model generator
#
import numpy as np
import sys
from save_model import saveModel

data_type = 'float32'


def pdpd_scale(name : str, x, scale, bias, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
        out = pdpd.scale(x=node_x, scale=scale, bias=bias, 
                         bias_after_scale=attrs['bias_after_scale'])

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
    scale = 2.0
    bias = 1.0
    data = np.random.random([2, 3]).astype("float32")
    pdpd_attrs = {
        'bias_after_scale': True,
    }  
    pdpd_scale("scale_test1", data, scale, bias, pdpd_attrs)

    pdpd_attrs = {
        'bias_after_scale': False,
    }
    pdpd_scale("scale_test2", data, scale, bias, pdpd_attrs)


if __name__ == "__main__":
    main()     