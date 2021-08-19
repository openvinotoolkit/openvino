#
# pool2d paddle model generator
#
import numpy as np
import sys
from save_model import saveModel


def pdpd_scale(name : str, x, scale, bias, attrs : dict, data_type):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        out = pdpd.scale(x=node_x, scale=scale, bias=bias, 
                         bias_after_scale=attrs['bias_after_scale'])
        #FuzzyTest only support FP32 now, so cast result to fp32
        out = pdpd.cast(out, "float32")
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def pdpd_scale_tensor(name : str, x, scale, bias, attrs : dict, data_type):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=data_type)
        node_scale = pdpd.static.data(name='scale', shape=[1], dtype='float32')
        out = pdpd.scale(x=node_x, scale=node_scale, bias=bias,
                         bias_after_scale=attrs['bias_after_scale'])
        #FuzzyTest only support FP32 now, so cast result to fp32
        out = pdpd.cast(out, "float32")
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'scale': scale},
            fetch_list=[out])

        saveModel(name, exe, feedkeys=['x', 'scale'], fetchlist=[out], inputs=[x, np.array([scale]).astype('float32')], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    scale = 2.0
    bias = 1.0
    data = np.random.random([2, 3]).astype("float32")

    test_cases = [
        "float32",
        "int32",
        "int64"
    ]

    pdpd_attrs = {
        'bias_after_scale': True,
    }
    pdpd_scale_tensor("scale_tensor_bias_after", data, scale, bias, pdpd_attrs, 'float32')

    pdpd_attrs = {
        'bias_after_scale': False,
    }
    pdpd_scale_tensor("scale_tensor_bias_before", data, scale, bias, pdpd_attrs, 'float32')

    for test in test_cases:
        data = np.random.random([2, 3]).astype(test)
        pdpd_attrs = {
            'bias_after_scale': True,
        }
        pdpd_scale("scale_bias_after_" + test, data, scale, bias, pdpd_attrs, test)

        pdpd_attrs = {
            'bias_after_scale': False,
        }
        pdpd_scale("scale_bias_before_" + test, data, scale, bias, pdpd_attrs, test)



if __name__ == "__main__":
    main()     