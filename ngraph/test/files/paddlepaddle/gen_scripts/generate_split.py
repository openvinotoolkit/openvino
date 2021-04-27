#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def split(name : str, x, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.fluid.layers.split(node_x, num_or_sections=attrs['num_or_sections'], dim=attrs['axis'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])
        print("outputs: ", type(outs),len(outs))
        print("out: ", type(out), len(out))          

        saveModel(name, exe, feedkeys=['x'], fetchlist=out, inputs=[x], outputs=outs)

    return outs[0]

def main():
    # split
    data_types = ['float32'] #TODOD: ['bool', 'float16', 'float32', 'float64', 'int32', 'int64']
    num_or_sections = [3, [2, 3, 4], [2, 3, -1]]
    axes = [1, -2]    

    idx = 1
    for t in data_types:
        for s in num_or_sections:
            for i in axes:
                pdpd_attrs = {
                        'num_or_sections': s,
                        'axis': i
                }
                print(idx, t, s, i)
                data_NCHW = np.random.rand(3,9,5).astype(t)
                split("split_test{}".format(idx), data_NCHW, pdpd_attrs)
                idx+=1


if __name__ == "__main__":
    main()     