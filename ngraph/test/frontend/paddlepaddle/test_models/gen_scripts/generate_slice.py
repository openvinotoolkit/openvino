#
# slice paddle model generator
#
import numpy as np
from save_model import saveModel
import paddle as pdpd
import sys
import os

data_type = 'float32'

def slice(name : str, x, axes : list, start : list, end : list):
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype = data_type)
        out = pdpd.fluid.layers.slice(node_x, axes = axes, starts = start, ends = end)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x},
            fetch_list=[out])             

        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


'''
export dyn model, long with input and output for reference.
'''
def exportModel(name, dyn_func, inputspecs:list, data):
    model_dir = os.path.join(sys.argv[1], name)
    save_path = '{}/{}'.format(model_dir, name)
    pdpd.jit.save(dyn_func, save_path, inputspecs)

    model = pdpd.jit.load(save_path)
    result = model(data)
    print(result.numpy().shape)
    np.save(os.path.join(model_dir, "input{}".format(0)), data.numpy())
    np.save(os.path.join(model_dir, "output{}".format(0)), result.numpy())

def slice_dyn(test_shape=[2,8,10,10]):
    pdpd.disable_static()

    input = pdpd.static.InputSpec(shape=test_shape, dtype='float32', name='x')
    data = pdpd.rand(shape=test_shape, dtype='float32')

    '''
    slice w/ decrease_axis
    '''
    @pdpd.jit.to_static
    def test_slice_decrease_axis(x):
        return x[0, 1:3, :, 5]
    exportModel('slice_decrease_axis', test_slice_decrease_axis, [input], data) # output shape (2, 10)

    '''
    slice w/o decrease_axis
    '''
    @pdpd.jit.to_static
    def test_slice(x):
        return pdpd.slice(x, axes=[0,1,3], starts=[0,1,5], ends=[1,3,6])
    # exportModel('slice_dyn', test_slice, [input], data) # output shape (1, 2, 10, 1)  # disable it by default as this kind of test model already there. It's for comparsion only.

    '''
    slice w/ decrease_axis of all dims
    '''
    @pdpd.jit.to_static
    def test_slice_decrease_axis_all(x):
        return x[0, 0, 0, 0]
    exportModel('slice_decrease_axis_all', test_slice_decrease_axis_all, [input], data) # output shape (1,)

    '''
    slice w/o decrease_axis of all dims
    '''
    @pdpd.jit.to_static
    def test_slice_alldim(x):
        return pdpd.slice(x, axes=[0,1,2,3], starts=[0,0,0,0], ends=[1,1,1,1])
    # exportModel('slice_alldim', test_slice_alldim, [input], data) # output shape (1, 1, 1, 1) # disable it by default as this kind of test model already there. It's for comparsion only.

def main():
    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(4, 3, 5).astype(data_type)
    slice("slice", x, axes=[1, 2], start=(0, 1), end=(-1, 3))

    x = np.linspace(1, 60, num = 60, dtype=np.int32).reshape(2, 30).astype(data_type)
    slice("slice_1d", x, axes=[0], start=[0], end=[1])

if __name__ == "__main__":
    main()
    slice_dyn()