import numpy as np
import sys
from save_model import saveModel


def reduce_max(name : str, x, axis=None, keepdim=False):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.max(data_x, axis=axis, keepdim=keepdim)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def reduce_mean(name : str, x, axis=None, keepdim=False):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.mean(data_x, axis=axis, keepdim=keepdim)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def reduce_min(name : str, x, axis=None, keepdim=False):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.min(data_x, axis=axis, keepdim=keepdim)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def reduce_prod(name : str, x, axis=None, keepdim=False):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.prod(data_x, axis=axis, keepdim=keepdim)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x'], fetchlist=[out], inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def reduce_sum(name : str, x, axis=None, keepdim=False):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        data_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        out = pdpd.sum(data_x, axis=axis, keepdim=keepdim)

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
    data = np.array([[[1.0,2.0], [3.0, 4.0]], [[5.0,6.0], [7.0, 8.0]]]).astype(np.float32)

    reduce_max("reduce_max_test_0", data)
    reduce_max("reduce_max_test_1", data, axis=0, keepdim=False)
    reduce_max("reduce_max_test_2", data, axis=-1, keepdim=False)
    reduce_max("reduce_max_test_3", data, axis=1, keepdim=True)
    reduce_max("reduce_max_test_4", data, axis=[1,2], keepdim=False)
    reduce_max("reduce_max_test_5", data, axis=[0,1], keepdim=True)

    reduce_mean("reduce_mean_test_0", data)
    reduce_mean("reduce_mean_test_1", data, axis=0, keepdim=False)
    reduce_mean("reduce_mean_test_2", data, axis=-1, keepdim=False)
    reduce_mean("reduce_mean_test_3", data, axis=1, keepdim=True)
    reduce_mean("reduce_mean_test_4", data, axis=[1,2], keepdim=False)
    reduce_mean("reduce_mean_test_5", data, axis=[0,1], keepdim=True)

    reduce_min("reduce_min_test_0", data)
    reduce_min("reduce_min_test_1", data, axis=0, keepdim=False)
    reduce_min("reduce_min_test_2", data, axis=-1, keepdim=False)
    reduce_min("reduce_min_test_3", data, axis=1, keepdim=True)
    reduce_min("reduce_min_test_4", data, axis=[1,2], keepdim=False)
    reduce_min("reduce_min_test_5", data, axis=[0,1], keepdim=True)

    reduce_prod("reduce_prod_test_0", data)
    reduce_prod("reduce_prod_test_1", data, axis=0, keepdim=False)
    reduce_prod("reduce_prod_test_2", data, axis=-1, keepdim=False)
    reduce_prod("reduce_prod_test_3", data, axis=1, keepdim=True)
    reduce_prod("reduce_prod_test_4", data, axis=[1,2], keepdim=False)
    reduce_prod("reduce_prod_test_5", data, axis=[0,1], keepdim=True)

    reduce_sum("reduce_sum_test_0", data)
    reduce_sum("reduce_sum_test_1", data, axis=0, keepdim=False)
    reduce_sum("reduce_sum_test_2", data, axis=-1, keepdim=False)
    reduce_sum("reduce_sum_test_3", data, axis=1, keepdim=True)
    reduce_sum("reduce_sum_test_4", data, axis=[1,2], keepdim=False)
    reduce_sum("reduce_sum_test_5", data, axis=[0,1], keepdim=True)

if __name__ == "__main__":
    main()
