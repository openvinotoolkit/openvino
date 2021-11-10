import numpy as np
import sys
from save_model import saveModel


def logical_and(name : str, x, y):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_y = pdpd.static.data(name='y', shape=y.shape, dtype=y.dtype)
        bool_x = pdpd.cast(node_x, np.bool)
        bool_y = pdpd.cast(node_y, np.bool)
        out = pdpd.logical_and(bool_x, bool_y)
        out = pdpd.cast(out, x.dtype)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def logical_or(name : str, x, y):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_y = pdpd.static.data(name='y', shape=y.shape, dtype=y.dtype)
        bool_x = pdpd.cast(node_x, np.bool)
        bool_y = pdpd.cast(node_y, np.bool)
        out = pdpd.logical_or(bool_x, bool_y)
        out = pdpd.cast(out, x.dtype)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def logical_xor(name : str, x, y):
    import paddle as pdpd
    pdpd.enable_static()

    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_y = pdpd.static.data(name='y', shape=y.shape, dtype=y.dtype)
        bool_x = pdpd.cast(node_x, np.bool)
        bool_y = pdpd.cast(node_y, np.bool)
        out = pdpd.logical_xor(bool_x, bool_y)
        out = pdpd.cast(out, x.dtype)

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])

        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])
        saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out], inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def equal_logical_not(name : str, x, y):
    import paddle as pdpd
    pdpd.enable_static()

    node_x = pdpd.static.data(name='x', shape=x.shape, dtype='float32')
    node_y = pdpd.static.data(name='y', shape=y.shape, dtype='float32')

    out = pdpd.equal(node_x, node_y)
    out = pdpd.logical_not(out)
    out = pdpd.cast(out, np.float32)

    cpu = pdpd.static.cpu_places(1)
    exe = pdpd.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(pdpd.static.default_startup_program())

    outs = exe.run(
        feed={'x': x, 'y': y},
        fetch_list=[out])

    saveModel(name, exe, feedkeys=['x', 'y'], fetchlist=[out],
              inputs=[x, y], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]

def main():
    data_x = np.array([[[[-1, 0, 1]], [[2, 3, 4]]]]).astype(np.float32)
    data_y = np.array([[[[2, 0, 3]], [[3, 1, 4]]]]).astype(np.float32)

    logical_and("logical_and", data_x, data_y)
    logical_or("logical_or", data_x, data_y)
    logical_xor("logical_xor", data_x, data_y)
    equal_logical_not("logical_not", data_x, data_y)

if __name__ == "__main__":
    main()
