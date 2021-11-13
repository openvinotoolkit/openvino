import numpy as np
from save_model import saveModel
import sys


def pdpd_assign_value(name, test_x):
    import paddle as pdpd
    pdpd.enable_static()
    main_program = pdpd.static.Program()
    startup_program = pdpd.static.Program()
    with pdpd.static.program_guard(main_program, startup_program):
        node_x = pdpd.static.data(name='x', shape=test_x.shape, dtype=test_x.dtype if test_x.dtype != np.bool else np.int32)
        node_x = pdpd.cast(node_x, dtype=test_x.dtype)
        const_value = pdpd.assign(test_x, output=None)
        result = pdpd.cast(pdpd.concat([node_x, const_value], 0), dtype=np.float32)
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        if test_x.dtype == np.bool:
            test_x = test_x.astype(np.int32)

        outs = exe.run(
            feed={'x': test_x},
            fetch_list=[result]
        )

        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[test_x], outputs=[outs[0]], target_dir=sys.argv[1])


def compare():

    test_cases = [
        {
            "name": "assign_value_fp32",
            "input": np.ones([1, 1, 4, 4]).astype(np.float32)
        },
        {
            "name": "assign_value_int32",
            "input": np.ones([1, 1, 4, 4]).astype(np.int32)
        },
        {
            "name": "assign_value_int64",
            "input": np.ones([1, 1, 4, 4]).astype(np.int64)
        },
        {
            "name": "assign_value_boolean",
            "input": np.array([False, True, False])
        }
    ]
    for test in test_cases:
        pdpd_assign_value(test['name'], test['input'])


if __name__ == "__main__":
    compare()
