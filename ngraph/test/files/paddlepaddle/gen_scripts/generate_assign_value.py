import numpy as np
from save_model import saveModel

def pdpd_assign_value(name, test_x):
    import paddle as pdpd
    pdpd.enable_static()
    main_program = pdpd.static.Program()
    startup_program = pdpd.static.Program()
    with pdpd.static.program_guard(main_program, startup_program):
        node_x = pdpd.static.data(name='x', shape=test_x.shape, dtype=test_x.dtype)
        const_value = pdpd.assign(test_x, output=None)
        result = pdpd.add(node_x, const_value);
        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())
        outs = exe.run(
            feed={'x': test_x},
            fetch_list=[result]
        )

        saveModel(name, exe, feedkeys=['x'], fetchlist=[result], inputs=[test_x], outputs=[outs[0]])

    print(outs[0])


def compare():
    x = np.ones([1, 1, 4, 4]).astype(np.float32)
    test_cases = [
        {
            "name": "assign_value_fp32",
            "input": np.ones([1, 1, 4, 4]).astype(np.float32)
        }
    ]
    for test in test_cases:
        pdpd_assign_value(test['name'], test['input'])


if __name__ == "__main__":
    compare()
