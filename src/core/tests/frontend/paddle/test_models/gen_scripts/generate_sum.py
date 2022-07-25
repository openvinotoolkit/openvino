from __future__ import print_function
import sys
import paddle
import paddle.fluid as fluid

from save_model import saveModel

def sum(name: str, shape: list, dtype, values):
    paddle.enable_static()
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        if len(values) == 1:
            inputs = fluid.layers.fill_constant(shape=shape, dtype=dtype, value=values[0])
        else:
            inputs = []
            for i in values:
                inputs.append(fluid.layers.fill_constant(shape=shape, dtype=dtype, value=i))
        out = fluid.layers.sum(inputs)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())
        outs = exe.run(
            fetch_list=[out])
        saveModel(name, exe, feedkeys=[], fetchlist=[out], inputs=[], outputs=[outs[0]], target_dir=sys.argv[1])
    return outs[0]


def main():
    # single tensor
    sum("sum_1", [2, 3], 'float32', [random.random()])
    # multiple tensors
    values = [random.random(), random.random(), random.random()]
    sum("sum_2", [2, 3], 'float32', values)


if __name__ == "__main__":
    main()
