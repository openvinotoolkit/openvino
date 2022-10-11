# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# relu paddle model generator
#
import os.path

import sys

import os
import numpy as np
import paddle


# print numpy array like C structure
def print_alike(arr):
    shape = arr.shape
    rank = len(shape)

    # print("shape: ", shape, "rank: %d" %(rank))

    # for idx, value in np.ndenumerate(arr):
    #    print(idx, value)

    def print_array(arr, end=' '):
        shape = arr.shape
        rank = len(arr.shape)
        if rank > 1:
            line = "{"
            for i in range(arr.shape[0]):
                line += print_array(arr[i, :], end="},\n" if i < arr.shape[0] - 1 else "}")
            line += end
            return line
        else:
            line = "{"
            for i in range(arr.shape[0]):
                line += "{:.2f}".format(arr[i])  # str(arr[i])
                line += ", " if i < shape[0] - 1 else ' '
            line += end
            # print(line)
            return line

    print(print_array(arr, "}"))


def saveModel(name, exe, feedkeys: list, fetchlist: list, inputs: list, outputs: list, target_dir: str):
    model_dir = os.path.join(target_dir, name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    print("\n\n------------- %s -----------\n" % (name))
    for i, input in enumerate(inputs):
        print("INPUT %s :" % (feedkeys[i]), input.shape, input.dtype, "\n")
        print_alike(input)
        np.save(os.path.join(model_dir, "input{}".format(i)), input)
        np.save(os.path.join(model_dir, "input{}.{}.{}".format(i, feedkeys[i], input.dtype)), input)
    print("\n")

    for i, output in enumerate(outputs):
        print("OUTPUT %s :" % (fetchlist[i]), output.shape, output.dtype, "\n")
        print_alike(output)
        np.save(os.path.join(model_dir, "output{}".format(i)), output)

        # composited model + scattered model
    paddle.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe)
    paddle.fluid.io.save_inference_model(model_dir, feedkeys, fetchlist, exe, model_filename=name + ".pdmodel",
                                       params_filename=name + ".pdiparams")


def relu(name: str, x):
    import paddle
    paddle.enable_static()

    node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
    out = paddle.nn.functional.relu(node_x)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(paddle.static.default_startup_program())

    outs = exe.run(
        feed={'x': x},
        fetch_list=[out])

    saveModel(name, exe, feedkeys=['x'], fetchlist=[out],
              inputs=[x], outputs=[outs[0]], target_dir=sys.argv[1])

    return outs[0]


def main():
    data = np.array([-2, 0, 1]).astype('float32')

    relu("relu_unsupported", data)

    with open(os.path.join(sys.argv[1], "relu_unsupported", "relu_unsupported.pdmodel"), mode='rb') as file:
        modelContent = file.read()

    modelContent = modelContent.replace(b"relu", b"rxyz")

    with open(os.path.join(sys.argv[1], "relu_unsupported", "relu_unsupported.pdmodel"), mode='wb') as file:
        file.write(modelContent)


if __name__ == "__main__":
    main()