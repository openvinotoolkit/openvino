# ref: https://www.paddlepaddle.org.cn/tutorials/projectdetail/1998893#anchor-2

import os
import sys

import numpy as np
import paddle

from save_model import exportModel, saveModel


def loop():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=0, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, dummy):
        i = i + 1
        return i, dummy

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        if paddle.__version__ >= '2.0.0':
            node_i = paddle.add(node_i, node_x)
        else:
            node_i = paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out, dummy = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=out)

        saveModel('loop', exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[res[0]], target_dir=sys.argv[1])

    return res

def loop_x():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=1, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, t):
        i = i + x
        return i, t

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        if paddle.__version__ >= '2.0.0':
            node_i = paddle.add(node_i, node_x)
        else:
            node_i = paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out, dummy = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=out)

        saveModel('loop_x', exe, feed_vars=[node_x], fetchlist=[out], inputs=[x], outputs=[res[0]], target_dir=sys.argv[1])

    return res

def loop_t():
    paddle.enable_static()
    x = np.full(shape=[1], fill_value=0, dtype='int64')

    def cond(i, ten):
        return ten >= i

    def body(i, t):
        i = i + 1
        t = t - 1
        return i, t

    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_i = paddle.full(shape=[1], fill_value=0, dtype='int64', name='i')
        if paddle.__version__ >= '2.0.0':
            node_i = paddle.add(node_i, node_x)
        else:
            paddle.fluid.layers.nn.elementwise_add(node_i, node_x)
        node_ten = paddle.full(shape=[1], fill_value=10, dtype='int64', name='ten')

        out_i,out_t = paddle.static.nn.while_loop(cond, body, [node_i, node_ten], name='while_loop')

        exe = paddle.static.Executor(place=paddle.CPUPlace())
        res = exe.run(paddle.static.default_main_program(), feed={'x':x}, fetch_list=[out_i,out_t])

        saveModel('loop_t', exe, feed_vars=[node_x], fetchlist=[out_i,out_t], inputs=[x], outputs=res, target_dir=sys.argv[1])

    return res

def loop_dyn():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')
        j = i + 1

        while t >= i:
            i = i + 1

        return i, j

    x = np.full(shape=[1], fill_value=0, dtype='int64')
    return exportModel('loop_dyn', test_model, [x], target_dir=sys.argv[1])


def loop_dyn_x():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        while t >= i:
            i = i + x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_dyn_x', test_model, [x], target_dir=sys.argv[1])

def loop_if():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        while t >= i:
            if i < 5:
                i = i + x
            else:
                i = i + 2 * x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_if', test_model, [x], target_dir=sys.argv[1])

def loop_if_loop():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        if x < 5:
            while t >= i:
                i = i + x * 2
        else:
            while t >= i:
                i = i + x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_if_loop', test_model, [x], target_dir=sys.argv[1])

def loop_if_loop_if():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        if x < 5:
            while t >= i:
                if x == 0:
                    i = i + x * 2
                else:
                    i = i + x * 3
        else:
            while t >= i:
                i = i + x

        return i

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_if_loop_if', test_model, [x], target_dir=sys.argv[1])

def loop_if_loop_complex():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x, y):
        i = paddle.full(shape=[1], fill_value=0, dtype='int64')
        j = paddle.full(shape=[1], fill_value=0, dtype='int64')
        
        i = i + x
        j = j + y
        t = paddle.full(shape=[1], fill_value=10, dtype='int64')

        if x < 5:
            if y < 44:
                while t >= i:
                    while t >= i:
                        if x == 0:
                            i = i + x * 2
                            j = j + y * 2
                        else:
                            i = i + x * 3
                            j = j + y * 3
        else:
            while t >= i:
                i = i + x

        return i, j

    x = np.full(shape=[1], fill_value=1, dtype='int64')
    y = np.full(shape=[1], fill_value=1, dtype='int64')
    return exportModel('loop_if_loop_complex', test_model, [x, y], target_dir=sys.argv[1])

def loop_tensor_array():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int32')
        
        t = paddle.full(shape=[1], fill_value=10, dtype='int32')
        y = paddle.full(shape=[30,3], fill_value=2, dtype='float32')

        result = []
        while t >= i:
            i = i + 1
            result.append(x[0:2,:])

        return paddle.concat(result)

    x = np.full(shape=[30,3], fill_value=1, dtype='float32')
    return exportModel('loop_tensor_array', test_model, [x], target_dir=sys.argv[1])

def loop_if_tensor_array():
    paddle.disable_static()

    @paddle.jit.to_static
    def test_model(x):
        i = paddle.full(shape=[1], fill_value=0, dtype='int32')
        
        t = paddle.full(shape=[1], fill_value=10, dtype='int32')
        y = paddle.full(shape=[30,3], fill_value=2, dtype='float32')

        result1 = []
        result2 = []
        while t >= i:
            i = i + 1
            if i >= 0:
                v = i + 1
                result1.append(x[0:i,:])
                result2.append(x[0:v,:] + 2)
            else:
                result1.append(x[0:i,:])
                #result2.append(x[0:i,:])
        return paddle.concat(result1), paddle.concat(result2)

    x = np.full(shape=[30,3], fill_value=1, dtype='float32')
    return exportModel('loop_if_tensor_array', test_model, [x], target_dir=sys.argv[1])

if __name__ == "__main__":
    # 95436: sporadic failure
    print(loop())
    print(loop_dyn())

    print(loop_t())
    print(loop_x())

    print(loop_dyn_x().numpy())
    print(loop_if().numpy())
    print(loop_if_loop().numpy())
    print(loop_if_loop_if().numpy())
    print(loop_if_loop_complex())
    print(loop_tensor_array().numpy())
    x, y = loop_if_tensor_array()
