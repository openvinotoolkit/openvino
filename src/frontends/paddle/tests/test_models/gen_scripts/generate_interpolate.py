# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import paddle
from paddle.nn.functional import interpolate
from save_model import saveModel
import sys

def run_and_save_model(input_x, name, feed, fetch_list, main_prog, start_prog):
    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    exe.run(start_prog)
    outs = exe.run(
        feed={'x': input_x},
        fetch_list=fetch_list,
        program=main_prog)

    with paddle.static.program_guard(main_prog, start_prog):
        saveModel(name, exe, feed_vars=[feed], fetchlist=fetch_list, inputs=[input_x],
                  outputs=[outs[0]], target_dir=sys.argv[1])

    return outs


def paddle_interpolate(x, sizes=None, scale_factor=None, mode='nearest', align_corners=True,
                     align_mode=0, data_format='NCHW', name=None):
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        interp = interpolate(node_x, size=sizes, scale_factor=scale_factor,
                             mode=mode, align_corners=align_corners, align_mode=align_mode,
                             data_format=data_format, name=name)
        out = paddle.static.nn.batch_norm(interp, use_global_stats=True, epsilon=0)
    outs = run_and_save_model(x, name, node_x, out, main_program, startup_program)
    return outs[0]


def resize_upsample_bilinear():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)

    test_case = [{'name': 'bilinear_upsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'bilinear_upsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'bilinear_upsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, [64, 64], None, mode='bilinear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])


def resize_downsample_bilinear():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)
    data_28 = data.reshape([1, 1, 2, 8])
    test_case = [{'name': 'bilinear_downsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'bilinear_downsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'bilinear_downsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data_28, [2, 4], None, mode='bilinear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])

def resize_upsample_nearest():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)

    test_case = [
        {'name': 'nearest_upsample_false_0', 'size': [64, 64], 'align_corners': False, 'align_mode': 0},
        {'name': 'nearest_upsample_false_1', 'size': [16, 64], 'align_corners': False, 'align_mode': 0}
    ]

    for test in test_case:
        paddle_result = paddle_interpolate(data, test['size'], None, mode='nearest', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])


def resize_downsample_nearest():
    data = np.arange(0, 4096).astype(np.float32)
    data_64 = data.reshape([1, 1, 64, 64])
    test_case = [
        {'name': 'nearest_downsample_false_0', 'size': [8, 8], 'align_corners': False, 'align_mode': 1},
        {'name': 'nearest_downsample_false_1', 'size': [4, 8], 'align_corners': False, 'align_mode': 1}
    ]

    for test in test_case:
        paddle_result = paddle_interpolate(data_64, test['size'], None, mode='nearest', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])

def paddle_interpolate_tensor_size(data, sizes, mode='nearest', align_corners=True, align_mode=0, data_format='NCHW', name=None):
    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        node_x = paddle.static.data(name='x', shape=data.shape, dtype='float32')
        node_sizes = paddle.static.data(name='sizes', shape=sizes.shape, dtype='int32')
        interp = interpolate(node_x, size=node_sizes, scale_factor=None,
                             mode=mode, align_corners=align_corners, align_mode=align_mode,
                             data_format=data_format, name=name)
        out = paddle.static.nn.batch_norm(interp, use_global_stats=True, epsilon=0)
        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        exe.run(startup_program)
        outs = exe.run(
            feed={'x': data, 'sizes': sizes},
            fetch_list=out,
            program=main_program)
        saveModel(name, exe, feed_vars=[node_x, node_sizes], fetchlist=out, inputs=[data, sizes], outputs=[outs[0]], target_dir=sys.argv[1])

def nearest_upsample_tensor_size():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)
    sizes = np.array([8, 8], dtype=np.int32)
    test_case = [{'name': 'nearest_upsample_tensor_size', 'align_corners': False, 'align_mode': 0}]
    for test in test_case:
        paddle_interpolate_tensor_size(data, sizes, 'nearest', test['align_corners'], test['align_mode'], 'NCHW', test['name'])

def bilinear_upsample_tensor_size():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)
    sizes = np.array([8, 8], dtype="int32")

    test_case = [{'name': 'bilinear_upsample_tensor_size', 'align_corners': False, 'align_mode': 1}]

    for test in test_case:
        paddle_interpolate_tensor_size(data, sizes, 'bilinear', test['align_corners'], test['align_mode'], 'NCHW', test['name'])

def bilinear_upsample_scales():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)

    test_case = [{'name': 'bilinear_upsample_scales', 'align_corners': False, 'align_mode': 1, "scales": 2},
                 {'name': 'bilinear_upsample_scales2', 'align_corners': False, 'align_mode': 1, "scales": [2, 2]}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, None, 2, mode='bilinear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])

# trilinear
def resize_upsample_trilinear():
    data = np.array([[[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ],[
        [13, 14, 15, 16],
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4],
    ]]]], dtype=np.float32)

    test_case = [{'name': 'trilinear_upsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'trilinear_upsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'trilinear_upsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, [4, 64, 64], None, mode='TRILINEAR', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCDHW', name=test['name'])


def resize_downsample_trilinear():
    data = np.array([[[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ],[
        [13, 14, 15, 16],
        [9, 10, 11, 12],
        [5, 6, 7, 8],
        [1, 2, 3, 4]
    ]]]], dtype=np.float32)
    data_28 = data.reshape([1, 1, 2, 2, 8])
    test_case = [{'name': 'trilinear_downsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'trilinear_downsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'trilinear_downsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data_28, [2, 2, 4], None, mode='TRILINEAR', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCDHW', name=test['name'])

def trilinear_upsample_tensor_size():
    data = np.array([[[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]]], dtype=np.float32)
    sizes = np.array([2, 8, 8], dtype="int32")

    test_case = [{'name': 'trilinear_upsample_tensor_size', 'align_corners': False, 'align_mode': 1}]

    for test in test_case:
        paddle_interpolate_tensor_size(data, sizes, 'TRILINEAR', test['align_corners'], test['align_mode'], 'NCDHW', test['name'])

def trilinear_upsample_scales():
    data = np.array([[[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]]], dtype=np.float32)

    test_case = [{'name': 'trilinear_upsample_scales', 'align_corners': False, 'align_mode': 1, "scales": 2},
                 {'name': 'trilinear_upsample_scales2', 'align_corners': False, 'align_mode': 1, "scales": [1, 2, 2]}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, None, 3, mode='TRILINEAR', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCDHW', name=test['name'])


# bicubic
def resize_upsample_bicubic():
    data = np.array([[[
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]]], dtype=np.float32)

    test_case = [{'name': 'bicubic_upsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'bicubic_upsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'bicubic_upsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, [6, 6], None, mode='bicubic', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])


def resize_downsample_bicubic():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)
    data_28 = data.reshape([1, 1, 2, 8])
    test_case = [{'name': 'bicubic_downsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'bicubic_downsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'bicubic_downsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data_28, [2, 4], None, mode='bicubic', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])

def bicubic_upsample_tensor_size():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)
    sizes = np.array([8, 8], dtype="int32")

    test_case = [{'name': 'bicubic_upsample_tensor_size', 'align_corners': False, 'align_mode': 1}]

    for test in test_case:
        paddle_interpolate_tensor_size(data, sizes, 'bicubic', test['align_corners'], test['align_mode'], 'NCHW', test['name'])

def bicubic_upsample_scales():
    data = np.array([[[
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]]], dtype=np.float32)

    test_case = [{'name': 'bicubic_upsample_scales', 'align_corners': False, 'align_mode': 1, "scales": 2},
                 {'name': 'bicubic_upsample_scales2', 'align_corners': False, 'align_mode': 1, "scales": [2, 2]}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, None, 2, mode='bicubic', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCHW', name=test['name'])

# linear
def resize_upsample_linear():
    data = np.array([[
        [1, 2, 3]
    ]], dtype=np.float32)

    test_case = [{'name': 'linear_upsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'linear_upsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'linear_upsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, [6,], None, mode='linear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCW', name=test['name'])


def resize_downsample_linear():
    data = np.array([[
        [1, 2, 3, 4],
        [5, 6, 7, 8]
    ]], dtype=np.float32)
    data_28 = data.reshape([1, 1, 8])
    test_case = [{'name': 'linear_downsample_false_1', 'align_corners': False, 'align_mode': 1},
                 {'name': 'linear_downsample_false_0', 'align_corners': False, 'align_mode': 0},
                 {'name': 'linear_downsample_true_0', 'align_corners': True, 'align_mode': 0}]

    for test in test_case:
        paddle_result = paddle_interpolate(data_28, [4,], None, mode='linear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCW', name=test['name'])

def linear_upsample_tensor_size():
    data = np.array([[
        [1, 2, 3, 4]
    ]], dtype=np.float32)
    sizes = np.array([8,], dtype="int32")

    test_case = [{'name': 'linear_upsample_tensor_size', 'align_corners': False, 'align_mode': 1}]

    for test in test_case:
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            node_x = paddle.static.data(name='x', shape=data.shape, dtype='float32')
            node_sizes = paddle.static.data(name='sizes', shape=sizes.shape, dtype='int32')
            interp = interpolate(node_x, size=node_sizes, scale_factor=None,
                                 mode='linear', align_corners=test['align_corners'], align_mode=test['align_mode'],
                                 data_format='NCW', name=test['name'])
            out = paddle.static.nn.batch_norm(interp, use_global_stats=True, epsilon=0)
            cpu = paddle.static.cpu_places(1)
            exe = paddle.static.Executor(cpu[0])
            exe.run(startup_program)
            outs = exe.run(
                feed={'x': data, 'sizes': sizes},
                fetch_list=out,
                program=main_program)
            saveModel(test['name'], exe, feed_vars=[node_x, node_sizes], fetchlist=out, inputs=[data, sizes], outputs=[outs[0]], target_dir=sys.argv[1])

def linear_upsample_scales():
    data = np.array([[
        [1, 2, 3, 4]
    ]], dtype=np.float32)

    test_case = [{'name': 'linear_upsample_scales', 'align_corners': False, 'align_mode': 1, "scales": 2},
                 {'name': 'linear_upsample_scales2', 'align_corners': False, 'align_mode': 1, "scales": [2, 2]}]

    for test in test_case:
        paddle_result = paddle_interpolate(data, None, 2, mode='linear', align_corners=test['align_corners'],
                                       align_mode=test['align_mode'], data_format='NCW', name=test['name'])

if __name__ == "__main__":
    # bilinear
    resize_downsample_bilinear()
    resize_upsample_bilinear()
    bilinear_upsample_tensor_size()
    bilinear_upsample_scales()
    # nearest
    resize_downsample_nearest()
    resize_upsample_nearest()
    nearest_upsample_tensor_size()
    # trilinear
    resize_downsample_trilinear()
    resize_upsample_trilinear()
    trilinear_upsample_tensor_size()
    trilinear_upsample_scales()
    # bicubic
    resize_downsample_bicubic()
    resize_upsample_bicubic()
    bicubic_upsample_tensor_size()
    bicubic_upsample_scales()
    # linear
    resize_downsample_linear()
    resize_upsample_linear()
    linear_upsample_tensor_size()
    linear_upsample_scales()
