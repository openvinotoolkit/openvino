# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel
import sys

def yolo_box(name : str, x, img_size, attrs : dict):
    import paddle
    paddle.enable_static()
    
    with paddle.static.program_guard(paddle.static.Program(), paddle.static.Program()):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_img_size = paddle.static.data(name='img_size', shape=img_size.shape, dtype=img_size.dtype)
        boxes, scores = paddle.vision.ops.yolo_box(node_x,
                                                node_img_size,
                                                anchors=attrs['anchors'],
                                                class_num=attrs['class_num'],
                                                conf_thresh=attrs['conf_thresh'],
                                                downsample_ratio=attrs['downsample_ratio'],
                                                clip_bbox=attrs['clip_bbox'],
                                                name=None, 
                                                scale_x_y=attrs['scale_x_y'])

        cpu = paddle.static.cpu_places(1)
        exe = paddle.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(paddle.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'img_size': img_size},
            fetch_list=[boxes, scores])
        
        # Save inputs in order of OpenVINO model, to facilitate Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        saveModel(name, exe, feed_vars=[node_x, node_img_size], fetchlist=[boxes, scores],
                  inputs=[x, img_size], outputs=outs, target_dir=sys.argv[1])

    return outs


def TEST1():
    # yolo_box
    paddle_attrs = {
            'name': "yolo_box_default",
            'anchors': [10, 13, 16, 30, 33, 23],
            'class_num': 2,
            'conf_thresh': 0.5,
            'downsample_ratio': 32,
            'clip_bbox': False,
            'scale_x_y': 1.0
    }

    paddle_attrs_clip_box = {
        'name': "yolo_box_clip_box",
        'anchors': [10, 13, 16, 30, 33, 23],
        'class_num': 2,
        'conf_thresh': 0.5,
        'downsample_ratio': 32,
        'clip_bbox': True,
        'scale_x_y': 1.0
    }

    paddle_attrs_scale_xy = {
        'name': "yolo_box_scale_xy",
        'anchors': [10, 13, 16, 30, 33, 23],
        'class_num': 2,
        'conf_thresh': 0.5,
        'downsample_ratio': 32,
        'clip_bbox': True,
        'scale_x_y': 1.2
    }

    paddle_attrs_list = [paddle_attrs, paddle_attrs_clip_box, paddle_attrs_scale_xy]
    
    N = 32
    num_anchors = int(len(paddle_attrs['anchors'])//2)
    x_shape = (N, num_anchors * (5 + paddle_attrs['class_num']), 13, 13)
    imgsize_shape = (N, 2)

    data = np.random.random(x_shape).astype('float32')
    data_ImSize = np.random.randint(10, 20, imgsize_shape).astype('int32') 

    for item in paddle_attrs_list:
        pred_paddle = yolo_box(item['name'], data, data_ImSize, item)


def TEST2():
    # yolo_box uneven spatial width and height
    paddle_attrs = {
            'name': "yolo_box_uneven_wh",
            'anchors': [10, 13, 16, 30, 33, 23],
            'class_num': 2,
            'conf_thresh': 0.5,
            'downsample_ratio': 32,
            'clip_bbox': False,
            'scale_x_y': 1.0
    }

    N = 16
    SPATIAL_WIDTH = 13
    SPATIAL_HEIGHT = 9
    num_anchors = int(len(paddle_attrs['anchors'])//2)
    x_shape = (N, num_anchors * (5 + paddle_attrs['class_num']), SPATIAL_HEIGHT, SPATIAL_WIDTH)
    imgsize_shape = (N, 2)

    data = np.random.random(x_shape).astype('float32')
    data_ImSize = np.random.randint(10, 20, imgsize_shape).astype('int32')
    
    pred_paddle = yolo_box(paddle_attrs['name'], data, data_ImSize, paddle_attrs)

if __name__ == "__main__":
    TEST1()
    TEST2()
