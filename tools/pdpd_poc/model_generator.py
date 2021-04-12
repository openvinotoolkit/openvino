'''
A paddlepaddle model generator for unit test.
You can extend to other Ops of interest.
'''
import numpy as np

import paddle
from paddle import fluid
from paddle.fluid.framework import Program, program_guard  


# yolo_box generator
def gen_yolo_box():  
    from paddle.vision.ops import yolo_box

    x = np.random.random([1, 255, 19, 19]).astype('float32')
    img_size = np.array([
        [608,608]
    ], dtype=np.int32)    
    #img_size = np.ones((1, 2)).astype('int32')
    print(img_size.shape)

    #x = paddle.to_tensor(x)
    #img_size = paddle.to_tensor(img_size)

    paddle.enable_static()
    main_program = paddle.static.Program()
    startup_program = paddle.static.Program()
    with paddle.static.program_guard(main_program, startup_program):
        node_x = paddle.static.data(name='x', shape=x.shape, dtype='float32')
        node_img_size = paddle.static.data(name='img_size', shape=img_size.shape, dtype='int32')

        boxes, scores = paddle.vision.ops.yolo_box(node_x,
                                                node_img_size,
                                                anchors=[116, 90, 156, 198, 373, 326],
                                                class_num=80,
                                                conf_thresh=0.01,
                                                downsample_ratio=32,
                                                clip_bbox=True,
                                                name=None, 
                                                scale_x_y=1.)
    print(boxes, scores)

    cpu = paddle.static.cpu_places(1)
    exe = paddle.static.Executor(cpu[0])
    # startup program will call initializer to initialize the parameters.
    exe.run(startup_program)

    outs = exe.run(
        feed={'x': x, 'img_size': img_size},
        fetch_list=[boxes, scores],
        program=main_program)    
    print(outs)

    '''
    option D: convert compact model representation to scattered.
    '''    
    path = "./yolo_box/decomposed_model"
    fluid.io.save_inference_model(dirname=path, main_program=main_program, feeded_var_names=['x', 'img_size'], target_vars=[boxes, scores], executor=exe)

    path = "./yolo_box/combined_model"
    fluid.io.save_inference_model(dirname=path, main_program=main_program, feeded_var_names=['x', 'img_size'], target_vars=[boxes, scores], executor=exe, 
                        model_filename="yolo_box.pdmodel", params_filename="yolo_box.pdiparams")



if __name__ == "__main__":
    gen_yolo_box()
