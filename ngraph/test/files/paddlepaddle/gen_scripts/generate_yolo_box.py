#
# pool2d paddle model generator
#
import numpy as np
from save_model import saveModel

def yolo_box(name : str, x, img_size, attrs : dict):
    import paddle as pdpd
    pdpd.enable_static()
    
    with pdpd.static.program_guard(pdpd.static.Program(), pdpd.static.Program()):
        node_x = pdpd.static.data(name='x', shape=x.shape, dtype=x.dtype)
        node_img_size = pdpd.static.data(name='img_size', shape=img_size.shape, dtype=img_size.dtype)
        boxes, scores = pdpd.vision.ops.yolo_box(node_x,
                                                node_img_size,
                                                anchors=attrs['anchors'],
                                                class_num=attrs['class_num'],
                                                conf_thresh=attrs['conf_thresh'],
                                                downsample_ratio=attrs['downsample_ratio'],
                                                clip_bbox=attrs['clip_bbox'],
                                                name=None, 
                                                scale_x_y=attrs['scale_x_y'])

        cpu = pdpd.static.cpu_places(1)
        exe = pdpd.static.Executor(cpu[0])
        # startup program will call initializer to initialize the parameters.
        exe.run(pdpd.static.default_startup_program())

        outs = exe.run(
            feed={'x': x, 'img_size': img_size},
            fetch_list=[boxes, scores])
        
        # Save inputs in order of ngraph function, to facilite Fuzzy test, 
        # which accepts inputs and outputs in this order as well. 
        saveModel(name, exe, feedkeys=['x', 'img_size'], fetchlist=[boxes, scores], inputs=[x, img_size], outputs=outs)

    return outs


def main():
    # yolo_box
    pdpd_attrs = {
            'anchors': [10, 13, 16, 30, 33, 23],
            'class_num': 2,
            'conf_thresh': 0.5,
            'downsample_ratio': 32,
            'clip_bbox': False, #There is bug in Paddle2ONN where clip_bbox is always ignored.
            'scale_x_y': 1.0
    }

    N = 1
    num_anchors = int(len(pdpd_attrs['anchors'])//2)
    x_shape = (N, num_anchors * (5 + pdpd_attrs['class_num']), 13, 13)
    imgsize_shape = (N, 2)

    data = np.random.random(x_shape).astype('float32')
    data_ImSize = np.random.randint(10, 20, imgsize_shape).astype('int32') 

    # For any change to pdpd_attrs, do -
    # step 1. generate paddle model
    pred_pdpd = yolo_box('yolo_box_test1', data, data_ImSize, pdpd_attrs)



if __name__ == "__main__":
    main()     