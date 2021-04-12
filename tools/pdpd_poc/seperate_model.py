import os
import numpy as np

import paddle
import paddle.fluid as fluid

def load_image():
    import cv2

    img = cv2.imread("cat3.bmp")
    img = cv2.resize(img, (608, 608))
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std

    return img

input_h, input_w = 608, 608
tensor_img = load_image()
image_shape = np.array([input_h, input_w], dtype='int32')
scale_factor = np.array([1, 1], dtype='int32')

paddle.enable_static()
exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

model_path = "/home/cecilia/explore/PDPD/paddle_models/object_detection/yolov3_darknet53_270e_coco/model.pdmodel"

dir_path = os.path.dirname(os.path.realpath(model_path))
model_name = os.path.basename(model_path)
model_name = os.path.splitext(model_name)[0]+".pdmodel"
params_name = os.path.splitext(model_name)[0]+".pdiparams"
[inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dir_path, exe, model_filename=model_name, params_filename=params_name)

for i in range(len(feed_target_names)):
    print("feed_targets: ", feed_target_names[i])

batch_outputs = exe.run(inference_program,
              feed={feed_target_names[0]: image_shape[np.newaxis, :], feed_target_names[1]: tensor_img.astype(np.float32), feed_target_names[2]: scale_factor[np.newaxis, :]},
              fetch_list=fetch_targets, 
              return_numpy=False)

bboxes = np.array(batch_outputs[0])
print(bboxes)


global_block = inference_program.global_block()


'''
convert compact model representation to scattered.
'''
path = "./infer_model"
fluid.io.save_inference_model(dirname=path, main_program=inference_program, feeded_var_names=feed_target_names,
             target_vars=fetch_targets, executor=exe)
