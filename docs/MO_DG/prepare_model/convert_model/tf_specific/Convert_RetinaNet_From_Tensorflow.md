# Converting RetinaNet Model from TensorFlow* to the Intermediate Representation {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_RetinaNet_From_Tensorflow}

This tutorial explains how to convert RetinaNet model to Intermediate Representation (IR).

[Public RetinaNet model](https://github.com/fizyr/keras-retinanet) does not contain pre-trained TensorFlow* weights. To convert this model to the IR you can use [ Reproduce Keras* to TensorFlow* Conversion tutorial](https://docs.openvinotoolkit.org/latest/omz_models_model_retinanet_tf.html)

After you convert model to TensorFlow* you can run the Model-Optimizer command below:
```sh
python mo.py --input input_1 --input_model retinanet_resnet50_coco_best_v2.1.0.pb --output_dir C:\projects\pycharm\models\IRs --data_type FP32 --input_shape [1,1333,1333,3] --output filtered_detections/map/TensorArrayStack/TensorArrayGatherV3,filtered_detections/map/TensorArrayStack_1/TensorArrayGatherV3,iltered_detections/map/TensorArrayStack_2/TensorArrayGatherV3 --transformations_config ./extensions/front/tf/retinanet.json
```
