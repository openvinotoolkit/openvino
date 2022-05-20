# Converting a TensorFlow RetinaNet Model {#openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_RetinaNet_From_Tensorflow}

This tutorial explains how to convert a RetinaNet model to the Intermediate Representation (IR).

[Public RetinaNet model](https://github.com/fizyr/keras-retinanet) does not contain pretrained TensorFlow weights.
To convert this model to the TensorFlow format, follow the [Reproduce Keras to TensorFlow Conversion tutorial](@ref omz_models_model_retinanet_tf).

After converting the model to TensorFlow format, run the Model Optimizer command below:
```sh
mo --input "input_1[1 1333 1333 3]" --input_model retinanet_resnet50_coco_best_v2.1.0.pb --data_type FP32 --transformations_config front/tf/retinanet.json
```

Where `transformations_config` command-line parameter specifies the configuration json file containing model conversion hints for the Model Optimizer.
The json file contains some parameters that need to be changed if you train the model yourself. It also contains information on how to match endpoints
to replace the subgraph nodes. After the model is converted to the IR, the output nodes will be replaced with DetectionOutput layer.
