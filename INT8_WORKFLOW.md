OpenVINO Int8 Workflow In a Nutshell
-----------------------------------
To operate with int8, all the data (weights, inputs, activations, etc) should be carefully quantized. The quantization process is driven by:

* Normalization (or scaling) factor, determined by range of the data
* Quantization level, which depends on whether data is signed or unsigned, and destination precision.

OpenVINO supports two main sources of this information, and thus two main sources of the int8 models:

* Conversion of the framework-quantized models. This approach relies on the training for low precision and subsequent conversion of the resulting model with [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html) tool. This approach usually gives optimal accuracy and performance, but requires careful model re-training/fine-tuning. Then for inference, both normalization and quantization factors are deduced fully from the model data (e.g. FakeQuantize layers) and no additional steps are required.

* Post-training quantization of the floating point models with the [Calibration tool](https://docs.openvinotoolkit.org/latest_docs_IE_DG_Int8Inference.html#low_precision_8_bit_integer_inference_workflow). Just like approach described earlier, the calibration is also fully offline additional step to equip a model with (optional) int8 information.  The approach is somewhat more universal, requiring just floating point model and no retraining to leverage the int8. The calibration is iterative process of gathering _activations_ statistics like histogram (for determining scaling/parameters), applying the quantization parameters and evaluating resulting model accuracy to keep it as close to original as possible. For _weights_, in contrast, the maximum abs value per output channel m is found. The per-channel range is then [-m,m]. This calibration process trades the performance vs accuracy and results in a mixed precision model which are a combination of fp32 (high accuracy) and int8 (high performance) layers.

Notice that OpenVINO assumes the symmetrically quantized models (with respect to weights) and either symmetric (signed) or fully unsigned activations.

Quantized Model Example
-----------------------------------
For the MLPerf 0.5 submission, the only directly converted quantized model is ssd-mobilenet from Habana ("ssd-mobilenet 300x300 symmetrically quantized finetuned"), referenced at https://github.com/mlperf/inference/tree/master/v0.5/classification_and_detection.

To convert the model, just call the [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html). There are certain specifics for [converting The TensorFlow Object Detection API models](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html). For example, the original pipeline.config is needed. For the symmetrically quantized it is actually the same as for another Habana's model  ("ssd-mobilenet 300x300 quantized finetuned").

Conversion command-line is as follows:
```
$ python3  <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/>mo.py
--input_model <path_to_model/>ssd_mobilenet_v1_quant_ft_no_zero_point_frozen_inference_graph.pb
--input_shape [1,300,300,3]
--reverse_input_channels
--tensorflow_use_custom_operations_config <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/>extensions/front/tf/ssd_v2_support.json
--tensorflow_object_detection_api_pipeline_config <path_to_model/>pipeline.config
```

Model Calibration Example
-----------------------------------
To give an example of the [calibration workflow](https://docs.openvinotoolkit.org/latest/_inference_engine_tools_calibration_tool_README.html), let's consider ResNet-50 (v1.5) example ("resnet50-v1.5	tensorflowfp32 NHWC").

* First, the model is converted from an original framework format using the Model Optimizer tool. Since this is classification (and not detection) model, the command-line is really simple:

```
$ python3  <OPENVINO_INSTALL_DIR/deployment_tools/model_optimizer/> mo.py --input_model ./resnet50_v1.pb --input_shape [1,224,224,3] --reverse_input_channels
```

This outputs the model in Intermediate Representation (IR) format ( *.xml and *.bin file). FP32 is default precision
(use '--data_type FP16' to get fp16 model instead, which is more GPU-friendly).

* Secondly, perform model calibration using the [Calibration tool](https://docs.openvinotoolkit.org/latest_docs_IE_DG_Int8Inference.html#low_precision_8_bit_integer_inference_workflow). The tool is framework-agnostic and accepts the model in the IR format. Model calibration requires a validation dataset (to keep track of the accuracy during calibration). Currently, the calibration tool comes with example support of classification and object detection  models on ImageNet and VOC2007/COCO data sets respectively and associated accuracy metrics. It is relatively straightforward to add another datasets and metrics.

The accuracy validation in turn comes via [Accuracy Checker](https://github.com/opencv/open_model_zoo/tree/develop/tools/accuracy_checker/accuracy_checker/) tool.
For that, the dataset specific annotations [are converted in the common format](https://github.com/opencv/open_model_zoo/tree/develop/tools/accuracy_checker/accuracy_checker/annotation_converters).
Specifically for the ImageNet required for the ResNet, the command-line is as follows:
```
$ convert_annotation imagenet --annotation_file <PATH_TO_IMAGES>/ILSVRC2012_val.txt --labels_file <PATH_TO_IMAGES>/synset_words.txt --has_background True
```
This outputs *.pickle and *.json files used in calibration via
[configuration files in YML](https://docs.openvinotoolkit.org/latest/_inference_engine_tools_calibration_tool_README.html).
Alternatively, you can specify the annotation conversion parameters in the config file and let the calibration tool call the 'convert_annotation' tool.
Similarly, the calibration tool can either accept the converted model as an IR, or the original model directly and perform conversion on the flight.
Both ways are governed by the 'launchers' section of the config file.

Care must be taken on the configuration in general, as there are many items like pre-processing
(mean and scale values, RGB vs BGR), resizing (with and without crop, etc), and so on, that can severely
affect the resulting accuracy. Notice that the pre-processing applied during calibration should match the pre-processing that is later used for inference.
Also, the pre-processing parameters (like mean/scale, or RGB-BGR conversion) can be either part of the Model Optimizer cmd-line
('mo_params' section of the config file) and this will bake the input transformations directly _into the resulting model_,
or 'preprocessing' section of the 'dataset'. The latter doesn't not include the pre-processing into the model,
but applies it to _every loaded dataset image_ instead (before using within the calibration).
The choice depends on your inference pipeline: if the pre-processing is explicitly performed in the code,
the model shouldn't include that, to avoid double pre-processing.

See example YML files for the MLPerf models in the 'example_calibration_files' folder.
The files define the original models, govern conversion to the IR, dataset annotations conversion,
and finally the calibration itself. You only have to patch the paths to your local machines.
*Notice that the pre-processing is not included into a model
(and thus assumed to be applied to an input image before inferencing that), see earlier this section*.

Finally, the calibration command-line is as simple as:
```
$ python3 calibrate.py
-c <PATH_TO_CONFIG>/resnet_v1.5_50.yml
-M <PATH_TO_MODEL_OPTIMIZER>
-C <PATH_TO_OUTPUT_FP_IR>
--output_dir <PATH_TO_OUTPUT_I8_IR>
```
Resulting IR contains original floating point (that all OpenVINO device plugins should support) and (optional) int8 statistics, that some devices might ignore (if int8 is not supported on the device), falling back to the original model.
