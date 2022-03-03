# Converting a PaddlePaddle Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

@sphinxdirective

.. _convert model paddle:

.. toctree::
   :maxdepth: 1
   :hidden:

   Supported Topologies <openvino_docs_MO_DG_prepare_model_convert_model_paddle_specific_supported_topologies>

@endsphinxdirective

A summary of the steps for optimizing and deploying a model trained with PaddlePaddle:

1. [Configure Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md) for PaddlePaddle.
2. [Convert a PaddlePaddle Model](#Convert_From_Paddle) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases.
3. Test the model in the Intermediate Representation format using the [OpenVINO™ Runtime](../../../OV_Runtime_UG/openvino_intro.md) in the target environment via provided [OpenVINO Samples](../../../OV_Runtime_UG/Samples_Overview.md).
4. [Integrate](../../../OV_Runtime_UG/Samples_Overview.md) the [OpenVINO™ Runtime](../../../OV_Runtime_UG/openvino_intro.md) in your application to deploy the model in the target environment.

## Convert a PaddlePaddle Model <a name="Convert_From_Paddle"></a>

To convert a PaddlePaddle model:

1. Activate environment with installed OpenVINO™ if needed
2. Use the `mo` script to simply convert a model, specifying the framework, the path to the input model `.pdmodel` file and the path to an output directory with write permissions:
```sh
 mo --input_model <INPUT_MODEL>.pdmodel --output_dir <OUTPUT_MODEL_DIR> --framework=paddle
```

Parameters to convert your model:

* [Framework-agnostic parameters](Converting_Model.md): These parameters are used to convert a model trained with any supported framework.
> **NOTE:** `--scale`, `--scale_values`, `--mean_values` are not supported in the current version of mo_paddle.

### Example of Converting a PaddlePaddle Model
Below is the example command to convert yolo v3 PaddlePaddle network to OpenVINO IR network with Model Optimizer.
```sh
 mo --model_name yolov3_darknet53_270e_coco --output_dir <OUTPUT_MODEL_DIR> --framework=paddle --data_type=FP32 --reverse_input_channels --input_shape=[1,3,608,608],[1,2],[1,2] --input=image,im_shape,scale_factor --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1 --input_model=yolov3.pdmodel
```

## Supported PaddlePaddle Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

When Model Optimizer is unable to run to completion due to issues like typographical errors, incorrectly used options, etc., it provides explanatory messages. They describe the potential cause of the problem and give a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md), which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.
