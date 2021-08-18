# Converting a Paddle* Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

A summary of the steps for optimizing and deploying a model that was trained with Paddle\*:

1. [Configure the Model Optimizer](../Config_Model_Optimizer.md) for Paddle\*.
2. [Convert a Paddle\* Model](#Convert_From_Paddle) to produce an optimized [Intermediate Representation (IR)](../../IR_and_opsets.md) of the model based on the trained network topology, weights, and biases values.
3. Test the model in the Intermediate Representation format using the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in the target environment via provided Inference Engine [sample applications](../../../IE_DG/Samples_Overview.md).
4. [Integrate](../../../IE_DG/Samples_Overview.md) the [Inference Engine](../../../IE_DG/Deep_Learning_Inference_Engine_DevGuide.md) in your application to deploy the model in the target environment.

## Supported Topologies

| Model Name| Model Type| Description|
| ------------- | ------------ | ------------- |
|ppocr-det| optical character recognition| Models are exported from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/). Refer to [READ.md](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15).|
|ppocr-rec| optical character recognition| Models are exported from [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/). Refer to [READ.md](https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15).|
|ResNet-50| classification| Models are exported from [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/). Refer to [getting_started_en.md](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict)|
|MobileNet v2| classification| Models are exported from [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/). Refer to [getting_started_en.md](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict)|
|MobileNet v3| classification| Models are exported from [PaddleClas](https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/). Refer to [getting_started_en.md](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict)|
|BiSeNet v2| semantic segmentation| Models are exported from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1). Refer to [model_export.md](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#)|
|DeepLab v3 plus| semantic segmentation| Models are exported from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1). Refer to [model_export.md](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#)|
|Faster-SCNN| semantic segmentation| Models are exported from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1). Refer to [model_export.md](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#)|
|OCRNET| semantic segmentation| Models are exported from [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1). Refer to [model_export.md](https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#)|
|Yolo v3| detection| Models are exported from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1). Refer to [EXPORT_MODEL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#).|
|ppyolo| detection| Models are exported from [PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1). Refer to [EXPORT_MODEL.md](https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#).|

> **NOTE:** The verified models are exported from the repository of branch release/2.1.

## Convert a Paddle* Model <a name="Convert_From_Paddle"></a>

To convert a Paddle\* model:

1. Go to the `$INTEL_OPENVINO_DIR/deployment_tools/model_optimizer` directory.
2. Use the `mo.py` script to simply convert a model, specifying the framework, the path to the input model `.pdmodel` file and the path to an output directory with write permissions:
```sh
python3 mo.py --input_model <INPUT_MODEL>.pdmodel --output_dir <OUTPUT_MODEL_DIR> --framework=paddle
```

Parameters to convert your model:

* [Framework-agnostic parameters](Converting_Model_General.md): These parameters are used to convert a model trained with any supported framework.
> **NOTE:** `--scale`, `--scale_values`, `--mean_values`, `--mean_file` are unsupported in the current version of mo_paddle.

### Example of Converting a Paddle* Model
Below is the example command to convert yolo v3 Paddle\* network to OpenVINO IR network with Model Optimizer.
```sh
python3 mo.py --model_name yolov3_darknet53_270e_coco --output_dir <OUTPUT_MODEL_DIR> --framework=paddle --data_type=FP32 --reverse_input_channels --input_shape=[2,3,608,608],[1,2],[1,2] --input=image,im_shape,scale_factor --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1 --input_model=yolov3.pdmodel
```

## Supported Paddle\* Layers
Refer to [Supported Framework Layers](../Supported_Frameworks_Layers.md) for the list of supported standard layers.

## Frequently Asked Questions (FAQ)

The Model Optimizer provides explanatory messages if it is unable to run to completion due to issues like typographical errors, incorrectly used options, or other issues. The message describes the potential cause of the problem and gives a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md). The FAQ has instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## Summary

In this document, you learned:

* Basic information about how the Model Optimizer works with Paddle\* models
* Which Paddle\* models are supported
* How to convert a trained Paddle\* model using the Model Optimizer with framework-agnostic command-line options
