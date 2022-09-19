# Converting a PaddlePaddle Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

To convert a PaddlePaddle model, use the `mo` script and specify the path to the input `.pdmodel` model file:

```sh
 mo --input_model <INPUT_MODEL>.pdmodel
```
**For example,** this command converts a yolo v3 PaddlePaddle network to OpenVINO IR network:

```sh
 mo --input_model=yolov3.pdmodel --input=image,im_shape,scale_factor --input_shape=[1,3,608,608],[1,2],[1,2] --reverse_input_channels --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1
```

## Supported PaddlePaddle Layers
For the list of supported standard layers, refer to the [Supported Framework Layers](../Supported_Frameworks_Layers.md) page.

## Officially Supported PaddlePaddle Models
The following PaddlePaddle models have been officially validated and confirmed to work (as of OpenVINO 2022.1):

@sphinxdirective
.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - Model Name
     - Model Type
     - Description
   * - ppocr-det
     - optical character recognition
     - Models are exported from `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/>`_. Refer to `READ.md <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15>`_.
   * - ppocr-rec
     - optical character recognition
     - Models are exported from `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/>`_. Refer to `READ.md <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15>`_.
   * - ResNet-50
     - classification
     - Models are exported from `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_. Refer to `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_.
   * - MobileNet v2
     - classification
     - Models are exported from `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_. Refer to `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_.
   * - MobileNet v3
     - classification
     - Models are exported from `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_. Refer to `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_.
   * - BiSeNet v2
     - semantic segmentation
     - Models are exported from `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_. Refer to `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_.
   * - DeepLab v3 plus
     - semantic segmentation
     - Models are exported from `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_. Refer to `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_.
   * - Fast-SCNN
     - semantic segmentation
     - Models are exported from `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_. Refer to `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_.
   * - OCRNET
     - semantic segmentation
     - Models are exported from `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_. Refer to `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_.
   * - Yolo v3
     - detection
     - Models are exported from `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1>`_. Refer to `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#>`_.
   * - ppyolo
     - detection
     - Models are exported from `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1>`_. Refer to `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#>`_.
   * - MobileNetv3-SSD
     - detection
     - Models are exported from `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2>`_. Refer to `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/deploy/EXPORT_MODEL.md#>`_.
   * - U-Net
     - semantic segmentation
     - Models are exported from `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3>`_. Refer to `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/docs/model_export.md#>`_.
   * - BERT
     - language representation
     -  Models are exported from `PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1>`_. Refer to `README.md <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#readme>`_.   
@endsphinxdirective

## Frequently Asked Questions (FAQ)
When Model Optimizer is unable to run to completion due to typographical errors, incorrectly used options, or other issues, it provides explanatory messages. They describe the potential cause of the problem and give a link to the [Model Optimizer FAQ](../Model_Optimizer_FAQ.md), which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

## See Also
[Model Conversion Tutorials](Convert_Model_Tutorials.md)
