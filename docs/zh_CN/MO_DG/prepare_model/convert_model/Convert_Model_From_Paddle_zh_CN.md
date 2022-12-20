# 转换 PadlePaddle 模型 {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle_zh_CN}

如需转换 PaddlePaddle 模型，请使用 `mo` 脚本并指定输入 `.pdmodel` 模型文件的路径：

```sh
 mo --input_model <INPUT_MODEL>.pdmodel
```
**例如，**此命令会将 yolo v3 PaddlePaddle 网络转换为 OpenVINO™ IR 网络：

```sh
 mo --input_model=yolov3.pdmodel --input=image,im_shape,scale_factor --input_shape=[1,3,608,608],[1,2],[1,2] --reverse_input_channels --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1
```

## 支持的 PaddlePaddle 层
有关支持的标准层列表，请参阅[支持的框架层](@ref openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers)页面。

## 正式支持的 PaddlePaddle 模型
以下 PaddlePaddle 模型已通过正式验证并确认可运行（自 OpenVINO™ 2022.1 起）：

@sphinxdirective
.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * - 模型名称
     - 模型类型
     - 描述
   * - ppocr-det
     - 光学字符识别
     - 模型从 `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/>`_ 中导出。请参阅 `READ.md <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15>`_。
   * - ppocr-rec
     - 光学字符识别
     - 模型从 `PaddleOCR <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/>`_ 中导出。请参阅 `READ.md <https://github.com/PaddlePaddle/PaddleOCR/tree/release/2.1/#pp-ocr-20-series-model-listupdate-on-dec-15>`_。
   * - ResNet-50
     - classification
     - 模型从 `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_ 中导出。请参阅 `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_。
   * - MobileNet v2
     - classification
     - 模型从 `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_ 中导出。请参阅 `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_。
   * - MobileNet v3
     - classification
     - 模型从 `PaddleClas <https://github.com/PaddlePaddle/PaddleClas/tree/release/2.1/>`_ 中导出。请参阅 `getting_started_en.md <https://github.com/PaddlePaddle/PaddleClas/blob/release/2.1/docs/en/tutorials/getting_started_en.md#4-use-the-inference-model-to-predict>`_。
   * - BiSeNet v2
     - 语义分割
     - 模型从 `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_ 中导出。请参阅 `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_。
   * - DeepLab v3 plus
     - 语义分割
     - 模型从 `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_ 中导出。请参阅 `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_。
   * - Fast-SCNN
     - 语义分割
     - 模型从 `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_ 中导出。请参阅 `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_。
   * - OCRNET
     - 语义分割
     - 模型从 `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.1>`_ 中导出。请参阅 `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.1/docs/model_export.md#>`_。
   * - Yolo v3
     - 检测
     - 模型从 `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1>`_ 中导出。请参阅 `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#>`_。
   * - ppyolo
     - 检测
     - 模型从 `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.1>`_ 中导出。请参阅 `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.1/deploy/EXPORT_MODEL.md#>`_。
   * - MobileNetv3-SSD
     - 检测
     - 模型从 `PaddleDetection <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.2>`_ 中导出。请参阅 `EXPORT_MODEL.md <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.2/deploy/EXPORT_MODEL.md#>`_。
   * - U-Net
     - 语义分割
     - 模型从 `PaddleSeg <https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.3>`_ 中导出。请参阅 `model_export.md <https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.3/docs/model_export.md#>`_。
   * - BERT
     - 语言表示
     - 模型从 `PaddleNLP <https://github.com/PaddlePaddle/PaddleNLP/tree/v2.1.1>`_ 中导出。请参阅 `README.md <https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/language_model/bert#readme>`_。   
@endsphinxdirective

## 常见问题解答 (FAQ)
模型优化器由于拼写错误、选项使用不当或其他问题而无法完成运行时，会提供解释性消息。这些消息描述了问题的潜在原因，并提供了[模型优化器常见问题解答](@ref openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ)的链接，对如何解决大多数问题进行了说明。常见问题解答还包括了模型优化器开发人员指南相关部分的链接，来帮助您了解哪里发生了错误。

## 其他资源
请参阅[模型转换教程](@ref openvino_docs_MO_DG_prepare_model_convert_model_tutorials)页面获取一系列教程，了解转换特定 PaddlePaddle 模型相关的分步指导。
