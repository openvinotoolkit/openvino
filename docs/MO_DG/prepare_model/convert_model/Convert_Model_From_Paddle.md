# Converting a PaddlePaddle Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle}

@sphinxdirective

This page provides general instructions on how to convert a model from a PaddlePaddle format to the OpenVINO IR format using Model Optimizer. The instructions are different depending on PaddlePaddle model format.

Converting PaddlePaddle Model Inference Format
##############################################

PaddlePaddle inference model includes ``.pdmodel`` (storing model structure) and ``.pdiparams`` (storing model weight). For how to export PaddlePaddle inference model, please refer to the `Exporting PaddlePaddle Inference Model <https://www.paddlepaddle.org.cn/documentation/docs/zh/develop/guides/beginner/model_save_load_cn.html>`__ Chinese guide.

To convert a PaddlePaddle model, use the ``mo`` script and specify the path to the input ``.pdmodel`` model file:

.. code-block:: sh

  mo --input_model <INPUT_MODEL>.pdmodel

**For example**, this command converts a yolo v3 PaddlePaddle network to OpenVINO IR network:

.. code-block:: sh

  mo --input_model=yolov3.pdmodel --input=image,im_shape,scale_factor --input_shape=[1,3,608,608],[1,2],[1,2] --reverse_input_channels --output=save_infer_model/scale_0.tmp_1,save_infer_model/scale_1.tmp_1

Converting PaddlePaddle Model From Memory Using Python API
##########################################################

MO Python API supports passing PaddlePaddle models directly from memory.

Following PaddlePaddle model format are supported:

* ``paddle.hapi.model.Model``

  Example of converting ``paddle.hapi.model.Model`` format model:

  .. code-block:: python

    import paddle
    from paddle.metirc import Accuracy
    from openvino.tools.mo import convert_model
    
    # create a paddle.hapi.model.Model format model
    model = paddle.Model(paddle.vision.models.resnet50(pretrained=False, num_classes=10))
    # prepare dataset
    train_dataset = paddle.vision.datasets.Cifar10(mode='train')
    val_dataset = paddle.vision.datasets.Cifar10(mode='test')
    # define optimizer
    optimizer = paddle.optimizer.Momentum(learning_rate=0.01,
                                          momentum=0.9,
                                          weight_decay=paddle.regularizer.L2Decay(1e-4),
                                          parameters=model.parameters())
    model.prepare(optimizer, paddle.nn.CrossEntropy(), paddle.metric.Accuracy(topk=(1, 5)))
    # start training
    model.fit(train_dataset,
              val_dataset,
              epochs=50,
              batch_size=64,
              save_dir="./output",
              num_workers=8)
    # convert to OpenVINO IR format
    ov_model = convert_model(model)

    # optional: serialize OpenVINO IR to *.xml & *.bin
    from openvino.runtime import serialize
    serialize(ov_model, "ov_model.xml", "ov_model.bin")

* ``paddle.fluid.dygraph.layers.Layer``

  ``example_input`` is required while ``example_output`` is optional, which accept the following formats:

  ``list`` with tensor(``paddle.Tensor``) or InputSpec(``paddle.static.input.InputSpec``)

  Example of converting ``paddle.fluid.dygraph.layers.Layer`` format model:

  .. code-block:: python
  
  	import paddle
  	from openvino.tools.mo import convert_model
  
    # create a paddle.fluid.dygraph.layers.Layer format model
  	model = paddle.vision.models.resnet50()
  	x = paddle.rand([1,3,224,224])

    # convert to OpenVINO IR format
  	ov_model = convert_model(model, example_input=[x])

    # optional: serialize OpenVINO IR to *.xml & *.bin
    from openvino.runtime import serialize
    serialize(ov_model, "ov_model.xml", "ov_model.bin")
  
* ``paddle.fluid.executor.Executor``

  ``example_input`` and ``example_output`` are required, which accept the following formats:

  ``list`` or ``tuple`` with variable(``paddle.static.data``)

  Example of converting ``paddle.fluid.executor.Executor`` format model:

  .. code-block:: python

    import paddle
  	from openvino.tools.mo import convert_model

    paddle.enable_static()

    # create a paddle.fluid.executor.Executor format model
    x = paddle.static.data(name="x", shape=shape)
    y = paddle.static.data(name="y", shape=shape)
    relu = paddle.nn.ReLU()
    sigmoid = paddle.nn.Sigmoid()
    y = sigmoid(relu(x))
    
    exe = paddle.static.Executor(paddle.CPUPlace())
    exe.run(paddle.static.default_startup_program())

    # convert to OpenVINO IR format
    ov_model = convert_model(exe, example_input=[x], example_output=[y])

    # optional: serialize OpenVINO IR to *.xml & *.bin
    from openvino.runtime import serialize
    serialize(ov_model, "ov_model.xml", "ov_model.bin")
  
Supported PaddlePaddle Layers
#############################

For the list of supported standard layers, refer to the :doc:`Supported Framework Layers <openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers>` page.

Officially Supported PaddlePaddle Models
########################################

The following PaddlePaddle models have been officially validated and confirmed to work (as of OpenVINO 2022.1):

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

Frequently Asked Questions (FAQ)
################################

When Model Optimizer is unable to run to completion due to typographical errors, incorrectly used options, or other issues, it provides explanatory messages. They describe the potential cause of the problem and give a link to the :doc:`Model Optimizer FAQ <openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ>`, which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in the Model Optimizer Developer Guide to help you understand what went wrong.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific PaddlePaddle models.

@endsphinxdirective

