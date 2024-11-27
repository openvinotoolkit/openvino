Legacy Conversion API
=====================


.. toctree::
   :maxdepth: 1
   :hidden:

   Setting Input Shapes <legacy-conversion-api/[legacy]-setting-input-shapes>
   Troubleshooting Reshape Errors <legacy-conversion-api/[legacy]-troubleshooting-reshape-errors>
   Cutting Off Parts of a Model <legacy-conversion-api/[legacy]-cutting-parts-of-a-model>
   Embedding Preprocessing Computation <legacy-conversion-api/[legacy]-embedding-preprocessing-computation>
   Compressing a Model to FP16 <legacy-conversion-api/[legacy]-compressing-model-to-fp16>
   Convert Models Represented as Python Objects <legacy-conversion-api/[legacy]-convert-models-as-python-objects>
   Model Optimizer Frequently Asked Questions <legacy-conversion-api/[legacy]-model-optimizer-faq>
   Supported Model Formats <legacy-conversion-api/[legacy]-supported-model-formats>

.. meta::
   :description: Model conversion (MO) furthers the transition between training and
                 deployment environments, it adjusts deep learning models for
                 optimal execution on target devices.

.. note::
   This part of the documentation describes a legacy approach to model conversion. Starting with OpenVINO 2023.1, a simpler alternative API for model conversion is available: ``openvino.convert_model`` and OpenVINO Model Converter ``ovc`` CLI tool. Refer to :doc:`Model preparation <../../../openvino-workflow/model-preparation>` for more details. If you are still using `openvino.tools.mo.convert_model` or `mo` CLI tool, you can still refer to this documentation. However, consider checking the :doc:`transition guide <../transition-legacy-conversion-api>` to learn how to migrate from the legacy conversion API to the new one. Depending on the model topology, the new API can be a better option for you.

To convert a model to OpenVINO model format (``ov.Model``), you can use the following command:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model(INPUT_MODEL)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --input_model INPUT_MODEL


If the out-of-the-box conversion (only the ``input_model`` parameter is specified) is not successful, use the parameters mentioned below to override input shapes and cut the model:

- ``input`` and ``input_shape`` - the model conversion API parameters used to override original input shapes for model conversion,

  For more information about the parameters, refer to the :doc:`Setting Input Shapes <legacy-conversion-api/[legacy]-setting-input-shapes>` guide.

- ``input`` and ``output`` - the model conversion API parameters used to define new inputs and outputs of the converted model to cut off unwanted parts (such as unsupported operations and training sub-graphs),

  For a more detailed description, refer to the :doc:`Cutting Off Parts of a Model <legacy-conversion-api/[legacy]-cutting-parts-of-a-model>` guide.

- ``mean_values``, ``scales_values``, ``layout`` - the parameters used to insert additional input pre-processing sub-graphs into the converted model,

  For more details, see the :doc:`Embedding Preprocessing Computation <legacy-conversion-api/[legacy]-embedding-preprocessing-computation>` article.

- ``compress_to_fp16`` - a compression parameter in ``mo`` command-line tool, which allows generating IR with constants (for example, weights for convolutions and matrix multiplications) compressed to ``FP16`` data type.

  For more details, refer to the :doc:`Compression of a Model to FP16 <legacy-conversion-api/[legacy]-compressing-model-to-fp16>` guide.

To get the full list of conversion parameters, run the following command:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          from openvino.tools.mo import convert_model
          ov_model = convert_model(help=True)

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          mo --help


Examples of model conversion parameters
#######################################

Below is a list of separate examples for different frameworks and model conversion parameters:

1. Launch model conversion for a TensorFlow MobileNet model in the binary protobuf format:

   .. tab-set::

       .. tab-item:: Python
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools.mo import convert_model
             ov_model = convert_model("MobileNet.pb")

       .. tab-item:: CLI
          :sync: cli

          .. code-block:: sh

             mo --input_model MobileNet.pb


   Launch model conversion for a TensorFlow BERT model in the SavedModel format with three inputs. Specify input shapes explicitly where the batch size and the sequence length equal 2 and 30 respectively:

   .. tab-set::

       .. tab-item:: Python
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools.mo import convert_model
             ov_model = convert_model("BERT", input_shape=[[2,30],[2,30],[2,30]])

       .. tab-item:: CLI
          :sync: cli

          .. code-block:: sh

             mo --saved_model_dir BERT --input_shape [2,30],[2,30],[2,30]


   For more information, refer to the :doc:`Converting a TensorFlow Model <legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-convert-tensorflow>` guide.

2. Launch model conversion for an ONNX OCR model and specify new output explicitly:

   .. tab-set::

       .. tab-item:: Python
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools.mo import convert_model
             ov_model = convert_model("ocr.onnx", output="probabilities")

       .. tab-item:: CLI
          :sync: cli

          .. code-block:: sh

             mo --input_model ocr.onnx --output probabilities


   For more information, refer to the :doc:`Converting an ONNX Model <legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-convert-onnx>` guide.

   .. note::

      PyTorch models must be exported to the ONNX format before conversion into IR. More information can be found in :doc:`Converting a PyTorch Model <legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-convert-pytorch>`.

3. Launch model conversion for a PaddlePaddle UNet model and apply mean-scale normalization to the input:

   .. tab-set::

       .. tab-item:: Python
          :sync: py

          .. code-block:: py
             :force:

             from openvino.tools.mo import convert_model
             ov_model = convert_model("unet.pdmodel", mean_values=[123,117,104], scale=255)

       .. tab-item:: CLI
          :sync: cli

          .. code-block:: sh

             mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255


   For more information, refer to the :doc:`Converting a PaddlePaddle Model <legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-convert-paddle>` guide.

- To get conversion recipes for specific TensorFlow, ONNX, and PyTorch models, refer to the :doc:`Model Conversion Tutorials <legacy-conversion-api/[legacy]-supported-model-formats/[legacy]-conversion-tutorials>`.
- For more information about IR, see :doc:`Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ <../../openvino-ir-format/operation-sets>`.
- For more information about support of neural network models trained with various frameworks, see :doc:`OpenVINO Extensibility Mechanism <../../openvino-extensibility>`

