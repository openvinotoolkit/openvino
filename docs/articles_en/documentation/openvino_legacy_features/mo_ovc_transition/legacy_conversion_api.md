# Legacy Conversion API {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_FP16_Compression
   openvino_docs_MO_DG_Python_API
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ
   Supported_Model_Formats_MO_DG

.. meta::
   :description: Model conversion (MO) furthers the transition between training and
                 deployment environments, it adjusts deep learning models for
                 optimal execution on target devices.

.. note::
   This part of the documentation describes a legacy approach to model conversion. Starting with OpenVINO 2023.1, a simpler alternative API for model conversion is available: ``openvino.convert_model`` and OpenVINO Model Converter ``ovc`` CLI tool. Refer to :doc:`Model preparation <openvino_docs_model_processing_introduction>` for more details. If you are still using `openvino.tools.mo.convert_model` or `mo` CLI tool, you can still refer to this documentation. However, consider checking the :doc:`transition guide <openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition>` to learn how to migrate from the legacy conversion API to the new one. Depending on the model topology, the new API can be a better option for you.

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

  For more information about the parameters, refer to the :doc:`Setting Input Shapes <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>` guide.

- ``input`` and ``output`` - the model conversion API parameters used to define new inputs and outputs of the converted model to cut off unwanted parts (such as unsupported operations and training sub-graphs),

  For a more detailed description, refer to the :doc:`Cutting Off Parts of a Model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>` guide.

- ``mean_values``, ``scales_values``, ``layout`` - the parameters used to insert additional input pre-processing sub-graphs into the converted model,

  For more details, see the :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>` article.

- ``compress_to_fp16`` - a compression parameter in ``mo`` command-line tool, which allows generating IR with constants (for example, weights for convolutions and matrix multiplications) compressed to ``FP16`` data type.

  For more details, refer to the :doc:`Compression of a Model to FP16 <openvino_docs_MO_DG_FP16_Compression>` guide.

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


   For more information, refer to the :doc:`Converting a TensorFlow Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow>` guide.

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


   For more information, refer to the :doc:`Converting an ONNX Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX>` guide.

   .. note::

      PyTorch models must be exported to the ONNX format before conversion into IR. More information can be found in :doc:`Converting a PyTorch Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch>`.

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


   For more information, refer to the :doc:`Converting a PaddlePaddle Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle>` guide.

- To get conversion recipes for specific TensorFlow, ONNX, and PyTorch models, refer to the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>`.
- For more information about IR, see :doc:`Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ <openvino_docs_MO_DG_IR_and_opsets>`.
- For more information about support of neural network models trained with various frameworks, see :doc:`OpenVINO Extensibility Mechanism <openvino_docs_Extensibility_UG_Intro>`

@endsphinxdirective
