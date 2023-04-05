# Model Optimizer Usage {#openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. _deep learning model optimizer:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_model_inputs_outputs
   openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model
   openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model
   openvino_docs_MO_DG_Additional_Optimization_Use_Cases
   openvino_docs_MO_DG_FP16_Compression
   openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ



Model Optimizer is a cross-platform command-line tool that facilitates the transition between training and deployment environments, performs static model analysis, and adjusts deep learning models for optimal execution on end-point target devices.

To use it, you need a pre-trained deep learning model in one of the supported formats: TensorFlow, PyTorch, PaddlePaddle, MXNet, Caffe, Kaldi, or ONNX. Model Optimizer converts the model to the OpenVINO Intermediate Representation format (IR), which you can infer later with :doc:`OpenVINO™ Runtime <openvino_docs_OV_UG_OV_Runtime_User_Guide>`.

Note that Model Optimizer does not infer models.

The figure below illustrates the typical workflow for deploying a trained deep learning model:

.. image:: _static/images/BASIC_FLOW_MO_simplified.svg

where IR is a pair of files describing the model:

* ``.xml`` - Describes the network topology.

* ``.bin`` - Contains the weights and biases binary data.

The OpenVINO IR can be additionally optimized for inference by :doc:`Post-training optimization <pot_introduction>` that applies post-training quantization methods.

How to Run Model Optimizer
##########################

To convert a model to IR, you can run Model Optimizer by using the following command:

.. code-block:: sh

   mo --input_model INPUT_MODEL


If the out-of-the-box conversion (only the ``--input_model`` parameter is specified) is not successful, use the parameters mentioned below to override input shapes and cut the model:

- Model Optimizer provides two parameters to override original input shapes for model conversion: ``--input`` and ``--input_shape``.
For more information about these parameters, refer to the :doc:`Setting Input Shapes <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>` guide.

- To cut off unwanted parts of a model (such as unsupported operations and training sub-graphs),
use the ``--input`` and ``--output`` parameters to define new inputs and outputs of the converted model.
For a more detailed description, refer to the :doc:`Cutting Off Parts of a Model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>` guide.

You can also insert additional input pre-processing sub-graphs into the converted model by using
the ``--mean_values``, ``scales_values``, ``--layout``, and other parameters described
in the :doc:`Embedding Preprocessing Computation <openvino_docs_MO_DG_Additional_Optimization_Use_Cases>` article.

The ``--compress_to_fp16`` compression parameter in Model Optimizer allows generating IR with constants (for example, weights for convolutions and matrix multiplications) compressed to ``FP16`` data type. For more details, refer to the :doc:`Compression of a Model to FP16 <openvino_docs_MO_DG_FP16_Compression>` guide.

To get the full list of conversion parameters available in Model Optimizer, run the following command:

.. code-block:: sh

   mo --help


Examples of CLI Commands
########################

Below is a list of separate examples for different frameworks and Model Optimizer parameters:

1. Launch Model Optimizer for a TensorFlow MobileNet model in the binary protobuf format:

   .. code-block:: sh

      mo --input_model MobileNet.pb


   Launch Model Optimizer for a TensorFlow BERT model in the SavedModel format with three inputs. Specify input shapes explicitly where the batch size and the sequence length equal 2 and 30 respectively:

   .. code-block:: sh

      mo --saved_model_dir BERT --input mask,word_ids,type_ids --input_shape [2,30],[2,30],[2,30]

      For more information, refer to the :doc:`Converting a TensorFlow Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow>` guide.

2. Launch Model Optimizer for an ONNX OCR model and specify new output explicitly:

   .. code-block:: sh

      mo --input_model ocr.onnx --output probabilities


   For more information, refer to the :doc:`Converting an ONNX Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX>` guide.

   .. note::

      PyTorch models must be exported to the ONNX format before conversion into IR. More information can be found in :doc:`Converting a PyTorch Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_PyTorch>`.

3. Launch Model Optimizer for a PaddlePaddle UNet model and apply mean-scale normalization to the input:

   .. code-block:: sh

      mo --input_model unet.pdmodel --mean_values [123,117,104] --scale 255


   For more information, refer to the :doc:`Converting a PaddlePaddle Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle>` guide.

4. Launch Model Optimizer for an Apache MXNet SSD Inception V3 model and specify first-channel layout for the input:

   .. code-block:: sh

      mo --input_model ssd_inception_v3-0000.params --layout NCHW


   For more information, refer to the :doc:`Converting an Apache MXNet Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet>` guide.

5. Launch Model Optimizer for a Caffe AlexNet model with input channels in the RGB format which needs to be reversed:

   .. code-block:: sh

      mo --input_model alexnet.caffemodel --reverse_input_channels


   For more information, refer to the :doc:`Converting a Caffe Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe>` guide.

6. Launch Model Optimizer for a Kaldi LibriSpeech nnet2 model:

   .. code-block:: sh

      mo --input_model librispeech_nnet2.mdl --input_shape [1,140]


   For more information, refer to the :doc:`Converting a Kaldi Model <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi>` guide.

- To get conversion recipes for specific TensorFlow, ONNX, PyTorch, Apache MXNet, and Kaldi models, refer to the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>`.
- For more information about IR, see :doc:`Deep Learning Network Intermediate Representation and Operation Sets in OpenVINO™ <openvino_docs_MO_DG_IR_and_opsets>`.
- For more information about support of neural network models trained with various frameworks, see :doc:`OpenVINO Extensibility Mechanism <openvino_docs_Extensibility_UG_Intro>`

@endsphinxdirective
