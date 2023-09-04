# Convert a Model {#openvino_docs_OV_Converter_UG_Deep_Learning_Model_Optimizer_DevGuide}

@sphinxdirective

.. _deep learning model optimizer:

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_OV_Converter_UG_prepare_model_convert_model_Converting_Model
   openvino_docs_OV_Converter_UG_FP16_Compression

.. meta::
   :description: Model Conversion API provides a way to import pre-trained models
                 from various deep learning frameworks to OpenVINO runtime for accelerated inference.

To convert a model to OpenVINO model, you can use the following commands as described in :doc:`Model preparation <openvino_docs_model_processing_introduction>` chapter:

.. tab-set::

    .. tab-item:: Python
       :sync: py

       .. code-block:: py
          :force:

          import openvino as ov

          ov_model = ov.convert_model('path_to_your_model')
          # or, when model is a Python model object
          ov_model = ov.convert_model(model)

          # Optionally adjust model by embedding pre-post processing here...

          ov.save_model(ov_model, 'model.xml')

    .. tab-item:: CLI
       :sync: cli

       .. code-block:: sh

          ovc path_to_your_model

Providing just a path to the model or model object as ``openvino.convert_model`` argument is frequently enough to make a successful conversion. However, depending on the model topology and original deep learning framework, additional parameters may be required:

- ``example_input`` parameter available in Python ``openvino.convert_model`` only is intended to trace the model to obtain its graph representation. This parameter is crucial for the successful conversion of a model from PyTorch and is sometimes required for TensorFlow. Refer to the :doc:`PyTorch Model Conversion <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_PyTorch>` or :doc:`TensorFlow Model Conversion <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Convert_Model_From_TensoFlow>`.

- ``input`` parameter to set or override shapes for model inputs. It configures dynamic and static dimensions in model inputs depending on your needs for the most efficient inference. For more information about using this parameter, refer to the :doc:`Setting Input Shapes <openvino_docs_OV_Converter_UG_prepare_model_convert_model_Converting_Model>` guide.

- ``output`` parameter to select one or multiple outputs from the original model. Useful if the model has outputs that are not required for inference in a deployment scenario. Specifying only necessary outputs may lead to a more compact model that infers faster.

- ``compress_to_fp16`` parameter that is provided by ``ovc`` CLI tool and ``openvino.save_model`` Python function, gives controls over the compression of model weights to FP16 format when saving OpenVINO model to IR. This option is enabled by default which means all produced IRs are saved using FP16 data type for weights which saves up to 2x storage space for the model file and in most cases doesn't sacrifice model accuracy. In case it does affect accuracy, the compression can be disabled by setting this flag to ``False``. Refer to the :doc:`Compression of a Model to FP16 <openvino_docs_OV_Converter_UG_FP16_Compression>` guide.

- ``extension`` parameter which makes possible conversion of the models consisting of operations that are not supported by OpenVINO out-of-the-box. It requires implementing of an OpenVINO extension first, please refer to :doc:`Setting Input Shapes <openvino_docs_Extensibility_UG_Frontend_Extensions>` guide.

Refer to ``openvino.convert_model`` reference documentation **TODO: INSERT LINK HERE** for more information about all available parameters. Run ``ovc -h`` to get all supported parameter for ``ovc`` CLI tool.

@endsphinxdirective
