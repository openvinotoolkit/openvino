#  OpenVINO™ API 2.0 Transition Guide {#openvino_2_0_transition_guide}

@sphinxdirective

.. meta::
   :description: A detailed information on a new version of OpenVINO™ API 2.0, 
                 as well as the new OpenVINO IR model format: IR v11.


.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_2_0_deployment
   openvino_2_0_inference_pipeline
   openvino_2_0_configure_devices
   openvino_2_0_preprocessing
   openvino_2_0_model_creation


This guide introduces the new OpenVINO™ API: API 2.0, as well as the new OpenVINO IR model format: IR v11. Here, you will find comparisons of their "old" and "new" versions.

Introduction of API 2.0
#######################

Versions of OpenVINO prior to 2022.1 required changes in the application logic when migrating an app from other frameworks, such as TensorFlow, ONNX Runtime, PyTorch, PaddlePaddle, etc. The changes were required because:

- Model conversion API changed input precisions for some inputs. For example, neural language processing models with ``I64`` inputs were changed to include ``I32`` ones.
- Model conversion API changed layouts for TensorFlow models (see the :doc:`Layouts in OpenVINO <openvino_docs_OV_UG_Layout_Overview>`). It lead to unusual requirement of using the input data with a different layout than that of the framework:

.. image:: _static/images/tf_openvino.svg
   :alt: tf_openvino

- Inference Engine API (`InferenceEngine::CNNNetwork <classInferenceEngine_1_1CNNNetwork.html#doxid-class-inference-engine-1-1-c-n-n-network>`__) applied some conversion rules for input and output precisions due to limitations in device plugins.
- Users needed to specify input shapes during model conversions in model conversion API, and work with static shapes in the application.

OpenVINO™ 2022.1 has introduced API 2.0 (also called OpenVINO API v2) to align the logic of working with models as it is done in their origin frameworks - no layout and precision changes, operating with tensor names and indices to address inputs and outputs. OpenVINO Runtime has combined Inference Engine API used for inference and nGraph API targeted to work with models and operations. API 2.0 has a common structure, naming convention styles, namespaces, and removes duplicated structures. For more details, see the :doc:`Changes to Inference Pipeline in OpenVINO API v2 <openvino_2_0_inference_pipeline>`.

.. note::

   Your existing applications will continue to work with OpenVINO Runtime 2022.1, as normal. Although, migration to API 2.0 is strongly recommended. This will allow you to use additional features, such as :doc:`Preprocessing <openvino_docs_OV_UG_Preprocessing_Overview>` and :doc:`Dynamic shapes support <openvino_docs_OV_UG_DynamicShapes>`.


The New OpenVINO IR v11
#######################

To support these features, OpenVINO has introduced OpenVINO IR v11, which is now the default version for model conversion API. The model represented in OpenVINO IR v11 fully matches the original model in the original framework format in terms of inputs and outputs. It is also not required to specify input shapes during conversion, which results in OpenVINO IR v11 containing ``-1`` to denote undefined dimensions. For more details on how to fully utilize this feature, see :doc:`Working with dynamic shapes <openvino_docs_OV_UG_DynamicShapes>`. For information on how to reshape to static shapes in application, see :doc:`Changing input shapes <openvino_docs_OV_UG_ShapeInference>`.

OpenVINO IR v11 is fully compatible with applications written with the Inference Engine API used by older versions of OpenVINO. This backward compatibility is allowed thanks to additional runtime information included in OpenVINO IR v11. This means that when OpenVINO IR v11 is read by an application based on Inference Engine, it is internally converted to OpenVINO IR v10.

OpenVINO IR v11 is supported by all OpenVINO Development tools including Post-Training Optimization Tool, Benchmark app, etc.

Backward Compatibility for OpenVINO IR v10
##########################################

API 2.0 also supports backward compatibility for models of OpenVINO IR v10. If you have OpenVINO IR v10 files, they can also be fed to OpenVINO Runtime. For more details, see the :doc:`migration steps <openvino_2_0_inference_pipeline>`.

Some of the OpenVINO Development Tools also support both OpenVINO IR v10 and v11 as an input:

- Accuracy checker uses API 2.0 for model accuracy measurement by default. It also supports switching to the old API by using the ``--use_new_api False`` command-line parameter. Both launchers accept OpenVINO IR v10 and v11, but in some cases configuration files should be updated. For more details, see the `Accuracy Checker documentation <https://github.com/openvinotoolkit/open_model_zoo/blob/master/tools/accuracy_checker/openvino/tools/accuracy_checker/launcher/openvino_launcher_readme.md>`__.
- :doc:`Compile tool <openvino_ecosystem>` compiles the model to be used in API 2.0 by default. To use the resulting compiled blob under the Inference Engine API, the additional ``ov_api_1_0`` option should be passed.

However, Post-Training Optimization Tool of OpenVINO 2022.1 does not support OpenVINO IR v10. They require the latest version of model conversion API to generate OpenVINO IR v11 files.

.. note::

   To quantize your OpenVINO IR v10 models to run with OpenVINO 2022.1, download and use Post-Training Optimization Tool of OpenVINO 2021.4.


.. _differences_api20_ie:

Differences in API 2.0 and Inference Engine API Behaviors
#########################################################

Inference Engine and nGraph APIs do not become deprecated with the introduction of the new API, and they can still be used in applications. However, it is highly recommended to migrate to API 2.0, as it offers more features (further extended in future releases), such as:

- :doc:`Working with dynamic shapes <openvino_docs_OV_UG_DynamicShapes>`, which increases performance when working with compatible models such as NLP (Neural Language Processing) and super-resolution models.
- :doc:`Preprocessing of the model <openvino_docs_OV_UG_Preprocessing_Overview>`, which adds preprocessing operations to inference models and fully occupies the accelerator, freeing CPU resources.

To understand the differences between Inference Engine API and API 2.0, see the definitions of two types of behaviors first:

- **Old behavior** of OpenVINO assumes that:

  - Model Conversion API can change input element types and order of dimensions (layouts) for the model from the original framework.
  - Inference Engine can override input and output element types.
  - Inference Engine API uses operation names to address inputs and outputs (e.g. `InferenceEngine::InferRequest::GetBlob <classInferenceEngine_1_1InferRequest.html#doxid-class-inference-engine-1-1-infer-request-1a9601a4cda3f309181af34feedf1b914c>`__).
  - Inference Engine API does not support compiling of models with dynamic input shapes.

- **New behavior** implemented in 2022.1 assumes full model alignment with the framework:

  - Model Conversion API preserves input element types and order of dimensions (layouts), and stores tensor names from the original models.
  - OpenVINO Runtime 2022.1 reads models in any format (OpenVINO IR v10, OpenVINO IR v11, TensorFlow (check :doc:`TensorFlow Frontend Capabilities and Limitations <openvino_docs_MO_DG_TensorFlow_Frontend>`), ONNX, PaddlePaddle, etc.).
  - API 2.0 uses tensor names for addressing, which is the standard approach among the compatible model frameworks.
  - API 2.0 can also address input and output tensors by the index. Some model formats like ONNX are sensitive to the input and output order, which is preserved by OpenVINO 2022.1.

The table below demonstrates which behavior, **old** or **new**, is used for models based on the two APIs.

+--------------------------------+-----------------+-----------------+-----------------+------------------------+
| API                            | OpenVINO IR v10 | OpenVINO IR v11 | ONNX Files      | Models Created in Code |
+================================+=================+=================+=================+========================+
| Inference Engine / nGraph APIs | Old             | Old             | Old             | Old                    |
+--------------------------------+-----------------+-----------------+-----------------+------------------------+
| API 2.0                        | Old             | New             | New             | New                    |
+--------------------------------+-----------------+-----------------+-----------------+------------------------+

More Information
####################

See the following pages to understand how to migrate Inference Engine-based applications to API 2.0:

- :doc:`Installation & Deployment <openvino_2_0_deployment>`
- :doc:`OpenVINO™ Common Inference pipeline <openvino_2_0_inference_pipeline>`
- :doc:`Preprocess your model <openvino_2_0_preprocessing>`
- :doc:`Configure device <openvino_2_0_configure_devices>`
- :doc:`OpenVINO™ Model Creation <openvino_2_0_model_creation>`

@endsphinxdirective
