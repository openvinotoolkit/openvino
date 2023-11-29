# [LEGACY] Model Optimizer Extensions {#openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions}


.. meta::
   :description: Learn about deprecated extensions, which enable injecting logic 
                 to the model conversion pipeline without changing the Model 
                 Optimizer core code.

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Operation
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Extractor
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Transformation_Extensions

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated TensorFlow conversion method. The guide on the new and recommended method, using a new frontend, can be found in the  :doc:`Frontend Extensions <openvino_docs_Extensibility_UG_Frontend_Extensions>` article. 

Model Optimizer extensions enable you to inject some logic to the model conversion pipeline without changing the Model
Optimizer core code. There are three types of the Model Optimizer extensions:

1. :doc:`Model Optimizer operation <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Operation>`.
2. A :doc:`framework operation extractor <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Extractor>`.
3. A :doc:`model transformation <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Model_Optimizer_Extensions_Model_Optimizer_Transformation_Extensions>`, which can be executed during front, middle or back phase of the model conversion.

An extension is just a plain text file with a Python code. The file should contain a class (or classes) inherited from
one of extension base classes. Extension files should be saved to a directory with the following structure:

.. code-block:: sh
   
   ./<MY_EXT>/
              ops/                  - custom operations
              front/                - framework independent front transformations
                    <FRAMEWORK_1>/  - front transformations for <FRAMEWORK_1> models only and extractors for <FRAMEWORK_1> operations
                    <FRAMEWORK_2>/  - front transformations for <FRAMEWORK_2> models only and extractors for <FRAMEWORK_2> operations
                    ...
              middle/               - middle transformations
              back/                 - back transformations

Model Optimizer uses the same layout internally to keep built-in extensions. The only exception is that the 
``mo/ops/`` directory is also used as a source of the Model Optimizer operations due to historical reasons.

.. note:: 
   The name of a root directory with extensions should not be equal to "extensions" because it will result in a name conflict with the built-in Model Optimizer extensions.

.. note:: 
   Model Optimizer itself is built by using these extensions, so there is a huge number of examples of their usage in the Model Optimizer code.

====================
Additional Resources
====================

* :doc:`Model Optimizer Extensibility <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer>`
* :doc:`Graph Traversal and Modification Using Ports and Connections <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Customize_Model_Optimizer_Model_Optimizer_Ports_Connections>`
* :doc:`Extending Model Optimizer with Caffe Python Layers <openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Extending_Model_Optimizer_With_Caffe_Python_Layers>`

