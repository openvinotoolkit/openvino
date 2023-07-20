# Converting an MXNet Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the
                 MXNet format to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Apache MXNet is currently being deprecated and will be removed entirely in the future.

To convert an MXNet model, run Model Optimizer with the path to the ``.params`` file of the input model:

.. code-block:: sh

  mo --input_model model-file-0000.params


Using MXNet-Specific Conversion Parameters
##########################################

The following list provides the MXNet-specific parameters.

.. code-block:: sh

  MXNet-specific parameters:
    --input_symbol <SYMBOL_FILE_NAME>
              Symbol file (for example, "model-symbol.json") that contains a topology structure and layer attributes
    --nd_prefix_name <ND_PREFIX_NAME>
              Prefix name for args.nd and argx.nd files
    --pretrained_model_name <PRETRAINED_MODEL_NAME>
              Name of a pre-trained MXNet model without extension and epoch
              number. This model will be merged with args.nd and argx.nd
              files
    --save_params_from_nd
              Enable saving built parameters file from .nd files
    --legacy_mxnet_model
              Enable Apache MXNet loader to make a model compatible with the latest Apache MXNet version.
              Use only if your model was trained with Apache MXNet version lower than 1.0.0
    --enable_ssd_gluoncv
              Enable transformation for converting the gluoncv ssd topologies.
              Use only if your topology is one of ssd gluoncv topologies


.. note::

   By default, model conversion API does not use the Apache MXNet loader. It transforms the topology to another format which is compatible with the latest version of Apache MXNet. However, the Apache MXNet loader is required for models trained with lower version of Apache MXNet. If your model was trained with an Apache MXNet version lower than 1.0.0, specify the ``--legacy_mxnet_model`` key to enable the Apache MXNet loader. Note that the loader does not support models with custom layers. In this case, you must manually recompile Apache MXNet with custom layers and install it in your environment.

Custom Layer Definition
#######################

For the definition of custom layers, refer to the :doc:`Cutting Off Parts of a Model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>` page.

Supported MXNet Layers
#######################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations>` page.

Frequently Asked Questions (FAQ)
################################

Model conversion API provides explanatory messages when it is unable to complete conversions due to typographical errors, incorrectly used options, or other issues. A message describes the potential cause of the problem and gives a link to :doc:`Model Optimizer FAQ <openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ>` which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>` to help you understand what went wrong.

Summary
########

In this document, you learned:

* Basic information about how model conversion API works with MXNet models.
* Which MXNet models are supported.
* How to convert a trained MXNet model by using model conversion API with both framework-agnostic and MXNet-specific command-line parameters.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>` page for a set of tutorials providing step-by-step instructions for converting specific MXNet models. Here are some examples:

* :doc:`Convert MXNet GluonCV Model <openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_Convert_GluonCV_Models>`
* :doc:`Convert MXNet Style Transfer Model <openvino_docs_MO_DG_prepare_model_convert_model_mxnet_specific_Convert_Style_Transfer_From_MXNet>`

@endsphinxdirective
