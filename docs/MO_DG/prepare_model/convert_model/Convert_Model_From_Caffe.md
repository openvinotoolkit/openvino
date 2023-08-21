# Converting a Caffe Model {#openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe}

@sphinxdirective

.. meta::
   :description: Learn how to convert a model from the 
                 Caffe format to the OpenVINO Intermediate Representation.


.. warning::

   Note that OpenVINO support for Caffe is currently being deprecated and will be removed entirely in the future.

To convert a Caffe model, run ``mo`` with the path to the input model ``.caffemodel`` file:

.. code-block:: cpp

   mo --input_model <INPUT_MODEL>.caffemodel


The following list provides the Caffe-specific parameters.

.. code-block:: cpp

  Caffe-specific parameters:
    --input_proto INPUT_PROTO, -d INPUT_PROTO
                          Deploy-ready prototxt file that contains a topology
                          structure and layer attributes
    --caffe_parser_path CAFFE_PARSER_PATH
                          Path to python Caffe parser generated from caffe.proto
    -k K                  Path to CustomLayersMapping.xml to register custom
                          layers
    --disable_omitting_optional
                          Disable omitting optional attributes to be used for
                          custom layers. Use this option if you want to transfer
                          all attributes of a custom layer to IR. Default
                          behavior is to transfer the attributes with default
                          values and the attributes defined by the user to IR.
    --enable_flattening_nested_params
                          Enable flattening optional params to be used for
                          custom layers. Use this option if you want to transfer
                          attributes of a custom layer to IR with flattened
                          nested parameters. Default behavior is to transfer the
                          attributes without flattening nested parameters.


CLI Examples Using Caffe-Specific Parameters
++++++++++++++++++++++++++++++++++++++++++++

* Launching model conversion for `bvlc_alexnet.caffemodel <https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet>`__ with a specified `prototxt` file. This is needed when the name of the Caffe model and the `.prototxt` file are different or are placed in different directories. Otherwise, it is enough to provide only the path to the input `model.caffemodel` file.
  
  .. code-block:: cpp
      
    mo --input_model bvlc_alexnet.caffemodel --input_proto bvlc_alexnet.prototxt
   
* Launching model conversion for `bvlc_alexnet.caffemodel <https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet>`__ with a specified `CustomLayersMapping` file. This is the legacy method of quickly enabling model conversion if your model has custom layers. This requires the Caffe system on the computer. Example of ``CustomLayersMapping.xml`` can be found in ``<OPENVINO_INSTALLATION_DIR>/mo/front/caffe/CustomLayersMapping.xml.example``. The optional parameters without default values and not specified by the user in the ``.prototxt`` file are removed from the Intermediate Representation, and nested parameters are flattened:

  .. code-block:: cpp

    mo --input_model bvlc_alexnet.caffemodel -k CustomLayersMapping.xml --disable_omitting_optional --enable_flattening_nested_params
   
   This example shows a multi-input model with input layers: ``data``, ``rois``

  .. code-block:: cpp

    layer {
      name: "data"
      type: "Input"
      top: "data"
      input_param {
        shape { dim: 1 dim: 3 dim: 224 dim: 224 }
      }
    }
    layer {
      name: "rois"
      type: "Input"
      top: "rois"
      input_param {
        shape { dim: 1 dim: 5 dim: 1 dim: 1 }
      }
    }

* Launching model conversion for a multi-input model with two inputs and providing a new shape for each input in the order they are passed to the model conversion API. In particular, for data, set the shape to ``1,3,227,227``. For rois, set the shape to ``1,6,1,1``:

  .. code-block:: cpp

    mo --input_model /path-to/your-model.caffemodel --input data,rois --input_shape (1,3,227,227),[1,6,1,1]

Custom Layer Definition
########################

For the definition of custom layers, refer to the :doc:`Cutting Off Parts of a Model <openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model>` page.

Supported Caffe Layers
#######################

For the list of supported standard layers, refer to the :doc:`Supported Operations <openvino_resources_supported_operations_frontend>` page.

Frequently Asked Questions (FAQ)
################################

Model conversion API provides explanatory messages when it is unable to complete conversions due to typographical errors, incorrectly used options, or other issues. A message describes the potential cause of the problem and gives a link to :doc:`Model Optimizer FAQ <openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ>`  which provides instructions on how to resolve most issues. The FAQ also includes links to relevant sections in :doc:`Convert a Model <openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide>`to help you understand what went wrong.

Summary
#######

In this document, you learned:

* Basic information about how model conversion works with Caffe models.
* Which Caffe models are supported.
* How to convert a trained Caffe model by using model conversion API with both framework-agnostic and Caffe-specific command-line parameters.

Additional Resources
####################

See the :doc:`Model Conversion Tutorials <openvino_docs_MO_DG_prepare_model_convert_model_tutorials>`  page for a set of tutorials providing step-by-step instructions for converting specific Caffe models.


@endsphinxdirective
