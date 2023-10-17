.. {#openvino_docs_ie_plugin_dg_overview}

Overview of OpenVINO Plugin Library
===================================


.. meta::
   :description: Develop and implement independent inference solutions for 
                 different devices with the components of plugin architecture 
                 of OpenVINO.


.. toctree::
   :maxdepth: 1
   :caption: Converting and Preparing Models
   :hidden:

   Implement Plugin Functionality <openvino_docs_ov_plugin_dg_plugin>
   Implement Compiled Model Functionality <openvino_docs_ov_plugin_dg_compiled_model>
   Implement Synchronous Inference Request <openvino_docs_ov_plugin_dg_infer_request>
   Implement Asynchronous Inference Request <openvino_docs_ov_plugin_dg_async_infer_request>
   Provide Plugin Specific Properties <openvino_docs_ov_plugin_dg_properties>
   Implement Remote Context <openvino_docs_ov_plugin_dg_remote_context>
   Implement Remote Tensor <openvino_docs_ov_plugin_dg_remote_tensor>
   openvino_docs_ov_plugin_dg_plugin_build
   openvino_docs_ov_plugin_dg_plugin_testing
   openvino_docs_ie_plugin_detailed_guides
   openvino_docs_ie_plugin_api_references


The plugin architecture of OpenVINO allows to develop and plug independent inference 
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library 
exporting the single ``CreatePluginEngine`` function that allows to create a new plugin instance.

OpenVINO Plugin Library
#######################

OpenVINO plugin dynamic library consists of several main components:

1.  :doc:`Plugin class <openvino_docs_ov_plugin_dg_plugin>`:

    * Provides information about devices of a specific type.
    * Can create an  :doc:`compiled model <openvino_docs_ov_plugin_dg_compiled_model>` instance which represents a Neural Network backend specific graph structure for a particular device in opposite to the ov::Model which is backend-independent.
    * Can import an already compiled graph structure from an input stream to a :doc:`compiled model <openvino_docs_ov_plugin_dg_compiled_model>` object.


2.  :doc:`Compiled Model class <openvino_docs_ov_plugin_dg_compiled_model>`:

    * Is an execution configuration compiled for a particular device and takes into account its capabilities.
    * Holds a reference to a particular device and a task executor for this device.
    * Can create several instances of  :doc:`Inference Request <openvino_docs_ov_plugin_dg_infer_request>`.
    * Can export an internal backend specific graph structure to an output stream.


3.  :doc:`Inference Request class <openvino_docs_ov_plugin_dg_infer_request>`:

    * Runs an inference pipeline serially.
    * Can extract performance counters for an inference pipeline execution profiling.


4.  :doc:`Asynchronous Inference Request class <openvino_docs_ov_plugin_dg_async_infer_request>`:

    * Wraps the  :doc:`Inference Request <openvino_docs_ov_plugin_dg_infer_request>` class and runs pipeline stages in parallel on several task executors based on a device-specific pipeline structure.


5.  :doc:`Plugin specific properties <openvino_docs_ov_plugin_dg_properties>`:

    * Provides the plugin specific properties.


6.  :doc:`Remote Context <openvino_docs_ov_plugin_dg_remote_context>`:

    * Provides the device specific remote context. Context allows to create remote tensors.


7.  :doc:`Remote Tensor <openvino_docs_ov_plugin_dg_remote_tensor>`

    * Provides the device specific remote tensor API and implementation.


.. note::  

   This documentation is written based on the ``Template`` plugin, which demonstrates plugin development details. Find the complete code of the ``Template``, which is fully compilable and up-to-date, at ``<openvino source dir>/src/plugins/template``.


Detailed Guides
###############

*  :doc:`Build <openvino_docs_ov_plugin_dg_plugin_build>` a plugin library using CMake
*  Plugin and its components :doc:`testing <openvino_docs_ov_plugin_dg_plugin_testing>`
*  :doc:`Quantized networks <openvino_docs_ov_plugin_dg_quantized_models>`
*  :doc:`Low precision transformations <openvino_docs_OV_UG_lpt>` guide
*  :doc:`Writing OpenVINO™ transformations <openvino_docs_transformations>` guide
*  `Integration with AUTO Plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto/docs/integration.md>`__

API References
##############

*  `OpenVINO Plugin API <https://docs.openvino.ai/2023.1/groupov_dev_api.html>`__
*  `OpenVINO Transformation API <https://docs.openvino.ai/2023.1/groupie_transformation_api.html>`__

