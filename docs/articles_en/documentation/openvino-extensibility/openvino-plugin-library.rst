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

   Implement Plugin Functionality <openvino-plugin-library/plugin>
   Implement Compiled Model Functionality <openvino-plugin-library/compiled-model>
   Implement Synchronous Inference Request <openvino-plugin-library/synch-inference-request>
   Implement Asynchronous Inference Request <openvino-plugin-library/asynch-inference-request>
   Provide Plugin Specific Properties <openvino-plugin-library/plugin-properties>
   Implement Remote Context <openvino-plugin-library/remote-context>
   Implement Remote Tensor <openvino-plugin-library/remote-tensor>
   openvino-plugin-library/build-plugin-using-cmake
   openvino-plugin-library/plugin-testing
   openvino-plugin-library/advanced-guides
   openvino-plugin-library/plugin-api-references


The plugin architecture of OpenVINO allows to develop and plug independent inference
solutions dedicated to different devices. Physically, a plugin is represented as a dynamic library
exporting the single ``create_plugin_engine`` function that allows to create a new plugin instance.

OpenVINO Plugin Library
#######################

OpenVINO plugin dynamic library consists of several main components:

1.  :doc:`Plugin class <openvino-plugin-library/plugin>`:

    * Provides information about devices of a specific type.
    * Can create an  :doc:`compiled model <openvino-plugin-library/compiled-model>` instance which represents a Neural Network backend specific graph structure for a particular device in opposite to the ov::Model which is backend-independent.
    * Can import an already compiled graph structure from an input stream to a :doc:`compiled model <openvino-plugin-library/compiled-model>` object.


2.  :doc:`Compiled Model class <openvino-plugin-library/compiled-model>`:

    * Is an execution configuration compiled for a particular device and takes into account its capabilities.
    * Holds a reference to a particular device and a task executor for this device.
    * Can create several instances of  :doc:`Inference Request <openvino-plugin-library/synch-inference-request>`.
    * Can export an internal backend specific graph structure to an output stream.


3.  :doc:`Inference Request class <openvino-plugin-library/synch-inference-request>`:

    * Runs an inference pipeline serially.
    * Can extract performance counters for an inference pipeline execution profiling.


4.  :doc:`Asynchronous Inference Request class <openvino-plugin-library/asynch-inference-request>`:

    * Wraps the  :doc:`Inference Request <openvino-plugin-library/synch-inference-request>` class and runs pipeline stages in parallel on several task executors based on a device-specific pipeline structure.


5.  :doc:`Plugin specific properties <openvino-plugin-library/plugin-properties>`:

    * Provides the plugin specific properties.


6.  :doc:`Remote Context <openvino-plugin-library/remote-context>`:

    * Provides the device specific remote context. Context allows to create remote tensors.


7.  :doc:`Remote Tensor <openvino-plugin-library/remote-tensor>`

    * Provides the device specific remote tensor API and implementation.


.. note::

   This documentation is written based on the ``Template`` plugin, which demonstrates plugin development details. Find the complete code of the ``Template``, which is fully compilable and up-to-date, at ``<openvino source dir>/src/plugins/template``.


Detailed Guides
###############

*  :doc:`Build <openvino-plugin-library/build-plugin-using-cmake>` a plugin library using CMake
*  Plugin and its components :doc:`testing <openvino-plugin-library/plugin-testing>`
*  :doc:`Quantized networks <openvino-plugin-library/advanced-guides/quantized-models>`
*  :doc:`Low precision transformations <openvino-plugin-library/advanced-guides/low-precision-transformations>` guide
*  :doc:`Writing OpenVINO™ transformations <transformation-api>` guide
*  `Integration with AUTO Plugin <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/auto/docs/integration.md>`__

API References
##############

*  `OpenVINO Plugin API <https://docs.openvino.ai/2025/api/c_cpp_api/group__ov__dev__api.html>`__
*  `OpenVINO Transformation API <https://docs.openvino.ai/2025/api/c_cpp_api/group__ie__transformation__api.html>`__

