# Arm® CPU Device {#openvino_docs_OV_UG_supported_plugins_ARM_CPU}


@sphinxdirective



Introducing the Arm® CPU Plugin
#######################################

The Arm® CPU plugin is developed in order to enable deep neural networks inference on Arm® CPU, using `Compute Library <https://github.com/ARM-software/ComputeLibrary>`__ as a backend.

.. note::

   This is a community-level add-on to OpenVINO™. Intel® welcomes community participation in the OpenVINO™ ecosystem, 
   as well as technical questions and code contributions on community forums. However, this component has not undergone 
   full release validation or qualification from Intel®, hence no official support is offered.


Supported Inference Data Types
#######################################

The Arm® CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  
  - f32
  - f16

- Integer data types:
  
  - i32

- Quantized data types:
  
  - i8 (support is experimental)

:doc:`<Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>` can be used to print out supported data types for all detected devices.


Supported Properties
#######################################

Read-write Properties
+++++++++++++++++++++++++++++++++++++++

In order to take effect, all parameters must be set before calling ``ov::Core::compile_model()`` or passed as additional argument to ``ov::Core::compile_model()``

- ov::enable_profiling
- ov::hint::inference_precision
- ov::hint::performance_mode
- ov::hint::num_request
- ov::num_streams
- ov::affinity
- ov::inference_num_threads
- ov::cache_dir

Read-only Properties
+++++++++++++++++++++++++++++++++++++++

- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::device::full_name
- ov::device::capabilities


Additional Resources
#######################################

* `CPU plugin developers documentation <https://github.com/openvinotoolkit/openvino/blob/master/src/plugins/intel_cpu/README.md>`__.


@endsphinxdirective


