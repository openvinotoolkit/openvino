# Arm® CPU Device {#openvino_docs_OV_UG_supported_plugins_ARM_CPU}


@sphinxdirective



Introducing the Arm® CPU Plugin
#######################################

The Arm® CPU plugin is developed in order to enable deep neural networks inference on Arm® CPU, using `Compute Library <https://github.com/ARM-software/ComputeLibrary>`__ as a backend.

.. note::

   This is a community-level add-on to OpenVINO™. Intel® welcomes community participation in the OpenVINO™ ecosystem, 
   as well as technical questions and code contributions on community forums. However, this component has not undergone 
   full release validation or qualification from Intel®, hence no official support is offered.

The set of supported layers and their limitations is defined in the 
`Op-set specification page <https://github.com/openvinotoolkit/openvino_contrib/wiki/ARM-plugin-operation-set-specification>`__.


Supported Inference Data Types
#######################################

The Arm® CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  
  - f32
  - f16
  
- Quantized data types:
  
  - i8 (support is experimental)

:doc:`<Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>` can be used to print out supported data types for all detected devices.

Supported Features
#######################################

Preprocessing Acceleration
+++++++++++++++++++++++++++++++++++++++

The Arm® CPU plugin supports the following accelerated preprocessing operations:

- Precision conversion:
  
  - u8  -> u16, s16, s32
  - u16 -> u8, u32
  - s16 -> u8, s32
  - f16 -> f32

- Transposition of tensors with dims < 5
- Interpolation of 4D tensors with no padding (``pads_begin`` and ``pads_end`` equal 0).

The Arm® CPU plugin supports the following preprocessing operations, however they are not accelerated:

- Precision conversion that is not mentioned above
- Color conversion:

  - NV12 to RGB
  - NV12 to BGR
  - i420 to RGB
  - i420 to BGR

For more details, see the :doc:`preprocessing API guide <openvino_docs_OV_UG_Preprocessing_Overview>`.

Supported Properties
#######################################

Read-write Properties
+++++++++++++++++++++++++++++++++++++++

In order to take effect, all parameters must be set before calling ``ov::Core::compile_model()`` or passed as additional argument to ``ov::Core::compile_model()``

- ov::enable_profiling

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

* `Arm® plugin developer documentation <https://github.com/openvinotoolkit/openvino_contrib/blob/master/modules/arm_plugin/README.md>`__.
* `How to run YOLOv4 model inference using OpenVINO™ and OpenCV on Arm® <https://opencv.org/how-to-run-yolov4-using-openvino-and-opencv-on-arm/>`__.
* `Face recognition on Android™ using OpenVINO™ toolkit with Arm® plugin <https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/>`__.


@endsphinxdirective


