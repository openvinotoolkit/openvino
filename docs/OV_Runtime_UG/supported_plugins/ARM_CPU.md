# Arm® CPU Device {#openvino_docs_OV_UG_supported_plugins_ARM_CPU}


## Introducing the Arm® CPU Plugin
The Arm® CPU plugin is developed in order to enable deep neural networks inference on Arm® CPU, using [Compute Library](https://github.com/ARM-software/ComputeLibrary) as a backend.

> **NOTE**: This is a community-level add-on to OpenVINO™. Intel® welcomes community participation in the OpenVINO™ ecosystem, technical questions and code contributions on community forums. However, this component has not undergone full release validation or qualification from Intel®, hence no official support is offered. 

The Arm® CPU plugin is not a part of the Intel® Distribution of OpenVINO™ toolkit and is not distributed in pre-built form. The plugin should be built from the source code for use. Plugin build procedure is described on [How to build Arm® CPU plugin](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin) guide. 

The set of supported layers is defined on the [Operation set specification](https://github.com/openvinotoolkit/openvino_contrib/wiki/ARM-plugin-operation-set-specification) page.


## Supported Inference Data Types
The Arm® CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - f16
- Quantized data types:
  - i8 (support is experimental)

[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out supported data types for all detected devices.

## Supported Features

### Preprocessing Acceleration
The Arm® CPU plugin supports the following accelerated preprocessing operations:
- Precision conversion:
    - u8  -> u16, s16, s32
    - u16 -> u8, u32
    - s16 -> u8, s32
    - f16 -> f32
- Transposition of tensors with dims < 5
- Interpolation of 4D tensors with no padding (`pads_begin` and `pads_end` equal 0).

The Arm® CPU plugin supports the following preprocessing operations, however they are not accelerated:
- Precision conversion that is not mentioned above
- Color conversion:
    - NV12 to RGB
    - NV12 to BGR
    - i420 to RGB
    - i420 to BGR

For more details, see the [preprocessing API guide](../preprocessing_overview.md).

## Supported Properties
The plugin supports the properties listed below.

### Read-write Properties
In order to take effect, all parameters must be set before calling `ov::Core::compile_model()` or passed as additional argument to `ov::Core::compile_model()`

- ov::enable_profiling

### Read-only Properties
- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::device::full_name
- ov::device::capabilities


## Known Layers Limitation
* `AvgPool` layer is supported via arm_compute library for 4D input tensor and via reference implementation for other cases.
* `BatchToSpace` layer is supported for 4D tensors only and constant nodes: `block_shape` with `N` = 1 and `C`= 1, `crops_begin` with zero values and `crops_end` with zero values.
* `ConvertLike` layer is supported for configuration like `Convert`.
* `DepthToSpace` layer is supported for 4D tensors only and for `BLOCKS_FIRST` of `mode` attribute.
* `Equal` does not support `broadcast` for inputs.
* `Gather` layer is supported for constant scalar or 1D indices axes only. Layer is supported via arm_compute library for non negative indices and via reference implementation otherwise.
* `Less` does not support `broadcast` for inputs.
* `LessEqual` does not support `broadcast` for inputs.
* `LRN` layer is supported for `axes = {1}` or `axes = {2, 3}` only.
* `MaxPool-1` layer is supported via arm_compute library for 4D input tensor and via reference implementation for other cases.
* `Mod` layer is supported for f32 only.
* `MVN` layer is supported via arm_compute library for 2D inputs and `false` value of `normalize_variance` and `false` value of `across_channels`, for other cases layer is implemented via runtime reference.
* `Normalize` layer is supported via arm_compute library with `MAX` value of `eps_mode` and `axes = {2 | 3}`, and for `ADD` value of `eps_mode` layer uses `DecomposeNormalizeL2Add`. For other cases layer is implemented via runtime reference.
* `NotEqual` does not support `broadcast` for inputs.
* `Pad` layer works with `pad_mode = {REFLECT | CONSTANT | SYMMETRIC}` parameters only.
* `Round` layer is supported via arm_compute library with `RoundMode::HALF_AWAY_FROM_ZERO` value of `mode`, for other cases layer is implemented via runtime reference.
* `SpaceToBatch` layer is supported for 4D tensors only and constant nodes: `shapes`, `pads_begin` or `pads_end` with zero paddings for batch or channels and one values `shapes` for batch and channels.
* `SpaceToDepth` layer is supported for 4D tensors only and for `BLOCKS_FIRST` of `mode` attribute.
* `StridedSlice` layer is supported via arm_compute library for tensors with dims < 5 and zero values of `ellipsis_mask` or zero values of `new_axis_mask` and `shrink_axis_mask`. For other cases, layer is implemented via runtime reference.
* `FakeQuantize` layer is supported via arm_compute library, in Low Precision evaluation mode for suitable models, and via runtime reference otherwise.

## Additional Resources
* [How to run YOLOv4 model inference using OpenVINO™ and OpenCV on Arm®](https://opencv.org/how-to-run-yolov4-using-openvino-and-opencv-on-arm/)
* [Face recognition on Android™ using OpenVINO™ toolkit with Arm® plugin](https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/)
