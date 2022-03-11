# Arm&reg; CPU device {#openvino_docs_OV_UG_supported_plugins_ARM_CPU}


## Introducing the Arm&reg; CPU Plugin
The ARM&reg; CPU plugin is developed in order to enable deep neural networks inference on Arm&reg; CPU, using [Compute Library](https://github.com/ARM-software/ComputeLibrary) as a backend.

The Arm&reg; CPU plugin is not a part of the Intel® Distribution of OpenVINO™ toolkit and is not distributed in pre-built form. To use the plugin, it should be buid from source code. Plugin build procedure is described on [How to build Arm&reg; CPU plugin](https://github.com/openvinotoolkit/openvino_contrib/wiki/How-to-build-ARM-CPU-plugin). 

The set of supported layers is defined on [Operation set specification](https://github.com/openvinotoolkit/openvino_contrib/wiki/ARM-plugin-operation-set-specification).


## Supported inference data types
The Arm&reg; CPU plugin supports the following data types as inference precision of internal primitives:

- Floating-point data types:
  - f32
  - f16
- Quantized data types:
  - i8


> **NOTE**: i8 support is experimental.

[Hello Query Device C++ Sample](../../../samples/cpp/hello_query_device/README.md) can be used to print out supported data types for all detected devices.


## Supported properties
The plugin supports the properties listed below.

### Read-write properties
All parameters must be set before calling `ov::Core::compile_model()` in order to take effect or passed as additional argument to `ov::Core::compile_model()`

- ov::enable_profiling

### Read-only properties
- ov::supported_properties
- ov::available_devices
- ov::range_for_async_infer_requests
- ov::range_for_streams
- ov::device::full_name
- ov::device::capabilities


## Known Layers Limitation
* `AvgPool` layer is supported via arm_compute library for 4D input tensor and via reference implementation for another cases.
* `BatchToSpace` layer is supported 4D tensors only and constant nodes: `block_shape` with `N` = 1 and `C`= 1, `crops_begin` with zero values and `crops_end` with zero values.
* `Convert` layer is supported via arm_compute library with configuration: src data type -> dst data type
    - U8  -> U16, S16, S32
    - U16 -> U8, U32
    - S16 -> U8, S32
    - F16 -> F32
    
    for another cases layer is implemented via runtime reference.
* `ConvertLike` layer is supported configuration like `Convert`.
* `DepthToSpace` layer is supported 4D tensors only and for `BLOCKS_FIRST` of `mode` attribute.
* `Equal` does not support `broadcast` for inputs.
* `Gather` layer is supported constant scalar or 1D indices axes only. Layer is supported as via arm_compute library for non negative indices and via reference implementation otherwise.
* `Interpolate` layer is supported 4D tensors and zero value of `pads_begin` and `pads_end` attributes only.
* `Less` does not support `broadcast` for inputs.
* `LessEqual` does not support `broadcast` for inputs.
* `LRN` layer is supported `axes = {1}` or `axes = {2, 3}` only.
* `MaxPool-1` layer is supported via arm_compute library for 4D input tensor and via reference implementation for another cases.
* `Mod` layer is supported for FP32 only.
* `MVN` layer is supported via arm_compute library for 2D inputs and `false` value of `normalize_variance` and `false` value of `across_channels`, for another cases layer is implemented via runtime reference.
* `Normalize` layer is supported via arm_compute library with `MAX` value of `eps_mode` and `axes = {2 | 3}`, and for `ADD` value of `eps_mode` layer uses `'DecomposeNormalizeL2Add'`, for another cases layer is implemented via runtime reference.
* `NotEqual` does not support `broadcast` for inputs.
* `Pad` layer works with `pad_mode = {REFLECT | CONSTANT | SYMMETRIC}` parameters only.
* `Round` layer is supported via arm_compute library with `RoundMode::HALF_AWAY_FROM_ZERO` value of `mode`, for another cases layer is implemented via runtime reference.
* `SpaceToBatch` layer is supported 4D tensors only and constant nodes: `shapes`, `pads_begin` or `pads_end` with zero paddings for batch or channels and one values `shapes` for batch and channels.
* `SpaceToDepth` layer is supported 4D tensors only and for `BLOCKS_FIRST` of `mode` attribute.
* `StridedSlice` layer is supported via arm_compute library for tensors with dims < 5 and zero values of `ellipsis_mask` or zero values of `new_axis_mask` and `shrink_axis_mask`, for another cases layer is implemented via runtime reference.
* `Transpose` layer is supported via arm_compute library for tensors with dims < 5, for another cases layer is implemented via runtime reference.

## See Also
* [How to run YOLOv4 model inference using OpenVINO&trade; and OpenCV on Arm&reg;](https://opencv.org/how-to-run-yolov4-using-openvino-and-opencv-on-arm/)
* [Face recognition on Android&trade; using OpenVINO&trade; toolkit with Arm&reg; plugin](https://opencv.org/face-recognition-on-android-using-openvino-toolkit-with-arm-plugin/)
