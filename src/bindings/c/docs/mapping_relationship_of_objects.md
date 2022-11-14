# Mapping Relationship of Objects

Here is the details introduction about the matching relationship between C structures and C++ classes: 
| C Structure | C++ Class | Description | File Location
|:---         |:---       |:---         |:---
| `ov_core_t` | `ov::Core` | structure represents `ov::Core` by a shared pointer | [ov_core.h](../include/openvino/c/ov_core.h)
| `ov_version_t` | `ov::Version` | structure represents `ov::Version` by C implementation | [ov_core.h](../include/openvino/c/ov_core.h)
| `ov_core_version_t` | NAN | structure represents plugin version by C implementation | [ov_core.h](../include/openvino/c/ov_core.h)
| `ov_core_version_list_t` | NAN | structure represents plugins version by C implementation | [ov_core.h](../include/openvino/c/ov_core.h)
| `ov_available_devices_t` | NAN | structure represents devices' names by C implementation | [ov_core.h](../include/openvino/c/ov_core.h)
| `ov_model_t` | `ov::Model` | structure represents the `ov::Model` by a shared pointer | [ov_model.h](../include/openvino/c/ov_model.h)
| `ov_compiled_model_t` | `ov::CompiledModel` | structure represents the `ov::CompiledModel` by a shared pointer | [ov_compiled_model.h](../include/openvino/c/ov_compiled_model.h)
| `ov_infer_request_t` | `ov::InferRequest` | structure represents the `ov::InferRequest` by a shared pointer | [ov_infer_request.h](../include/openvino/c/ov_infer_request.h)
| `ov_callback_t` | NAN | structure represents call back func by C implementation | [ov_infer_request.h](../include/openvino/c/ov_infer_request.h)
| `ov_profiling_info_t` | `ov::ProfilingInfo` | structure represents `ov::ProfilingInfo` by C implementation | [ov_infer_request.h](../include/openvino/c/ov_infer_request.h)
| `ov_profiling_info_list_t` | NAN | structure represents the vector of `ov::ProfilingInfo` by C implementation | [ov_infer_request.h](../include/openvino/c/ov_infer_request.h)
| `ov_dimension_t` | `ov::Dimension` | structure represents the `ov::Dimension` by C implementation | [ov_dimension.h](../include/openvino/c/ov_dimension.h)
| `ov_rank_t` | `ov::Rank` | structure represents the `ov::Rank` by C implementation | [ov_rank.h](../include/openvino/c/ov_rank.h)
| `ov_shape_t` | `ov::shape` | structure represents the `ov::shape` by C implementation | [ov_shape.h](../include/openvino/c/ov_shape.h)
 `ov_partial_shape_t` | `ov::PartialShape` | structure represents the `ov::PartialShape` by C implementation | [ov_partial_shape.h](../include/openvino/c/ov_partial_shape.h)
 | `ov_layout_t` | `ov::Layout` | represents the `ov::Layout` by C structure include the object | [ov_layout.h](../include/openvino/c/ov_layout.h)
| `ov_output_port_t`/`ov_output_const_port_t` | `ov::Output\<ov::Node\>`/`ov::Output\<const ov::Node\>` | structure represents the `ov::Output\<ov::Node\>`/`ov::Output\<const ov::Node\>` by a shared pointer | [ov_node.h](../include/openvino/c/ov_node.h)
| `ov_tensor_t` | `ov::Tensor` | structure represents the `ov::Tensor` by a shared pointer | [ov_tensor.h](../include/openvino/c/ov_tensor.h)
| `ov_property_key_*` | `ov::Property` | char* represents the properties provided by OpenVINO | [ov_property.h](../include/openvino/c/ov_property.h)
| `ov_preprocess_prepostprocessor_t` | `ov::preprocess::PrePostProcessor` | structure represents the `ov::preprocess::PrePostProcessor` by a pointer | [ov_prepostprocess.h](../include/openvino/c/ov_prepostprocess.h)

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [C API Reference](https://docs.openvino.ai/latest/api/api_reference.html)


