# Developer Guide of OpenVINO C API

> **NOTE**: This guide provides a instroduction about OpenVINO C API.
C API provides simplified interface for OpenVINO functionality that allows to:

- handle the input network models & input/output tensors.
- load and configure OpenVINO plugins based on device name.
- perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities).

## Supported OSes

Currently the OpenVINO C API is supported on Ubuntu* 20.04, Microsoft Windows* 10 and CentOS* 7.3 OSes.
Supported Python* versions:

- On Ubuntu 20.04: 2.7, 3.5, 3.6
- On Windows 10: 3.5, 3.6
- On CentOS 7.3: 3.4, 3.5, 3.6

## Instances Mapping Relationships
| C Structure    | C++ Class  | Description |
|:---     |:--- |:---
| `ov_core_t` | `ov::Core` | An structure represents the 'ov::Core' pointer
| `ov_compiled_model_t` | `ov::CompiledModel` | An structure represents the 'ov::CompiledModel' pointer
| `ov_dimension_t` | `ov::Dimension` | An structure represents the 'ov::Dimension' object
| `ov_infer_request_t` | `ov::InferRequest` | An structure represents the 'ov::InferRequest' pointer
| `ov_layout_t` | `ov::Layout` | An structure represents the 'ov::Layout' object
| `ov_model_t` | `ov::Model` | An structure represents the 'ov::Model' pointer
| `ov_output_port_t`/`ov_output_const_port_t` | `ov::Output\<ov::Node\>`/`ov::Output\<const ov::Node\>` | Structure represents the `ov::Output\<ov::Node\>`/`ov::Output\<const ov::Node\>` pointer
| `ov_partial_shape_t` | `ov::Dimension` | An structure represents the vector of `ov::Dimension`
| `ov_preprocess_prepostprocessor_t` | `ov::preprocess::PrePostProcessor` | An structure represents the `ov::preprocess::PrePostProcessor` pointer
| `ov_property_key_*` | `ov::Any` | An properties provided by OpenVINO
| `ov_rank_t` | `ov::Dimension` | The size of `ov::Dimension`
| `ov_shape_t` | `ov::shape` | An structure represents the `ov::shape` object
| `ov_tensor_t` | `ov::Tensor` | An structure represents the `ov::Tensor` pointer


## Provided APIs
| Component    | API  | Description |
|:---     |:--- |:---
| common | `ov_get_error_info()`, `ov_free()`  | Common APIs, details in `ov_common.h`
| core | `ov_get_openvino_version()`, `ov_version_free()`, `ov_core_create()`, `ov_core_create_with_config()`, `ov_core_create_with_config_unicode()`, `ov_core_free()`, `ov_core_read_model()`, `ov_core_read_model_unicode()`, `ov_core_read_model_from_memory()`, `ov_core_compile_model()`, `ov_core_compile_model_from_file()`, `ov_core_compile_model_from_file_unicode()`, `ov_core_set_property()`, `ov_core_get_property()`, `ov_core_get_available_devices()`, `ov_available_devices_free()`, `ov_core_import_model()`, `ov_core_get_versions_by_device_name()`,`ov_core_versions_free()` | Operations of core, details in `ov_core.h`
| compiled model | `ov_compiled_model_inputs_size()`, `ov_compiled_model_input()`, `ov_compiled_model_input_by_index()`, `ov_compiled_model_input_by_name()`, `ov_compiled_model_outputs_size()`, `ov_compiled_model_output()`, `ov_compiled_model_output_by_index()`, `ov_compiled_model_output_by_name()`, `ov_compiled_model_get_runtime_model()`, `ov_compiled_model_create_infer_request()`, `ov_compiled_model_set_property()`, `ov_compiled_model_get_property()`, `ov_compiled_model_export_model()`, `ov_compiled_model_free` | Opertions of compiled model, details in `ov_compiled_model.h`
| model | `ov_model_free()`, `ov_model_const_input()`, `ov_model_const_input_by_name()`, `ov_model_const_input_by_index()`, `ov_model_input()`, `ov_model_input_by_name()`, `ov_model_input_by_index()`, `ov_model_const_output()`, `ov_model_const_output_by_index()`, `ov_model_const_output_by_name()`, `ov_model_output()`, `ov_model_output_by_index()`, `ov_model_output_by_name()`, `ov_model_inputs_size()`, `ov_model_outputs_size()`, `ov_model_is_dynamic()`, `ov_model_reshape()`, `ov_model_reshape_input_by_name`, `ov_model_reshape_single_input()`, `ov_model_reshape_by_port_indexes()`, `ov_model_reshape_by_ports()`, `ov_model_get_friendly_name()` | Opertions of model, details in `ov_model.h`
| infer request | `ov_infer_request_set_tensor()`, `ov_infer_request_set_tensor_by_port()`, `ov_infer_request_set_tensor_by_const_port()`, `ov_infer_request_set_input_tensor_by_index()`, `ov_infer_request_set_input_tensor()`, `ov_infer_request_set_output_tensor_by_index()`, `ov_infer_request_set_output_tensor()`, `ov_infer_request_get_tensor()`, `ov_infer_request_get_tensor_by_const_port()`, `ov_infer_request_get_tensor_by_port()`, `ov_infer_request_get_input_tensor_by_index()`, `ov_infer_request_get_input_tensor()`, `ov_infer_request_get_output_tensor_by_index()`, `ov_infer_request_get_out_tensor()`, `ov_infer_request_infer()`, `ov_infer_request_cancel()`, `ov_infer_request_start_async()`, `ov_infer_request_wait()`, `ov_infer_request_wait_for()`, `ov_infer_request_set_callback()`, `ov_infer_request_free()`, `ov_infer_request_get_profiling_info()`, `ov_profiling_info_list_free()` | Opertions of infer request, details in `ov_infer_request.h`
| dimension | `ov_dimension_is_dynamic()` | Opertions of dimensions, details in `ov_dimension.h`
| layout | `ov_layout_create()`, `ov_layout_free()`, `ov_layout_to_string()` | Opertions of layout, details in `ov_layout.h`
| node | `ov_const_port_get_shape()`, `ov_port_get_shape()`, `ov_port_get_any_name()`, `ov_port_get_partial_shape()`, `ov_port_get_element_type()`, `ov_output_port_free()`, `ov_output_const_port_free()` | Opertions of nodel, details in `ov_node.h`
| partial shape | `ov_partial_shape_create()`, `ov_partial_shape_create_dynamic()`, `ov_partial_shape_create_static()`, `ov_partial_shape_free()`, `ov_partial_shape_to_shape()`, `ov_shape_to_partial_shape()`, `ov_partial_shape_is_dynamic()`, `ov_partial_shape_to_string` | Opertions of partial shape, details in `ov_partial_shape.h`
| prepostprocess | `ov_preprocess_prepostprocessor_create()`, `ov_preprocess_prepostprocessor_free()`, `ov_preprocess_prepostprocessor_get_input_info()`, `ov_preprocess_prepostprocessor_get_input_info_by_name()`, `ov_preprocess_prepostprocessor_get_input_info_by_index()`, `ov_preprocess_input_info_free()`, `ov_preprocess_input_info_get_tensor_info()`, `ov_preprocess_input_tensor_info_free()`, `ov_preprocess_input_info_get_preprocess_steps()`, `ov_preprocess_preprocess_steps_free()`, `ov_preprocess_preprocess_steps_resize()`, `ov_preprocess_input_tensor_info_set_element_type()`, `ov_preprocess_input_tensor_info_set_color_format()`, `ov_preprocess_input_tensor_info_set_spatial_static_shape()`, `ov_preprocess_preprocess_steps_convert_element_type()`, `ov_preprocess_preprocess_steps_convert_color()`, `ov_preprocess_input_tensor_info_set_from()`, `ov_preprocess_input_tensor_info_set_layout()`, `ov_preprocess_prepostprocessor_get_output_info()`, `ov_preprocess_prepostprocessor_get_output_info_by_index()`, `ov_preprocess_prepostprocessor_get_output_info_by_name()`, `ov_preprocess_output_info_free()`, `ov_preprocess_output_info_get_tensor_info()`, `ov_preprocess_output_tensor_info_free()`, `ov_preprocess_output_set_element_type()`, `ov_preprocess_input_info_get_model_info()`, `ov_preprocess_input_model_info_free()`, `ov_preprocess_input_model_info_set_layout()`, `ov_preprocess_prepostprocessor_build()` | Operations of prepostprocess, details in `ov_prepostprocess.h`
| property | `ov_property_key_supported_properties`, `ov_property_key_available_devices`, `ov_property_key_optimal_number_of_infer_requests`, `ov_property_key_range_for_async_infer_requests`, `ov_property_key_range_for_streams`, `ov_property_key_device_full_name`, `ov_property_key_device_capabilities`, `ov_property_key_model_name`, `ov_property_key_optimal_batch_size`, `ov_property_key_max_batch_size`, `ov_property_key_cache_dir`, `ov_property_key_num_streams`, `ov_property_key_affinity`, `ov_property_key_inference_num_threads`, `ov_property_key_hint_performance_mode`, `ov_property_key_hint_inference_precision`, `ov_property_key_hint_num_requests`, `ov_property_key_log_level`, `ov_property_key_hint_model_priority`, `ov_property_key_enable_profiling`, `ov_property_key_device_priorities` | Supported properties, details in `ov_property.h`
| rank | `ov_rank_is_dynamic()` | Operations of rank, details in `ov_rank.h`
| shape | `ov_shape_create()`, `ov_shape_free` | Operations of shape, details in `ov_shape.h`
| tensor | `ov_tensor_create_from_host_ptr()`, `ov_tensor_create()`, `ov_tensor_set_shape()`, `ov_tensor_get_shape()`, `ov_tensor_get_element_type()`, `ov_tensor_get_size()`, `ov_tensor_get_byte_size()`, `ov_tensor_data()`, `ov_tensor_free()` | Operations of tensor, details in `ov_tensor.h`

## The mapping relationship between C++ api 2.0 and C api
| C++ API  | C API  |
|:---     |:--- |
| `Core(const std::string& xml_config_file = {})`| `ov_core_create(ov_core_t** core)` or `ov_core_create_with_config(const char* xml_config_file, ov_core_t** core)`|
|`~Core()`| `ov_core_free(ov_core_t* core)`|
|`get_versions(const std::string& device_name)`|`ov_core_get_versions_by_device_name(const ov_core_t* core, const char* device_name, ov_core_version_list_t* versions)`|
|` read_model(const std::string& model_path, const std::string& bin_path = {})`|`ov_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model)`|
|`compile_model(const std::shared_ptr<const ov::Model>& model, const std::string& device_name, const AnyMap& properties = {})`|`ov_core_compile_model(const ov_core_t* core, const ov_model_t* model, const char* device_name, const ov_properties_t* property, ov_compiled_model_t** compiled_model)`|

## Roles in development
* Most of APIs should return "ov_status_e" and the actual output will be put in input/output parameters. 
* Some specified APIs will return void or boolean per API's funtionality.
* structure name plus  "_create" suffix, and input/output parameter is "**" format. Such as C++ `ov::Core()` to C `ov_core_create(ov_core_t** core)`.
* Don't support the vector of hidden structure.
* As much as possible to apply pass by value  for input parameter of C APIs.
* Pass by pointer will be used in output parameters
* Snake_case style, strict unfold of C++ class plus this class method (can add additional action word to make its readable),  the first paramter should be the C structure pointer of the coresponding C++ class. Such as `OPENVINO_C_API(ov_status_e)  <class_name>_<method_name>_<additional_info>(<sturcture_object_pointer>, <input/output parameter>,... );`
* Don't support class's  template methods.
* With same name but different parameters, choose the popular one name first, and others will be add edadditional info to distinguish them.
* Function parameter order:  input parameters first and then output parameters.
* All unchanged input arguments shoud add "const" specificator.

## Tutorials
Two C samples provided for customer reference, and the main steps about how to create C sample with C API in [here](./How_to_Create_C_Sample_with_OpenVINO_C_API.md)
- [Hello Classification C Sample](../../../../samples/c/hello_classification/README.md)
- [Hello NV12 Input Classification C Sample](../../../../samples/c/hello_nv12_input_classification/README.md)

## See also
 * [OpenVINO Runtime C API Developer Guide](./OpenVINO_Runtime_C_API_Developer_Guide.md)
 * [How to Create C Sample with C API](./How_to_Create_C_Sample_with_OpenVINO_C_API.md)
