# Mapping Relationship of Interfaces

Here provides the details mapping relationship between C API and C++ API in OpenVINO including components `core`, `model`, `compiled model`, `infer request`, `partial shape`, `prepostprocess`, `tensor`. This introduction doesn't provide all interfaces' mapping relationship from C++ to C in OpenVINO, but the most important & common interfaces provided.

**NOTE**: For more matching details can be found in [Mapping Relationship of Interfaces](./docs/mapping_relationship_of_interfaces.md).

## Contents:

 - [The mapping relationship for core](#the-mapping-relationship-for-core)
 - [The mapping relationship for model](#the-mapping-relationship-for-model)
 - [The mapping relationship for compiled model](#the-mapping-relationship-for-compiled-model)
 - [The mapping relationship for infer request](#the-mapping-relationship-for-infer-request)
 - [The mapping relationship for partial shape](#the-mapping-relationship-for-partial-shape)
 - [The mapping relationship for prepostprocess](#the-mapping-relationship-for-prepostprocess)
 - [The mapping relationship for tensor](#the-mapping-relationship-for-tensor)
 - [See also](#see-also)

## The mapping relationship for core

Here is the mapping relationship for core related,
 * All core related APIs implemented with C++ can be found [here](../../../inference/include/openvino/runtime/core.hpp)
 * All core related APIs implemented with C can be found [here](../include/openvino/c/ov_core.h)

| C++ API | C API | Description |
|:---     |:---   |:---
| `Core(const std::string& xml_config_file = {})`| `ov_core_create(ov_core_t** core)` or `ov_core_create_with_config(const char* xml_config_file, ov_core_t** core)`| Constructs OpenVINO Core instance
|`~Core()`| `ov_core_free(ov_core_t* core)`| C need to free the allocated core instance abviously, C++ not
|`read_model(const std::string& model_path, const std::string& bin_path = {})`|`v_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model)`| Reads models from IR/ONNX/PDPD formats (also provides method read from memory or unicode)
|`compile_model(const std::shared_ptr<const ov::Model>& model, const std::string& deviceName, const AnyMap& config)`|`ov_core_compile_model(const ov_core_t* core, const ov_model_t* model, const char* device_name, const size_t property_args_size, ov_compiled_model_t** compiled_model, ...)`| Creates a compiled model from model object (also provides method compile from file or unicode)
|`set_property(const std::string& device_name, const AnyMap& properties)`|`ov_core_set_property(const ov_core_t* core, const char* device_name, ...)`| Sets properties for a device (or core with device name NULL)
|`get_property(const std::string& deviceName, const std::string& name, const AnyMap& arguments)`|`ov_core_get_property(const ov_core_t* core, const char* device_name, const char* property_key, char** property_value)`| Gets properties related to the key
|`get_versions(const std::string& deviceName)`|`ov_core_get_versions_by_device_name(const ov_core_t* core, const char* device_name, ov_core_version_list_t* versions)`| get device plugins version information (need be freed by `ov_core_versions_free(ov_core_version_list_t* versions)`)

## The mapping relationship for model
 * All model related APIs implemented with C++ can be found [here](../../../core/include/openvino/core/model.hpp)
 * All model related APIs implemented with C can be found [here](../include/openvino/c/ov_model.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`~Model()`|`ov_model_free(ov_model_t* model)`| C need to free the allocated model instance abviously, C++ not
|`ov::Model::input()`|`ov_model_input(const ov_model_t* model, ov_output_port_t** input_port)`| Get input port from model (also provides method get by index/name)
|`ov::Model::output()`|`ov_model_output(const ov_model_t* model, ov_output_port_t** output_port)`|  Get output port from model (also provides method get by index/name)
|`ov::Model::inputs()`|`ov_model_inputs_size(const ov_model_t* model, size_t* input_size)`| Get the input size of model
|`ov::Model::outputs()`|` ov_model_outputs_size(const ov_model_t* model, size_t* output_size)`| Get the output size of model
|`ov::Model::is_dynamic()`|`ov_model_is_dynamic(const ov_model_t* model)`| check the model is dynamic or not
|`reshape(const ov::PartialShape& partial_shape)`|`ov_model_reshape(const ov_model_t* model, const char** tensor_names, const ov_partial_shape_t* partial_shapes,size_t size)`| reshape model with special shape (also provides method reshape by index/name)
|`get_friendly_name()`|`ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name)`| Gets the friendly name for a model

## The mapping relationship for compiled model
 * All compiled model related APIs implemented with C++ can be found [here](../../../inference/include/openvino/runtime/compiled_model.hpp)
 * All compiled model related APIs implemented with C can be found [here](../include/openvino/c/ov_compiled_model.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`~CompiledModel()`|`ov_compiled_model_free(ov_compiled_model_t* compiled_model)`| C need to free the allocated compiled model instance abviously, C++ not
|`CompiledModel::inputs()`|`ov_compiled_model_inputs_size(const ov_compiled_model_t* compiled_model, size_t* size)`| C++ gets all inputs from compiled model save in vector, but C just get the inputs size
|`CompiledModel::input()`|`ov_compiled_model_input(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** input_port)`| get the input port from compiled model (also provides method get by index/name)
|`CompiledModel::outputs()`|`ov_compiled_model_outputs_size(const ov_compiled_model_t* compiled_model, size_t* size)`| C++ gets all outputs from compiled model save in vector, but C just get the outputs size
|`CompiledModel::output()`|`ov_compiled_model_output(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** output_port)`| get the output port from compiled model (also provides method get by index/name)
|`CompiledModel::get_runtime_model()`|`ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model)`| gets runtime model information
|`CompiledModel::create_infer_request()`|`ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request)`| creates an inference request object used to infer the compiled model
|`CompiledModel::set_property(const AnyMap& properties)`|`ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, ...)`| sets properties for the current compiled model
|`CompiledModel::get_property(const std::string& name)`|`ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model, const char* property_key, char** property_value)`| gets properties for current compiled model
|`CompiledModel::export_model(std::ostream& model_stream)`|`ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path)`| exports the current compiled model

## The mapping relationship for infer request
 * All infer request related APIs implemented with C++ can be found [here](../../../inference/include/openvino/runtime/infer_request.hpp)
 * All infer request related APIs implemented with C can be found [here](../include/openvino/c/ov_infer_request.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`~InferRequest()`|`ov_infer_request_free(ov_infer_request_t* infer_request)`| C need to free the allocated infer request instance abviously, C++ not
|`set_input_tensor(const Tensor& tensor)`|`ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor)`| sets an input tensor (also provides method set by index/port/name)
|`set_output_tensor(const Tensor& tensor)`|`ov_infer_request_set_output_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor)`| sets an output tensor (also provides method set by index/port/name)
|`get_input_tensor()`|`ov_infer_request_get_input_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor)`| get an input tensor (also provides method get by index/port/name)
|`get_output_tensor()`|`ov_infer_request_get_output_tensor(const ov_infer_request_t* infer_request, ov_tensor_t** tensor)`| get an output tensor (also provides method get by index/port/name)
|`infer()`|`ov_infer_request_infer(ov_infer_request_t* infer_request)`| infer specified input(s) in synchronous mode
|`start_async()`|`ov_infer_request_start_async(ov_infer_request_t* infer_request)`| start inference of specified input(s) in asynchronous mode
|`cancel()`|`ov_infer_request_cancel(ov_infer_request_t* infer_request)`| cancel inference request
|`wait()`|`ov_infer_request_wait(ov_infer_request_t* infer_request)`| wait for the result to become available
|`wait_for(const std::chrono::milliseconds timeout)`|`ov_infer_request_wait_for(ov_infer_request_t* infer_request, const int64_t timeout)`| Blocks until the specified timeout has elapsed or the result becomes available, whichever comes first
|`set_callback(std::function<void(std::exception_ptr)> callback)`|`ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback)`| set callback function, which will be called when inference is done
|`get_profiling_info()`|`ov_infer_request_get_profiling_info(const ov_infer_request_t* infer_request, ov_profiling_info_list_t* profiling_infos)` | Query performance measures per layer to identify the most time consuming operation, need to be freed by `ov_profiling_info_list_free()`

## The mapping relationship for partial shape
 * All partial shape related APIs implemented with C++ can be found [here](../../../core/include/openvino/core/partial_shape.hpp)
 * All partial shape related APIs implemented with C can be found [here](../include/openvino/c/ov_partial_shape.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`PartialShape(std::initializer_list<Dimension> init)`|`ov_partial_shape_create(const int64_t rank, const ov_dimension_t* dims, ov_partial_shape_t* partial_shape_obj)`| create partial shape with static rank and dynamic dimension
|`PartialShape(std::vector<Dimension> dimensions)`|`ov_partial_shape_create_dynamic(const ov_rank_t rank, const ov_dimension_t* dims, ov_partial_shape_t* partial_shape_obj)`| create partial shape with static rank from a vector of Dimension
|`PartialShape(const std::vector<Dimension::value_type>& dimensions)`|`ov_partial_shape_create_static(const int64_t rank, const int64_t* dims, ov_partial_shape_t* partial_shape_obj)`| create partial shape with static rank and static dimension
|`~PartialShape()`|`ov_partial_shape_free(ov_partial_shape_t* partial_shape)`| C need to free the allocated partial shape instance abviously, C++ not
|`to_shape()`|`ov_partial_shape_to_shape(const ov_partial_shape_t partial_shape, ov_shape_t* shape)`| convert a static PartialShape to a Shape
|NAN|`ov_shape_to_partial_shape(const ov_shape_t shape, ov_partial_shape_t* partial_shape)`| convert a Shape to PartialShape
|`is_dynamic()`|`ov_partial_shape_is_dynamic(const ov_partial_shape_t partial_shape)`| check this partial shape whether is dynamic
|`std::ostream& operator<<(std::ostream& str, const PartialShape& shape)`|`ov_partial_shape_to_string(const ov_partial_shape_t partial_shape)`| convert a partial shape to readable string

## The mapping relationship for prepostprocess
 * All prepostprocess related APIs implemented with C++ can be found [here](../../../core/include/openvino/core/preprocess/pre_post_process.hpp)
 * All prepostprocess related APIs implemented with C can be found [here](../include/openvino/c/ov_prepostprocess.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`PrePostProcessor(const std::shared_ptr<Model>& function)`|`ov_preprocess_prepostprocessor_create(const ov_model_t* model, ov_preprocess_prepostprocessor_t** preprocess)`| create a prepostprocessor instance
|`~PrePostProcessor()`|`ov_preprocess_prepostprocessor_free(ov_preprocess_prepostprocessor_t* preprocess)`| C need to free the allocated prepostprocessor instance abviously, C++ not
|`PrePostProcessor::input()`|`ov_preprocess_prepostprocessor_get_input_info(const ov_preprocess_prepostprocessor_t* preprocess, ov_preprocess_input_info_t** preprocess_input_info)`| get the input info of prepostprocessor (also provides method get by index/name), the input info need to be free by `ov_preprocess_input_info_free()`
|`PreProcessSteps::resize(ResizeAlgorithm alg)`|`ov_preprocess_preprocess_steps_resize(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps, const ov_preprocess_resize_algorithm_e resize_algorithm)`| add resize operation to model's dimensions
|`InputTensorInfo::set_element_type(const element::Type& type)`|`ov_preprocess_input_tensor_info_set_element_type(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info, const ov_element_type_e element_type)`| set preprocess input tensor precesion
|`InputTensorInfo::set_layout(const Layout& layout)`|`ov_preprocess_input_tensor_info_set_layout(ov_preprocess_input_tensor_info_t* preprocess_input_tensor_info, ov_layout_t* layout)`| set preprocess input tensor layout
|`PrePostProcessor::output()`|`ov_preprocess_prepostprocessor_get_output_info(const ov_preprocess_prepostprocessor_t* preprocess, ov_preprocess_output_info_t** preprocess_output_info)`| get the output info of preprocess output instance (also provides method get by index/name)
|`PrePostProcessor::build()`|`ov_preprocess_prepostprocessor_build(const ov_preprocess_prepostprocessor_t* preprocess, ov_model_t** model)`| adds pre/post-processing operations to function passed in constructor

> **NOTE**: For prepostprocesser, only basic operations are wraped for C, if you need some other methods, please refer [How to wrap OpenVINO interfaces with C](./docs/how_to_wrap_openvino_interfaces_with_c.md) to wrape the interfaces.

## The mapping relationship for tensor
 * All tensor related APIs implemented with C++ can be found [here](../../../core/include/openvino/runtime/tensor.hpp)
 * All tensor related APIs implemented with C can be found [here](../include/openvino/c/ov_tensor.h)

| C++ API | C API | Description |
|:---     |:---   |:---
|`Tensor(const element::Type type, const Shape& shape, const Allocator& allocator = {})`|`ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor)`| create tensor using element type and shape
|`Tensor(const element::Type type, const Shape& shape, void* host_ptr, const Strides& strides = {})`|`ov_tensor_create_from_host_ptr(const ov_element_type_e type, const ov_shape_t shape, void* host_ptr, ov_tensor_t** tensor)`| create tensor using element type and shape and pre allocated memory
|`Tensor::set_shape(const ov::Shape& shape)`|`ov_tensor_set_shape(ov_tensor_t* tensor, const ov_shape_t shape)`| set new shape for tensor
|Tensor::get_shape()|`ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape)`| get shape for tensor
|`Tensor::get_element_type()`|`ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type)`| get type for tensor
|`Tensor::get_size()`|`ov_tensor_get_size(const ov_tensor_t* tensor, size_t* elements_size)`| get the total number of elements
|`Tensor::get_byte_size()`|`ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size)`| the size of the current Tensor in bytes
|`Tensor::data(const element::Type element_type)`|`ov_tensor_data(const ov_tensor_t* tensor, void** data)`| provides an access to the underlaying host memory
|`~Tensor()`|`ov_tensor_free(ov_tensor_t* tensor)`| C need to free the allocated tensor instance abviously, C++ not

## See also
 * [OpenVINO™ README](../../../../README.md)
 * [C API developer guide](../README.md)
 * [C API Reference](https://docs.openvino.ai/latest/api/api_reference.html)