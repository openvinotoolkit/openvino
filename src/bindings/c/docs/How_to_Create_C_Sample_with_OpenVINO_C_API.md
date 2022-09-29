# How to Create C Sample with OpenVINO C API

## Setting Up the development Environment

To configure the environment for developing with OpenVINO C API, OpenVINO provided script automatically detects the installed C API and configures required environment, run:
- On Ubuntu: `source <INSTALL_DIR>/setupvars.sh .`
- On Windows: `setupvars.bat`

For CMakeFile should include C API header file & link library:
```ruby
target_include_directories(${TARGET_NAME} PRIVATE ${OV_CPACK_INCLUDEDIR}/openvino/c)
target_link_libraries(${TARGET_NAME} PRIVATE openvino::runtime::c)
```

## Main Processes for Create Sample with C API 

### Step 1 Include the header file of C API
All C API provided interfaces are included in this file.
```ruby
#include "openvino/c/openvino.h"
```

### Step 2 Create OpenVINO Core "ov_core_t"
Use the following code to create OpenVINO™ Core to manage available devices and read model objects. `ov_core_t` represents an OpenVINO runtime entity and allows you to manipulate with plugins using unified interfaces.
```ruby
ov_core_create(const char* xml_config_file, ov_core_t** core);
```

### Step 3 Read input model to "ov_model_t"
`ov_model_t` represent the input model read from IR/ONNX/PDPD formats.
```ruby
ov_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model);
```

### Step 4 Compile the input model to "ov_compiled_model_t"
`ov_compiled_model_t` represents a device specific compiled model, which allows you to get information inputs or output ports by a tensor name or index. Creates a compiled model from the source model object with device
```ruby
ov_core_compile_model(const ov_core_t* core, const ov_model_t* model, const char* device_name, ov_compiled_model_t** compiled_model, const ov_property_t* property);
```

### Step 5 Create an infer request "ov_infer_request_t"
`ov_infer_request_t` Creates an inference request instance used to infer the network, which provides  methods for model inference in OpenVINO™ Runtime and serves to handle infer requests execution and to set and get output data. Create an infer request using the following code:
```ruby
ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request);
```

### Step 6 Set input tensor for infer request
Sets an input tensor to infer on.
```ruby
ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, size_t idx, const ov_tensor_t* tensor);
```

### Step 7 Do inference synchronously
Infers specified input(s) in synchronous mode.
```ruby
ov_infer_request_infer(ov_infer_request_t* infer_request);
```

### Step 8 Get inference results
Gets an output tensor after infer and process the inference results.
```ruby
ov_infer_request_get_out_tensor(const ov_infer_request_t* infer_request, size_t idx, ov_tensor_t** tensor);
```

### Step 9 Free related object
To avoid memory leak, the allocated objects needed to be freed, like:
```ruby
        ov_infer_request_free(infer_request);       // free infer request
        ov_compiled_model_free(compiled_model);     // free compile model
        ov_tensor_free(tensor);                     // free tensor
        ov_model_free(model);                       // free model
        ov_core_free(core);                         // free core
```
> **NOTE**: the object free need in order, because of the dependent relationship between objects. 