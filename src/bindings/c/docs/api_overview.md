# Overview of OpenVINO C* API

This API provides a simplified interface for OpenVINO functionality that allows to:

- handle the models
- load and configure OpenVINO plugins based on device names
- perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities)

## Supported OSes

Currently the OpenVINO C API is supported on Ubuntu* 18.04/20.04/22.04 Microsoft Windows* 10/11 and CentOS* 7.3/10.15 and above OSes.
Supported Python* versions:

  - Ubuntu 22.04 long-term support (LTS), 64-bit (Kernel 5.15+)
  - Ubuntu 20.04 long-term support (LTS), 64-bit (Kernel 5.15+)
  - Ubuntu 18.04 long-term support (LTS) with limitations, 64-bit (Kernel 5.4+)
  - Windows* 10
  - Windows* 11
  - macOS* 12.6 and above, 64-bit and ARM64
  - Red Hat Enterprise Linux* 8, 64-bit
  - Debian 9 ARM64 and ARM
  - CentOS 7 64-bit

## Setting Up the Environment

To configure the environment for the OpenVINO C* API, run:

- On Ubuntu 20.04/22.04: `source <INSTALL_DIR>/setupvars.sh .`
- On Windows 10/11:

  * `. <path-to-setupvars-folder>/setupvars.ps1` in PowerShell
  * `<INSTALL_DIR>\setupvars.bat ` in Command Prompt

The script automatically detects latest installed C* version and configures required environment if the version is supported.


## Struct

```
typedef struct ov_version {

    const char* buildNumber;

    const char* description;

} ov_version_t;
```

```
typedef struct {

    const char* device_name;

    ov_version_t version;

} ov_core_version_t;
```

```
typedef struct {

    ov_core_version_t* versions;

    size_t size;

} ov_core_version_list_t;
```

```
typedef struct {

    char** devices;

    size_t size;

} ov_available_devices_t;
```

```
typedef struct ov_dimension {

    int64_t min;

    int64_t max;

} ov_dimension_t;
```

```
typedef struct {

    int64_t rank;

    int64_t* dims;

} ov_shape_t;
```

```
typedef struct ov_partial_shape {

    ov_rank_t rank;

    ov_dimension_t* dims;

} ov_partial_shape_t;
```

```
typedef struct {

    enum Status {

        NOT_RUN,

        OPTIMIZED_OUT,

        EXECUTED

    } status;

    int64_t real_time;

    int64_t cpu_time;

    const char* node_name;

    const char* exec_type;

    const char* node_type;

} ov_profiling_info_t;
```

```
typedef struct {

    ov_profiling_info_t* profiling_infos;

    size_t size;

} ov_profiling_info_list_t;
```

```
typedef struct {

    void(OPENVINO_C_API_CALLBACK* callback_func)(void* args);

    void* args;

} ov_callback_t;
```

```
typedef enum {

    UNDEFINED = 0U,  //!< Undefined element type

    DYNAMIC,         //!< Dynamic element type

    BOOLEAN,         //!< boolean element type

    BF16,            //!< bf16 element type

    F16,             //!< f16 element type

    F32,             //!< f32 element type

    F64,             //!< f64 element type

    I4,              //!< i4 element type

    I8,              //!< i8 element type

    I16,             //!< i16 element type

    I32,             //!< i32 element type

    I64,             //!< i64 element type

    U1,              //!< binary element type

    U2,              //!< u2 element type

    U3,              //!< u3 element type

    U4,              //!< u4 element type

    U6,              //!< u6 element type

    U8,              //!< u8 element type

    U16,             //!< u16 element type

    U32,             //!< u32 element type

    U64,             //!< u64 element type

    NF4,             //!< nf4 element type

    F8E4M3,          //!< f8e4m3 element type

    F8E5M3,          //!< f8e5m2 element type

    STRING,          //!< string element type

    F4E2M1,          //!< f4e2m1 element type

    F8E8M0,          //!< f8e8m0 element type

} ov_element_type_e;
```

```
typedef enum {

    OK = 0,  //!< SUCCESS

    GENERAL_ERROR = -1,       //!< GENERAL_ERROR

    NOT_IMPLEMENTED = -2,     //!< NOT_IMPLEMENTED

    NETWORK_NOT_LOADED = -3,  //!< NETWORK_NOT_LOADED

    PARAMETER_MISMATCH = -4,  //!< PARAMETER_MISMATCH

    NOT_FOUND = -5,           //!< NOT_FOUND

    OUT_OF_BOUNDS = -6,       //!< OUT_OF_BOUNDS

    UNEXPECTED = -7,          //!< UNEXPECTED

    REQUEST_BUSY = -8,        //!< REQUEST_BUSY

    RESULT_NOT_READY = -9,    //!< RESULT_NOT_READY

    NOT_ALLOCATED = -10,      //!< NOT_ALLOCATED

    INFER_NOT_STARTED = -11,  //!< INFER_NOT_STARTED

    NETWORK_NOT_READ = -12,   //!< NETWORK_NOT_READ

    INFER_CANCELLED = -13,    //!< INFER_CANCELLED

    INVALID_C_PARAM = -14,         //!< INVALID_C_PARAM

    UNKNOWN_C_ERROR = -15,         //!< UNKNOWN_C_ERROR

    NOT_IMPLEMENT_C_METHOD = -16,  //!< NOT_IMPLEMENT_C_METHOD

    UNKNOW_EXCEPTION = -17,        //!< UNKNOW_EXCEPTION

} ov_status_e;

```

```
typedef enum {

    UNDEFINE = 0U,      //!< Undefine color format

    NV12_SINGLE_PLANE,  //!< Image in NV12 format as single tensor

    NV12_TWO_PLANES,    //!< Image in NV12 format represented as separate tensors for Y and UV planes.

    I420_SINGLE_PLANE,  //!< Image in I420 (YUV) format as single tensor

    I420_THREE_PLANES,  //!< Image in I420 format represented as separate tensors for Y, U and V planes.

    RGB,                //!< Image in RGB interleaved format (3 channels)

    BGR,                //!< Image in BGR interleaved format (3 channels)

    GRAY,               //!< Image in GRAY format (1 channel)

    RGBX,               //!< Image in RGBX interleaved format (4 channels)

    BGRX                //!< Image in BGRX interleaved format (4 channels)

} ov_color_format_e;
```

```
typedef enum {

    RESIZE_LINEAR,  //!< linear algorithm

    RESIZE_CUBIC,   //!< cubic algorithm

    RESIZE_NEAREST  //!< nearest algorithm

} ov_preprocess_resize_algorithm_e;
```

## Properties

### common properties
```
OPENVINO_C_VAR(const char*) ov_property_key_supported_properties;

OPENVINO_C_VAR(const char*) ov_property_key_available_devices;

OPENVINO_C_VAR(const char*) ov_property_key_optimal_number_of_infer_requests;

OPENVINO_C_VAR(const char*) ov_property_key_range_for_async_infer_requests;

OPENVINO_C_VAR(const char*) ov_property_key_range_for_streams;

OPENVINO_C_VAR(const char*) ov_property_key_device_full_name;

OPENVINO_C_VAR(const char*) ov_property_key_device_capabilities;

OPENVINO_C_VAR(const char*) ov_property_key_model_name;

OPENVINO_C_VAR(const char*) ov_property_key_optimal_batch_size;

OPENVINO_C_VAR(const char*) ov_property_key_max_batch_size;

OPENVINO_C_VAR(const char*) ov_property_key_cache_dir;

OPENVINO_C_VAR(const char*) ov_property_key_num_streams;

OPENVINO_C_VAR(const char*) ov_property_key_affinity;

OPENVINO_C_VAR(const char*) ov_property_key_inference_num_threads;

OPENVINO_C_VAR(const char*) ov_property_key_hint_enable_cpu_pinning;

OPENVINO_C_VAR(const char*) ov_property_key_hint_enable_hyper_threading;

OPENVINO_C_VAR(const char*) ov_property_key_hint_performance_mode;

OPENVINO_C_VAR(const char*) ov_property_key_hint_scheduling_core_type;

OPENVINO_C_VAR(const char*) ov_property_key_hint_inference_precision;

OPENVINO_C_VAR(const char*) ov_property_key_hint_num_requests;

OPENVINO_C_VAR(const char*) ov_property_key_log_level;

OPENVINO_C_VAR(const char*) ov_property_key_hint_model_priority;

OPENVINO_C_VAR(const char*) ov_property_key_enable_profiling;

OPENVINO_C_VAR(const char*) ov_property_key_device_priorities;

OPENVINO_C_VAR(const char*) ov_property_key_hint_execution_mode;

OPENVINO_C_VAR(const char*) ov_property_key_force_tbb_terminate;

OPENVINO_C_VAR(const char*) ov_property_key_enable_mmap;

OPENVINO_C_VAR(const char*) ov_property_key_auto_batch_timeout;
```

### AUTO plugin specified properties
```
OPENVINO_C_VAR(const char*) ov_property_key_intel_auto_device_bind_buffer;

OPENVINO_C_VAR(const char*) ov_property_key_intel_auto_enable_startup_fallback;

OPENVINO_C_VAR(const char*) ov_property_key_intel_auto_enable_runtime_fallback;
```

### GPU plugin specified properties
```
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_context_type;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_context_device_id;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_tile_id;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_ocl_queue;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_device;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_shared_mem_type;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_mem_handle;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_dev_object_handle;

OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_va_plane;
```


## OV Core

This struct represents OpenVINO entity and allows you to manipulate with plugins using unified interfaces.

### Create

- `ov_status_e ov_core_create(ov_core_t** core)`

  > Note: Constructs OpenVINO Core instance by default.

  - Parameters:

    - `core` - A pointer to the newly created `ov_core_t`.

  - Return value: Status code of the operation: OK(0) for success.


- `ov_status_e ov_core_create_with_config(const char* xml_config_file, ov_core_t** core)`

  > Note: Constructs OpenVINO Core instance using XML configuration file with devices description.

  - Parameters:

    - `xml_config_file`- A full path to`.xml` file containing plugins configuration. If the parameter is not specified, the default configuration is handled automatically.
    - `core` - A pointer to the newly created `ov_core_t`.

  - Return value: Status code of the operation: OK(0) for success.

  - Usage examples:

    Create an `ov_core_t` t instance with a custom configuration location specified:

    ```
    char *xml_config_file="/localdisk/plugins/my_custom_cfg.xml";
    ov_core_t* core;
    ov_status_e status = ov_core_create_with_config(xml_config_file, &core);
    ```

    .`xml` file has the following structure:

    ```
    <ie>
    	<plugins>
    		<plugin name="" location="" optional="yes/no">
    			<extensions>
    				<extension location=""/>
    			</extensions>
                <properties>
                	<property key="" value=""/>
                </properties>
             </plugin>
        </plugins>
    </ie>
    ```


### Methods

- `ov_status_e ov_get_openvino_version(ov_version_t* version)`

  - Description: Get version of OpenVINO.

  - Parameters:

    - `ov_version_t` - a pointer to the version.

  - Return value: Status  of the operation: OK(0) for success.

  - Usage example:

    ```
      ov_version_t version = {.description = NULL, .buildNumber = NULL};
      ov_get_openvino_version(&version);
      printf("description : %s \n", version.description);
      printf("build number: %s \n", version.buildNumber);
      ov_version_free(&version);
    ```

- `ov_status_e ov_core_read_model(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model)`

  - Description: Reads models from IR / ONNX / PDPD / TF / TFLite formats to create ov_model_t.
    You can create as many ov_model_t as you need and use them simultaneously (up to the limitation of the hardware resources).

  - Parameters:

    - `core` - A pointer to `ov_core_t` instance.
    - `model_path` - Path to a model.
    - `bin_path` - Path to a data file.
    - `model` - A pointer to the newly created model.

  - Return value: Status code of the operation: OK(0) for success.

  - Usage example:

    ```
      ov_core_t* core = NULL;
      ov_core_create(&core);
      ov_model_t* model = NULL;
      ov_core_read_model(core, "model.xml", "model.bin", &model);
    ```

- `ov_status_e ov_core_compile_model(const ov_core_t* core,
                      const ov_model_t* model,
                      const char* device_name,
                      const size_t property_args_size,
                      ov_compiled_model_t** compiled_model,
                      ...);`

  - Description: Creates a compiled model from a source model object.
  - Parameters:
    - `core`- A pointer to `ov_core_t` instance.
    - `model Model` - An object acquired from Core::read_model.
    - `device_name` - Name of a device to load a model to.
    - `property_args_size` - How many properties args will be passed, each property contains 2 args: key and value.
    - `compiled_model` - A pointer to the newly created compiled_model.
    - `...` - property paramater, optional pack of pairs: <char* property_key, char* property_value> relevant only for this load operation operation.

  - Return value: Status code of the operation: OK(0) for success.

  - Usage example:

    ```
      ov_core_t* core = nullptr;
      ov_core_create(&core);
      ov_model_t* model = nullptr;
      ov_core_read_model(core, xml_file_name.c_str(), bin_file_name.c_str(), &model);
      const char* key = ov_property_key_hint_performance_mode;
      const char* num = "LATENCY";
      ov_compiled_model_t* compiled_model = nullptr;
      ov_core_compile_model(core, model, "CPU", 2, &compiled_model, key, num);
      ...
      ov_compiled_model_free(compiled_model);
      ov_model_free(model);
      ov_core_free(core);
    ```

- `ov_status_e ov_core_set_property(const ov_core_t* core, const char* device_name, ...)`

  - Description: Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.
  - Parameters:
    - `core` - A pointer to `ov_core_t` instance.
    - `device_name` - Name of a device.
    - `...` - property paramaters, optional pack of pairs: <char* property_key, char* property_value>.
  - Return value: Status code of the operation: OK(0) for success.

  - Usage example:

    ```
      ov_core_t* core = nullptr;
      ov_core_create(&core);
      const char* key_1 = ov_property_key_inference_num_threads;
      const char* value_1 = "12";
      const char* key_2 = ov_property_key_num_streams;
      const char* value_2 = "7";
      ov_core_set_property(core, "CPU", key_1, value_1, key_2, value_2);
      ...
      ov_core_free(core);
    ```

- `ov_status_e ov_core_get_property(const ov_core_t* core, const char* device_name, const char* property_key, char** property_value)`

  - Description: Gets properties related to device behaviour.
  - Parameters:
    - `core` - A pointer to `ov_core_t` instance.
    - `device_name` - Name of a device.
    - `property_key` - Property key.
    - `property_value` - A pointer to property value with string format.
  - Return value: Status code of the operation: 0 for success.

  - Usage example:

    ```
      ov_core_t* core = nullptr;
      ov_core_create(&core);
      const char* key = ov_property_key_hint_performance_mode;
      const char* mode = "LATENCY";
      ov_core_set_property(core, "CPU", key, mode);
      char* ret = nullptr;
      ov_core_get_property(core, "CPU", key, &ret);
      ov_free(ret);
      ...
      ov_core_free(core);
    ```

- `ov_status_e ov_core_import_model(const ov_core_t* core,
                     const char* content,
                     const size_t content_size,
                     const char* device_name,
                     ov_compiled_model_t** compiled_model);`

  - Description: Imports a compiled model from the previously exported one.
  - Parameters:
    - `core` -  A pointer `ov_core_t` instance.
    - `content` - A pointer to content of the exported model.
    - `content_size` - Number of bytes in the exported network.
    - `device_name` - Name of a device to import a compiled model for.
    - `compiled_model` - A pointer to the newly created compiled_model.

  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_core_get_versions_by_device_name(const ov_core_t* core, const char* device_name, ov_core_version_list_t* versions)`

  - Description: Returns device plugins version information.
  - Parameters:
    - `core` - A pointer `ov_core_t` instance.
    - `device_name` -  A device name to identify a plugin.
    - `versions` - A pointer to versions corresponding to device_name.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_core_create_context(const ov_core_t* core,
                       const char* device_name,
                       const size_t context_args_size,
                       ov_remote_context_t** context,
                       ...);`

  - Description: Creates a new remote shared context object on the specified accelerator device using specified plugin-specific low-level device API parameters (device handle, pointer, context, etc.).
  - Parameters:
    - `core` - A pointer `ov_core_t` instance.
    - `device_name` - Device name to identify a plugin.
    - `context_args_size` - How many property args will be for this remote context creation.
    - `context` - A pointer to the newly created remote context.
    - `...` - variadic parmameters Actual property parameter for remote context
  -  Return value: Status code of the operation: OK(0) for success.


- `ov_status_e ov_core_compile_model_with_context(const ov_core_t* core,
                                   const ov_model_t* model,
                                   const ov_remote_context_t* context,
                                   const size_t property_args_size,
                                   ov_compiled_model_t** compiled_model,
                                   ...);`

  - Description: Creates a compiled model from a source model within a specified remote context.

  - Parameters:
    - `core` - A pointer `ov_core_t` instance.
    - `model` - Model object acquired from ov_core_read_model.
    - `context` - A pointer to the newly created remote context.
    - `property_args_size` - How many args will be for this compiled model.
    - `compiled_model` - A pointer to the newly created compiled_model.
    - `...` - variadic parmameters Actual property parameter for remote context

  - Return value: Status code of the operation: OK(0) for success.

## OV Model

This struct contains the information about the model read from IR and allows you to manipulate with some model parameters such as layers affinity and output layers.

### Methods

- `ov_status_e ov_model_free(ov_model_t* model)`
  - Description: Release the memory allocated by ov_model_t.
  - Parameters:
    - `model` -  A pointer to the ov_model_t to free memory..
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_const_input(const ov_model_t* model, ov_output_const_port_t** input_port);`
  - Description: Get a const input port of ov_model_t,which only support single input model.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `input_port` - A pointer to the `ov_output_const_port_t`.
  - Return value: Status code of the operation: OK(0) for success.

-  `ov_status_e ov_model_input(const ov_model_t* model, ov_output_port_t** input_port);`
  - Description: Get single input port of ov_model_t, which only support single input model.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `input_port` - A pointer to the `ov_output_port_t`.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_input_by_name(const ov_model_t* model, const char* tensor_name, ov_output_port_t** input_port)`
  - Description: Get an input port of ov_model_t by name.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `tensor_name` - Input tensor name (char *).
    - `input_port` - A pointer to the `ov_output_port_t`.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_input_by_index(const ov_model_t* model, const size_t index, ov_output_port_t** input_port)`
  - Description: Get an input port of ov_model_t by name.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `index` - Input tensor index.
    - `input_port` - A pointer to the `ov_output_port_t`.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_const_output(const ov_model_t* model, ov_output_const_port_t** output_port);`
  - Description: Get a single const output port of ov_model_t, which only support single output model.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `output_port` - A pointer to the ov_output_const_port_t.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_output(const ov_model_t* model, ov_output_port_t** output_port);`
  - Description: Get a single output port of ov_model_t, which only support single output model.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `output_port` - A pointer to the ov_output_port_t.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_inputs_size(const ov_model_t* model, size_t* input_size);`
  - Description: Get the input size of ov_model_t.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `input_size` - The model's input size.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_outputs_size(const ov_model_t* model, size_t* output_size);`
  - Description: Get the input size of ov_model_t.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `output_size` - The model's output size.
  - Return value: Status code of the operation: OK(0) for success.

- `bool ov_model_is_dynamic(const ov_model_t* model)`
  - Description: Returns true if any of the ops defined in the model is dynamic shape.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
  - Return value: true if model contains dynamic shapes.

- `ov_status_e ov_model_reshape(const ov_model_t* model,
                 const char** tensor_names,
                 const ov_partial_shape_t* partial_shapes,
                 size_t size)`
  - Description: Do reshape in model with a list of <name, partial shape>.
  - Parameters:
    - `model` - A pointer to the ov_model_t.
    - `tensor_names` - The list of input tensor names.
    - `partialShape` - A PartialShape list.
    - `size` - The item count in the list.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_model_get_friendly_name(const ov_model_t* model, char** friendly_name)`
  - Description: Gets the friendly name for a model.
  - Parameters:
     - `model` - A pointer to the ov_model_t.
    - `friendly_name` - The model's friendly name.
  - Return value: Status code of the operation: OK(0) for success.


## Node

This struct contains the information about the model's port.

### Methods

- `ov_status_e ov_const_port_get_shape(const ov_output_const_port_t* port, ov_shape_t* tensor_shape)`
  - Description: Get the shape of port object.
  - Parameters:
    - `port` - A pointer to ov_output_const_port_t.
    - `tensor_shape` - Returned tensor shape.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_port_get_shape(const ov_output_port_t* port, ov_shape_t* tensor_shape)`
  - Description: Get the shape of port object.
  - Parameters:
    - `port` - A pointer to ov_output_port_t.
    - `tensor_shape` - Returned tensor shape.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_port_get_any_name(const ov_output_const_port_t* port, char** tensor_name)`
  - Description: Get the tensor name of port.
  - Parameters:
    - `port` - A pointer to ov_output_port_t.
    - `tensor_name` - Returned tensor name.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_port_get_partial_shape(const ov_output_const_port_t* port, ov_partial_shape_t* partial_shape)`
  - Description: Get the partial shape of port.
  - Parameters:
    - `port` - A pointer to ov_output_const_port_t.
    - `partial_shape` - Partial shape.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_port_get_element_type(const ov_output_const_port_t* port, ov_element_type_e* tensor_type)`
  - Description: Get the tensor type of port.
  - Parameters:
    - `port` - A pointer to ov_output_const_port_t.
    - `tensor_type` - Returned tensor type.
  - Return value: Status code of the operation: OK(0) for success.

- `void ov_output_port_free(ov_output_port_t* port)`
  - Description: free port object.
  - Parameters:
    - `port` - A pointer to ov_output_port_t.
  - Return value: no return.

- `void ov_output_const_port_free(ov_output_const_port_t* port)`
  - Description: free const port object.
  - Parameters:
    - `port` - A pointer to ov_output_const_port_t.
  - Return value: no return.

## CompiledModel

This struct represents a compiled model instance loaded to plugin and ready for inference.

### Methods

- `ov_status_e ov_compiled_model_inputs_size(const ov_compiled_model_t* compiled_model, size_t* size)`
  - Description: Get the input size of ov_compiled_model_t.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `input_size` - The compiled_model's input size.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_input(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** input_port)`
  - Description: - Get the single const input port of ov_compiled_model_t, which only support single input model.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `input_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_input_by_index(const ov_compiled_model_t* compiled_model,
                                 const size_t index,
                                 ov_output_const_port_t** input_port)`
  - Description: Get a const input port of ov_compiled_model_t by port index.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `index`: Input index.
    - `input_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_input_by_name(const ov_compiled_model_t* compiled_model,
                                const char* name,
                                ov_output_const_port_t** input_port)`
  - Description: - Get a const input port of ov_compiled_model_t by name.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `name` - input tensor name.
    - `input_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_compiled_model_outputs_size(const ov_compiled_model_t* compiled_model, size_t* size)`
  - Description: - Get the output size of ov_compiled_model_t.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `size` - The compiled_model's output size.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_output(const ov_compiled_model_t* compiled_model, ov_output_const_port_t** output_port)`
  - Description: - Get the single const output port of ov_compiled_model_t, which only support single output model.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `output_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_output_by_index(const ov_compiled_model_t* compiled_model,
                                  const size_t index,
                                  ov_output_const_port_t** output_port)`
  - Description: Get a const input port of ov_compiled_model_t by port index.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `index`: Output index.
    - `output_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_output_by_name(const ov_compiled_model_t* compiled_model,
                                 const char* name,
                                 ov_output_const_port_t** output_port)`
  - Description: - Get a const output port of ov_compiled_model_t by name.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `name` - input tensor name.
    - `output_port` - A pointer to the `ov_output_const_port_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_get_runtime_model(const ov_compiled_model_t* compiled_model, ov_model_t** model)`
  - Description: - Gets runtime model information from a device.
  - Parameters:
    - `compiled_model` - A pointer to the `ov_compiled_model_t` instance.
    - `model` - A pointer to the `ov_model_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_create_infer_request(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request)`
  - Description: - Creates an inference request object used to infer the compiled model.
  - Parameters:
    - `compiled_model` - A pointer to `ov_compiled_model_t` instance.
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_set_property(const ov_compiled_model_t* compiled_model, ...)`
  - Description: - Sets properties for a device, acceptable keys can be found in ov_property_key_xxx.
  - Parameters:
    - `compiled_model` - A pointer to `ov_compiled_model_t` instance.
    - `...` variadic paramaters, the format is <char *property_key, char* property_value>.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_get_property(const ov_compiled_model_t* compiled_model,
                               const char* property_key,
                               char** property_value)`
  - Description: - Gets properties for current compiled model.
  - Parameters:
    - `compiled_model` - A pointer to `ov_compiled_model_t` instance.
    - `property_key` - Property key.
    - `property_value` - A pointer to property value.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_compiled_model_export_model(const ov_compiled_model_t* compiled_model, const char* export_model_path)`
  - Description: - Exports the current compiled model to an output stream `std::ostream`.
  - Parameters:
    - `compiled_model` - A pointer to `ov_compiled_model_t` instance.
    - `export_model_path` - Path to the file.
  - Return value: Status code of the operation: OK(0) for success.

- `void ov_compiled_model_free(ov_compiled_model_t* compiled_model)`
  - Description: - Release the memory allocated by ov_compiled_model_t`.
  - Parameters:
    - `compiled_model` - A pointer to `ov_compiled_model_t` instance.
  - Return value: None

## InferRequest

This struct provides an interface to infer requests of `ov_compiled_model_t` and serves to handle infer requests execution and to set and get output data.

### Methods

- `ov_status_e ov_infer_request_set_tensor(ov_infer_request_t* infer_request, const char* tensor_name, const ov_tensor_t* tensor)`

  - Description: Set an input/output tensor to infer on by the name of tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `tensor_name` - Name of the input or output tensor.
    - `tensor` - Reference to the tensor.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_tensor_by_port(ov_infer_request_t* infer_request,
                                    const ov_output_port_t* port,
                                    const ov_tensor_t* tensor)`

  - Description: Set an input/output tensor to infer request for the port.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `port ` -   Port of the input or output tensor, which can be got by calling ov_model_t/ov_compiled_model_t interface.
    - `tensor` - Reference to the tensor.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_input_tensor_by_index(ov_infer_request_t* infer_request,
                                           const size_t idx,
                                           const ov_tensor_t* tensor)`

  - Description: Set an input tensor to infer on by the index of tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `idx` - Index of the input port.
    - `tensor` - Reference to the tensor.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_input_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor)`

  - Description: Set an input tensor for the model with single input to infer on.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `tensor` - Reference to the tensor.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_output_tensor_by_index(ov_infer_request_t* infer_request,
                                            const size_t idx,
                                            const ov_tensor_t* tensor)`

  - Description: Set an output tensor to infer by the index of output tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `idx` - Index of the input port.
    - `tensor` - Reference to the tensor.
  - Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_output_tensor(ov_infer_request_t* infer_request, const ov_tensor_t* tensor)`

  - Description: Set an output tensor to infer models with single output.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `tensor` - Reference to the tensor.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_get_tensor(const ov_infer_request_t* infer_request, const char* tensor_name, ov_tensor_t** tensor)`

  - Description: Get an input/output tensor by the name of tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `tensor_name` - Name of the input or output tensor.
    - `tensor` - Reference to the tensor.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_get_tensor_by_port(const ov_infer_request_t* infer_request,
                                    const ov_output_port_t* port,
                                    ov_tensor_t** tensor)`

  - Description: Get an input/output tensor by port.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `port ` -  Port of the tensor to get.
    - `tensor` - Reference to the tensor.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_get_input_tensor_by_index(const ov_infer_request_t* infer_request,
                                           const size_t idx,
                                           ov_tensor_t** tensor)`

  - Description: Get an input tensor by the index of input tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `idx ` - Index of the tensor to get.
    - `tensor` - Reference to the tensor.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_get_output_tensor_by_index(const ov_infer_request_t* infer_request,
                                            const size_t idx,
                                            ov_tensor_t** tensor)`

  - Description: Get an output tensor by the index of output tensor.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `idx ` - Index of the tensor to get.
    - `tensor` - Reference to the tensor.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_infer(ov_infer_request_t* infer_request)`

  - Description: Infer specified input(s) in synchronous mode.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_start_async(ov_infer_request_t* infer_request)`

  - Description: Start inference of specified input(s) in asynchronous mode.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_cancel(ov_infer_request_t* infer_request)`

  - Description: Cancel inference request.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_wait_for(ov_infer_request_t* infer_request, const int64_t timeout);`

  - Description: Waits for the result to become available. Blocks until the specified timeout has elapsed or the result becomes available, whichever comes first.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `timeout` - Maximum duration, in milliseconds, to block for.
  - Return value:  Status code of the operation: OK(0) for success.

- `ov_status_e ov_infer_request_set_callback(ov_infer_request_t* infer_request, const ov_callback_t* callback)`

  - Description: Set callback function, which will be called when inference is done.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `callback`  A function to be called.
  - Return value:  Status code of the operation: OK(0) for success.

- `void ov_infer_request_free(ov_infer_request_t* infer_request)`

  - Description: Release the memory allocated by ov_infer_request_t.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
  - Return value:  None.

- `void ov_infer_request_get_profiling_info(const ov_infer_request_t* infer_request, ov_profiling_info_list_t* profiling_infos)`

  - Description: Query performance measures per layer to identify the most time consuming operation.
  - Parameters:
    - `infer_request` - A pointer to `ov_infer_request_t` instance.
    - `profiling_infos` - Vector of profiling information for operations in a model.
  - Return value:  None.

## Tensor

### Methods

- `ov_status_e ov_tensor_create_from_host_ptr(const ov_element_type_e type,
                               const ov_shape_t shape,
                               void* host_ptr,
                               ov_tensor_t** tensor)`
  - Description: Constructs Tensor using element type, shape and external host ptr.
  - Parameters:
    - `type` - Tensor element type
    - `shape` - Tensor shape
    - `host_ptr` - Pointer to pre-allocated host memory
    - `tensor` - A point to ov_tensor_t
  -  Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_tensor_create(const ov_element_type_e type, const ov_shape_t shape, ov_tensor_t** tensor)`
  - Description: Constructs Tensor using element type and shape. Allocate internal host storage using default allocator.
  - Parameters:
    - `type` - Tensor element type
    - `shape` - Tensor shape
    - `tensor` - A point to ov_tensor_t
  -  Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_tensor_get_shape(const ov_tensor_t* tensor, ov_shape_t* shape)`
  - Description: Get shape for tensor.
  - Parameters:
    - `tensor` - A point to ov_tensor_t
    - `shape` - Tensor shape
  -  Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_tensor_get_element_type(const ov_tensor_t* tensor, ov_element_type_e* type)`
  - Description: Get type for tensor.
  - Parameters:
    - `tensor` - A point to ov_tensor_t
    - `type` - Tensor element type.
  -  Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_tensor_get_byte_size(const ov_tensor_t* tensor, size_t* byte_size)`
  - Description: Get byte size for tensor.
  - Parameters:
    - `tensor` - A point to ov_tensor_t
    - `byte_size` - The size of the current Tensor in bytes.
  -  Return value: Status code of the operation: OK(0) for success.

- `ov_status_e ov_tensor_data(const ov_tensor_t* tensor, void** data)`
  - Description: Provides an access to the underlaying host memory.
  - Parameters:
    - `tensor` - A point to ov_tensor_t
    - `data` - A point to host memory.
  -  Return value: Status code of the operation: OK(0) for success.

- `void ov_tensor_free(ov_tensor_t* tensor)`
  - Description: Free ov_tensor_t.
  - Parameters:
    - `tensor` - A point to ov_tensor_t
  -  Return value: None.
