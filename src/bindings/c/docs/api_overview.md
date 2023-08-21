# Overview of Inference Engine C* API

> **NOTE**: It is a preview version of the Inference Engine C* API for evaluation purpose only.
> Module structure and API itself may be changed in future releases.

This API provides a simplified interface for Inference Engine functionality that allows to:

- handle the models
- load and configure Inference Engine plugins based on device names
- perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities)

## Supported OSes

Currently the Inference Engine C* API is supported on Ubuntu* 16.04, Microsoft Windows* 10 and CentOS* 7.3 OSes.
Supported Python* versions:

- On Ubuntu 16.04: 2.7, 3.5, 3.6
- On Windows 10: 3.5, 3.6
- On CentOS 7.3: 3.4, 3.5, 3.6

## Setting Up the Environment

To configure the environment for the Inference Engine C* API, run:

- On Ubuntu 16.04: `source <INSTALL_DIR>/setupvars.sh .`
- On Windows 10: XXXX

The script automatically detects latest installed C* version and configures required environment if the version is supported.
If you want to use certain version of C*, set the environment variable XXXXX
after running the environment configuration script.

## Struct

```
typedef struct ie_core_version {

​    size_t major;

​    size_t minor;

​    const char *build_number;

​    const char *description;

}ie_core_version_t;
```

```
typedef struct ie_config {

​    char *name;

​    char *value;

}ie_config_t;
```

```
typedef struct ie_param {

​    union { //To be continue, to collect metric and config parameters

​    };

}ie_param_t;
```

```
typedef struct ie_param_config {

​    char *name;

​    ie_param_t *param;

}ie_param_config_t;
```

```
typedef struct desc {

​    char msg[256];

}desc_t;
```

```
typedef struct dimensions {

​    size_t dims[8];

}dimensions_t;
```

```
struct tensor_desc {

​    layout_t layout;

​    dimensions_t dims;

​    precision_e precision;

};


```

```
typedef void (*completeCallBackFunc)(ie_infer_request_t *infer_request, int *status);
```

```
enum precision_e{

​    UNSPECIFIED = 255, /**< Unspecified value. Used by default */

​    MIXED = 0,  /**< Mixed value. Can be received from network. No applicable for tensors */

​    FP32 = 10,  /**< 32bit floating point value */

​    FP16 = 11,  /**< 16bit floating point value */

    BF16 = 12,  /**< 16bit floating point value, 8 bit for exponent, 7 bit for mantisa*/

    FP64 = 13,  /**< 64bit floating point value */

​    Q78 = 20,   /**< 16bit specific signed fixed point precision */

​    I16 = 30,   /**< 16bit signed integer value */

​    U4 = 39,    /**< 4bit unsigned integer value */

​    U8 = 40,    /**< 8bit unsigned integer value */

​    I4 = 49,    /**< 4bit signed integer value */

​    I8 = 50,    /**< 8bit signed integer value */

​    U16 = 60,   /**< 16bit unsigned integer value */

​    I32 = 70,   /**< 32bit signed integer value */

​    I64 = 72,   /**< 64bit signed integer value */

​    U64 = 73,   /**< 64bit unsigned integer value */

​    U32 = 74,   /**< 32bit unsigned integer value */

​    BIN = 71,   /**< 1bit integer value */

​    CUSTOM = 80 /**< custom precision has it's own name and size of elements */

};
```

```
enum layout_t {

​    ANY = 0,	// "any" layout

​    // I/O data layouts

​    NCHW = 1,

​    NHWC = 2,

​    NCDHW = 3,

​    NDHWC = 4,

​    // weight layouts

​    OIHW = 64,

​    // Scalar

​    SCALAR = 95,

​    // bias layouts

​    C = 96,

​    // Single image layout (for mean image)

​    CHW = 128,

​    // 2D

​    HW = 192,

​    NC = 193,

​    CN = 194,


​    BLOCKED = 200,

};
```

```
enum colorformat_e {

​    RAW = 0u,    ///< Plain blob (default), no extra color processing required

​    RGB,         ///< RGB color format

​    BGR,         ///< BGR color format, default in OpenVINO

​    GRAY,        ///< GRAY color format

​    RGBX,        ///< RGBX color format with X ignored during inference

​    BGRX,        ///< BGRX color format with X ignored during inference
};
```

```
enum resize_alg_e {

​    NO_RESIZE = 0,

​    RESIZE_BILINEAR,

​    RESIZE_AREA

};
```

```
struct roi_e {

​    size_t id;     // ID of a roi

​    size_t posX;   // W upper left coordinate of roi

​    size_t posY;   // H upper left coordinate of roi

​    size_t sizeX;  // W size of roi

​    size_t sizeY;  // H size of roi

};
```

```
enum IEStatusCode {

​    OK = 0,

​    GENERAL_ERROR = -1,

​    NOT_IMPLEMENTED = -2,

​    NETWORK_NOT_LOADED = -3,

​    PARAMETER_MISMATCH = -4,

​    NOT_FOUND = -5,

​    OUT_OF_BOUNDS = -6,

​    /*

​     \* @brief exception not of std::exception derived type was thrown

​     */

​    UNEXPECTED = -7,

​    REQUEST_BUSY = -8,

​    RESULT_NOT_READY = -9,

​    NOT_ALLOCATED = -10,

​    INFER_NOT_STARTED = -11,

​    NETWORK_NOT_READ = -12

};
```



- `const char *ie_c_api_version(void)`

  - Description: Returns number of version that is exported.

  - Parameters: None.

  - Return value: Version number of the API.

  - Usage example:

    ```
    const char *ver_num=ie_c_api_version();
    ```

## IECore

This strcut represents an Inference Engine entity and allows you to manipulate with plugins using unified interfaces.

### Create

- `IEStatusCode ie_core_create(char *xml_config_file, ie_core_t *core_result)`

  > Note: create an ie_core_t instance with default configuration when xml_config_file=null.

  - Parameters:

    - `xml_config_file`- A full path to`.xml` file containing plugins configuration. If the parameter is not specified, the default configuration is handled automatically.
    - `core_result` - A pointer to the newly created `ie_core_t`.

  - Return value: Status code of the operation: OK(0) for success.

  - Usage examples:

    Create an `ie_core_t` t instance with a custom configuration location specified:

    ```
    char *xml_config_file="/localdisk/plugins/my_custom_cfg.xml";
    ie_core_t ie;
    IEStatusCode status = ie_core_create(xml_config_file,ie);
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


### <a name="iecore-methods"></a>Methods

- `IEStatusCode ie_core_get_versions(ie_core_t *core, char *device_name, ie_core_version_t *version_result)`

  - Description: Returns a `ie_core_version_t` with versions of the plugin specified.

  - Parameters:

    - `core` -A pointer to `ie_core_t` instance.
    - `device_name` - Name of the registered plugin.
  - `version_result` - Dictionary mapping a plugin name .

  - Return value: Status  of the operation: OK(0) for success.

  - Usage example:

    ```
    char *xml_config_file="/localdisk/plugins/my_custom_cfg.xml";
  char *device_name="CPU";
    ie_core_t *ie;
  ie_core_version_t *version;
    IEStatusCode status= ie_core_create(xml_config_file, ie);
    IEStatusCode status2=ie_core_get_versions(ie,device_name, version);
    print("description:%s, major:%d, minor:%d, build_number:%s.\n",version-		  >description, version->major, version->minor, version->build_number);
    ```

- `IEStatusCode ie_core_load_network(ie_core_t *core, ie_network_t *network, const char *device_name,  ie_config_t config, ie_executable_network_t *exec_network_result)`

  - Description: Loads a network that was read from the Intermediate Representation (IR) to the plugin with specified device name and creates an `ie_executable_network_t` instance of the `ie_network_t` struct.
    You can create as many networks as you need and use them simultaneously (up to the limitation of the hardware resources).

  - Parameters:

    - `core` - A pointer to `ie_core_t` instance.
    - `network` - A pointer to `ie_network_t` instance.
    - `device_name` - A device name of a target plugin.
    - `config` - A dictionary of plugin configuration keys and their values.
    - `exec_network_result` - A pointer to the newly loaded network.

  - Return value: Status code of the operation: OK(0) for success.

  - Usage example:

    ```

    ```

- `IEStatusCode ie_core_set_config(ie_core_t *core, ie_config_t *ie_core_config, const char *device_name)`

  - Description: Sets a configuration for a plugin.
  - Parameters:
    - `core`- A pointer to `ie_core_t` instance.
    - `ie_core_config` - A dictionary of configuration parameters as keys and their values.
    - `device_name` - A device name of a target plugin.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_core_register_plugin(ie_core_t *core, const char *plugin, const char *device_name )`

  - Description: Registers a new device and a plugin which implement this device inside Inference Engine.
  - Parameters:
    - `core` - A pointer to `ie_core_t` instance.
    - `plugin` - A path (absolute or relative) or name of a plugin. Depending on platform, plugin is wrapped with shared library suffix and prefix to identify library full name
    - `device_name` - A target device name for the plugin. If not specified, the method registers.
      a plugin with the default name.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_core_register_plugins(ie_core_t *core, const char *xml_config_file)`

  - Description: Registers plugins specified in an `.xml` configuration file
  - Parameters:
    - `core` - A pointer to `ie_core_t` instance.
    - `xml_config_file` -  A full path to `.xml` file containing plugins configuration.
  - Return value: Status code of the operation: 0 for success.

- `IEStatusCode ie_core_unregister_plugin(ie_core_t *core, const char *device_name)`

  - Description: Unregisters a plugin with a specified device name
  - Parameters:
    - `core` -  A pointer `ie_core_t` instance.
    - `device_name` -  A device name of the plugin to unregister.

  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_core_add_extension(ie_core_t *core, const char *extension_path, const char *device_name)`

  - Description:  Loads extension library to the plugin with a specified device name.
  - Parameters:
    - `core` - A pointer `ie_core_t` instance.
    - `extension_path` -  Path to the extensions library file to load to a plugin.
    - `device_name` -  A device name of a plugin to load the extensions to.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_core_get_metric(ie_core_t *core, const char *device_name, const char *metric_name, ie_param_t *param_result)`

  - Description: Gets a general runtime metric for dedicated hardware. Enables to request common device properties, which are `ie_executable_network_t` agnostic, such as device name, temperature, and other devices-specific values.
  - Parameters:
    - `core` - A pointer `ie_core_t` instance.
    - `device_name` - A name of a device to get a metric value.
    - `metric_name` - A metric name to request.
    - `param_result` - A metric value corresponding to a metric key.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_core_get_config(ie_core_t *core, const char *device_name, const char *config_name, ie_param_t *param_result)`

  - Description:  Gets a configuration dedicated to device behavior. The method targets to extract information which can be set via SetConfig method.

  - Parameters:
    - `core` - A pointer `ie_core_t` instance.
    - `device_name` - A name of a device to get a metric value.
    - `config_name` - A configuration value corresponding to a configuration key.
    - `param_result` - A metric value corresponding to a metric key.

  - Return value: Status code of the operation: OK(0) for success.



## IENetwork

This struct contains the information about the network model read from IR and allows you to manipulate with some model parameters such as layers affinity and output layers.

### Methods

- `IEStatusCode ie_network_read(char *xml, char *weights_file, ie_network_t *network_result)`
  - Description: Reads the model from the `.xml` and `.bin` files of the IR.
  - Parameters:
    - `xml_file` -  `.xml` file's path of the IR.
    - `weights_file` - `.bin` file's path of the IR.
    - `network_result` - A pointer to the newly created network.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_free(ie_network_t *network)`
  - Description: When network is loaded into the Inference Engine, it is not required anymore and should be released.
  - Parameters:
    - `network` - The pointer to the instance of the `ie_network_t` to free.
  - Return value: Status code of the operation: OK(0) for success.
-  `IEStatusCode ie_network_get_input_numbers(ie_network_t *network, size_t *size_result)`
  - Description: Gets number of inputs for the `IENetwork` instance.
  - Parameters:
    - `network` - The instance of the `ie_network_t` to get size of input information for this instance.
    - `size_result` - A number of the instance's input information.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_input_name(ie_network_t *network, size_t number, char *name_result)`
  - Description: Gets name corresponding to the "number".
  - Parameters:
    - `network` - The instance of the `ie_network_t` to get input information.
    - `number` - An id of input  information .
    - `name_result` - Input name corresponding to the "number".
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_output_numbers(ie_network_t *network, size_t size_result)`
  - Description: Gets number of output for the `ie_network_t` instance.
  - Parameters:
    - `network` - The instance of the `ie_network_t` to get size of output information for this instance.
    - `size_result` - A number of the instance's output information.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_output_name(ie_network_t *network, size_t number, char *name_result)`
  - Description: Gets output name corresponding to the "number".
  - Parameters:
    - `network` - The instance of the `ie_network_t` to get out information of nth layer for this instance.
    - `number` - An id of output  information.
    - `name_result` - A output name corresponding to the "number".
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_input_precision(ie_network_t *network, char *input_name, precision_e *prec_result)`
  - Description: Gets a precision of the input data named "input_name".
  - Parameters:
    - `network` - A pointer to ie_network_t instance.
    - `input_name` - Name of input data.
    - `prec_result` - A pointer to the precision used for input blob creation.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_input_precision(ie_network_t *network, char *input_name, precision_e p)`
  - Description: Changes the precision of the input data named "input_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `p` - A new precision of the input data to set (eg. precision_e.FP16).
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_input_layout(ie_network_t *network, char *input_name, layout_t *layout_result)`
  - Description: Gets a layout of the input data named "input_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `layout_result` - A pointer to the layout used for input blob creation.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_input_layout(ie_network_t *network, char *input_name, layout_t l)`
  - Description: Changes the layout of the input data named "input_name". This function should be called before loading the network to the plugin
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` -  Name of input data.
    - `layout` - Network layer layout (eg. layout_t.NCHW).
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_input_dims(ie_network_t *network, char *input_name, dimensions_t *dims_result)`
  - Description: Gets dimensions/shape of the input data with reversed order.
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `dims_result` - A pointer to the dimensions used for input blob creation.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_input_resize_algorithm(ie_network_t *network, char *input_name, resize_alg_e *resize_alg_result)`
  - Description: Gets pre-configured resize algorithm.
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `resize_alg_result` - The pointer to the resize algorithm used for input blob creation.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_input_resize_algorithm(ie_network_t *network, char *input_name, resize_alg_e resize_algo)`
  - Description: Sets resize algorithm to be used during pre-processing
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `resize_algo` - Resize algorithm.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_color_format(ie_network_t *network, char *input_name, colorformat_e *colformat_result)`
  - Description: Gets color format of the input data named "input_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input` - Name of input data.
    - `colformat_result` - Input color format of the input data named "input_name".
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_color_format(ie_network_t *network, char *input_name, colorformat_e color_format)`
  - Description: Changes the color format of the input data named "input_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `input_name` - Name of input data.
    - `color_format` - Color format of the input data .
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_output_precision(ie_network_t *network, char *output_name, precision_e *prec_result)`
  - Description: Get output precision of the output data named "output_name".
  - Parameters:
    - `network` - A pointer `ie_network_t` instance.
    - `output_name` - Name of output date.
    - `precision_e` - Output precision of the output data named "output_name".
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_output_precision(ie_network_t *network, char *output_name, precision_e p)`
  - Description: Sets a precision type of the output date named "output_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `outputName` - Name of output data.
    - `p` - Precision of the output data (eg. precision_e.FP16).
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_output_layout(ie_network_t *network, char *output_name, layout_t *layout_result)`
  - Description: Get output layout of the output date named "output_name" in the network.
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `output_name` - Name of output data.
    - `layout_result` - Layout value of the output data named "output_name".
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_set_output_layout(ie_network_t *network, char *output_name, c l)`
  - Description: Sets the layout value for output data named "output_name".
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `output_name` - Name of output data.
    - `l` - Layout value to set (eg. output_name.NCHW).
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_network_get_output_dims(ie_network_t *network, char *output_name, dimensions_t *dims_result)`
  - Description: Get output dimension of output data named "output_name" in the network.
  - Parameters:
    - `network` - A pointer to `ie_network_t` instance.
    - `output_name` - Name of output data.
    - `dims_result` - Dimensions value of the output data named "output_name".
  - Return value: Status code of the operation: OK(0) for success.

## ExecutableNetwork

This struct represents a network instance loaded to plugin and ready for inference.

### Methods

- `IEStatusCode ie_exec_network_create_infer_request(ie_executable_network_t *ie_exec_network, desc_t *desc, ie_infer_request_t **req)`

  - Description:  Creates an inference request instance used to infer the network. The created request has allocated input and output blobs (that can be changed later).
  - Parameters:
    - `ie_exec_network` - A pointer to `ie_executable_network_t` instance.
    - `desc` - A pointer to a `desc_t` instance.
    - `req`  - A pointer to the newly created `ie_infer_request_t` instance.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_exec_network_get_metric(ie_executable_network_t *ie_exec_network, const char *metric_name, ie_param_t *param_result)`

  - Description: - Gets general runtime metric for an executable network. It can be network name, actual device ID on which executable network is running or all other properties which cannot be changed dynamically.
  - Parameters:
    - `ie_exec_network`: A pointer to `ie_executable_network_t` instance.
    - `metric_name` - A metric name to request.
    - `param_result` - A metric value corresponding to a metric key.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_exec_network_set_config(ie_executable_network_t *ie_exec_network, ie_param_config_t *param_config, desc_t *desc)`

  - Description: Sets a configuration for current executable network.
  - Parameters:
    - `ie_exec_network`: A pointer to `ie_executable_network_t` instance.
    - `config`:  An  config for current executable network.
  - Return value: Status code of the operation: OK(0) for success.
- `IEStatusCode ie_exec_network_get_config(ie_executable_network_t *ie_exec_network, const char *metric_config, ie_param_t *param_result)`

  - Description: - Gets configuration for current executable network. The method is responsible to extract information
    - which affects executable network execution
  - Parameters:
    - `ie_exec_network` - A pointer to `ie_executable_network_t` instance.
    - `metric_config` - A configuration parameter name to request.
    - `param_result` - A configuration value corresponding to a configuration key.
  - Return value: Status code of the operation: OK(0) for success.




## InferRequest

This struct provides an interface to infer requests of `ExecutableNetwork` and serves to handle infer requests execution and to set and get output data.

### Methods

- `IEStatusCode *ie_infer_request_get_blob(ie_infer_request_t *infer_request, const char *name, ie_blob_t **blob_result)`

  - Description: Get a Blob corresponding to blob name.
  - Parameters:
    - `infer_request` - A pointer to `ie_infer_request_t` instance
    - `name` - Blob name.
    -  `blob_result` - A pointer to the blob corresponding to the blob name.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_infer_request_set_blob(ie_infer_request_t *infer_request, ie_blob_t *blob)`

  - Description: Sets the blob in a inference request.
  - Parameters:
    - `infer_request`: A pointer to `ie_infer_request_t` instance.
    - `blob ` -   A pointer to `ie_blob_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_infer_request_infer(ie_infer_request_t *infer_request)`

  - Description:  Starts synchronous inference of the infer request and fill outputs array
  - Parameters:
    - `infer_request`: A pointer to `ie_infer_request_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_infer_request_infer_async(ie_infer_request_t *infer_request)`

  -  Description: Starts asynchronous inference of the infer request and fill outputs array.
  - Parameters:
    - `infer_request` - A pointer to `ie_infer_request_t` instance.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_infer_set_completion_callback(ie_infer_request_t *infer_request,completeCallBackFunc callback)`

  - Description: Sets a callback function that will be called on success or failure of asynchronous request.
  - Parameters:
    - `infer_request` - A pointer to a `ie_infer_request_t` instance.
    - `callback` -  A function to be called.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_infer_request_wait(ie_infer_request_t *infer_request, int64_t timeout)`

  - Description:  Waits for the result to become available. Blocks until specified timeout elapses or the result becomes available, whichever comes first.

    NOTE:** There are special values of the timeout parameter:

    - 0 - Immediately returns the inference status. It does not block or interrupt execution.
      ind statuses meaning.
    - -1 - Waits until inference result becomes available (default value).

  - Parameters:

    - `infer_request` -A pointer to a `ie_infer_request_t` instance.
    - `timeout` - Time to wait in milliseconds or special (0, -1) cases described above. If not specified, `timeout` value is set to -1 by default.

  - Return value:  Status code of the operation: OK(0) for success.

## Blob

### Methods

/*The structure of the blobs has complex structure, below functions represent creation of memory blobs from the scratch or on top of existing memory These functions return handle to the blob to be used in other ie_* functions*/

- `IEStatusCode make_memory_blob(const tensor_desc *tensorDesc, ie_blob_t *blob_result)`
  - Description: Creates a `ie_blob_t` instance with the specified dimensions and layout but does not allocate the memory. Use the allocate() method to allocate memory. `tensor_desc` Defines the layout and dims of the blob.
  - Parameters:
    - `tensorDesc` - Defines the layout and dims of the blob.
    - `blob_result` - A pointer to an empty ie_blob_t instance.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode make_memory_blob_from_preallocated_memory(const tensor_desc *tensorDesc, void *ptr, size_t size = 0, ie_blob_t *blob_result)`
  - Description: The constructor creates a `ie_blob_t` instance with the specified dimensions and layout on the pre-allocated memory. The allocate() call is not required.
  - Parameters:
    - `tensorDesc` - Tensor description for Blob creation.
    - `ptr` - A pointer to the pre-allocated memory.
    - `size` -Length of the pre-allocated array. If not set, size is assumed equal to the dot product of dims.
    - `blob_result` - A pointer to the newly created  ie_blob_t instance.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode make_memory_blob_with_roi(const ie_blob_t **inputBlob, const roi_e *roi, ie_blob_t *blob_result)`
  - Description:  Creates a blob describing given roi instance based on the given blob with pre-allocated memory.
  - Parameters:
    - `inputBlob` - Original blob with pre-allocated memory.
    - `roi` - A roi object inside of the original blob.
    - `blob_result` - A  pointer to the newly created blob.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_size(ie_blob_t *blob, int *size_result)`
  - Description: Gets the total number of elements, which is a product of all the dimensions.
  - Parameters:
    - `blob` -  A  pointer to the blob.
    - `size_result` - The total number of elements.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_byte_size(ie_blob_t *blob, int *bsize_result)`
  - Description: Gets the size of the current Blob in bytes.
  - Parameters:
    - `blob` -  A  pointer to the blob.
    - `bsize_result` - The size of the current Blob in bytes.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_allocate(ie_blob_t *blob)`
  - Description:  Allocates memory for blob.
  - Parameters:
    - `blob` - A pointer to an empty blob.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_deallocate(ie_blob_t *blob)`
  - Description:  Releases previously allocated data.
  - Parameters:
    - `blob` - A  pointer to the blob.
  - Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_buffer(ie_blob_t *blob, void *buffer)`
  - Description: Gets access to the allocated memory .
  - Parameters:
    - `blob` - A  pointer to the blob.
    - `buffer` - A pointer  to the coped date from the given pointer to the blob.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_cbuffer(ie_blob_t *blob, const void *cbuffer)`
  - Description:   Gets read-only access to the allocated memory.
  - Parameters:
    - `blob` - A  pointer to the blob.
    - `cbuffer` - A pointer  to the coped date from the given pointer to the blob and the date is read-only.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_get_dims(ie_blob_t *blob, dimensions_t *dims_result)`
  - Description: Gets dimensions of blob instance's tensor.
  - Parameters:
    - `blob` - A  pointer to the blob.
    - `dims_result` - A pointer to the dimensions of blob instance's tensor.
  -  Return value:  Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_get_layout(ie_blob_t *blob, layout_t *layout_result)`
  - Description: Gets layout of blob instance's tensor.
  - Parameters:
    - `blob` - A  pointer to the blob.
    - `layout_result` -  A pointer to the layout of blob instance's tensor.
  -  Return value: Status code of the operation: OK(0) for success.

- `IEStatusCode ie_blob_get_precision(ie_blob_t *blob, precision_e *prec_result)`
  - Description: Gets precision of blob instance's tensor.
  - Parameters:
    - `blob` - A  pointer to the blob.
    - `prec_result` - A pointer to the precision of blob instance's tensor.
  - Return value: Status code of the operation: OK(0) for success.
