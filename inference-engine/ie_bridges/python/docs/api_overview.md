# Overview of Inference Engine Python* API {#InferEnginePythonAPI}

**NOTE:** It is a preview version of the Inference Engine Python\* API for evaluation purpose only. 
Module structure and API itself may be changed in future releases.  

This API provides a simplified interface for Inference Engine functionality that allows to:

* handle the models
* load and configure Inference Engine plugins based on device names
* perform inference in synchronous and asynchronous modes with arbitrary number of infer requests (the number of infer requests may be limited by target device capabilities)

## Supported OSes

Currently the Inference Engine Python\* API is supported on Ubuntu* 16.04, Microsoft Windows* 10 and CentOS* 7.3 OSes.   
Supported Python* versions:  

* On Ubuntu 16.04: 2.7, 3.5, 3.6  
* On Windows 10: 3.5, 3.6
* On CentOS 7.3: 3.4, 3.5, 3.6

## Setting Up the Environment

To configure the environment for the Inference Engine Python\* API, run:
 * On Ubuntu 16.04: `source <INSTALL_DIR>/bin/setupvars.sh .` 
 * On Windows 10: `call <INSTALL_DIR>\deployment_tools\inference_engine\python_api\setenv.bat`
 
The script automatically detects latest installed Python\* version and configures required environment if the version is supported.  
If you want to use certain version of Python\*, set the environment variable `PYTHONPATH=<INSTALL_DIR>/deployment_tools/inference_engine/python_api/<desired_python_version>`
after running the environment configuration script.
   
## <a name="ienetlayer-class"></a>IENetLayer
This class stores main information about the layer and allow to modify some layer parameters 
### Class attributes:
 
* `name` - name of the layer 
* `type` - layer type
* `precision` - layer base operating precision
* `affinity` - layer affinity set by user or default affinity set by IEPlugin.set_initial_affinity() method.             
             The affinity attribute provides getter and setter interface, so the layer affinity can be modified directly in following way
             
```py
    >>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
    >>> plugin = IEPlugin(device="HETERO:FPGA,CPU")
    >>> plugin.set_config({"TARGET_FALLBACK": "HETERO:FPGA,CPU"})
    >>> plugin.set_initial_affinity(net) 
    >>> for l in net.layers.values():
    ...     if l.type == "Convolution":
    ...         l.affinity = "CPU"

```
		
To correctly set affinity for the network, you must first initialize and properly configure the HETERO plugin. 
`set_config({"TARGET_FALLBACK": "HETERO:FPGA,GPU"})` function configures the plugin fallback devices and their order. 
`plugin.set_initial_affinity(net)` function sets affinity parameter of model layers according to its support 
on specified devices. 

After default affinity is set by the plugin, override the default values by setting affinity manually how it's 
described in example above  

To understand how default and non-default affinities are set: 

1. Call `net.layers` function right after model loading and check that layer affinity parameter is empty.
2. Call `plugin.set_default_affinity(net)`.
3. Call `net.layers` and check layer affinity parameters to see how plugin set default affinity
4. Set layer affinity how it's described above
5. Call `net.layers` again and check layer affinity parameters to see how it was changed after manual affinity 
   setting
          
Please refer to `affinity_setting_sample.py` to see the full usage pipeline.
    
* `weights` - dictionary with layer weights, biases or custom blobs if any
* `params` - layer specific parameters. Provides getter and setter interface which allows to get and\or modify layer parameters.
            Please note that some modifications can be ignored and\or overwriten by target plugin (e.g. modification of 
            convolution kernel size will be reflected in layer parameters but finally the plugin will ignore it and will
            use initial kernel size) 

## <a name="ienetwork-class"></a>IENetwork 

This class contains the information about the network model read from IR and allows you to manipulate with some model parameters such as
layers affinity and output layers.

### Class Constructor

There is no explicit class constructor. Use `from_ir` class method to read the Intermediate Representation (IR) and initialize a correct instance of the `IENetwork` class.

### Class attributes:

* `name` - Name of the loaded network
* `inputs` - a dictionary of input layer name as a key and input data shape as a value

    * Usage example:
    ```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> net.inputs
		{'data': [1, 3, 224, 224]}
    ```
* `outputs` - a list of output layer names

    * Usage example:
    ```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> net.outputs
		['prob']
    ```
    
* `batch_size` - Batch size of the network. Provides getter and setter interface which allows to get and modify the 
                 network batch size in the following way:
    ```py
    >>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
    >>> net.batch_size
    1
    >>> net.batch_size = 4
    >>> net.batch_size
    4
    ```
* `layers` - return dictionary with the network layer names as key and <a name="ienetlayer-class"></a>IENetLayer objects containing layer properties 
             as value
             
    ```py
		 >>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		 >>> net.layers
		 {'conv0': <inference_engine.ie_api.IENetLayer object at 0x7f3a4c102370>}
		 ```
### Class Methods

* `from_ir(model: str, weights: str)` 

    * Description:
            
        The class method serves to read the model from the `.xml` and `.bin` files of the IR.
		
    * Parameters:
        
        * model - path to `.xml` file  of the IR
        * weights - path to `.bin` file  of the IR
    
    * Return value:
            
        An instance of the `IENetwork` class
            
    * Usage example:
		```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> net
		<inference_engine.ie_api.IENetwork object at 0x7fd7dbce54b0>
		```
            
### Instance Methods
	 
* `add_outputs(outputs)`:

    * Description:
                
        The method serves to mark any intermediate layer as output layer to retrieve the inference results 
        from the specified layers.
		
    * Parameters:
            
        * `outputs` - a list of layer names to be set as model outputs. In case of setting one layer as output, string with one layer can be provided.
            
    * Return value:
    
        None
            
    * Usage example:
		```py
        >>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
        >>> net.add_outputs(["conv5_1/dwise', conv2_1/expand'])]
        >>> net.outputs
        ['prob', 'conv5_1/dwise', 'conv2_1/expand']
        ```  
		
        Note that the last layers (nodes without successors in graph representation of the model) are set as output 
        by default. In the case above, `prob` layer is a default output and `conv5_1/dwise`, `conv2_1/expand` are user-defined
		outputs. 

## <a name="ieplugin-class"></a>IEPlugin Class

This class is the main plugin interface and serves to initialize and configure the plugin.
 
### Class Constructor

* `__init__(device: str, plugin_dirs=None)`

    * Parameters:
    
        * `device` - target device name. Supported devices: CPU, GPU, FPGA, MYRIAD, HETERO
        * `plugin_dirs` - list of paths to plugin directories 
        
### Properties

* `device` - a name of the device that was specified to initialize IEPlugin
* `version` -  a version of the plugin 

### Instance Methods

*  `load(network: IENetwork, num_requests: int=1, config=None)`

    * Description:
        
        Loads a network that was read from the IR to the plugin and creates an executable network from a network object. 
        You can create as many networks as you need and use them simultaneously (up to the limitation of the hardware 
        resources).
    
    * Parameters:
	
        * `network` - a valid IENetwork instance created by `IENetwork.from_ir()` method
        * `num_requests` - a positive integer value of infer requests to be created. Number of infer requests may be limited 
        by device capabilities.        
        * `config` - a dictionary of plugin configuration keys and their values
        
    * Return value:
        
        None
    
    * Usage example:
		```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> plugin = IEPlugin(device="CPU")
		>>> exec_net = plugin.load(network=net, num_requsts=2)
		>>> exec_net
		<inference_engine.ie_api.ExecutableNetwork object at 0x7f5140bbcd38>
		```
* `set_initial_affinity(net: IENetwork)`
    
    * Description:
        
        Sets initial affinity for model layers according to the HETERO plugin logic. Applicable only if 
        IEPlugin was initialized for HETERO device.
        
    * Parameters:
	
        * `net` - a valid instance of IENetwork 
    
    * Return value:
        
        None
        
    * Usage example: 
	
		See `affinity` attribute of the `IENetLayer` class.
    
* `add_cpu_extension(extension_path: str)`

    * Description:
        
        Loads extensions library to the plugin. Applicable only for CPU device and HETERO device with CPU
        
    * Parameters:
    
        * `extension_path` - a full path to CPU extensions library   
        
     * Return value:
        
        None
        
    * Usage example:
		```py
		>>> plugin = IEPlugin(device="CPU")
		>>> plugin.add_cpu_extenstions(ext_lib_path)
		```
* `set_config(config: dict)`

    * Description: 
     
        Sets a configuration for the plugin. Refer to `SetConfig()` in Inference Engine C++ documentation for acceptable 
        keys and values list.
        
    * Parameters: 
        
        * `config` - a dictionary of keys and values of acceptable configuration parameters
        
    * Return value:
        
        None
    
    * Usage examples: 
		
		See `set_affinity` method of the `IENetwork` class.

* `get_supported_layers(net: IENetwork)`
    * Description:
        
        Returns the set of layers supported by the plugin. Please note that in case of CPU plugin support of 
        a layer may depends on extension loaded by `add_cpu_extenstion()` method 
        
    * Parameters:
	
        * `net` - a valid instance of IENetwork 
    
    * Return value:
        
        Set of layers supported by the plugin
        
    * Usage example: 
	
		See `affinity` attribute of the `IENetLayer` class.
   
## <a name="executablenetwork"></a>ExecutableNetwork Class

This class represents a network instance loaded to plugin and ready for inference. 

### Class Constructor

There is no explicit class constructor. To make a valid instance of `ExecutableNetwork`, use `load()` method of the `IEPlugin` class.

### Class attributes

* `requests` - a tuple of InferRequest instances

    * Usage example:
		```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> plugin = IEPlugin(device="CPU")
		>>> exec_net = plugin.load(network=net, num_requsts=2)
		>>> exec_net.requests
		(<inference_engine.ie_api.InferRequest object at 0x7f66f56c57e0>, <inference_engine.ie_api.InferRequest object at 0x7f66f56c58b8>, <inference_engine.ie_api.InferRequest object at 0x7f66f56c5900>)
		```
		
### Instance Methods

* `infer(inputs=None)`

    * Description:
        
        Starts synchronous inference for the first infer request of the executable network and returns output data.
        Wraps `infer()` method of the `InferRequest` class
    
    * Parameters:
        * `inputs` - a dictionary of input layer name as a key and `numpy.ndarray` of proper shape with input data for the layer as a value
        
    * Return value:
        
        A dictionary of output layer name as a key and `numpy.ndarray` with output data of the layer as a value
        
    * Usage example:
		```py
		>>> net = IENetwork.from_ir(model=path_to_xml_file, weights=path_to_bin_file)
		>>> plugin = IEPlugin(device="CPU")
		>>> exec_net = plugin.load(network=net, num_requsts=2)
		>>> res = exec_net.infer({'data': img})
		>>> res
		{'prob': array([[[[2.83426580e-08]],
						 [[2.40166020e-08]],
						 [[1.29469613e-09]],
						 [[2.95946148e-08]]
						 ......
					  ]])}
		```
      For illustration of input data preparation, please see samples (for example, `classification_sample.py`).
      
* `start_async(request_id, inputs=None)`

    * Description:
        
        Starts asynchronous inference for specified infer request.
        Wraps `async_infer()` method of the `InferRequest` class
        
    * Parameters:
	
        * `request_id` - index of infer request to start inference
        * `inputs` - a dictionary of input layer name as a key and `numpy.ndarray` of proper shape with input data for the layer as a value
        
    * Return value:
        
        A handler of specified infer request, which is an instance of the `InferRequest` class.
        
    * Usage example:
		```py
		>>> infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: image})
		>>> infer_status = infer_request_handle.wait()
		>>> res = infer_request_handle.outputs[out_blob]
		```
        For more details about infer requests processing, see `classification_sample_async.py` (simplified case) and 
        `object_detection_demo_ssd_async.py` (real synchronous use case) samples.
        
## <a name="inferrequest"></a>InferRequest Class

This class provides an interface to infer requests of `ExecutableNetwork` and serves to handle infer requests execution
and to set and get output data.   

### Class Constructor

There is no explicit class constructor. To make a valid `InferRequest` instance, use `load()` method of the `IEPlugin` 
class with specified number of requests to get `ExecutableNetwork` instance which stores infer requests. 

### Class attributes

* `inputs` - a dictionary of input layer name as a key and `numpy.ndarray` of proper shape with input data for the layer as a value
* `outputs` - a dictionary of output layer name as a key and `numpy.ndarray` with output data of the layer as a value

* Usage example:
	```py    
	>>> exec_net.requests[0].inputs['data'][:] = image
	>>> exec_net.requests[0].infer()
	>>> res = exec_net.requests[0].outputs['prob']
	>>> np.flip(np.sort(np.squeeze(res)),0) 
	array([4.85416055e-01, 1.70385033e-01, 1.21873841e-01, 1.18894853e-01,
		   5.45198545e-02, 2.44456064e-02, 5.41366823e-03, 3.42589128e-03,
		   2.26027006e-03, 2.12283316e-03 ...])
	``` 
	
### Instance Methods

It is not recommended to run inference directly on `InferRequest` instance. 
To run inference, please use simplified methods `infer()` and `start_async()` of `ExecutableNetwork`. 

* `infer(inputs=None)`

    * Description: 
    
         Starts synchronous inference of the infer request and fill outputs array
		 
     * Parameters:
	 
        * `inputs` - a dictionary of input layer name as a key and `numpy.ndarray` of proper shape with input data for the layer as a value
        
    * Return value:
        
        None
        
    * Usage example:
		```py
		>>> exec_net = plugin.load(network=net, num_requests=2)
		>>> exec_net.requests[0].infer({input_blob: image})
		>>> res = exec_net.requests[0].outputs['prob']
		>>> np.flip(np.sort(np.squeeze(res)),0) 
		array([4.85416055e-01, 1.70385033e-01, 1.21873841e-01, 1.18894853e-01,
			   5.45198545e-02, 2.44456064e-02, 5.41366823e-03, 3.42589128e-03,
			   2.26027006e-03, 2.12283316e-03 ...]) 
		```       
                   
* `async_infer(inputs=None)`

    * Description: 
    
        Starts asynchronous inference of the infer request and fill outputs array
		
     * Parameters:
	 
        * `inputs` - a dictionary of input layer name as a key and `numpy.ndarray` of proper shape with input data for the layer as a value
        
    * Return value:
        
        None
        
    * Usage example:
		```py
		>>> exec_net = plugin.load(network=net, num_requests=2)
		>>> exec_net.requests[0].async_infer({input_blob: image})
		>>> exec_net.requests[0].wait()
		>>> res = exec_net.requests[0].outputs['prob']
		>>> np.flip(np.sort(np.squeeze(res)),0) 
		array([4.85416055e-01, 1.70385033e-01, 1.21873841e-01, 1.18894853e-01,
			   5.45198545e-02, 2.44456064e-02, 5.41366823e-03, 3.42589128e-03,
			   2.26027006e-03, 2.12283316e-03 ...]) 
		```
			
* `wait(timeout=-1)`

    * Description:
        
        Waits for the result to become available. Blocks until specified timeout elapses or the result 
        becomes available, whichever comes first. 
        
        **Note:**
        
        There are special values of the timeout parameter:
        
        * 0 - immediately returns the inference status. It does not block or interrupt execution. 
        To find statuses meaning, please refer to InferenceEngine::StatusCode in Inference Engine C++ documentation
        
        * -1 - waits until inference result becomes available (default value)
        
    * Parameters:
	
        * `timeout` - time to wait in milliseconds or special (0, -1) cases described above. 
          If not specified, `timeout` value is set to -1 by default.
      
    * Usage example: 
	
		See `async_infer()` method of the the `InferRequest` class.
		

* `get_perf_counts()`

    * Description:
        
        Queries performance measures per layer to get feedback of what is the most time consuming layer. . 
        
        **Note**:
            
        Performance counters data and format depends on the plugin
        
    * Parameters:
	
        None
      
    * Usage example: 
	
		```py
		>>> exec_net = plugin.load(network=net, num_requests=2)
		>>> exec_net.requests[0].infer({input_blob: image})
		>>> exec_net.requests[0].get_perf_counts()
        {'Conv2D': {'exec_type': 'jit_avx2_1x1', 
                    'real_time': 154, 
                    'cpu_time': 154, 
                    'status': 'EXECUTED', 
                    'layer_type': 'Convolution'},
         'Relu6':  {'exec_type': 'undef', 
                    'real_time': 0, 
                    'cpu_time': 0, 
                    'status': 'NOT_RUN', 
                    'layer_type': 'Clamp'}
        ...
        }