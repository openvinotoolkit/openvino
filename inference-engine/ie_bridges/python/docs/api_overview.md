# Overview of Inference Engine Python* API

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
 
* `name` - Name of the layer 
* `type`- Layer type
* `precision` - Layer base operating precision. Provides getter and setter interfaces.
* `layout` - Returns the layout of shape of the layer.
* `shape` -  Return the list of the shape of the layer.
* `parents` - Returns a list, which contains names of layers preceding this layer.
* `children` - Returns a list, which contains names of layers following this layer. 
* `affinity` - Layer affinity set by user or a default affinity set by the `IEPlugin.set_initial_affinity()` method.             
               The affinity attribute provides getter and setter interfaces, so the layer affinity can be modified directly.
               For example:                          
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
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
3. Call `net.layers` and check layer affinity parameters to see how plugin set a default affinity
4. Set layer affinity how it's described above
5. Call `net.layers` again and check layer affinity parameters to see how it was changed after manual affinity 
   setting
          
Please refer to `affinity_setting_demo.py` to see the full usage pipeline.
    
* `weights`- Dictionary with layer weights, biases or custom blobs if any
* `params` - Layer specific parameters. Provides getter and setter interfaces to get and modify layer parameters.
             Please note that some modifications can be ignored and\or overwriten by target plugin (e.g. modification of 
             convolution kernel size will be reflected in layer parameters but finally the plugin will ignore it and will
             use initial kernel size) 

## <a name="ienetwork-class"></a>IENetwork 

This class contains the information about the network model read from IR and allows you to manipulate with some model parameters such as
layers affinity and output layers.

### Class Constructor

* `__init__(model: str, weights: str)`

    * Parameters:
        
        * model - Path to `.xml` file  of the IR
        * weights - Path to `.bin` file  of the IR

### Class attributes:

* `name` - Name of the loaded network
* `inputs` - A dictionary that maps input layer names to <a name="inputinfo-class"></a>InputInfo objects. 
             For example, to get a shape of the input layer:

```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net.inputs
{'data': <inference_engine.ie_api.InputInfo object at 0x7efe042dedd8>}
>>> net.inputs['data'].shape
[1, 3, 224, 224]
```

* `outputs` - A dictionary that maps output layer names to <a name="inputinfo-class"></a>OutputInfo objects
              For example, to get a shape of the output layer:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net.inputs
{'prob': <inference_engine.ie_api.OutputInfo object at 0x7efe03ab95d0>}
>>> net.outputs['prob'].shape
[1, 1000]
```
    
* `batch_size` - Batch size of the network. Provides getter and setter interfaces to get and modify the 
                 network batch size. For example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net.batch_size
1
>>> net.batch_size = 4
>>> net.batch_size
4
>>> net.inputs['data'].shape
    [4, 3, 224, 224]
```
    
* `layers` - Return dictionary that maps network layer names to <a name="ienetlayer-class"></a>`IENetLayer` 
             objects containing layer properties in topological order. For example, to list all network layers:
             
```py
 >>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
 >>> net.layers
 {'conv0': <inference_engine.ie_api.IENetLayer object at 0x7f3a4c102370>
 ...
 }
 ```
 
 * `stats` - Returns `LayersStatsMap` object containing dictionary that maps network layer names to calibration statistics 
            represented by <a name="layerstats-class"></a> `LayerStats` objects.
            `LayersStatsMap` class inherited from built-in python `dict` and overrides default `update()`method to allow
            to set or modify layers calibration statistics. 
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net.stats.update({
        "conv1_2d" : LayserStats(min=(-25, -1, 0), max=(63, 124, 70)),
        "conv2_2d" : LayserStats(min=(-5, -1, 0, 1, -7, 2), max=(63, 124, 70, 174, 99, 106)),
    })
```
For more details about low precision inference please refer to "Low-Precision 8-bit Integer Inference" 
section in Inference Engine Developers Guide documentation. 

             
### Class Methods

* `from_ir(model: str, weights: str)` 

**Note:** The function is deprecated. Please use `IENetwork()` class constructor to create valid instance of `IENetwork`

    * Description:
            
        The class method serves to read the model from the `.xml` and `.bin` files of the IR.
		
    * Parameters:
        
        * model - Path to `.xml` file  of the IR
        * weights - Path to `.bin` file  of the IR
    
    * Return value:
            
        An instance of the `IENetwork` class
            
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net
<inference_engine.ie_api.IENetwork object at 0x7fd7dbce54b0>
```
            
### Instance Methods
	 
* `add_outputs(outputs)`:

    * Description:
                
        The method serves to mark any intermediate layer as output layer to retrieve the inference results 
        from the specified layers.
		
    * Parameters:
            
        * `outputs` - List of layer names to be set as model outputs. In case of setting one layer as output, string with one layer can be provided.
            
    * Return value:
    
        None
            
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> net.add_outputs(["conv5_1/dwise', conv2_1/expand'])]
>>> net.outputs
['prob', 'conv5_1/dwise', 'conv2_1/expand']
```  
    
**Note**

The last layers (nodes without successors in graph representation of the model) are set as output 
by default. In the case above, `prob` layer is a default output and `conv5_1/dwise`, `conv2_1/expand` are user-defined
outputs.

* `reshape(input_shapes: dict)`:
    
    * Description: 
        
        The method reshapes the network to change spatial dimensions, batch size, or any dimension.
        
        **Note:**
        
        Before using this method, make sure that the target shape is applicable for the network
        Changing the network shape to an arbitrary value may lead to unpredictable behaviour. 
        
    * Parameters:
    
        * `input_shapes` - The dictionary that maps input layer names to tuples with the target shape

    * Return value:
    
        None
            
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> input_layer = next(iter(net.inputs))
>>> n, c, h, w = net.inputs[input_layer]
>>> net.reshape({input_layer: (n, c, h*2, w*2)}]
``` 

* `serialize(path_to_xml, path_to_bin)`:
    
    * Description: 
        
        The method serializes the network and stores it in files. 
        
    * Parameters:
    
        * `path_to_xml` - path to a file, where a serialized model will be stored. 
        * `path_to_bin` - path to a file, where serialized weights will be stored.

    * Return value:
    
        None
            
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_model, weights=path_to_weights)
>>> net.serialize(path_to_xml, path_to_bin)
``` 
## <a name="layerstats-class"></a>LayerStats
Layer calibration statistic container
### Class Constructor

* `__init__(min: tuple = (), max: tuple = ())`

    * Parameters:
        
        * min - Tuple with per-channel minimum layer activation values 
        * max - Tuple with per-channel maximum layer activation values

## <a name="inputinfo-class"></a>InputInfo 

This class contains the information about the network input layers

### Class attributes:

* `precision` - Precision of the input data provided by user. Provides setter and getter interfaces 
                to get and modify input layer precision.
                
    List of applicable precisions: FP32 FP16, I32, I16, I8, U32, U16
    
    **Note**:  Support of any calculation precision depends on the target plugin                 

* `layout` - Layout of the input data provided by user. Provides setter and getter interfaces  
             to get and modify input layer layout. 
             
    List of applicable layouts: NCHW, NHWC, OIHW, C, CHW, HW, NC, CN, BLOCKED

* `shape` - input layer data shape


## <a name="outputinfo-class"></a>OutputInfo 

This class contains the information about the network input layers

### Class attributes:

* `precision` - Precision of the output data. Provides setter and getter interfaces  
                to get and modify output layer precision.          

* `layout` - Layout of the output data provided by user

* `shape` - Input layer data shape
 
## <a name="ieplugin-class"></a>IEPlugin Class

This class is the main plugin interface and serves to initialize and configure the plugin.
 
### Class Constructor

* `__init__(device: str, plugin_dirs=None)`

    * Parameters:
    
        * `device` - Target device name. Supported devices: CPU, GPU, FPGA, MYRIAD, HETERO
        * `plugin_dirs` - List of paths to plugin directories 
        
### Properties

* `device` - a name of the device that was specified to initialize IEPlugin
* `version` -  a version of the plugin 

### Instance Methods

*  ```load(network: IENetwork, num_requests: int=1, config=None)```

    * Description:
        
        Loads a network that was read from the IR to the plugin and creates an executable network from a network object. 
        You can create as many networks as you need and use them simultaneously (up to the limitation of the hardware 
        resources).
    
    * Parameters:
	
        * `network` - A valid `IENetwork` instance
        * `num_requests` - A positive integer value of infer requests to be created. Number of infer requests may be limited 
        by device capabilities.        
        * `config` - A dictionary of plugin configuration keys and their values
        
    * Return value:
        
        None
    
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
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
	
        * `net` - A valid instance of IENetwork 
    
    * Return value:
        
        None
        
    * Usage example: 
	
		See `affinity` attribute of the `IENetLayer` class.
    
* `add_cpu_extension(extension_path: str)`

    * Description:
        
        Loads extensions library to the plugin. Applicable only for CPU device and HETERO device with CPU
        
    * Parameters:
    
        * `extension_path` - A full path to CPU extensions library   
        
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
        
        * `config` - A dictionary of keys and values of acceptable configuration parameters
        
    * Return value:
        
        None
    
    * Usage examples: 
		
		See `set_affinity` method of the `IENetwork` class.

* `get_supported_layers(net: IENetwork)`
    
    * Description:
        
        Returns the set of layers supported by the plugin. Please note that in case of CPU plugin support of 
        a layer may depends on extension loaded by `add_cpu_extenstion()` method 
        
    * Parameters:
	
        * `net` - A valid instance of IENetwork 
    
    * Return value:
        
        Set of layers supported by the plugin
        
    * Usage example: 
	
		See `affinity` attribute of the `IENetLayer` class.
   
## <a name="executablenetwork"></a>ExecutableNetwork Class

This class represents a network instance loaded to plugin and ready for inference. 

### Class Constructor

There is no explicit class constructor. To make a valid instance of `ExecutableNetwork`, use `load()` method of the `IEPlugin` class.

### Class attributes

* `requests` - A tuple of InferRequest instances

    * Usage example:
        
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
>>> plugin = IEPlugin(device="CPU")
>>> exec_net = plugin.load(network=net, num_requsts=3)
>>> exec_net.requests
(<inference_engine.ie_api.InferRequest object at 0x7f66f56c57e0>, 
<inference_engine.ie_api.InferRequest object at 0x7f66f56c58b8>, 
<inference_engine.ie_api.InferRequest object at 0x7f66f56c5900>)
```
		
### Instance Methods

* `infer(inputs=None)`

    * Description:
        
        Starts synchronous inference for the first infer request of the executable network and returns output data.
        Wraps `infer()` method of the `InferRequest` class
    
    * Parameters:
        * `inputs` - A dictionary that maps input layer names to `numpy.ndarray` objects of proper shape with input data for the layer
        
    * Return value:
        
        A dictionary that maps output layer names to `numpy.ndarray` objects with output data of the layer
        
    * Usage example:
    
```py
>>> net = IENetwork(model=path_to_xml_file, weights=path_to_bin_file)
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
	
        * `request_id` - Index of infer request to start inference
        * `inputs` - A dictionary that maps input layer names to `numpy.ndarray` objects of proper shape with input data for the layer
        
    * Return value:
        
        A handler of specified infer request, which is an instance of the `InferRequest` class.
        
    * Usage example:
		
```py
>>> infer_request_handle = exec_net.start_async(request_id=0, inputs={input_blob: image})
>>> infer_status = infer_request_handle.wait()
>>> res = infer_request_handle.outputs[out_blob]
```
		
For more details about infer requests processing, see `classification_sample_async.py` (simplified case) and 
`object_detection_demo_ssd_async.py` (real asynchronous use case) samples.
        
## <a name="inferrequest"></a>InferRequest Class

This class provides an interface to infer requests of `ExecutableNetwork` and serves to handle infer requests execution
and to set and get output data.   

### Class Constructor

There is no explicit class constructor. To make a valid `InferRequest` instance, use `load()` method of the `IEPlugin` 
class with specified number of requests to get `ExecutableNetwork` instance which stores infer requests. 

### Class attributes

* `inputs` - A dictionary that maps input layer names to `numpy.ndarray` objects of proper shape with input data for the layer
* `outputs` - A dictionary that maps output layer names to `numpy.ndarray` objects with output data of the layer

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
	 
        * `inputs` - A dictionary that maps input layer names to `numpy.ndarray` objects of proper shape with input data for the layer
        
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
	 
        * `inputs` - A dictionary that maps input layer names to `numpy.ndarray` objects of proper shape with input data for the layer
        
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
        
        * 0 - Immediately returns the inference status. It does not block or interrupt execution. 
        To find statuses meaning, please refer to InferenceEngine::StatusCode in Inference Engine C++ documentation
        
        * -1 - Waits until inference result becomes available (default value)
        
    * Parameters:
	
        * `timeout` - Time to wait in milliseconds or special (0, -1) cases described above. 
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
```

* `set_batch(size)`
    * Description:   
       Sets new batch size for certain infer request when dynamic batching is enabled in executable network that created this request.
       
       **Note:** Support of dynamic batch size depends on the target plugin.        
        
    * Parameters:
        * `batch` - new batch size to be used by all the following inference calls for this request.
        
    * Usage example:
```py
>>> plugin.set_config({"DYN_BATCH_ENABLED": "YES"})
>>> exec_net = plugin.load(network=net)
>>> exec_net.requests[0].set_batch(inputs_count)
```
Please refer to `dynamic_batch_demo.py` to see the full usage example.


