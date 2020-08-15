# Model Optimizer Frequently Asked Questions  {#openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ}

If your question is not covered by the topics below, use the [OpenVINO&trade; Support page](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started), where you can participate on a free forum.

#### 1. What does the message "[ ERROR ]: Current caffe.proto does not contain field" mean? <a name="question-1"></a>

Internally, the Model Optimizer uses a protobuf library to parse and load Caffe\* models. This library requires a file grammar and a generated parser. For a Caffe fallback, the Model Optimizer uses a Caffe-generated parser for a Caffe-specific `.proto` file (which is usually located in the `src/caffe/proto` directory). So, if you have Caffe installed on your machine with Python* interface available, make sure that this is exactly the version of Caffe that was used to create the model.

If you just want to experiment with the Model Optimizer and test a Python extension for working with your custom 
layers without building Caffe, add the layer description to the `caffe.proto` file and generate a parser for it.

For example, to add the description of the `CustomReshape` layer, which is an artificial layer not present in any `caffe.proto` files:

1.  Add the following lines to of the `caffe.proto` file:
```shell
    package mo_caffe; // to avoid conflict with system Caffe* it is highly recommended to specify different package name
    ...
    message LayerParameter {
      // other layers parameters description
      ...
      optional CustomReshapeParameter custom_reshape_param = 546; // 546 - ID is any number not present in caffe.proto
    }
    // these lines to end of the file - describing contents of this parameter
    message CustomReshapeParameter {
      optional BlobShape shape = 1; // we just use the same parameter type as some other Caffe layers
    }
```
    
2.  Generate a new parser:
```shell
cd <INSTALL_DIR>/deployment_tools/model_optimizer/mo/front/caffe/proto
python3 generate_caffe_pb2.py --input_proto <PATH_TO_CUSTOM_CAFFE>/src/caffe/proto/caffe.proto
```
where `PATH_TO_CUSTOM_CAFFE` is the path to the root directory of custom Caffe\*.
    
3.  Now, the Model Optimizer is able to load the model into memory and start working with your extensions if there are any.

However, because your model has custom layers, you must register your custom layers as custom. To learn more about it, refer to the section [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md). 

#### 2. How do I create a bare caffemodel, if I have only prototxt? <a name="question-2"></a>

You need the Caffe\* Python\* interface. In this case, do the following:
```shell
python3
import caffe
net = caffe.Net('<PATH_TO_PROTOTXT>/my_net.prototxt', caffe.TEST)
net.save('<PATH_TO_PROTOTXT>/my_net.caffemodel')
```
#### 3. What does the message "[ ERROR ]: Unable to create ports for node with id" mean? <a name="question-3"></a>

Most likely, the Model Optimizer does not know how to infer output shapes of some layers in the given topology. 
To lessen the scope, compile the list of layers that are custom for the Model Optimizer: present in the topology, 
absent in [list of supported layers](Supported_Frameworks_Layers.md) for the target framework. Then refer to available options in the corresponding section in [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md).

#### 4. What does the message "Input image of shape is larger than mean image from file" mean? <a name="question-4"></a>

Your model input shapes must be smaller than or equal to the shapes of the mean image file you provide. The idea behind the mean file is to subtract its values from the input image in an element-wise manner. When the mean file is smaller than the input image, there are not enough values to perform element-wise subtraction. Also, make sure that you use the mean file that was used during the network training phase. Note that the mean file is dataset dependent.

#### 5. What does the message "Mean file is empty" mean? <a name="question-5"></a>

Most likely, the mean file that you have is specified with `--mean_file` flag, while launching the Model Optimizer is empty. Make sure that this is exactly the required mean file and try to regenerate it from the given dataset if possible.

#### 6. What does the message "Probably mean file has incorrect format" mean? <a name="question-6"></a>

The mean file that you provide for the Model Optimizer must be in a `.binaryproto` format. You can try to check the content using recommendations from the BVLC Caffe\* ([#290](https://github.com/BVLC/caffe/issues/290)).

#### 7. What does the message "Invalid proto file: there is neither 'layer' nor 'layers' top-level messages" mean? <a name="question-7"></a>

The structure of any Caffe\* topology is described in the `caffe.proto` file of any Caffe version. For example, in the Model Optimizer, you can find the following proto file, used by default: `<INSTALL_DIR>/deployment_tools/model_optimizer/mo/front/caffe/proto/my_caffe.proto`. There you can find the structure:
```
message NetParameter {
  // ... some other parameters
  // The layers that make up the net.  Each of their configurations, including
  // connectivity and behavior, is specified as a LayerParameter.
  repeated LayerParameter layer = 100;  // ID 100 so layers are printed last.
  // DEPRECATED: use 'layer' instead.
  repeated V1LayerParameter layers = 2;
}
```
This means that any topology should contain layers as top-level structures in `prototxt`. For example, see the [LeNet topology](https://github.com/BVLC/caffe/blob/master/examples/mnist/lenet.prototxt).

#### 8. What does the message "Old-style inputs (via 'input_dims') are not supported. Please specify inputs via 'input_shape'" mean? <a name="question-8"></a>

The structure of any Caffe\* topology is described in the `caffe.proto` file for any Caffe version. For example, in the Model Optimizer you can find the following `.proto` file, used by default: `<INSTALL_DIR>/deployment_tools/model_optimizer/mo/front/caffe/proto/my_caffe.proto`. There you can find the structure:
```sh
message NetParameter {

 optional string name = 1; // consider giving the network a name
  // DEPRECATED. See InputParameter. The input blobs to the network.
  repeated string input = 3;
  // DEPRECATED. See InputParameter. The shape of the input blobs.
  repeated BlobShape input_shape = 8;
  // 4D input dimensions -- deprecated.  Use "input_shape" instead.
  // If specified, for each input blob there should be four
  // values specifying the num, channels, height and width of the input blob.
  // Thus, there should be a total of (4 * #input) numbers.
  repeated int32 input_dim = 4;
  // ... other parameters
}
```
So, the input layer of the provided model must be specified in one of the following styles:

*   
```sh
input: "data"
input_shape
{
    dim: 1
    dim: 3
    dim: 227
    dim: 227
}
```
    
*   
```sh
input: "data"
input_shape
{
    dim: 1
    dim: 3
    dim: 600
    dim: 1000
}
input: "im_info"
input_shape
{
     dim: 1
     dim: 3
}
```
*   
```sh
layer
{
    name: "data"
    type: "Input"
    top: "data"
    input_param {shape: {dim: 1 dim: 3 dim: 600 dim: 1000}}
}
layer
{
    name: "im_info"
    type: "Input"
    top: "im_info"
    input_param {shape: {dim: 1 dim: 3}}
}
```
*   
```sh
input: "data"
input_dim: 1
input_dim: 3
input_dim: 500
```

However, if your model contains more than one input, the Model Optimizer is able to convert the model with inputs specified in a form of 1, 2, 3 of the list above. The last form is not supported for multi-input topologies.

#### 9. What does the message "Mean file for topologies with multiple inputs is not supported" mean? <a name="question-9"></a>

Model Optimizer does not support mean file processing for topologies with more than one input. In this case, you need to perform preprocessing of the inputs for a generated Intermediate Representation in the Inference Engine to perform subtraction for every input of your multi-input model.

#### 10. What does the message "Cannot load or process mean file: value error" mean? <a name="question-10"></a>

There are multiple reasons why the Model Optimizer does not accept the mean file. See FAQs [#4](#question-4), [#5](#question-5), and [#6](#question-6).

#### 11. What does the message "Invalid prototxt file: value error" mean? <a name="question-11"></a>

There are multiple reasons why the Model Optimizer does not accept a Caffe* topology. See FAQs [#7](#question-7) and [#20](#question-20).

#### 12. What does the message "Error happened while constructing caffe.Net in the Caffe fallback function" mean? <a name="question-12"></a>

Model Optimizer tried to infer a specified layer via the Caffe\* framework, however it cannot construct a net using the Caffe Python* interface. Make sure that your `caffemodel` and `prototxt` files are correct. To prove that the problem is not in the `prototxt` file, see FAQ [#2](#question-2).

#### 13. What does the message "Cannot infer shapes due to exception in Caffe" mean? <a name="question-13"></a>

Model Optimizer tried to infer a custom layer via the Caffe\* framework, however an error occurred, meaning that the model could not be inferred using the Caffe. It might happen if you try to convert the model with some noise weights and biases resulting in problems with layers with dynamic shapes. You should write your own extension for every custom layer you topology might have. For more details, refer to [Extending Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 14. What does the message "Cannot infer shape for node {} because there is no Caffe available. Please register python infer function for op or use Caffe for shape inference" mean? <a name="question-14"></a>

Your model contains a custom layer and you have correctly registered it with the `CustomLayersMapping.xml` file. These steps are required to offload shape inference of the custom layer with the help of the system Caffe\*. However, the Model Optimizer could not import a Caffe package. Make sure that you have built Caffe with a `pycaffe` target and added it into the `PYTHONPATH` environment variable. For more information, please refer to the [Configuring the Model Optimizer](customize_model_optimizer/Legacy_Mode_for_Caffe_Custom_Layers.md). At the same time, it is highly recommend to avoid dependency on Caffe and write your own Model Optimizer extension for your custom layer. For more information, refer to the FAQ [#45](#question-45).

#### 15. What does the message "Framework name can not be deduced from the given options. Use --framework to choose one of Caffe, TensorFlow, MXNet" mean? <a name="question-15"></a>

You have run the Model Optimizer without a flag `--framework caffe|tf|mxnet`. Model Optimizer tries to deduce the framework by the input model file extension (`.pb` for TensorFlow\*, `.caffemodel` for Caffe\*, `.params` for MXNet\*). Your input model might have a different extension and you need to explicitly set the source framework. For example, use `--framework caffe`.

#### 16. What does the message "Input shape is required to convert MXNet model. Please provide it with --input_shape" mean? <a name="question-16"></a>

Input shape was not provided. That is mandatory for converting an MXNet\* model to the Intermediate Representation, because MXNet models do not contain information about input shapes. Please, use the `--input_shape` flag to specify it. For more information about using the `--input_shape`, refer to the FAQ [#57](#question-57).

#### 17. What does the message "Both --mean_file and mean_values are specified. Specify either mean file or mean values" mean? <a name="question-17"></a>

`--mean_file` and `--mean_values` are two ways of specifying preprocessing for the input. However, they cannot be used together, as it would mean double subtraction and lead to ambiguity. Choose one of these options and pass it using the corresponding CLI option.

#### 18. What does the message "Negative value specified for --mean_file_offsets option. Please specify positive integer values in format '(x,y)'" mean? <a name="question-18"></a>

You might have specified negative values with `--mean_file_offsets`. Only positive integer values in format '(x,y)' must be used.

#### 19. What does the message "Both --scale and --scale_values are defined. Specify either scale factor or scale values per input channels" mean? <a name="question-19"></a>

`--scale` sets a scaling factor for all channels. `--scale_values` sets a scaling factor per each channel. Using both of them simultaneously produces ambiguity, so you must use only one of them. For more information, refer to the Using Framework-Agnostic Conversion Parameters: for <a href="ConvertFromCaffe.html#using-framework-agnostic-conv-param">Converting a Caffe* Model</a>, <a href="ConvertFromTensorFlow.html#using-framework-agnostic-conv-param">Converting a TensorFlow* Model</a>, <a href="ConvertFromMXNet.html#using-framework-agnostic-conv-param">Converting an MXNet* Model</a>.

#### 20. What does the message "Cannot find prototxt file: for Caffe please specify --input_proto - a protobuf file that stores topology and --input_model that stores pretrained weights" mean? <a name="question-20"></a>

Model Optimizer cannot find a `.prototxt` file for a specified model. By default, it must be located in the same directory as the input model with the same name (except extension). If any of these conditions is not satisfied, use `--input_proto` to specify the path to the `.prototxt` file.

#### 22. What does the message "Failed to create directory .. . Permission denied!" mean? <a name="question-22"></a>

Model Optimizer cannot create a directory specified via `--output_dir`. Make sure that you have enough permissions to create the specified directory.

#### 23. What does the message "Discovered data node without inputs and value" mean? <a name="question-23"></a>

One of the layers in the specified topology might not have inputs or values. Please make sure that the provided `caffemodel` and `protobuf` files are correct.

#### 24. What does the message "Part of the nodes was not translated to IE. Stopped" mean? <a name="question-24"></a>

Some of the layers are not supported by the Inference Engine and cannot be translated to an Intermediate Representation. You can extend the Model Optimizer by allowing generation of new types of layers and implement these layers in the dedicated Inference Engine plugins. For more information, refer to [Extending the Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md) page and [Inference Engine Extensibility Mechanism](../../IE_DG/Extensibility_DG/Intro.md)

#### 25. What does the message "While creating an edge from .. to .. : node name is undefined in the graph. Check correctness of the input model" mean? <a name="question-25"></a>

Model Optimizer cannot build a graph based on a specified model. Most likely, it is incorrect.

#### 26. What does the message "Node does not exist in the graph" mean? <a name="question-26"></a>

You might have specified an output node via the `--output` flag that does not exist in a provided model. Make sure that the specified output is correct and this node exists in the current model.

#### 27. What does the message "--input parameter was provided. Other inputs are needed for output computation. Provide more inputs or choose another place to cut the net" mean? <a name="question-27"></a>

Most likely, the Model Optimizer tried to cut the model by a specified input. However, other inputs are needed.

#### 28. What does the message "Placeholder node does not have an input port, but input port was provided" mean?  <a name="question-28"></a>

You might have specified a placeholder node with an input node, while the placeholder node does not have it the model.

#### 29. What does the message "Port index is out of number of available input ports for node" mean? <a name="question-29"></a>

This error occurs when an incorrect input port is specified with the `--input` command line argument. When using `--input`, you can optionally specify an input port in the form: `X:node_name`, where `X` is an integer index of the input port starting from 0 and `node_name` is the name of a node in the model. This error occurs when the specified input port `X` is not in the range 0..(n-1), where n is the number of input ports for the node. Please, specify a correct port index, or do not use it if it is not needed.

#### 30. What does the message "Node has more than 1 input and input shapes were provided. Try not to provide input shapes or specify input port with PORT:NODE notation, where PORT is an integer" mean? <a name="question-30"></a>

This error occurs when an incorrect combination of the `--input` and `--input_shape` command line options is used. Using both `--input` and `--input_shape` is valid only if `--input` points to the `Placeholder` node, a node with one input port or `--input` has the form `PORT:NODE`, where `PORT` is an integer port index of input for node `NODE`. Otherwise, the combination of `--input` and `--input_shape` is incorrect.

#### 31. What does the message "Input port > 0 in --input is not supported if --input_shape is not provided. Node: NAME_OF_THE_NODE. Omit port index and all input ports will be replaced by placeholders. Or provide --input_shape" mean? <a name="question-31"></a>

When using the `PORT:NODE` notation for the `--input` command line argument and `PORT` > 0, you should specify `--input_shape` for this input. This is a limitation of the current Model Optimizer implementation.

#### 32. What does the message "No or multiple placeholders in the model, but only one shape is provided, cannot set it" mean? <a name="question-32"></a>

Looks like you have provided only one shape for the placeholder, however there are no or multiple inputs in the model. Please, make sure that you have provided correct data for placeholder nodes.

#### 33. What does the message "The amount of input nodes for port is not equal to 1" mean? <a name="question-33"></a>

This error occurs when the `SubgraphMatch.single_input_node` function is used for an input port that supplies more than one node in a sub-graph. The `single_input_node` function can be used only for ports that has a single consumer inside the matching sub-graph. When multiple nodes are connected to the port, use the `input_nodes` function or `node_by_pattern` function instead of `single_input_node`. Please, refer to [Sub-Graph Replacement in the Model Optimizer](customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md) for more details.

#### 34. What does the message "Output node for port has already been specified" mean? <a name="question-34"></a>

This error occurs when the `SubgraphMatch._add_output_node` function is called manually from user's extension code. This is an internal function, and you should not call it directly.

#### 35. What does the message "Unsupported match kind.... Match kinds "points" or "scope" are supported only" mean? <a name="question-35"></a>

While using configuration file to implement a TensorFlow\* front replacement extension, an incorrect match kind was used. Only `points` or `scope` match kinds are supported. Please, refer to [Sub-Graph Replacement in the Model Optimizer](customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md) for more details.

#### 36. What does the message "Cannot write an event file for the TensorBoard to directory" mean? <a name="question-36"></a>

Model Optimizer tried to write an event file in the specified directory but failed to do that. That could happen because the specified directory does not exist or you do not have enough permissions to write in it.

#### 37. What does the message "There is no registered 'infer' function for node  with op = .. . Please implement this function in the extensions" mean? <a name="question-37"></a>

Most likely, you tried to extend Model Optimizer with a new primitive, but did not specify an infer function. For more information on extensions, see [Extending the Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 38. What does the message "Stopped shape/value propagation at node" mean? <a name="question-38"></a>

Model Optimizer cannot infer shapes or values for the specified node. It can happen because of a bug in the custom shape infer function, because the node inputs have incorrect values/shapes, or because the input shapes are incorrect.

#### 39. What does the message "The input with shape .. does not have the batch dimension" mean? <a name="question-39"></a>

Batch dimension is the first dimension in the shape and it should be equal to 1 or undefined. In your case, it is not equal to either 1 or undefined, which is why the `-b` shortcut produces undefined and unspecified behavior. To resolve the issue, specify full shapes for each input with the `--input_shape` option. Run Model Optimizer with the `--help` option to learn more about the notation for input shapes.

#### 40. What does the message "Not all output shapes were inferred or fully defined for node" mean? <a name="question-40"></a>

Most likely, the shape is not defined (partially or fully) for the specified node. You can use `--input_shape` with positive integers to override model input shapes.

#### 41. What does the message "Shape for tensor is not defined. Can not proceed" mean? <a name="question-41"></a>

This error occurs when the `--input` command line option is used to cut a model and `--input_shape` is not used to override shapes for a node and a shape for the node cannot be inferred by Model Optimizer. You need to help Model Optimizer and specify shapes with `--input_shape` for each node that is specified with the `--input` command line option.

#### 42. What does the message "Module TensorFlow was not found. Please install TensorFlow 1.2 or higher" mean? <a name="question-42"></a>

To convert TensorFlow\* models with Model Optimizer, TensorFlow 1.2 or newer must be installed. For more information on prerequisites, see [Configuring the Model Optimizer](Config_Model_Optimizer.md).

#### 43. What does the message "Cannot read the model file: it is incorrect TensorFlow model file or missing" mean? <a name="question-43"></a>

The model file should contain a frozen TensorFlow\* graph in the text or binary format. Make sure that `--input_model_is_text` is provided for a model in the text format. By default, a model is interpreted as binary file.

#### 44. What does the message "Cannot pre-process TensorFlow graph after reading from model file. File is corrupt or has unsupported format" mean? <a name="question-44"></a>

Most likely, there is a problem with the specified file for model. The file exists, but it has bad formatting or is corrupted.

#### 45. What does the message "Found custom layer. Model Optimizer does not support this layer. Please, register it in CustomLayersMapping.xml or implement extension" mean? <a name="question-45"></a>

This means that the layer `{layer_name}` is not supported in the Model Optimizer. You can find a list of all unsupported layers in the corresponding section. You should add this layer to `CustomLayersMapping.xml` ([Legacy Mode for Caffe* Custom Layers](customize_model_optimizer/Legacy_Mode_for_Caffe_Custom_Layers.md)) or implement the extensions for this layer ([Extending Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md)).

#### 46. What does the message "Custom replacement configuration file does not exist" mean? <a name="question-46"></a>

Path to the custom replacement configuration file was provided with the `--transformations_config` flag, but the file could not be found. Please, make sure that the specified path is correct and the file exists.

#### 47. What does the message "Extractors collection have case insensitive duplicates" mean? <a name="question-47"></a>

When extending Model Optimizer with new primitives keep in mind that their names are case insensitive. Most likely, another operation with the same name is already defined. For more information, see [Extending the Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 48. What does the message "Input model name is not in an expected format, cannot extract iteration number" mean? <a name="question-48"></a>

Model Optimizer can not load an MXNet\* model in the specified file format. Please, use the `.json` or `.param` format.

#### 49. What does the message "Cannot convert type of placeholder because not all of its outputs are 'Cast' to float operations" mean? <a name="question-49"></a>

There are models where `Placeholder` has the UINT8 type and the first operation after it is 'Cast', which casts the input to FP32. Model Optimizer detected that the `Placeholder` has the UINT8 type, but the next operation is not 'Cast' to float. Model Optimizer does not support such a case. Please, change the model to have placeholder FP32 data type.

#### 50. What does the message "Data type is unsupported" mean? <a name="question-50"></a>

Model Optimizer cannot convert the model to the specified data type. Currently, FP16 and FP32 are supported. Please, specify the data type with the `--data_type` flag. The available values are: FP16, FP32, half, float.

#### 51. What does the message "No node with name ..." mean? <a name="question-51"></a>

Model Optimizer tried to access a node that does not exist. This could happen if you have incorrectly specified placeholder, input or output node name.

#### 52. What does the message "Module mxnet was not found. Please install MXNet 1.0.0" mean? <a name="question-52"></a>

To convert MXNet\* models with Model Optimizer, MXNet 1.0.0 must be installed. For more information about prerequisites, see [Configuring the Model Optimizer](Config_Model_Optimizer.md).

#### 53. What does the message "The following error happened while loading MXNet model .." mean? <a name="question-53"></a>

Most likely, there is a problem with loading of the MXNet\* model. Please, make sure that the specified path is correct, the model exists, it is not corrupted, and you have sufficient permissions to work with it.

#### 54. What does the message "The following error happened while processing input shapes: .." mean? <a name="question-54"></a>

Please, make sure that inputs are defined and have correct shapes. You can use `--input_shape` with positive integers to override model input shapes.

#### 55. What does the message "Attempt to register of custom name for the second time as class. Note that custom names are case-insensitive" mean? <a name="question-55"></a>

When extending Model Optimizer with new primitives keep in mind that their names are case insensitive. Most likely, another operation with the same name is already defined. For more information, see [Extending the Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md) .

#### 56. What does the message "Both --input_shape and --batch were provided. Please, provide only one of them" mean? <a name="question-56"></a>

You cannot specify the batch and the input shape at the same time. You should specify a desired batch as the first value of the input shape.

#### 57. What does the message "Input shape .. cannot be parsed" mean? <a name="question-57"></a>

The specified input shape cannot be parsed. Please, define it in one of the following ways:

*   
```shell
python3 mo.py --input_model <INPUT_MODEL>.caffemodel --input_shape (1,3,227,227)
```
*
```shell
python3 mo.py --input_model <INPUT_MODEL>.caffemodel --input_shape [1,3,227,227]
```
*   In case of multi input topology you should also specify inputs:
```shell
python3 mo.py --input_model /path-to/your-model.caffemodel --input data,rois --input_shape (1,3,227,227),(1,6,1,1)
```

Keep in mind that there is no space between and inside the brackets for input shapes.

#### 58. What does the message "Please provide input layer names for input layer shapes" mean? <a name="question-58"></a>

When specifying input shapes for several layers, you must provide names for inputs, whose shapes will be overwritten. For usage examples, see [Converting a Caffe\* Model](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe.html). Additional information for `--input_shape` is in FAQ [#57](#question-57).

#### 59. What does the message "Values cannot be parsed" mean? <a name="question-59"></a>

Mean values for the given parameter cannot be parsed. It should be a string with a list of mean values. For example, in '(1,2,3)', 1 stands for the RED channel, 2 for the GREEN channel, 3 for the BLUE channel.

#### 60. What does the message ".. channels are expected for given values" mean? <a name="question-60"></a>

The number of channels and the number of given values for mean values do not match. The shape should be defined as '(R,G,B)' or '[R,G,B]'. The shape should not contain undefined dimensions (? or -1). The order of values is as follows: (value for a RED channel, value for a GREEN channel, value for a BLUE channel).

#### 61. What does the message "You should specify input for each mean value" mean? <a name="question-61"></a>

Most likely, you have not specified inputs using `--mean_values`. Please, specify inputs with the `--input` flag. For usage examples, please, refer to FAQ [#63](#question-63).

#### 62. What does the message "You should specify input for each scale value" mean? <a name="question-62"></a>

Most likely, you have not specified inputs using `--scale_values`. Please, specify inputs with the `--input` flag. For usage examples, please, refer to FAQ [#64](#question-64).

#### 63. What does the message "Number of inputs and mean values does not match" mean? <a name="question-63"></a>

The number of specified mean values and the number of inputs must be equal. Please, refer to [Converting a Caffe* Model](convert_model/Convert_Model_From_Caffe.md) for a usage example.

#### 64. What does the message "Number of inputs and scale values does not match" mean? <a name="question-64"></a>

The number of specified scale values and the number of inputs must be equal. Please, refer to [Converting a Caffe* Model](convert_model/Convert_Model_From_Caffe.md) for a usage example.

#### 65. What does the message "No class registered for match kind ... Supported match kinds are .. " mean? <a name="question-65"></a>

A replacement defined in the configuration file for sub-graph replacement using node names patterns or start/end nodes has the `match_kind` attribute. The attribute may have only one of the values: `scope` or `points`. If a different value is provided, this error is displayed.

#### 66. What does the message "No instance(s) is(are) defined for the custom replacement" mean? <a name="question-66"></a>

A replacement defined in the configuration file for sub-graph replacement using node names patterns or start/end nodes has the `instances` attribute. This attribute is mandatory, and it causes this error if it is missing. Refer to documentation with a description of the sub-graph replacement feature.

#### 67. What does the message "The instance must be a single dictionary for the custom replacement with id .." mean? <a name="question-67"></a>

A replacement defined in the configuration file for sub-graph replacement using start/end nodes has the `instances` attribute. For this type of replacement, the instance must be defined with a dictionary with two keys `start_points` and `end_points`. Values for these keys are lists with the start and end node names, respectively. Refer to documentation with a description of the sub-graph replacement feature.

#### 68. What does the message "No instances are defined for replacement with id .. " mean? <a name="question-68"></a>

A replacement for the specified id is not defined in the configuration file. Please, refer to FAQ [#66](#question-66) for more information.

#### 69. What does the message "Custom replacements configuration file .. does not exist" mean? <a name="question-69"></a>

Path to a custom replacement configuration file was provided with the `--transformations_config` flag, but it cannot be found. Please, make sure that the specified path is correct and the file exists.

#### 70. What does the message "Failed to parse custom replacements configuration file .." mean? <a name="question-70"></a>

The file for custom replacement configuration provided with the `--transformations_config` flag cannot be parsed. In particular, it should have a valid JSON structure. For more details, refer to [JSON Schema Reference](https://spacetelescope.github.io/understanding-json-schema/reference/index.html).

#### 71. What does the message "One of the custom replacements in the configuration file .. does not contain attribute 'id'" mean? <a name="question-71"></a>

Every custom replacement should declare a set of mandatory attributes and their values. For more details, refer to FAQ [#72](#question-72).

#### 72. What does the message "File .. validation failed" mean? <a name="question-72"></a>

The file for custom replacement configuration provided with the `--transformations_config` flag cannot pass validation. Make sure that you have specified `id`, `instances` and `match_kind` for all the patterns.

#### 73. What does the message "Cannot update the file .. because it is broken" mean? <a name="question-73"></a>

The custom replacement configuration file provided with the `--tensorflow_custom_operations_config_update` cannot be parsed. Please, make sure that the file is correct and refer to FAQs [#69](#question-69), [#70](#question-70), [#71](#question-71), and [#72](#question-72).

#### 74. What does the message "End node .. is not reachable from start nodes: .." mean? <a name="question-74"></a>

This error occurs when you try to make a sub-graph match. It is detected that between the start and end nodes that were specified as inputs/outputs of the subgraph to find, there are nodes that are marked as outputs but there is no path from them to the input nodes. Make sure that the subgraph you want to match does actually contain all the specified output nodes.

#### 75. What does the message "Sub-graph contains network input node .." mean? <a name="question-75"></a>

Start or end node for the sub-graph replacement using start/end nodes is specified incorrectly. Model Optimizer finds internal nodes of the sub-graph strictly "between" the start and end nodes. Then it adds all input nodes to the sub-graph (and inputs of their inputs and so on) for these "internal" nodes. The error reports, that the Model Optimizer reached input node during this phase. This means that the start/end points are specified incorrectly in the configuration file. Refer to documentation with a description of the sub-graph replacement feature.

#### 76. What does the message "... elements of ... were clipped to infinity while converting a blob for node [...] to ..." mean? <a name="question-76"></a>

This message may appear when the `--data_type=FP16` command line option is used. This option implies conversion of all the blobs in the node to FP16. If a value in a blob is out of the range of valid FP16 values, the value is converted to positive or negative infinity. It may lead to incorrect results of inference or may not be a problem, depending on the model. The number of such elements and the total number of elements in the blob is printed out together with the name of the node, where this blob is used.

#### 77. What does the message "... elements of ... were clipped to zero while converting a blob for node [...] to ..." mean? <a name="question-77"></a>

This message may appear when the `--data_type=FP16` command line option is used. This option implies conversion of all blobs in the mode to FP16. If a value in the blob is so close to zero that it cannot be represented as a valid FP16 value, it is converted to a true zero FP16 value. Depending on the model, it may lead to incorrect results of inference or may not be a problem. The number of such elements and the total number of elements in the blob are printed out together with a name of the node, where this blob is used.

#### 78. What does the message "The amount of nodes matched pattern ... is not equal to 1" mean? <a name="question-78"></a>

This error occurs when the `SubgraphMatch.node_by_pattern` function is used with a pattern that does not uniquely identify a single node in a sub-graph. Try to extend the pattern string to make unambiguous match to a single sub-graph node. For more details, refer to [Sub-graph Replacement in the Model Optimizer](customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md).

#### 79. What does the message "The topology contains no "input" layers" mean? <a name="question-79"></a>

Your Caffe\* topology `.prototxt` file is intended for training. Model Optimizer expects a deployment-ready `.prototxt` file. To fix the problem, prepare a deployment-ready `.prototxt` file. Usually, preparation of a deploy-ready topology results in removing `data` layer(s), adding `input` layer(s), and removing loss layer(s).

#### 80. What does the message "Warning: please expect that Model Optimizer conversion might be slow" mean? <a name="question-80"></a>

You are using an unsupported Python\* version. Use only versions 3.4 - 3.6 for the C++ `protobuf` implementation that is supplied with the OpenVINO Toolkit. You can still boost conversion speed by building protobuf library from sources. For complete instructions about building `protobuf` from sources, see the appropriate section in [Converting a Model to Intermediate Representation](Config_Model_Optimizer.md).

#### 81. What does the message "Arguments --nd_prefix_name, --pretrained_model_name and --input_symbol should be provided. Please provide all or do not use any." mean? <a name="question-81"></a>

This error occurs if you do not provide `--nd_prefix_name`, `--pretrained_model_name` and `--input_symbol` parameters. 
Model Optimizer requires both `.params` and `.nd` model files to merge into the result file (`.params`). Topology 
description (`.json` file) should be prepared (merged) in advance and provided with `--input_symbol` parameter.

If you add to your model additional layers and weights that are in `.nd` files, the Model Optimizer can build a model 
from one `.params` file and two additional `.nd` files (`*_args.nd`, `*_auxs.nd`).
To do that, provide both CLI options or do not pass them if you want to convert an MXNet model without additional weights.
For more information, refer to [Converting a MXNet* Model](convert_model/Convert_Model_From_MxNet.md).

#### 82. What does the message "You should specify input for mean/scale values" mean? <a name="question-82"></a>

In case when the model has multiple inputs and you want to provide mean/scale values, you need to pass those values for each input. More specifically, a number of passed values should be the same as the number of inputs of the model. 
For more information, refer to [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md).

#### 83. What does the message "Input with name ... not found!" mean? <a name="question-83"></a>

When you passed the mean/scale values and specify names of input layers of the model, you might have used the name that does not correspond to any input layer. Make sure that by passing values with `--input` option, you list only names of the input layers of your model.
For more information, refer to the [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md).

#### 84. What does the message "Specified input json ... does not exist" mean? <a name="question-84"></a>

Most likely, `.json` file does not exist or has a name that does not match the notation of MXNet. Make sure that the file exists and it has a correct name.
For more information, refer to [Converting a MXNet\* Model](convert_model/Convert_Model_From_MxNet.md).

#### 85. What does the message "Unsupported Input model file type ... Model Optimizer support only .params and .nd files format" mean? <a name="question-85"></a>

Model Optimizer for MXNet supports only `.params` and `.nd` files formats. Most likely, you specified some unsupported file format in `--input_model`.
For more information, refer to [Converting a MXNet* Model](convert_model/Convert_Model_From_MxNet.md).

#### 86. What does the message "Operation ... not supported. Please register it as custom op" mean? <a name="question-86"></a>

Model Optimizer tried to load the model that contains some unsupported operations. 
If you want to convert model that contains unsupported operations you need to prepare extension for all such operations.
For more information, refer to [Extending Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 87. What does the message "Can not register Op ... Please, call function 'register_caffe_python_extractor' with parameter 'name'" mean? <a name="question-87"></a>

This error appears if the class of implementation of op for Python Caffe layer could not be used by Model Optimizer. Python layers should be handled differently compared to ordinary Caffe layers.

In particular, you need to call the function `register_caffe_python_extractor` and pass `name` as the second argument of the function.
The name should be the compilation of the layer name and the module name separated by a dot. 

For example, your topology contains this layer with type `Python`:

```
layer {
  name: 'proposal'
  type: 'Python'
  ...
  python_param {
    module: 'rpn.proposal_layer'
    layer: 'ProposalLayer'
    param_str: "'feat_stride': 16"
  }
}
```

What you do first is implementing an extension for this layer in the Model Optimizer as an ancestor of `Op` class.
```
class ProposalPythonExampleOp(Op):
       op = 'Proposal'
   
       def __init__(self, graph: nx.MultiDiGraph, attrs: dict):
           ...
```

It is mandatory to call two functions right after the implementation of that class:
```
class ProposalPythonExampleOp(Op):
      ...

register_caffe_python_extractor(ProposalPythonExampleOp, 'rpn.proposal_layer.ProposalLayer')
Op.excluded_classes.append(ProposalPythonExampleOp)
```

Note that the first call <code>register_caffe_python_extractor(ProposalPythonExampleOp, 'rpn.proposal_layer.ProposalLayer')</code> registers extension of the layer in the Model Optimizer that will be found by the specific name (mandatory to join module name and layer name): <code>rpn.proposal_layer.ProposalLayer</code>.

The second call prevents Model Optimizer from using this extension as if it is an extension for 
a layer with type `Proposal`. Otherwise, this layer can be chosen as an implementation of extension that can lead to potential issues.
For more information, refer to the [Extending Model Optimizer with New Primitives](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 88. What does the message "Model Optimizer is unable to calculate output shape of Memory node .." mean? <a name="question-88"></a>

Model Optimizer supports only `Memory` layers, in which `input_memory` goes before `ScaleShift` or `FullyConnected` layer.  
This error message means that in your model the layer after input memory is not of type `ScaleShift` or `FullyConnected`.
This is a known limitation.

#### 89. What do the messages "File ...  does not appear to be a Kaldi file (magic number does not match)", "Kaldi model should start with <Nnet> tag" mean? <a name="question-89"></a>

These error messages mean that the Model Optimizer does not support your Kaldi\* model, because check sum of the model is not 
16896 (the model should start with this number) or model file does not contain tag `<Net>` as a starting one.
Double check that you provide a path to a true Kaldi model and try again.

#### 90. What do the messages "Expect counts file to be one-line file." or "Expect counts file to contain list of integers" mean? <a name="question-90"></a>

These messages mean that you passed the file counts containing not one line. The count file should start with 
`[` and end with  `]`,  and integer values should be separated by space between those signs.

#### 91. What does the message "Model Optimizer is not able to read Kaldi model .." mean? <a name="question-91"></a>

There are multiple reasons why the Model Optimizer does not accept a Kaldi topology:
file is not available or does not exist. Refer to FAQ [#89](#question-89).

#### 92. What does the message "Model Optimizer is not able to read counts file  .." mean? <a name="question-92"></a>

There are multiple reasons why the Model Optimizer does not accept a counts file:
file is not available or does not exist. Also refer to FAQ [#90](#question-90).

#### 93. What does the message "For legacy MXNet models Model Optimizer does not support conversion of old MXNet models (trained with 1.0.0 version of MXNet and lower) with custom layers." mean? <a name="question-93"></a>

This message means that if you have model with custom layers and its json file has been generated with MXNet version
lower than 1.0.0, Model Optimizer does not support such topologies. If you want to convert it you have to rebuild 
MXNet with unsupported layers or generate new json with MXNet version 1.0.0 and higher. Also you need to implement 
Inference Engine extension for used custom layers.
For more information, refer to the [appropriate section of Model Optimizer configuration](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md).

#### 97. What does the message "Graph contains a cycle. Can not proceed .." mean?  <a name="question-97"></a>

Model Optimizer supports only straightforward models without cycles.

There are multiple ways to avoid cycles:

For Tensorflow: 
* [Convert models, created with TensorFlow Object Detection API](convert_model/tf_specific/Convert_Object_Detection_API_Models.md)

For all frameworks: 
1. [Replace cycle containing Sub-graph in Model Optimizer](customize_model_optimizer/Subgraph_Replacement_Model_Optimizer.md)
2. [Extend Model Optimizer with New Primitives from first step](customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md)

or
* Edit network in original framework to exclude cycle.

#### 98. What does the message "Can not transpose attribute '..' with value .. for node '..' .." mean?  <a name="question-98"></a>

This message means that model is not supported. It may be caused by using shapes larger than 4-D.
There are two ways to avoid such message:

1. [Cut model part containing such layers in Model Optimizer](convert_model/Cutting_Model.md) 
2. Edit network in original framework to exclude such layers.

#### 99. What does the message "Expected token `</ParallelComponent>`, has `...`" mean?  <a name="question-99"></a>

This error messages mean that Model Optimizer does not support your Kaldi model, because the Net contains `ParallelComponent` that does not end by tag `</ParallelComponent>`.
Double check that you provide a path to a true Kaldi model and try again.

#### 100. What does the message "Interp layer shape inference function may be wrong, please, try to update layer shape inference function in the file (extensions/ops/interp.op at the line ...)." mean?  <a name="question-100"></a>

There are many flavors of Caffe framework, and most layers in them are implemented identically.
But there are exceptions. For example, output value of layer Interp is calculated differently in Deeplab-Caffe and classic Caffe. So if your model contain layer Interp and converting of your model has failed, please modify the 'interp_infer' function in the file extensions/ops/interp.op according to the comments of the file.

#### 101. What does the message "Mean/scale values should ..." mean? <a name="question-101"></a>

It means that your mean/scale values have wrong format. Specify mean/scale values using the form `layer_name(val1,val2,val3)`. 
You need to specify values for each input of the model. For more information, refer to [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md).

#### 102. What does the message "Operation _contrib_box_nms is not supported ..." mean? <a name="question-102"></a>

It means that you trying to convert the topology which contains '_contrib_box_nms' operation which is not supported directly. However the sub-graph of operations including the '_contrib_box_nms' could be replaced with DetectionOutput layer if your topology is one of the gluoncv topologies. Specify '--enable_ssd_gluoncv' command line parameter for the Model Optimizer to enable this transformation.
