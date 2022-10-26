# Model Optimizer Frequently Asked Questions  {#openvino_docs_MO_DG_prepare_model_Model_Optimizer_FAQ}

If your question is not covered by the topics below, use the [OpenVINO&trade; Support page](https://software.intel.com/en-us/openvino-toolkit/documentation/get-started), where you can participate on a free forum.

#### 1. What does the message "[ ERROR ]: Current caffe.proto does not contain field" mean? <a name="question-1"></a>

Internally, Model Optimizer uses a protobuf library to parse and load Caffe models. This library requires a file grammar and a generated parser. For a Caffe fallback, Model Optimizer uses a Caffe-generated parser for a Caffe-specific `.proto` file (which is usually located in the `src/caffe/proto` directory). Make sure that you install exactly the same version of Caffe (with Python interface) as that was used to create the model.

If you just want to experiment with Model Optimizer and test a Python extension for working with your custom
layers without building Caffe, add the layer description to the `caffe.proto` file and generate a parser for it.

For example, to add the description of the `CustomReshape` layer, which is an artificial layer not present in any `caffe.proto` files:

1.  Add the following lines to the `caffe.proto` file:
```shell
    package mo_caffe; // To avoid conflict with Caffe system, it is highly recommended to specify different package name.
    ...
    message LayerParameter {
      // Other layers parameters description.
      ...
      optional CustomReshapeParameter custom_reshape_param = 546; // 546 - ID is any number not present in caffe.proto.
    }
    // The lines from here to the end of the file are describing contents of this parameter.
    message CustomReshapeParameter {
      optional BlobShape shape = 1; // Just use the same parameter type as some other Caffe layers.
    }
```

2.  Generate a new parser:
```shell
cd <SITE_PACKAGES_WITH_INSTALLED_OPENVINO>/openvino/tools/mo/front/caffe/proto
python3 generate_caffe_pb2.py --input_proto <PATH_TO_CUSTOM_CAFFE>/src/caffe/proto/caffe.proto
```
where `PATH_TO_CUSTOM_CAFFE` is the path to the root directory of custom Caffe.

3.  Now, Model Optimizer is able to load the model into memory and start working with your extensions if there are any.

However, since your model has custom layers, you must register them as custom. To learn more about it, refer to [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md).

#### 2. How do I create a bare caffemodel, if I have only prototxt? <a name="question-2"></a>

You need the Caffe Python interface. In this case, do the following:
```shell
python3
import caffe
net = caffe.Net('<PATH_TO_PROTOTXT>/my_net.prototxt', caffe.TEST)
net.save('<PATH_TO_PROTOTXT>/my_net.caffemodel')
```
#### 3. What does the message "[ ERROR ]: Unable to create ports for node with id" mean? <a name="question-3"></a>

Most likely, Model Optimizer does not know how to infer output shapes of some layers in the given topology.
To lessen the scope, compile the list of layers that are custom for Model Optimizer: present in the topology,
absent in the [list of supported layers](Supported_Frameworks_Layers.md) for the target framework. Then, refer to available options in the corresponding section in the  [Custom Layers in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md) page.

#### 4. What does the message "Input image of shape is larger than mean image from file" mean? <a name="question-4"></a>

Your model input shapes must be smaller than or equal to the shapes of the mean image file you provide. The idea behind the mean file is to subtract its values from the input image in an element-wise manner. When the mean file is smaller than the input image, there are not enough values to perform element-wise subtraction. Also, make sure you use the mean file that was used during the network training phase. Note that the mean file is dependent on dataset.

#### 5. What does the message "Mean file is empty" mean? <a name="question-5"></a>

Most likely, the mean file specified with the `--mean_file` flag is empty while Model Optimizer is launched. Make sure that this is exactly the required mean file and try to regenerate it from the given dataset if possible.

#### 6. What does the message "Probably mean file has incorrect format" mean? <a name="question-6"></a>

The mean file that you provide for Model Optimizer must be in the `.binaryproto` format. You can try to check the content, using recommendations from the BVLC Caffe ([#290](https://github.com/BVLC/caffe/issues/290)).

#### 7. What does the message "Invalid proto file: there is neither 'layer' nor 'layers' top-level messages" mean? <a name="question-7"></a>

The structure of any Caffe topology is described in the `caffe.proto` file of any Caffe version. For example, the following `.proto` file in Model Optimizer is used by default: `mo/front/caffe/proto/my_caffe.proto`, with the structure:
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

The structure of any Caffe topology is described in the `caffe.proto` file for any Caffe version. For example, the following `.proto` file in Model Optimizer is used by default: `mo/front/caffe/proto/my_caffe.proto`, with the structure:
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
Therefore, the input layer of the provided model must be specified in one of the following styles:

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

However, if your model contains more than one input, Model Optimizer is able to convert the model with inputs specified in one of the first three forms in the above list. The 4th form is not supported for multi-input topologies.

#### 9. What does the message "Mean file for topologies with multiple inputs is not supported" mean? <a name="question-9"></a>

Model Optimizer does not support mean file processing for topologies with more than one input. In this case, you need to perform preprocessing of the inputs for a generated Intermediate Representation in OpenVINO Runtime to perform subtraction for every input of your multi-input model. See the [Overview of Preprocessing](../../OV_Runtime_UG/preprocessing_overview.md) for details.

#### 10. What does the message "Cannot load or process mean file: value error" mean? <a name="question-10"></a>

There are multiple reasons why Model Optimizer does not accept the mean file. See FAQs [#4](#question-4), [#5](#question-5), and [#6](#question-6).

#### 11. What does the message "Invalid prototxt file: value error" mean? <a name="question-11"></a>

There are multiple reasons why Model Optimizer does not accept a Caffe topology. See FAQs [#7](#question-7) and [#20](#question-20).

#### 12. What does the message "Error happened while constructing caffe.Net in the Caffe fallback function" mean? <a name="question-12"></a>

Model Optimizer tried to infer a specified layer via the Caffe framework. However, it cannot construct a net using the Caffe Python interface. Make sure that your `caffemodel` and `prototxt` files are correct. To ensure that the problem is not in the `prototxt` file, see FAQ [#2](#question-2).

#### 13. What does the message "Cannot infer shapes due to exception in Caffe" mean? <a name="question-13"></a>

Model Optimizer tried to infer a custom layer via the Caffe framework, but the model could not be inferred using Caffe. This might happen if you try to convert the model with some noise weights and biases, which conflict with layers that have dynamic shapes. You should write your own extension for every custom layer your topology might have. For more details, refer to the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) page.

#### 14. What does the message "Cannot infer shape for node {} because there is no Caffe available. Please register python infer function for op or use Caffe for shape inference" mean? <a name="question-14"></a>

Your model contains a custom layer and you have correctly registered it with the `CustomLayersMapping.xml` file. These steps are required to offload shape inference of the custom layer with the help of the system Caffe. However, Model Optimizer could not import a Caffe package. Make sure that you have built Caffe with a `pycaffe` target and added it to the `PYTHONPATH` environment variable. At the same time, it is highly recommended to avoid dependency on Caffe and write your own Model Optimizer extension for your custom layer. For more information, refer to FAQ [#44](#question-44).

#### 15. What does the message "Framework name can not be deduced from the given options. Use --framework to choose one of Caffe, TensorFlow, MXNet" mean? <a name="question-15"></a>

You have run Model Optimizer without a flag `--framework caffe|tf|mxnet`. Model Optimizer tries to deduce the framework by the extension of input model file (`.pb` for TensorFlow, `.caffemodel` for Caffe, `.params` for Apache MXNet). Your input model might have a different extension and you need to explicitly set the source framework. For example, use `--framework caffe`.

#### 16. What does the message "Input shape is required to convert MXNet model. Please provide it with --input_shape" mean? <a name="question-16"></a>

Input shape was not provided. That is mandatory for converting an MXNet model to the OpenVINO Intermediate Representation, because MXNet models do not contain information about input shapes. Use the `--input_shape` flag to specify it. For more information about using the `--input_shape`, refer to FAQ [#56](#question-56).

#### 17. What does the message "Both --mean_file and mean_values are specified. Specify either mean file or mean values" mean? <a name="question-17"></a>

The `--mean_file` and `--mean_values` options are two ways of specifying preprocessing for the input. However, they cannot be used together, as it would mean double subtraction and lead to ambiguity. Choose one of these options and pass it with the corresponding CLI option.

#### 18. What does the message "Negative value specified for --mean_file_offsets option. Please specify positive integer values in format '(x,y)'" mean? <a name="question-18"></a>

You might have specified negative values with `--mean_file_offsets`. Only positive integer values in format '(x,y)' must be used.

#### 19. What does the message "Both --scale and --scale_values are defined. Specify either scale factor or scale values per input channels" mean? <a name="question-19"></a>

The `--scale` option sets a scaling factor for all channels, while `--scale_values` sets a scaling factor per each channel. Using both of them simultaneously produces ambiguity, so you must use only one of them. For more information, refer to the **Using Framework-Agnostic Conversion Parameters** section: for <a href="ConvertFromCaffe.html#using-framework-agnostic-conv-param">Converting a Caffe Model</a>, <a href="ConvertFromTensorFlow.html#using-framework-agnostic-conv-param">Converting a TensorFlow Model</a>, <a href="ConvertFromMXNet.html#using-framework-agnostic-conv-param">Converting an MXNet Model</a>.

#### 20. What does the message "Cannot find prototxt file: for Caffe please specify --input_proto - a protobuf file that stores topology and --input_model that stores pre-trained weights" mean? <a name="question-20"></a>

Model Optimizer cannot find a `.prototxt` file for a specified model. By default, it must be located in the same directory as the input model with the same name (except extension). If any of these conditions is not satisfied, use `--input_proto` to specify the path to the `.prototxt` file.

#### 21. What does the message "Failed to create directory .. . Permission denied!" mean? <a name="question-21"></a>

Model Optimizer cannot create a directory specified via `--output_dir`. Make sure that you have enough permissions to create the specified directory.

#### 22. What does the message "Discovered data node without inputs and value" mean? <a name="question-22"></a>

One of the layers in the specified topology might not have inputs or values. Make sure that the provided `caffemodel` and `protobuf` files are correct.

#### 23. What does the message "Part of the nodes was not translated to IE. Stopped" mean? <a name="question-23"></a>

Some of the operations are not supported by OpenVINO Runtime and cannot be translated to OpenVINO Intermediate Representation. You can extend Model Optimizer by allowing generation of new types of operations and implement these operations in the dedicated OpenVINO plugins. For more information, refer to the [OpenVINO Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 24. What does the message "While creating an edge from .. to .. : node name is undefined in the graph. Check correctness of the input model" mean? <a name="question-24"></a>

Model Optimizer cannot build a graph based on a specified model. Most likely, it is incorrect.

#### 25. What does the message "Node does not exist in the graph" mean? <a name="question-25"></a>

You might have specified an output node via the `--output` flag that does not exist in a provided model. Make sure that the specified output is correct and this node exists in the current model.

#### 26. What does the message "--input parameter was provided. Other inputs are needed for output computation. Provide more inputs or choose another place to cut the net" mean? <a name="question-26"></a>

Most likely, Model Optimizer tried to cut the model by a specified input. However, other inputs are needed.

#### 27. What does the message "Placeholder node does not have an input port, but input port was provided" mean?  <a name="question-27"></a>

You might have specified a placeholder node with an input node, while the placeholder node does not have it in the model.

#### 28. What does the message "Port index is out of number of available input ports for node" mean? <a name="question-28"></a>

This error occurs when an incorrect input port is specified with the `--input` command line argument. When using `--input`, you may optionally specify an input port in the form: `X:node_name`, where `X` is an integer index of the input port starting from 0 and `node_name` is the name of a node in the model. This error occurs when the specified input port `X` is not in the range 0..(n-1), where n is the number of input ports for the node. Specify a correct port index, or do not use it if it is not needed.

#### 29. What does the message "Node has more than 1 input and input shapes were provided. Try not to provide input shapes or specify input port with PORT:NODE notation, where PORT is an integer" mean? <a name="question-29"></a>

This error occurs when an incorrect combination of the `--input` and `--input_shape` command line options is used. Using both `--input` and `--input_shape` is valid only if `--input` points to the `Placeholder` node, a node with one input port or `--input` has the form `PORT:NODE`, where `PORT` is an integer port index of input for node `NODE`. Otherwise, the combination of `--input` and `--input_shape` is incorrect.

#### 30. What does the message "Input port > 0 in --input is not supported if --input_shape is not provided. Node: NAME_OF_THE_NODE. Omit port index and all input ports will be replaced by placeholders. Or provide --input_shape" mean? <a name="question-30"></a>

When using the `PORT:NODE` notation for the `--input` command line argument and `PORT` > 0, you should specify `--input_shape` for this input. This is a limitation of the current Model Optimizer implementation.

**NOTE**: It is no longer relevant message since the limitation on input port index for model truncation has been resolved.

#### 31. What does the message "No or multiple placeholders in the model, but only one shape is provided, cannot set it" mean? <a name="question-31"></a>

You might have provided only one shape for the placeholder, while there are none or multiple inputs in the model. Make sure that you have provided the correct data for placeholder nodes.

#### 32. What does the message "The amount of input nodes for port is not equal to 1" mean? <a name="question-32"></a>

This error occurs when the `SubgraphMatch.single_input_node` function is used for an input port that supplies more than one node in a sub-graph. The `single_input_node` function can be used only for ports that has a single consumer inside the matching sub-graph. When multiple nodes are connected to the port, use the `input_nodes` function or `node_by_pattern` function instead of `single_input_node`. For more details, refer to the **Graph Transformation Extensions** section in the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide.

#### 33. What does the message "Output node for port has already been specified" mean? <a name="question-33"></a>

This error occurs when the `SubgraphMatch._add_output_node` function is called manually from user's extension code. This is an internal function, and you should not call it directly.

#### 34. What does the message "Unsupported match kind.... Match kinds "points" or "scope" are supported only" mean? <a name="question-34"></a>

While using configuration file to implement a TensorFlow front replacement extension, an incorrect match kind was used. Only `points` or `scope` match kinds are supported.  For more details, refer to the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide.

#### 35. What does the message "Cannot write an event file for the TensorBoard to directory" mean? <a name="question-35"></a>

Model Optimizer tried to write an event file in the specified directory but failed to do that. That could happen when the specified directory does not exist or you do not have permissions to write in it.

#### 36. What does the message "There is no registered 'infer' function for node  with op = .. . Please implement this function in the extensions" mean? <a name="question-36"></a>

Most likely, you tried to extend Model Optimizer with a new primitive, but you did not specify an infer function. For more information on extensions, see the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 37. What does the message "Stopped shape/value propagation at node" mean? <a name="question-37"></a>

Model Optimizer cannot infer shapes or values for the specified node. It can happen because of the following reasons: a bug exists in the custom shape infer function, the node inputs have incorrect values/shapes, or the input shapes are incorrect.

#### 38. What does the message "The input with shape .. does not have the batch dimension" mean? <a name="question-38"></a>

Batch dimension is the first dimension in the shape and it should be equal to 1 or undefined. In your case, it is not either equal to 1 or undefined, which is why the `-b` shortcut produces undefined and unspecified behavior. To resolve the issue, specify full shapes for each input with the `--input_shape` option. Run Model Optimizer with the `--help` option to learn more about the notation for input shapes.

#### 39. What does the message "Not all output shapes were inferred or fully defined for node" mean? <a name="question-39"></a>

Most likely, the shape is not defined (partially or fully) for the specified node. You can use `--input_shape` with positive integers to override model input shapes.

#### 40. What does the message "Shape for tensor is not defined. Can not proceed" mean? <a name="question-40"></a>

This error occurs when the `--input` command-line option is used to cut a model and `--input_shape` is not used to override shapes for a node, so a shape for the node cannot be inferred by Model Optimizer. You need to help Model Optimizer by specifying shapes with `--input_shape` for each node specified with the `--input` command-line option.

#### 41. What does the message "Module TensorFlow was not found. Please install TensorFlow 1.2 or higher" mean? <a name="question-41"></a>

To convert TensorFlow models with Model Optimizer, TensorFlow 1.2 or newer must be installed. For more information on prerequisites, see the [Configuring Model Optimizer](../Deep_Learning_Model_Optimizer_DevGuide.md) guide.

#### 42. What does the message "Cannot read the model file: it is incorrect TensorFlow model file or missing" mean? <a name="question-42"></a>

The model file should contain a frozen TensorFlow graph in the text or binary format. Make sure that `--input_model_is_text` is provided for a model in the text format. By default, a model is interpreted as binary file.

#### 43. What does the message "Cannot pre-process TensorFlow graph after reading from model file. File is corrupt or has unsupported format" mean? <a name="question-43"></a>

Most likely, there is a problem with the specified file for the model. The file exists, but it has an invalid format or is corrupted.

#### 44. What does the message "Found custom layer. Model Optimizer does not support this layer. Please, register it in CustomLayersMapping.xml or implement extension" mean? <a name="question-44"></a>

This means that the layer `{layer_name}` is not supported in Model Optimizer. You will find a list of all unsupported layers in the corresponding section. You should implement the extensions for this layer. See [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) for more information.

#### 45. What does the message "Custom replacement configuration file does not exist" mean? <a name="question-45"></a>

A path to the custom replacement configuration file was provided with the `--transformations_config` flag, but the file could not be found. Make sure the specified path is correct and the file exists.

#### 46. What does the message "Extractors collection have case insensitive duplicates" mean? <a name="question-46"></a>

When extending Model Optimizer with new primitives, keep in mind that their names are case-insensitive. Most likely, another operation with the same name is already defined. For more information, see the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 47. What does the message "Input model name is not in an expected format, cannot extract iteration number" mean? <a name="question-47"></a>

Model Optimizer cannot load an MXNet model in the specified file format. Make sure you use the `.json` or `.param` format.

#### 48. What does the message "Cannot convert type of placeholder because not all of its outputs are 'Cast' to float operations" mean? <a name="question-48"></a>

There are models where `Placeholder` has the UINT8 type and the first operation after it is 'Cast', which casts the input to FP32. Model Optimizer detected that the `Placeholder` has the UINT8 type, but the next operation is not 'Cast' to float. Model Optimizer does not support such a case. Make sure you change the model to have `Placeholder` for FP32.

#### 49. What does the message "Data type is unsupported" mean? <a name="question-49"></a>

Model Optimizer cannot convert the model to the specified data type. Currently, FP16 and FP32 are supported. Make sure you specify the data type with the `--data_type` flag. The available values are: FP16, FP32, half, float.

#### 50. What does the message "No node with name ..." mean? <a name="question-50"></a>

Model Optimizer tried to access a node that does not exist. This could happen if you have incorrectly specified placeholder, input or output node name.

#### 51. What does the message "Module MXNet was not found. Please install MXNet 1.0.0" mean? <a name="question-51"></a>

To convert MXNet models with Model Optimizer, Apache MXNet 1.0.0 must be installed. For more information about prerequisites, see the[Configuring Model Optimizer](../Deep_Learning_Model_Optimizer_DevGuide.md) guide.

#### 52. What does the message "The following error happened while loading MXNet model .." mean? <a name="question-52"></a>

Most likely, there is a problem with loading of the MXNet model. Make sure the specified path is correct, the model exists and is not corrupted, and you have sufficient permissions to work with it.

#### 53. What does the message "The following error happened while processing input shapes: .." mean? <a name="question-53"></a>

Make sure inputs are defined and have correct shapes. You can use `--input_shape` with positive integers to override model input shapes.

#### 54. What does the message "Attempt to register of custom name for the second time as class. Note that custom names are case-insensitive" mean? <a name="question-54"></a>

When extending Model Optimizer with new primitives, keep in mind that their names are case-insensitive. Most likely, another operation with the same name is already defined. For more information, see the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 55. What does the message "Both --input_shape and --batch were provided. Please, provide only one of them" mean? <a name="question-55"></a>

Specifying the batch and the input shapes at the same time is not supported. You must specify a desired batch as the first value of the input shape.

#### 56. What does the message "Input shape .. cannot be parsed" mean? <a name="question-56"></a>

The specified input shape cannot be parsed. Define it in one of the following ways:

*
```shell
 mo --input_model <INPUT_MODEL>.caffemodel --input_shape (1,3,227,227)
```
*
```shell
 mo --input_model <INPUT_MODEL>.caffemodel --input_shape [1,3,227,227]
```
*   In case of multi input topology you should also specify inputs:
```shell
 mo --input_model /path-to/your-model.caffemodel --input data,rois --input_shape (1,3,227,227),(1,6,1,1)
```

Keep in mind that there is no space between and inside the brackets for input shapes.

#### 57. What does the message "Please provide input layer names for input layer shapes" mean? <a name="question-57"></a>

When specifying input shapes for several layers, you must provide names for inputs, whose shapes will be overwritten. For usage examples, see the [Converting a Caffe Model](convert_model/Convert_Model_From_Caffe.md). Additional information for `--input_shape` is in FAQ [#56](#question-56).

#### 58. What does the message "Values cannot be parsed" mean? <a name="question-58"></a>

Mean values for the given parameter cannot be parsed. It should be a string with a list of mean values. For example, in '(1,2,3)', 1 stands for the RED channel, 2 for the GREEN channel, 3 for the BLUE channel.

#### 59. What does the message ".. channels are expected for given values" mean? <a name="question-59"></a>

The number of channels and the number of given values for mean values do not match. The shape should be defined as '(R,G,B)' or '[R,G,B]'. The shape should not contain undefined dimensions (? or -1). The order of values is as follows: (value for a RED channel, value for a GREEN channel, value for a BLUE channel).

#### 60. What does the message "You should specify input for each mean value" mean? <a name="question-60"></a>

Most likely, you didn't specify inputs using `--mean_values`. Specify inputs with the `--input` flag. For usage examples, refer to the FAQ [#62](#question-62).

#### 61. What does the message "You should specify input for each scale value" mean? <a name="question-61"></a>

Most likely, you didn't specify inputs using `--scale_values`. Specify inputs with the `--input` flag. For usage examples, refer to the FAQ [#63](#question-63).

#### 62. What does the message "Number of inputs and mean values does not match" mean? <a name="question-62"></a>

The number of specified mean values and the number of inputs must be equal. For a usage example, refer to the [Converting a Caffe Model](convert_model/Convert_Model_From_Caffe.md) guide.

#### 63. What does the message "Number of inputs and scale values does not match" mean? <a name="question-63"></a>

The number of specified scale values and the number of inputs must be equal.  For a usage example, refer to the [Converting a Caffe Model](convert_model/Convert_Model_From_Caffe.md) guide.

#### 64. What does the message "No class registered for match kind ... Supported match kinds are .. " mean? <a name="question-64"></a>

A replacement defined in the configuration file for sub-graph replacement, using node names patterns or start/end nodes, has the `match_kind` attribute. The attribute may have only one of the values: `scope` or `points`. If a different value is provided, this error is displayed.

#### 65. What does the message "No instance(s) is(are) defined for the custom replacement" mean? <a name="question-65"></a>

A replacement defined in the configuration file for sub-graph replacement, using node names patterns or start/end nodes, has the `instances` attribute. This attribute is mandatory. This error will occur if the attribute is missing. For more details, refer to the **Graph Transformation Extensions** section in the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide.

#### 66. What does the message "The instance must be a single dictionary for the custom replacement with id .." mean? <a name="question-66"></a>

A replacement defined in the configuration file for sub-graph replacement, using start/end nodes, has the `instances` attribute. For this type of replacement, the instance must be defined with a dictionary with two keys `start_points` and `end_points`. Values for these keys are lists with the start and end node names, respectively. For more details, refer to the **Graph Transformation Extensions** section in the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide.

#### 67. What does the message "No instances are defined for replacement with id .. " mean? <a name="question-67"></a>

A replacement for the specified id is not defined in the configuration file. For more information, refer to the FAQ [#65](#question-65).

#### 68. What does the message "Custom replacements configuration file .. does not exist" mean? <a name="question-68"></a>

The path to a custom replacement configuration file was provided with the `--transformations_config` flag, but it cannot be found. Make sure the specified path is correct and the file exists.

#### 69. What does the message "Failed to parse custom replacements configuration file .." mean? <a name="question-69"></a>

The file for custom replacement configuration provided with the `--transformations_config` flag cannot be parsed. In particular, it should have a valid JSON structure. For more details, refer to the [JSON Schema Reference](https://spacetelescope.github.io/understanding-json-schema/reference/index.html) page.

#### 70. What does the message "One of the custom replacements in the configuration file .. does not contain attribute 'id'" mean? <a name="question-70"></a>

Every custom replacement should declare a set of mandatory attributes and their values. For more details, refer to FAQ [#71](#question-71).

#### 71. What does the message "File .. validation failed" mean? <a name="question-71"></a>

The file for custom replacement configuration provided with the `--transformations_config` flag cannot pass validation. Make sure you have specified `id`, `instances`, and `match_kind` for all the patterns.

#### 72. What does the message "Cannot update the file .. because it is broken" mean? <a name="question-72"></a>

The custom replacement configuration file provided with the `--tensorflow_custom_operations_config_update` cannot be parsed. Make sure that the file is correct and refer to FAQ [#68](#question-68), [#69](#question-69), [#70](#question-70), and [#71](#question-71).

#### 73. What does the message "End node .. is not reachable from start nodes: .." mean? <a name="question-73"></a>

This error occurs when you try to make a sub-graph match. It is detected that between the start and end nodes that were specified as inputs/outputs for the subgraph to find, there are nodes marked as outputs but there is no path from them to the input nodes. Make sure the subgraph you want to match does actually contain all the specified output nodes.

#### 74. What does the message "Sub-graph contains network input node .." mean? <a name="question-74"></a>

The start or end node for the sub-graph replacement using start/end nodes is specified incorrectly. Model Optimizer finds internal nodes of the sub-graph strictly "between" the start and end nodes, and then adds all input nodes to the sub-graph (and the inputs of their inputs, etc.) for these "internal" nodes. This error reports that Model Optimizer reached input node during this phase. This means that the start/end points are specified incorrectly in the configuration file. For more details, refer to the **Graph Transformation Extensions** section in the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide. 

#### 75. What does the message "... elements of ... were clipped to infinity while converting a blob for node [...] to ..." mean? <a name="question-75"></a>

This message may appear when the `--data_type=FP16` command-line option is used. This option implies conversion of all the blobs in the node to FP16. If a value in a blob is out of the range of valid FP16 values, the value is converted to positive or negative infinity. It may lead to incorrect results of inference or may not be a problem, depending on the model. The number of such elements and the total number of elements in the blob is printed out together with the name of the node, where this blob is used.

#### 76. What does the message "... elements of ... were clipped to zero while converting a blob for node [...] to ..." mean? <a name="question-76"></a>

This message may appear when the `--data_type=FP16` command-line option is used. This option implies conversion of all blobs in the mode to FP16. If a value in the blob is so close to zero that it cannot be represented as a valid FP16 value, it is converted to a true zero FP16 value. Depending on the model, it may lead to incorrect results of inference or may not be a problem. The number of such elements and the total number of elements in the blob are printed out together with a name of the node, where this blob is used.

#### 77. What does the message "The amount of nodes matched pattern ... is not equal to 1" mean? <a name="question-77"></a>

This error occurs when the `SubgraphMatch.node_by_pattern` function is used with a pattern that does not uniquely identify a single node in a sub-graph. Try to extend the pattern string to make unambiguous match to a single sub-graph node. For more details, refer to the **Graph Transformation Extensions** section in the [Model Optimizer Extensibility](customize_model_optimizer/Customize_Model_Optimizer.md) guide.

#### 78. What does the message "The topology contains no "input" layers" mean? <a name="question-78"></a>

Your Caffe topology `.prototxt` file is intended for training. Model Optimizer expects a deployment-ready `.prototxt` file. To fix the problem, prepare a deployment-ready `.prototxt` file. Preparation of a deploy-ready topology usually results in removing `data` layer(s), adding `input` layer(s), and removing loss layer(s).

#### 79. What does the message "Warning: please expect that Model Optimizer conversion might be slow" mean? <a name="question-79"></a>

You are using an unsupported Python version. Use only versions 3.4 - 3.6 for the C++ `protobuf` implementation that is supplied with OpenVINO toolkit. You can still boost the conversion speed by building the protobuf library from sources. For complete instructions about building `protobuf` from sources, see the appropriate section in the[Converting a Model to Intermediate Representation](../Deep_Learning_Model_Optimizer_DevGuide.md) guide.

#### 80. What does the message "Arguments --nd_prefix_name, --pretrained_model_name and --input_symbol should be provided. Please provide all or do not use any." mean? <a name="question-80"></a>

This error occurs if you did not provide the `--nd_prefix_name`, `--pretrained_model_name`, and `--input_symbol` parameters.
Model Optimizer requires both `.params` and `.nd` model files to merge into the result file (`.params`). 
Topology description (`.json` file) should be prepared (merged) in advance and provided with the `--input_symbol` parameter.

If you add additional layers and weights that are in `.nd` files to your model, Model Optimizer can build a model
from one `.params` file and two additional `.nd` files (`*_args.nd`, `*_auxs.nd`).
To do that, provide both CLI options or do not pass them if you want to convert an MXNet model without additional weights.
For more information, refer to the [Converting an MXNet Model](convert_model/Convert_Model_From_MxNet.md) guide.

#### 81. What does the message "You should specify input for mean/scale values" mean? <a name="question-81"></a>

When the model has multiple inputs and you want to provide mean/scale values, you need to pass those values for each input. More specifically, the number of passed values should be the same as the number of inputs of the model.
For more information, refer to the [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md) guide.

#### 82. What does the message "Input with name ... not found!" mean? <a name="question-82"></a>

When you passed the mean/scale values and specify names of input layers of the model, you might have used the name that does not correspond to any input layer. Make sure that you list only names of the input layers of your model when passing values with the `--input` option. 
For more information, refer to the [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md) guide.

#### 83. What does the message "Specified input json ... does not exist" mean? <a name="question-83"></a>

Most likely, `.json` file does not exist or has a name that does not match the notation of Apache MXNet. Make sure the file exists and has a correct name.
For more information, refer to the [Converting an MXNet Model](convert_model/Convert_Model_From_MxNet.md) guide.

#### 84. What does the message "Unsupported Input model file type ... Model Optimizer support only .params and .nd files format" mean? <a name="question-84"></a>

Model Optimizer for Apache MXNet supports only `.params` and `.nd` files formats. Most likely, you specified an unsupported file format in `--input_model`.
For more information, refer to [Converting an MXNet Model](convert_model/Convert_Model_From_MxNet.md).

#### 85. What does the message "Operation ... not supported. Please register it as custom op" mean? <a name="question-85"></a>

Model Optimizer tried to load the model that contains some unsupported operations.
If you want to convert model that contains unsupported operations, you need to prepare extension for all such operations.
For more information, refer to the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 86. What does the message "Can not register Op ... Please, call function 'register_caffe_python_extractor' with parameter 'name'" mean? <a name="question-86"></a>

This error appears if the class of implementation of `Op` for Python Caffe layer could not be used by Model Optimizer. Python layers should be handled differently comparing to ordinary Caffe layers.

In particular, you need to call the function `register_caffe_python_extractor` and pass `name` as the second argument of the function.
The name should be the compilation of the layer name with the module name separated by a dot.

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

The first step is to implement an extension for this layer in Model Optimizer as an ancestor of `Op` class:
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

Note that the first call <code>register_caffe_python_extractor(ProposalPythonExampleOp, 'rpn.proposal_layer.ProposalLayer')</code> registers an extension of the layer in Model Optimizer, which will be found by the specific name (mandatory to join module name and layer name): <code>rpn.proposal_layer.ProposalLayer</code>.

The second call prevents Model Optimizer from using this extension as if it is an extension for
a layer with type `Proposal`. Otherwise, this layer can be chosen as an implementation of extension that can lead to potential issues.
For more information, refer to the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 87. What does the message "Model Optimizer is unable to calculate output shape of Memory node .." mean? <a name="question-87"></a>

Model Optimizer supports only `Memory` layers, in which `input_memory` goes before `ScaleShift` or the `FullyConnected` layer.
This error message means that in your model the layer after input memory is not of the `ScaleShift` or `FullyConnected` type.
This is a known limitation.

#### 88. What do the messages "File ...  does not appear to be a Kaldi file (magic number does not match)", "Kaldi model should start with <Nnet> tag" mean? <a name="question-88"></a>

These error messages mean that Model Optimizer does not support your Kaldi model, because the `checksum` of the model is not
16896 (the model should start with this number), or the model file does not contain the `<Net>` tag as a starting one.
Make sure that you provide a path to a true Kaldi model and try again.

#### 89. What do the messages "Expect counts file to be one-line file." or "Expect counts file to contain list of integers" mean? <a name="question-89"></a>

These messages mean that the file counts you passed contain not one line. The count file should start with
`[` and end with  `]`,  and integer values should be separated by spaces between those brackets.

#### 90. What does the message "Model Optimizer is not able to read Kaldi model .." mean? <a name="question-90"></a>

There are multiple reasons why Model Optimizer does not accept a Kaldi topology, including: 
the file is not available or does not exist. Refer to FAQ [#88](#question-88).

#### 91. What does the message "Model Optimizer is not able to read counts file  .." mean? <a name="question-91"></a>

There are multiple reasons why Model Optimizer does not accept a counts file, including: 
the file is not available or does not exist. Refer to FAQ [#89](#question-89).

#### 92. What does the message "For legacy MXNet models Model Optimizer does not support conversion of old MXNet models (trained with 1.0.0 version of MXNet and lower) with custom layers." mean? <a name="question-92"></a>

This message means that if you have a model with custom layers and its JSON file has been generated with Apache MXNet version
lower than 1.0.0, Model Optimizer does not support such topologies. If you want to convert it, you have to rebuild
MXNet with unsupported layers or generate a new JSON file with Apache MXNet version 1.0.0 or higher. You also need to implement
OpenVINO extension to use custom layers.
For more information, refer to the [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md) guide.

#### 93. What does the message "Graph contains a cycle. Can not proceed .." mean?  <a name="question-93"></a>

Model Optimizer supports only straightforward models without cycles.

There are multiple ways to avoid cycles:

For Tensorflow:
* [Convert models, created with TensorFlow Object Detection API](convert_model/tf_specific/Convert_Object_Detection_API_Models.md)

For all frameworks:
1. [Replace cycle containing Sub-graph in Model Optimizer](customize_model_optimizer/Customize_Model_Optimizer.md)
2. See [OpenVINO&trade; Extensibility Mechanism](../../Extensibility_UG/Intro.md)

or
* Edit the model in its original framework to exclude cycle.

#### 94. What does the message "Can not transpose attribute '..' with value .. for node '..' .." mean?  <a name="question-94"></a>

This message means that the model is not supported. It may be caused by using shapes larger than 4-D.
There are two ways to avoid such message:

* [Cut off parts of the model](convert_model/Cutting_Model.md).
* Edit the network in its original framework to exclude such layers.

#### 95. What does the message "Expected token `</ParallelComponent>`, has `...`" mean?  <a name="question-95"></a>

This error messages mean that Model Optimizer does not support your Kaldi model, because the Net contains `ParallelComponent` that does not end with the `</ParallelComponent>` tag.
Make sure that you provide a path to a true Kaldi model and try again.

#### 96. What does the message "Interp layer shape inference function may be wrong, please, try to update layer shape inference function in the file (extensions/ops/interp.op at the line ...)." mean?  <a name="question-96"></a>

There are many flavors of Caffe framework, and most layers in them are implemented identically.
However, there are exceptions. For example, the output value of layer Interp is calculated differently in Deeplab-Caffe and classic Caffe. Therefore, if your model contains layer Interp and the conversion of your model has failed, modify the `interp_infer` function in the `extensions/ops/interp.op` file according to the comments in the file.

#### 97. What does the message "Mean/scale values should ..." mean? <a name="question-97"></a>

It means that your mean/scale values have a wrong format. Specify mean/scale values in the form of `layer_name(val1,val2,val3)`.
You need to specify values for each input of the model. For more information, refer to the [Converting a Model to Intermediate Representation](convert_model/Converting_Model.md) guide.

#### 98. What does the message "Operation _contrib_box_nms is not supported ..." mean? <a name="question-98"></a>

It means that you are trying to convert a topology contains the `_contrib_box_nms` operation which is not supported directly. However, the sub-graph of operations including `_contrib_box_nms` could be replaced with the DetectionOutput layer if your topology is one of the `gluoncv` topologies. Specify the `--enable_ssd_gluoncv` command-line parameter for Model Optimizer to enable this transformation.

#### 99. What does the message "ModelOptimizer is not able to parse *.caffemodel" mean? <a name="question-99"></a>

If a `*.caffemodel` file exists and is correct, the error occurred possibly because of the use of Python protobuf implementation. In some cases, error messages may appear during model parsing, for example: "`utf-8` codec can't decode byte 0xe0 in position 4: invalid continuation byte in field: mo_caffe.SpatialTransformerParameter.transform_type". You can either use Python 3.7 or build the `cpp` implementation of `protobuf` yourself for your version of Python. For the complete instructions about building `protobuf` from sources, see the appropriate section in the [Converting Models with Model Optimizer](../Deep_Learning_Model_Optimizer_DevGuide.md) guide.

#### 100. What does the message "SyntaxError: 'yield' inside list comprehension" during MxNet model conversion mean? <a name="question-100"></a>

The issue "SyntaxError: `yield` inside list comprehension" might occur during converting MXNet models (`mobilefacedet-v1-mxnet`, `brain-tumor-segmentation-0001`) on Windows platform with Python 3.8 environment. This issue is caused by the API changes for `yield expression` in Python 3.8.
The following workarounds are suggested to resolve this issue:
1. Use Python 3.7 to convert MXNet models on Windows
2. Update Apache MXNet by using `pip install mxnet==1.7.0.post2`
Note that it might have conflicts with previously installed PyPI dependencies.

#### 101. What does the message "The IR preparation was executed by the legacy MO path. ..." mean? <a name="question-101"></a>

For the models in ONNX format, there are two available paths of IR conversion.
The old one is handled by the old Python implementation, while the new one uses new C++ frontends.
Starting from the 2022.1 version, the default IR conversion path for ONNX models is processed using the new ONNX frontend.
Certain features, such as `--extensions` and `--transformations_config`, are not yet fully supported on the new frontends.
The new frontends support only paths to shared libraries (.dll and .so) for `--extensions`. They support JSON configurations with defined library fields for `--transformations_config`.
Inputs freezing (enabled by `--freeze_placeholder_with_value` or `--input` arguments) is not supported by the new frontends.
The IR conversion falls back to the old path if a user does not select any expected path of conversion explicitly (with `--use_new_frontend` or `--use_legacy_frontend` MO arguments) and unsupported pre-defined scenario is detected on the new frontend path.
