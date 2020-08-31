# Converting a Model Using General Conversion Parameters {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model_General}

To simply convert a model trained by any supported framework, run the Model Optimizer launch script ``mo.py`` with
specifying a path to the input model file:
```sh
python3 mo.py --input_model INPUT_MODEL
```

> **NOTE:** The color channel order (RGB or BGR) of an input data should match the channel order of the model training dataset. If they are different, perform the `RGB<->BGR` conversion specifying the command-line parameter: `--reverse_input_channels`. Otherwise, inference results may be incorrect. For details, refer to [When to Reverse Input Channels](#when_to_reverse_input_channels).

To adjust the conversion process, you can also use the general (framework-agnostic) parameters:

```sh
optional arguments:
  -h, --help            show this help message and exit
  --framework {tf,caffe,mxnet,kaldi,onnx}
                        Name of the framework used to train the input model.

Framework-agnostic parameters:
  --input_model INPUT_MODEL, -w INPUT_MODEL, -m INPUT_MODEL
                        Tensorflow*: a file with a pre-trained model (binary
                        or text .pb file after freezing). Caffe*: a model
                        proto file with model weights
  --model_name MODEL_NAME, -n MODEL_NAME
                        Model_name parameter passed to the final create_ir
                        transform. This parameter is used to name a network in
                        a generated IR and output .xml/.bin files.
  --output_dir OUTPUT_DIR, -o OUTPUT_DIR
                        Directory that stores the generated IR. By default, it
                        is the directory from where the Model Optimizer is
                        launched.
  --input_shape INPUT_SHAPE
                        Input shape(s) that should be fed to an input node(s)
                        of the model. Shape is defined as a comma-separated
                        list of integer numbers enclosed in parentheses or
                        square brackets, for example [1,3,227,227] or
                        (1,227,227,3), where the order of dimensions depends
                        on the framework input layout of the model. For
                        example, [N,C,H,W] is used for Caffe* models and
                        [N,H,W,C] for TensorFlow* models. Model Optimizer
                        performs necessary transformations to convert the
                        shape to the layout required by Inference Engine
                        (N,C,H,W). The shape should not contain undefined
                        dimensions (? or -1) and should fit the dimensions
                        defined in the input operation of the graph. If there
                        are multiple inputs in the model, --input_shape should
                        contain definition of shape for each input separated
                        by a comma, for example: [1,3,227,227],[2,4] for a
                        model with two inputs with 4D and 2D shapes.
                        Alternatively, specify shapes with the --input
                        option.
  --scale SCALE, -s SCALE
                        All input values coming from original network inputs
                        will be divided by this value. When a list of inputs
                        is overridden by the --input parameter, this scale is
                        not applied for any input that does not match with the
                        original input of the model.
  --reverse_input_channels
                        Switch the input channels order from RGB to BGR (or
                        vice versa). Applied to original inputs of the model
                        if and only if a number of channels equals 3. Applied
                        after application of --mean_values and --scale_values
                        options, so numbers in --mean_values and
                        --scale_values go in the order of channels used in the
                        original model.
  --log_level {CRITICAL,ERROR,WARN,WARNING,INFO,DEBUG,NOTSET}
                        Logger level
  --input INPUT         Quoted list of comma-separated input nodes names with
                        shapes, data types, and values for freezing. The shape
                        and value are specified as space-separated lists. The
                        data type of input node is specified in braces and can
                        have one of the values: f64 (float64), f32 (float32),
                        f16 (float16), i64 (int64), i32 (int32), u8 (uint8),
                        boolean. For example, use the following format to set
                        input port 0 of the node `node_name1` with the shape
                        [3 4] as an input node and freeze output port 1 of the
                        node `node_name2` with the value [20 15] of the int32
                        type and shape [2]: "0:node_name1[3
                        4],node_name2:1[2]{i32}->[20 15]".
  --output OUTPUT       The name of the output operation of the model. For
                        TensorFlow*, do not add :0 to this name.
  --mean_values MEAN_VALUES, -ms MEAN_VALUES
                        Mean values to be used for the input image per
                        channel. Values to be provided in the (R,G,B) or
                        [R,G,B] format. Can be defined for desired input of
                        the model, for example: "--mean_values
                        data[255,255,255],info[255,255,255]". The exact
                        meaning and order of channels depend on how the
                        original model was trained.
  --scale_values SCALE_VALUES
                        Scale values to be used for the input image per
                        channel. Values are provided in the (R,G,B) or [R,G,B]
                        format. Can be defined for desired input of the model,
                        for example: "--scale_values
                        data[255,255,255],info[255,255,255]". The exact
                        meaning and order of channels depend on how the
                        original model was trained.
  --data_type {FP16,FP32,half,float}
                        Data type for all intermediate tensors and weights. If
                        original model is in FP32 and --data_type=FP16 is
                        specified, all model weights and biases are quantized
                        to FP16.
  --disable_fusing      Turn off fusing of linear operations to Convolution
  --disable_resnet_optimization
                        Turn off resnet optimization
  --finegrain_fusing FINEGRAIN_FUSING
                        Regex for layers/operations that won't be fused.
                        Example: --finegrain_fusing Convolution1,.*Scale.*
  --disable_gfusing     Turn off fusing of grouped convolutions
  --enable_concat_optimization
                        Turn on Concat optimization.
  --move_to_preprocess  Move mean values to IR preprocess section
  --extensions EXTENSIONS
                        Directory or a comma separated list of directories
                        with extensions. To disable all extensions including
                        those that are placed at the default location, pass an
                        empty string.
  --batch BATCH, -b BATCH
                        Input batch size
  --version             Version of Model Optimizer
  --silent              Prevent any output messages except those that
                        correspond to log level equals ERROR, that can be set
                        with the following option: --log_level. By default,
                        log level is already ERROR.
  --freeze_placeholder_with_value FREEZE_PLACEHOLDER_WITH_VALUE
                        Replaces input layer with constant node with provided
                        value, for example: "node_name->True". It will be
                        DEPRECATED in future releases. Use --input option to
                        specify a value for freezing.
  --generate_deprecated_IR_V7
                        Force to generate old deprecated IR V7 with layers
                        from old IR specification.
  --static_shape        Enables `ShapeOf` operation with all children folding to `Constant`.
                        This option makes model not reshapable in Inference Engine
  --disable_weights_compression
                        Disable compression and store weights with original
                        precision.
  --progress            Enable model conversion progress display.
  --stream_output       Switch model conversion progress display to a
                        multiline mode.
  --transformations_config TRANSFORMATIONS_CONFIG
                        Use the configuration file with transformations
                        description.
```

The sections below provide details on using particular parameters and examples of CLI commands.

## When to Specify Mean and Scale Values
Usually neural network models are trained with the normalized input data. This means that the input data values are converted to be in a specific range, for example, `[0, 1]` or `[-1, 1]`. Sometimes the mean values (mean images) are subtracted from the input data values as part of the pre-processing. There are two cases how the input data pre-processing is implemented.
 * The input pre-processing operations are a part of a topology. In this case, the application that uses the framework to infer the topology does not pre-process the input.
 * The input pre-processing operations are not a part of a topology and the pre-processing is performed within the application which feeds the model with an input data.
 
In the first case, the Model Optimizer generates the IR with required pre-processing layers and Inference Engine samples may be used to infer the model. 
 
In the second case, information about mean/scale values should be provided to the Model Optimizer to embed it to the generated IR. Model Optimizer provides a number of command line parameters to specify them: `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. 

If both mean and scale values are specified, the mean is subtracted first and then scale is applied. Input values are *divided* by the scale value(s). 

There is no a universal recipe for determining the mean/scale values for a particular model. The steps below could help to determine them:
* Read the model documentation. Usually the documentation describes mean/scale value if the pre-processing is required.
* Open the example script/application executing the model and track how the input data is read and passed to the framework.
* Open the model in a visualization tool and check for layers performing subtraction or multiplication (like `Sub`, `Mul`, `ScaleShift`, `Eltwise` etc) of the input data. If such layers exist, the pre-processing is most probably the part of the model.

## When to Specify Input Shapes <a name="when_to_specify_input_shapes"></a>
There are situations when the input data shape for the model is not fixed, like for the fully-convolutional neural networks. In this case, for example, TensorFlow\* models contain `-1` values in the `shape` attribute of the `Placeholder` operation. Inference Engine does not support input layers with undefined size, so if the input shapes are not defined in the model, the Model Optimizer fails to convert the model. The solution is to provide the input shape(s) using the `--input` or `--input_shape` command line parameter for all input(s) of the model or provide the batch size using the `-b` command line parameter if the model contains just one input with undefined batch size only. In the latter case, the `Placeholder` shape for the TensorFlow\* model looks like this `[-1, 224, 224, 3]`. 

## When to Reverse Input Channels <a name="when_to_reverse_input_channels"></a>
Input data for your application can be of RGB or BRG color input order. For example, Inference Engine samples load input images in the BGR channels order. However, the model may be trained on images loaded with the opposite order (for example, most TensorFlow\* models are trained with images in RGB order). In this case, inference results using the Inference Engine samples may be incorrect. The solution is to provide `--reverse_input_channels` command line parameter. Taking this parameter, the Model Optimizer performs first convolution or other channel dependent operation weights modification so these operations output will be like the image is passed with RGB channels order.

## When to Specify `--static_shape` Command Line Parameter
If the `--static_shape` command line parameter is specified the Model Optimizer evaluates shapes of all operations in the model (shape propagation) for a fixed input(s) shape(s). During the shape propagation the Model Optimizer evaluates operations *Shape* and removes them from the computation graph. With that approach, the initial model which can consume inputs of different shapes may be converted to IR working with the input of one fixed shape only. For example, consider the case when some blob is reshaped from 4D of a shape *[N, C, H, W]* to a shape *[N, C, H \* W]*. During the model conversion the Model Optimize calculates output shape as a constant 1D blob with values *[N, C, H \* W]*. So if the input shape changes to some other value *[N,C,H1,W1]* (it is possible scenario for a fully convolutional model) then the reshape layer becomes invalid.
Resulting Intermediate Representation will not be resizable with the help of Inference Engine.

## Examples of CLI Commands

Launch the Model Optimizer for the Caffe bvlc_alexnet model with debug log level:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --log_level DEBUG
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with the output IR called `result.*` in the specified `output_dir`:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --model_name result --output_dir /../../models/
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with one input with scale values:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --scale_values [59,59,59]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with multiple inputs with scale values:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --input data,rois --scale_values [59,59,59],[5,5,5]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with multiple inputs with scale and mean values specified for the particular nodes:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --input data,rois --mean_values data[59,59,59] --scale_values rois[5,5,5]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with specified input layer, overridden input shape, scale 5, batch 8 and specified name of an output operation:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --input "data[1 3 224 224]" --output pool5 -s 5 -b 8
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with disabled fusing for linear operations to Convolution and grouped convolutions:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --disable_fusing --disable_gfusing
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with reversed input channels order between RGB and BGR, specified mean values to be used for the input image per channel and specified data type for input tensor values:
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --reverse_input_channels --mean_values [255,255,255] --data_type FP16
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with extensions listed in specified directories, specified mean_images binaryproto.
 file For more information about extensions, please refer to [this](../customize_model_optimizer/Extending_Model_Optimizer_with_New_Primitives.md) page.
```sh
python3 mo.py --input_model bvlc_alexnet.caffemodel --extensions /home/,/some/other/path/ --mean_file /path/to/binaryproto
```

Launch the Model Optimizer for TensorFlow* FaceNet* model with a placeholder freezing value. 
It replaces the placeholder with a constant layer that contains the passed value.
For more information about FaceNet conversion, please refer to [this](tf_specific/Convert_FaceNet_From_Tensorflow.md) page
```sh
python3 mo.py --input_model FaceNet.pb --input "phase_train->False"
```

Launch the Model Optimizer for any model with a placeholder freezing tensor of values. 
It replaces the placeholder with a constant layer that contains the passed values.

Tensor here is represented in square brackets with each value separated from another by a whitespace. 
If data type is set in the model, this tensor will be reshaped to a placeholder shape and casted to placeholder data type.
Otherwise, it will be casted to data type passed to `--data_type` parameter (by default, it is FP32).
```sh
python3 mo.py --input_model FaceNet.pb --input "placeholder_layer_name->[0.1 1.2 2.3]"
```
