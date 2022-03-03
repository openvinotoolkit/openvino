# Converting a Model to Intermediate Representation (IR)  {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

@sphinxdirective

.. toctree::
   :maxdepth: 1
   :hidden:

   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Caffe
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_MxNet
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_ONNX
   openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Paddle
   openvino_docs_MO_DG_prepare_model_Model_Optimization_Techniques
   openvino_docs_MO_DG_prepare_model_convert_model_Cutting_Model
   openvino_docs_MO_DG_prepare_model_Supported_Frameworks_Layers
   openvino_docs_MO_DG_prepare_model_convert_model_IR_suitable_for_INT8_inference
   openvino_docs_MO_DG_prepare_model_customize_model_optimizer_Subgraph_Replacement_Model_Optimizer
   openvino_docs_MO_DG_prepare_model_convert_model_Legacy_IR_Layers_Catalog_Spec

@endsphinxdirective

To convert the model to the Intermediate Representation (IR), run Model Optimizer using the following command:

```sh
mo --input_model INPUT_MODEL
```

To adjust the conversion process, you may use general parameters defined in the [General Conversion Parameters](#general_conversion_parameters) and 
Framework-specific parameters for:
* [TensorFlow](Convert_Model_From_TensorFlow.md)
* [PyTorch](Convert_Model_From_ONNX.md)
* [ONNX](Convert_Model_From_ONNX.md)
* [PaddlePaddle](Convert_Model_From_Paddle.md)
* [MXNet](Convert_Model_From_MxNet.md)
* [Caffe](Convert_Model_From_Caffe.md)
* [Kaldi](Convert_Model_From_Kaldi.md)

The sections below provide details on using particular parameters and examples of CLI commands.

## When to Specify `--static_shape` Command Line Parameter
If the `--static_shape` command line parameter is specified the Model Optimizer evaluates shapes of all operations in the model (shape propagation) for a fixed input(s) shape(s). During the shape propagation the Model Optimizer evaluates operations *Shape* and removes them from the computation graph. With that approach, the initial model which can consume inputs of different shapes may be converted to IR working with the input of one fixed shape only. For example, consider the case when some blob is reshaped from 4D of a shape *[N, C, H, W]* to a shape *[N, C, H \* W]*. During the model conversion the Model Optimize calculates output shape as a constant 1D blob with values *[N, C, H \* W]*. So if the input shape changes to some other value *[N,C,H1,W1]* (it is possible scenario for a fully convolutional model) then the reshape layer becomes invalid.
Resulting Intermediate Representation will not be resizable with the help of OpenVINO Runtime API.

## Examples of CLI Commands

Launch the Model Optimizer for the Caffe bvlc_alexnet model with debug log level:
```sh
mo --input_model bvlc_alexnet.caffemodel --log_level DEBUG
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with the output IR called `result.*` in the specified `output_dir`:
```sh
mo --input_model bvlc_alexnet.caffemodel --model_name result --output_dir <OUTPUT_MODEL_DIR>
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with one input with scale values:
```sh
mo --input_model bvlc_alexnet.caffemodel --scale_values [59,59,59]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with multiple inputs with scale values:
```sh
mo --input_model bvlc_alexnet.caffemodel --input data,rois --scale_values [59,59,59],[5,5,5]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with multiple inputs with scale and mean values specified for the particular nodes:
```sh
mo --input_model bvlc_alexnet.caffemodel --input data,rois --mean_values data[59,59,59] --scale_values rois[5,5,5]
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with specified input layer, overridden input shape, scale 5, batch 8 and specified name of an output operation:
```sh
mo --input_model bvlc_alexnet.caffemodel --input data --output pool5 -s 5 -b 8
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with reversed input channels order between RGB and BGR, specified mean values to be used for the input image per channel and specified data type for input tensor values:
```sh
mo --input_model bvlc_alexnet.caffemodel --reverse_input_channels --mean_values [255,255,255] --data_type FP16
```

Launch the Model Optimizer for the Caffe bvlc_alexnet model with extensions listed in specified directories, specified mean_images binaryproto 
 file. For more information about extensions, please refer to the [OpenVINO™ Extensibility Mechanism](../../../Extensibility_UG/Intro.md).
```sh
mo --input_model bvlc_alexnet.caffemodel --extensions /home/,/some/other/path/ --mean_file /path/to/binaryproto
```

Launch the Model Optimizer for TensorFlow* FaceNet* model with a placeholder freezing value. 
It replaces the placeholder with a constant layer that contains the passed value.
For more information about FaceNet conversion, please refer to [this](tf_specific/Convert_FaceNet_From_Tensorflow.md) page.
```sh
mo --input_model FaceNet.pb --input "phase_train->False"
```
Launch the Model Optimizer for any model with a placeholder freezing tensor of values. 
It replaces the placeholder with a constant layer that contains the passed values.

Tensor here is represented in square brackets with each value separated from another by a whitespace. 
If data type is set in the model, this tensor will be reshaped to a placeholder shape and casted to placeholder data type.
Otherwise, it will be casted to data type passed to `--data_type` parameter (by default, it is FP32).
```sh
mo --input_model FaceNet.pb --input "placeholder_layer_name->[0.1 1.2 2.3]"
```


## See Also
* [Configuring the Model Optimizer](../../Deep_Learning_Model_Optimizer_DevGuide.md)
* [IR Notation Reference](../../IR_and_opsets.md)
* [Model Optimizer Extensibility](../customize_model_optimizer/Customize_Model_Optimizer.md)
* [Model Cutting](Cutting_Model.md)
