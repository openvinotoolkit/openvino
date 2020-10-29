# Converting a Model to Intermediate Representation (IR)  {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

Use the <code>mo.py</code> script from the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory to run the Model Optimizer and convert the model to the Intermediate Representation (IR). 
The simplest way to convert a model is to run <code>mo.py</code> with a path to the input model file:
```sh
python3 mo.py --input_model INPUT_MODEL
```

> **NOTE**: Some models require using additional arguments to specify conversion parameters, such as `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. To learn about when you need to use these parameters, refer to [Converting a Model Using General Conversion Parameters](Converting_Model_General.md).

The <code>mo.py</code> script is the universal entry point that can deduce the framework that has produced the input model by a standard extension of the model file:

* `.caffemodel` - Caffe\* models
* `.pb` - TensorFlow\* models
* `.params` - MXNet\* models
* `.onnx` - ONNX\* models
* `.nnet` - Kaldi\* models.

If the model files do not have standard extensions, you can use the ``--framework {tf,caffe,kaldi,onnx,mxnet}`` option to specify the framework type explicitly. 

For example, the following commands are equivalent: 
```sh
python3 mo.py --input_model /user/models/model.pb
```
```sh
python3 mo.py --framework tf --input_model /user/models/model.pb
```

To adjust the conversion process, you may use general parameters defined in the [Converting a Model Using General Conversion Parameters](Converting_Model_General.md) and 
Framework-specific parameters for:
* [Caffe](Convert_Model_From_Caffe.md),
* [TensorFlow](Convert_Model_From_TensorFlow.md),
* [MXNet](Convert_Model_From_MxNet.md),
* [ONNX](Convert_Model_From_ONNX.md),
* [Kaldi](Convert_Model_From_Kaldi.md).


## See Also
* [Configuring the Model Optimizer](../Config_Model_Optimizer.md)
* [IR Notation Reference](../../IR_and_opsets.md)
* [Custom Layers in Model Optimizer](../customize_model_optimizer/Customize_Model_Optimizer.md) 
* [Model Cutting](Cutting_Model.md)