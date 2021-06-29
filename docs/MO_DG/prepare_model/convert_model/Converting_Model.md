# Converting a Model to Intermediate Representation (IR)  {#openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model}

Use the <code>mo.py</code> script from the `<INSTALL_DIR>/deployment_tools/model_optimizer` directory to run the Model Optimizer and convert the model to the Intermediate Representation (IR): 
```sh
python3 mo.py --input_model INPUT_MODEL --output_dir <OUTPUT_MODEL_DIR>
```
You need to have have write permissions for an output directory.

> **NOTE**: Some models require using additional arguments to specify conversion parameters, such as `--input_shape`, `--scale`, `--scale_values`, `--mean_values`, `--mean_file`. To learn about when you need to use these parameters, refer to [Converting a Model Using General Conversion Parameters](Converting_Model_General.md).

To adjust the conversion process, you may use general parameters defined in the [Converting a Model Using General Conversion Parameters](Converting_Model_General.md) and 
Framework-specific parameters for:
* [Caffe](Convert_Model_From_Caffe.md)
* [TensorFlow](Convert_Model_From_TensorFlow.md)
* [MXNet](Convert_Model_From_MxNet.md)
* [ONNX](Convert_Model_From_ONNX.md)
* [Kaldi](Convert_Model_From_Kaldi.md)


## See Also
* [Configuring the Model Optimizer](../Config_Model_Optimizer.md)
* [IR Notation Reference](../../IR_and_opsets.md)
* [Model Optimizer Extensibility](../customize_model_optimizer/Customize_Model_Optimizer.md)
* [Model Cutting](Cutting_Model.md)
