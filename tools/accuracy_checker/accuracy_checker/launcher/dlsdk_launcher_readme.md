# How to configure OpenVINO™ launcher

For enabling OpenVINO™ launcher you need to add `framework: dlsdk` in launchers section of your configuration file and provide following parameters:

* `device` - specifies which device will be used for infer. Supported: `CPU`, `GPU`, `FPGA`, `MYRIAD` and Heterogeneous plugin as `HETERO:target_device,fallback_device`.
* `model` - path to xml file with Caffe model for your topology.
* `weights` - path to bin file with weights for your topology.

launcher may optionally provide model parameters in source framework format which will be converted to Inference Engine IR using Model Optimizer.
If you want to use Model Optimizer for model conversion, please view [Model Optimizer Developer Guide][openvino-mo].
You can provide:

* `caffe_model` and `caffe_weights` for Caffe model and weights (*.prototxt and *.caffemodel).
* `tf_model` for TensorFlow model (*.pb, *.pb.frozen, *.pbtxt).
* `mxnet_weights` for MXNet params (*.params).
* `onnx_model` for ONNX model (*.onnx).
* `kaldi_model` for Kaldi model (*.nnet).

In case when you want to determine additional parameters for model conversion (data_type, input_shape and so on), you can use `mo_params` for arguments with values and `mo_flags` for positional arguments like `legacy_mxnet_model` .
Full list of supported parameters you can find in Model Optimizer Developer Guide.

Model will be converted before every evaluation. 
You can provide `converted_model_dir` for saving converted model in specific folder, otherwise, converted models will be saved in path provided via `-C` command line argument or source model directory.

* `adapter` - approach how raw output will be converted to representation of dataset problem, some adapters can be specific to framework. You can find detailed instruction how to use adapters [here][adapters].

Launcher understands which batch size will be used from model intermediate representation (IR). If you want to use batch for infer, please, provide model with required batch or convert it using specific parameter in `mo_params`.

* `allow_reshape_input` - parameter, which allows to reshape input layer to data shape (default value is False).

Additionally you can provide device specific parameters:

* `cpu_extensions` (path to extension *.so file with custom layers for cpu).
* `gpu_extensions` (path to extension *.xml file with OpenCL kernel description for gpu).
* `bitstream` for running on FPGA.

OpenVINO™ launcher config example:

```yml
launchers:
  - framework: dlsdk
    device: HETERO:FPGA,CPU
    caffe_model: path_to_model/alexnet.prototxt
    caffe_weights: path_to_weights/alexnet.caffemodel
    adapter: classification
    mo_params:
      batch: 4
    mo_flags:
      - reverse_input_channels
    cpu_extensions: cpu_extentions_avx512.so
```

[adapters]: ./tools/accuracy_checker/accuracy_checker/adapters/README.md
[openvino-mo]: https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer
