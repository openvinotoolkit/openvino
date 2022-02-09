# Automatic Speech Recognition Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_speech_sample_README}

This sample demonstrates how to do a Synchronous Inference of acoustic model based on Kaldi\* neural networks and speech feature vectors.

The sample works with Kaldi ARK or Numpy* uncompressed NPZ files, so it does not cover an end-to-end speech recognition scenario (speech to text), requiring additional preprocessing (feature extraction) to get a feature vector from a speech signal, as well as postprocessing (decoding) to produce text from scores.

Automatic Speech Recognition Python sample application demonstrates how to use the following Inference Engine Python API in applications:

| Feature             | API                                                                                                   | Description                                                           |
| :------------------ | :---------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| Import/Export Model | [IECore.import_network], [ExecutableNetwork.export]                                                   | The GNA plugin supports loading and saving of the GNA-optimized model |
| Network Operations  | [IENetwork.batch_size], [CDataPtr.shape], [ExecutableNetwork.input_info], [ExecutableNetwork.outputs] | Managing of network: configure input and output blobs                 |
| Network Operations  | [IENetwork.add_outputs] | Managing of network: Change names of output layers in the network |
| InferRequest Operations|InferRequest.query_state, VariableState.reset| Gets and resets state control interface for given executable network |

Basic Inference Engine API is covered by [Hello Classification Python* Sample](../hello_classification/README.md).

| Options                    | Values                                                                                                |
| :------------------------- | :---------------------------------------------------------------------------------------------------- |
| Validated Models           | Acoustic model based on Kaldi* neural networks (see [Model Preparation](#model-preparation) section) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin) |
| Supported devices          | See [Execution Modes](#execution-modes) section below and [List Supported Devices](../../../docs/OV_Runtime_UG/supported_plugins/Supported_Devices.md) |
| Other language realization | [C++](../../../samples/cpp/speech_sample/README.md) |

## How It Works

At startup, the sample application reads command-line parameters, loads a specified model and input data to the Inference Engine plugin, performs synchronous inference on all speech utterances stored in the input file, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/OV_Runtime_UG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## GNA-specific details

### Quantization

If the GNA device is selected (for example, using the `-d` GNA flag), the GNA Inference Engine plugin quantizes the model and input feature vector sequence to integer representation before performing inference.

Several neural network quantization modes:

- *static* - The first utterance in the input file is scanned for dynamic range.  The scale factor (floating point scalar multiplier) required to scale the maximum input value of the first utterance to 16384 (15 bits) is used for all subsequent inputs. The neural network is quantized to accommodate the scaled input dynamic range.
- *user-defined* - The user may specify a scale factor via the `-sf` flag that will be used for static quantization.

The `-qb` flag provides a hint to the GNA plugin regarding the preferred target weight resolution for all layers.  
For example, when `-qb 8` is specified, the plugin will use 8-bit weights wherever possible in the
network.

> **NOTE**:
>
> - It is not always possible to use 8-bit weights due to GNA hardware limitations. For example, convolutional layers always use 16-bit weights (GNA hardware version 1 and 2).  This limitation will be removed in GNA hardware version 3 and higher.
>

### Execution Modes

Several execution modes are supported via the `-d` flag:

- `CPU` - All calculation are performed on CPU device using CPU Plugin.
- `GPU` - All calculation are performed on GPU device using GPU Plugin.
- `MYRIAD` - All calculation are performed on Intel® Neural Compute Stick 2 device using VPU MYRIAD Plugin.
- `GNA_AUTO` - GNA hardware is used if available and the driver is installed. Otherwise, the GNA device is emulated in fast-but-not-bit-exact mode.
- `GNA_HW` - GNA hardware is used if available and the driver is installed. Otherwise, an error will occur.
- `GNA_SW` - Deprecated. The GNA device is emulated in fast-but-not-bit-exact mode.
- `GNA_SW_FP32` - Substitutes parameters and calculations from low precision to floating point (FP32).
- `GNA_SW_EXACT` - GNA device is emulated in bit-exact mode.

### Loading and Saving Models

The GNA plugin supports loading and saving of the GNA-optimized model (non-IR) via the `-rg` and `-wg` flags.  
Thereby, it is possible to avoid the cost of full model quantization at run time.  
The GNA plugin also supports export of firmware-compatible embedded model images for the Intel® Speech Enabling Developer Kit and Amazon Alexa* Premium Far-Field Voice Development Kit via the `-we` flag (save only).

In addition to performing inference directly from a GNA model file, these options make it possible to:

- Convert from IR format to GNA format model file (`-m`, `-wg`)
- Convert from IR format to embedded format model file (`-m`, `-we`)
- Convert from GNA format to embedded format model file (`-rg`, `-we`)

## Running

Run the application with the `-h` option to see the usage message:

```
python <path_to_sample>/speech_sample.py -h
```

Usage message:

```
usage: speech_sample.py [-h] (-m MODEL | -rg IMPORT_GNA_MODEL) -i INPUT       
                        [-o OUTPUT] [-r REFERENCE] [-d DEVICE] [-bs [1-8]]    
                        [-qb [8, 16]] [-sf SCALE_FACTOR]
                        [-wg EXPORT_GNA_MODEL] [-we EXPORT_EMBEDDED_GNA_MODEL]
                        [-we_gen [GNA1, GNA3]]
                        [--exec_target [GNA_TARGET_2_0, GNA_TARGET_3_0]] [-pc]
                        [-a [CORE, ATOM]] [-iname INPUT_LAYERS]
                        [-oname OUTPUT_LAYERS] [-cw_l CONTEXT_WINDOW_LEFT]    
                        [-cw_r CONTEXT_WINDOW_RIGHT]

optional arguments:
  -m MODEL, --model MODEL
                        Path to an .xml file with a trained model (required if
                        -rg is missing).
  -rg IMPORT_GNA_MODEL, --import_gna_model IMPORT_GNA_MODEL
                        Read GNA model from file using path/filename provided
                        (required if -m is missing).

Options:
  -h, --help            Show this help message and exit.
  -i INPUT, --input INPUT
                        Required. Path to an input file (.ark or .npz).
  -o OUTPUT, --output OUTPUT
                        Optional. Output file name to save inference results (.ark or .npz).
  -r REFERENCE, --reference REFERENCE
                        Optional. Read reference score file and compare
                        scores.
  -d DEVICE, --device DEVICE
                        Optional. Specify a target device to infer on. CPU,
                        GPU, MYRIAD, GNA_AUTO, GNA_HW, GNA_SW_FP32,
                        GNA_SW_EXACT and HETERO with combination of GNA as the
                        primary device and CPU as a secondary (e.g.
                        HETERO:GNA,CPU) are supported. The sample will look
                        for a suitable plugin for device specified. Default
                        value is CPU.
  -bs [1-8], --batch_size [1-8]
                        Optional. Batch size 1-8 (default 1).
  -qb [8, 16], --quantization_bits [8, 16]
                        Optional. Weight bits for quantization: 8 or 16
                        (default 16).
  -sf SCALE_FACTOR, --scale_factor SCALE_FACTOR
                        Optional. The user-specified input scale factor for
                        quantization. If the network contains multiple inputs,
                        provide scale factors by separating them with commas.
  -wg EXPORT_GNA_MODEL, --export_gna_model EXPORT_GNA_MODEL
                        Optional. Write GNA model to file using path/filename
                        provided.
  -we EXPORT_EMBEDDED_GNA_MODEL, --export_embedded_gna_model EXPORT_EMBEDDED_GNA_MODEL
                        Optional. Write GNA embedded model to file using
                        path/filename provided.
  -we_gen [GNA1, GNA3], --embedded_gna_configuration [GNA1, GNA3]
                        Optional. GNA generation configuration string for
                        embedded export. Can be GNA1 (default) or GNA3.
  --exec_target [GNA_TARGET_2_0, GNA_TARGET_3_0]
                        Optional. Specify GNA execution target generation. By
                        default, generation corresponds to the GNA HW
                        available in the system or the latest fully supported
                        generation by the software. See the GNA Plugin's
                        GNA_EXEC_TARGET config option description.
  -pc, --performance_counter
                        Optional. Enables performance report (specify -a to
                        ensure arch accurate results).
  -a [CORE, ATOM], --arch [CORE, ATOM]
                        Optional. Specify architecture. CORE, ATOM with the
                        combination of -pc.
  -iname INPUT_LAYERS, --input_layers INPUT_LAYERS
                        Optional. Layer names for input blobs. The names are
                        separated with ",". Allows to change the order of
                        input layers for -i flag. Example: Input1,Input2
  -oname OUTPUT_LAYERS, --output_layers OUTPUT_LAYERS
                        Optional. Layer names for output blobs. The names are
                        separated with ",". Allows to change the order of
                        output layers for -o flag. Example:
                        Output1:port,Output2:port.
  -cw_l CONTEXT_WINDOW_LEFT, --context_window_left CONTEXT_WINDOW_LEFT
                        Optional. Number of frames for left context windows
                        (default is 0). Works only with context window
                        networks. If you use the cw_l or cw_r flag, then batch
                        size argument is ignored.
  -cw_r CONTEXT_WINDOW_RIGHT, --context_window_right CONTEXT_WINDOW_RIGHT
                        Optional. Number of frames for right context windows
                        (default is 0). Works only with context window
                        networks. If you use the cw_l or cw_r flag, then batch
                        size argument is ignored.
```

## Model Preparation

You can use the following model optimizer command to convert a Kaldi nnet1 or nnet2 neural network to Inference Engine Intermediate Representation format:

```sh
mo --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>
```

The following pre-trained models are available:

- wsj_dnn5b_smbr
- rm_lstm4f
- rm_cnn4a_smbr

All of them can be downloaded from [https://storage.openvinotoolkit.org/models_contrib/speech/2021.2](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2).

## Speech Inference

You can do inference on Intel® Processors with the GNA co-processor (or emulation library):

```
python <path_to_sample>/speech_sample.py -m <path_to_model>/wsj_dnn5b.xml -i <path_to_ark>/dev93_10.ark -r <path_to_ark>/dev93_scores_10.ark -d GNA_AUTO -o result.npz
```

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample supports input and output in numpy file format (.npz)

## Sample Output

The sample application logs each step in a standard output stream.

```
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: wsj_dnn5b.xml
[ INFO ] Configuring input and output blobs
[ INFO ] Using scale factor(s) calculated from first utterance
[ INFO ] For input 0 using scale factor of 2175.4322418
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] Utterance 0 (4k0c0301)
[ INFO ] Output blob name: affinetransform14/Fused_Add_
[ INFO ] Frames in utterance: 1294
[ INFO ] Total time in Infer (HW and SW): 6211.45ms
[ INFO ] max error: 0.7051840
[ INFO ] avg error: 0.0448388
[ INFO ] avg rms error: 0.0582387
[ INFO ] stdev error: 0.0371650
[ INFO ]
[ INFO ] Utterance 1 (4k0c0302)
[ INFO ] Output blob name: affinetransform14/Fused_Add_
[ INFO ] Frames in utterance: 1005
[ INFO ] Total time in Infer (HW and SW): 4742.27ms
[ INFO ] max error: 0.7575974
[ INFO ] avg error: 0.0452166
[ INFO ] avg rms error: 0.0586013
[ INFO ] stdev error: 0.0372769
...
[ INFO ] Total sample time: 40219.99ms
[ INFO ] File result.npz was created!
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/OV_Runtime_UG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/OV_Runtime_UG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[IENetwork.batch_size]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e
[IENetwork.add_outputs]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1IENetwork.html#ae8024b07f3301d6d5de5c0d153e2e6e6
[CDataPtr.shape]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1CDataPtr.html#aa6fd459edb323d1c6215dc7a970ebf7f
[ExecutableNetwork.input_info]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#ac76a04c2918607874018d2e15a2f274f
[ExecutableNetwork.outputs]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#a4a631776df195004b1523e6ae91a65c1
[IECore.import_network]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1IECore.html#afdeac5192bb1d9e64722f1071fb0a64a
[ExecutableNetwork.export]:https://docs.openvino.ai/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#afa78158252f0d8070181bafec4318413
