# Speech Recognition Python* Sample {#openvino_inference_engine_ie_bridges_python_sample_speech_sample_README}

This sample demonstrates how to do inference of acoustic model based on Kaldi* neural networks and speech feature vectors using Synchronous Inference Request API.  
Models with only 1 input and output are supported.

The following Inference Engine Python API is used in the application:

| Feature             | API                                                                                                   | Description                                                           |
| :------------------ | :---------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------- |
| Import/Export Model | [IECore.import_network], [ExecutableNetwork.export]                                                   | The GNA plugin supports loading and saving of the GNA-optimized model |
| Network Operations  | [IENetwork.batch_size], [CDataPtr.shape], [ExecutableNetwork.input_info], [ExecutableNetwork.outputs] | Managing of network: configure input and output blobs                 |

Basic Inference Engine API is covered by [Hello Classification Python* Sample](../hello_classification/README.md).

| Options                    | Values                                                                                                |
| :------------------------- | :---------------------------------------------------------------------------------------------------- |
| Validated Models           | Acoustic model based on Kaldi* neural networks (see [Model Preparation](##Model-Preparation) section) |
| Model Format               | Inference Engine Intermediate Representation (.xml + .bin), ONNX (.onnx)                              |
| Supported devices          | See [Execution Modes](###Execution-Modes) section below and [List Supported Devices](../../../docs/IE_DG/supported_plugins/Supported_Devices.md)                             |
| Other language realization | [C++](../../../../samples/speech_sample)                                                              |

## How It Works

At startup, the sample application reads command-line parameters, loads a specified model and input data to the Inference Engine plugin, performs synchronous inference on all speech utterances stored in the input file, logging each step in a standard output stream.

You can see the explicit description of
each sample step at [Integration Steps](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

## GNA-specific details

### Quantization

If the GNA device is selected (for example, using the `-d` GNA flag), the GNA Inference Engine plugin quantizes the model and input feature vector sequence to integer representation before performing inference.

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
Thereby, it is possible to avoid the cost of full model quantization at run time. The GNA plugin also supports export of firmware-compatible embedded model images for the Intel® Speech Enabling Developer Kit and Amazon Alexa* Premium Far-Field Voice Development Kit via the `-we` flag (save only).

In addition to performing inference directly from a GNA model file, these options make it possible to:

- Convert from IR format to GNA format model file (`-m`, `-wg`)
- Convert from IR format to embedded format model file (`-m`, `-we`)
- Convert from GNA format to embedded format model file (`-rg`, `-we`)

## Running

Run the application with the <code>-h</code> option to see the usage message:

```sh
python speech_sample.py -h
```

Usage message:

```sh
usage: speech_sample.py [-h] (-m MODEL | -rg IMPORT_GNA_MODEL) -i INPUT       
                        [-o OUTPUT] [-r REFERENCE] [-d DEVICE]
                        [-bs BATCH_SIZE] [-qb QUANTIZATION_BITS]
                        [-wg EXPORT_GNA_MODEL] [-we EXPORT_EMBEDDED_GNA_MODEL]
                        [-we_gen EMBEDDED_GNA_CONFIGURATION]

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
                        Required. Path to an utterance file.
  -o OUTPUT, --output OUTPUT
                        Optional. Output file name to save inference results.
  -r REFERENCE, --reference REFERENCE
                        Optional. Read reference score file and compare
                        scores.
  -d DEVICE, --device DEVICE
                        Optional. Specify the target device to infer on; CPU,
                        GPU, MYRIAD, HDDL, GNA_AUTO, GNA_HW, GNA_SW_FP32,
                        GNA_SW_EXACT or HETERO: is acceptable. The sample will
                        look for a suitable plugin for device specified.
                        Default value is CPU.
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        Optional. Batch size 1-8 (default 1).
  -qb QUANTIZATION_BITS, --quantization_bits QUANTIZATION_BITS
                        Optional. Weight bits for quantization: 8 or 16
                        (default 16).
  -wg EXPORT_GNA_MODEL, --export_gna_model EXPORT_GNA_MODEL
                        Optional. Write GNA model to file using path/filename
                        provided.
  -we EXPORT_EMBEDDED_GNA_MODEL, --export_embedded_gna_model EXPORT_EMBEDDED_GNA_MODEL
                        Optional. Write GNA embedded model to file using
                        path/filename provided.
  -we_gen EMBEDDED_GNA_CONFIGURATION, --embedded_gna_configuration EMBEDDED_GNA_CONFIGURATION
                        Optional. GNA generation configuration string for
                        embedded export. Can be GNA1 (default) or GNA3.
```

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).
>
> - The sample accepts models in ONNX format (.onnx) that do not require preprocessing.
>
> - The sample supports input and output in numpy file format (.npz)
>

You can do inference on Intel® Processors with the GNA co-processor (or emulation library):

```sh
python speech_sample.py -d GNA_AUTO -m wsj_dnn5b.xml -i dev93_10.ark -r dev93_scores_10.ark -o result.npz
```

## Model Preparation

You can use the following model optimizer command to convert a Kaldi nnet1 or nnet2 neural network to Inference Engine Intermediate Representation format:

```sh
python mo.py --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>
```

The following pre-trained models are available:

- wsj_dnn5b_smbr
- rm_lstm4f
- rm_cnn4a_smbr

All of them can be downloaded from [https://storage.openvinotoolkit.org/models_contrib/speech/2021.2](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2).

## Sample Output

The sample application logs each step in a standard output stream.

```sh
[ INFO ] Creating Inference Engine
[ INFO ] Reading the network: wsj_dnn5b.xml
[ INFO ] Configuring input and output blobs
[ INFO ] Using scale factor of 2175.4322417974627 calculated from first utterance.
[ INFO ] Loading the model to the plugin
[ INFO ] Starting inference in synchronous mode
[ INFO ] Utterance 0 (4k0c0301)
[ INFO ] Frames in utterance: 1294
[ INFO ] Total time in Infer (HW and SW): 5305.47ms
[ INFO ] max error: 0.7051839828491211
[ INFO ] avg error: 0.044838781794944625
[ INFO ] avg rms error: 0.05823873597800535        
[ INFO ] stdev error: 0.03716495682038603
[ INFO ]
[ INFO ] Utterance 1 (4k0c0302)
[ INFO ] Frames in utterance: 1005
[ INFO ] Total time in Infer (HW and SW): 5031.53ms
[ INFO ] max error: 0.7575974464416504
[ INFO ] avg error: 0.04521660570793721
[ INFO ] avg rms error: 0.05860129608000006
[ INFO ] stdev error: 0.03727694288040223
[ INFO ]
[ INFO ] Utterance 2 (4k0c0303)
[ INFO ] Frames in utterance: 1471
[ INFO ] Total time in Infer (HW and SW): 7950.45ms
[ INFO ] max error: 0.7191710472106934
[ INFO ] avg error: 0.047222570859711814
[ INFO ] avg rms error: 0.061299099888637014
[ INFO ] stdev error: 0.03908463187183161
[ INFO ]
[ INFO ] Utterance 3 (4k0c0304)
[ INFO ] Frames in utterance: 845
[ INFO ] Total time in Infer (HW and SW): 4023.39ms
[ INFO ] max error: 0.7436461448669434
[ INFO ] avg error: 0.04775810807647345
[ INFO ] avg rms error: 0.06213336382943745
[ INFO ] stdev error: 0.03974566660300375
[ INFO ]
[ INFO ] Utterance 4 (4k0c0305)
[ INFO ] Frames in utterance: 855
[ INFO ] Total time in Infer (HW and SW): 3420.64ms
[ INFO ] max error: 0.7071599960327148
[ INFO ] avg error: 0.044914698467230396
[ INFO ] avg rms error: 0.058504752434499985
[ INFO ] stdev error: 0.03748967749954517
[ INFO ]
[ INFO ] Utterance 5 (4k0c0306)
[ INFO ] Frames in utterance: 699
[ INFO ] Total time in Infer (HW and SW): 2769.76ms
[ INFO ] max error: 0.8870468139648438
[ INFO ] avg error: 0.04792428074365066
[ INFO ] avg rms error: 0.06254901777693281
[ INFO ] stdev error: 0.040195061140180025
[ INFO ]
[ INFO ] Utterance 6 (4k0c0307)
[ INFO ] Frames in utterance: 790
[ INFO ] Total time in Infer (HW and SW): 3219.15ms
[ INFO ] max error: 0.7648272514343262
[ INFO ] avg error: 0.04827024419632296
[ INFO ] avg rms error: 0.0629733825700257
[ INFO ] stdev error: 0.040442928152375
[ INFO ]
[ INFO ] Utterance 7 (4k0c0308)
[ INFO ] Frames in utterance: 622
[ INFO ] Total time in Infer (HW and SW): 2582.92ms
[ INFO ] max error: 0.7389559745788574
[ INFO ] avg error: 0.046554336083386876
[ INFO ] avg rms error: 0.060494128445356
[ INFO ] stdev error: 0.03862943655036005
[ INFO ]
[ INFO ] Utterance 8 (4k0c0309)
[ INFO ] Frames in utterance: 548
[ INFO ] Total time in Infer (HW and SW): 2272.36ms
[ INFO ] max error: 0.6680135726928711
[ INFO ] avg error: 0.04393408749747612
[ INFO ] avg rms error: 0.057461442316338734
[ INFO ] stdev error: 0.03703529814701154
[ INFO ]
[ INFO ] Utterance 9 (4k0c030a)
[ INFO ] Frames in utterance: 368
[ INFO ] Total time in Infer (HW and SW): 1457.43ms
[ INFO ] max error: 0.6550579071044922
[ INFO ] avg error: 0.046764278689150146
[ INFO ] avg rms error: 0.06050449413426664
[ INFO ] stdev error: 0.03839135383295311
[ INFO ]
[ INFO ] Total sample time: 38033.09ms
[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## See Also

- [Integrate the Inference Engine with Your Application](../../../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)

[IENetwork.batch_size]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IENetwork.html#a79a647cb1b49645616eaeb2ca255ef2e
[CDataPtr.shape]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1CDataPtr.html#aa6fd459edb323d1c6215dc7a970ebf7f
[ExecutableNetwork.input_info]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#ac76a04c2918607874018d2e15a2f274f
[ExecutableNetwork.outputs]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#a4a631776df195004b1523e6ae91a65c1
[IECore.import_network]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1IECore.html#afdeac5192bb1d9e64722f1071fb0a64a
[ExecutableNetwork.export]:https://docs.openvinotoolkit.org/latest/ie_python_api/classie__api_1_1ExecutableNetwork.html#afa78158252f0d8070181bafec4318413