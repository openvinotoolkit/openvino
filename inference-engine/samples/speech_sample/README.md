# Automatic Speech Recognition C++ Sample {#openvino_inference_engine_samples_speech_sample_README}

This sample demonstrates how to execute an Asynchronous Inference of acoustic model based on Kaldi\* neural networks and speech feature vectors.

The sample works with Kaldi ARK or Numpy* uncompressed NPZ files, so it does not cover an end-to-end speech recognition scenario (speech to text), requiring additional preprocessing (feature extraction) to get a feature vector from a speech signal, as well as postprocessing (decoding) to produce text from scores.

Automatic Speech Recognition C++ sample application demonstrates how to use the following Inference Engine C++ API in applications:

| Feature    | API  | Description |
|:---     |:--- |:---
|Inference Engine Version| `InferenceEngine::GetInferenceEngineVersion` | Get Inference Engine API version
|Available Devices|`InferenceEngine::Core::GetAvailableDevices`| Get version information of the devices for inference
| Network Operations | `InferenceEngine::CNNNetwork::setBatchSize`, `InferenceEngine::CNNNetwork::getBatchSize` |  Managing of network, operate with its batch size.
|Network Operations|`InferenceEngine::CNNNetwork::addOutput`| Change names of output layers in the network
|Import Network|`InferenceEngine::ExecutableNetwork::Export`,`InferenceEngine::Core::ImportNetwork`| Creates an executable network from a previously exported network
|Asynchronous Infer| `InferenceEngine::InferRequest::StartAsync`, `InferenceEngine::InferRequest::Wait`| Do asynchronous inference and waits until inference result becomes available
|InferRequest Operations|`InferenceEngine::InferRequest::QueryState`, `InferenceEngine::VariableState::Reset`| Gets and resets state control interface for given executable network
|InferRequest Operations|`InferenceEngine::InferRequest::GetPerformanceCounts`| Get performance counters for infer request

Basic Inference Engine API is covered by [Hello Classification C++ sample](../hello_classification/README.md).

| Options  | Values |
|:---                              |:---
| Validated Models                 | Acoustic model based on Kaldi\* neural networks (see [Model Preparation](#model-preparation) section)
| Model Format                     | Inference Engine Intermediate Representation (\*.xml + \*.bin)
| Supported devices                | See [Execution Modes](#execution-modes) section below and [List Supported Devices](../../../docs/IE_DG/supported_plugins/Supported_Devices.md) |

## How It Works

Upon the start-up, the application reads command line parameters, loads a specified model and input data to the Inference Engine plugin, performs synchronous inference on all speech utterances stored in the input file. Context-windowed speech frames are processed in batches of 1-8
frames according to the `-bs` parameter.  Batching across utterances is not supported by this sample.  When inference is done, the application creates an output file.  If the `-r` option is given, error
statistics are provided for each speech utterance as shown above.

You can see the explicit description of
each sample step at [Integration Steps](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md) section of "Integrate the Inference Engine with Your Application" guide.

### GNA-specific details

#### Quantization

If the GNA device is selected (for example, using the `-d` GNA flag), the GNA Inference Engine plugin quantizes the model and input feature vector sequence to integer representation before performing inference.
Several parameters control neural network quantization. The `-q` flag determines the quantization mode.
Three modes are supported:

- *static* - The first
utterance in the input file is scanned for dynamic range.  The scale factor (floating point scalar multiplier) required to scale the maximum input value of the first utterance to 16384 (15 bits) is used
for all subsequent inputs.  The neural network is quantized to accommodate the scaled input dynamic range.
- *dynamic* - The user may specify a scale factor via the `-sf` flag that will be used for static quantization.
- *user-defined* - The scale factor for each input batch is computed
just before inference on that batch.  The input and network are (re)quantized on-the-fly using an efficient procedure.

The `-qb` flag provides a hint to the GNA plugin regarding the preferred target weight resolution for all layers.  For example, when `-qb 8` is specified, the plugin will use 8-bit weights wherever possible in the
network.
> **NOTE**:
>
> - It is not always possible to use 8-bit weights due to GNA hardware limitations. For example, convolutional layers always use 16-bit weights (GNA hardware version 1 and 2).  This limitation
will be removed in GNA hardware version 3 and higher.

#### Execution Modes

Several execution modes are supported via the `-d` flag:

- `CPU` - All calculation are performed on CPU device using CPU Plugin.
- `GPU` - All calculation are performed on GPU device using GPU Plugin.
- `MYRIAD` - All calculation are performed on Intel® Neural Compute Stick 2 device using VPU MYRIAD Plugin.
- `GNA_AUTO` - GNA hardware is used if available and the driver is installed. Otherwise, the GNA device is emulated in fast-but-not-bit-exact mode.
- `GNA_HW` - GNA hardware is used if available and the driver is installed. Otherwise, an error will occur.
- `GNA_SW` - Deprecated. The GNA device is emulated in fast-but-not-bit-exact mode.
- `GNA_SW_FP32` - Substitutes parameters and calculations from low precision to floating point (FP32).
- `GNA_SW_EXACT` - GNA device is emulated in bit-exact mode.

#### Loading and Saving Models

The GNA plugin supports loading and saving of the GNA-optimized model (non-IR) via the `-rg` and `-wg` flags.  Thereby, it is possible to avoid the cost of full model quantization at run time. The GNA plugin also supports export of firmware-compatible embedded model images for the Intel® Speech Enabling Developer Kit and Amazon Alexa* Premium Far-Field Voice Development Kit via the `-we` flag (save only).

In addition to performing inference directly from a GNA model file, these options make it possible to:

- Convert from IR format to GNA format model file (`-m`, `-wg`)
- Convert from IR format to embedded format model file (`-m`, `-we`)
- Convert from GNA format to embedded format model file (`-rg`, `-we`)

## Building

To build the sample, please use instructions available at [Build the Sample Applications](../../../docs/IE_DG/Samples_Overview.md) section in Inference Engine Samples guide.

## Running

Running the application with the `-h` option yields the following usage message:

```
[ INFO ] InferenceEngine:
        API version ............ <version>
        Build .................. <number>
        Description ....... API
[ INFO ] Parsing input parameters

speech_sample [OPTION]
Options:

    -h                      Print a usage message.
    -i "<path>"             Required. Paths to input files. Example of usage: <file1.ark,file2.ark> or <file.ark> or <file.npz>.
    -m "<path>"             Required. Path to an .xml file with a trained model (required if -rg is missing).
    -o "<path>"             Optional. Output file name to save scores. Example of usage: <output.ark> or <output.npz>
    -d "<device>"           Optional. Specify a target device to infer on. CPU, GPU, MYRIAD, GNA_AUTO, GNA_HW, GNA_SW_FP32, GNA_SW_EXACT and HETERO with combination of GNA
     as the primary device and CPU as a secondary (e.g. HETERO:GNA,CPU) are supported. The list of available devices is shown below. The sample will look for a suitable plugin for device specified.
    -pc                     Optional. Enables per-layer performance report.
    -q "<mode>"             Optional. Input quantization mode:  static (default), dynamic, or user (use with -sf).
    -qb "<integer>"         Optional. Weight bits for quantization: 8 or 16 (default)
    -sf "<double>"          Optional. User-specified input scale factor for quantization (use with -q user). If the network contains multiple inputs, provide scale factors by separating them with commas.
    -bs "<integer>"         Optional. Batch size 1-8 (default 1)
    -r "<path>"             Optional. Read referefile and compare scores. Example of usage: <reference.ark> or <reference.npz>
    -rg "<path>"            Read GNA model from file using path/filename provided (required if -m is missing).
    -wg "<path>"            Optional. Write GNA model to file using path/filename provided.
    -we "<path>"            Optional. Write GNA embedded model to file using path/filename provided.
    -nthreads "<integer>"   Optional. Number of threads to use for concurrent async inference requests on the GNA.
    -cw_l "<integer>"       Optional. Number of frames for left context windows (default is 0). Works only with context window networks.
                            If you use the cw_l or cw_r flag, then batch size and nthreads arguments are ignored.
    -cw_r "<integer>"       Optional. Number of frames for right context windows (default is 0). Works only with context window networks.
                            If you use the cw_r or cw_l flag, then batch size and nthreads arguments are ignored.
    -oname "<string>"       Optional. Layer names for output blobs. The names are separated with "," Example: Output1:port,Output2:port
    -iname "<string>"       Optional. Layer names for input blobs. The names are separated with "," Example: Input1,Input2
    -pwl_me "<double>"      Optional. The maximum percent of error for PWL function.The value must be in <0, 100> range. The default value is 1.0.

Available target devices: <devices>

```

Running the application with the empty list of options yields the usage message given above and an error message.

### Model Preparation

You can use the following model optimizer command to convert a Kaldi nnet1 or nnet2 neural network to Inference Engine Intermediate Representation format:

```
python <path_to_mo>/mo.py --framework kaldi --input_model wsj_dnn5b.nnet --counts wsj_dnn5b.counts --remove_output_softmax --output_dir <OUTPUT_MODEL_DIR>
```

Assuming that the model optimizer (`mo.py`), Kaldi-trained neural network, `wsj_dnn5b.nnet`, and Kaldi class counts file, `wsj_dnn5b.counts`, are in the working directory this produces
the IE IR network consisting of `wsj_dnn5b.xml` and `wsj_dnn5b.bin`.

The following pre-trained models are available:

- wsj\_dnn5b\_smbr
- rm\_lstm4f
- rm\_cnn4a\_smbr

All of them can be downloaded from [https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/).

### Speech Inference

Once the IR is created, you can use the following command to do inference on Intel&reg; Processors with the GNA co-processor (or emulation library):

```
<path_to_sample>/speech_sample -d GNA_AUTO -bs 2 -i <path_to_ark>/dev93_10.ark -m <path_to_model>/wsj_dnn5b.xml -o scores.ark -r <path_to_ark>/dev93_scores_10.ark
```

Here, the floating point Kaldi-generated reference neural network scores (`dev93_scores_10.ark`) corresponding to the input feature file (`dev93_10.ark`) are assumed to be available
for comparison.

All of them can be downloaded from [https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr). Inference Engine Intermediate Representation `wsj_dnn5b.xml` file was generated in the previous [Model Preparation](#model-preparation) section.

> **NOTES**:
>
> - Before running the sample with a trained model, make sure the model is converted to the Inference Engine format (\*.xml + \*.bin) using the [Model Optimizer tool](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md).


## Sample Output

The acoustic log likelihood sequences for all utterances are stored in the file. Example `scores.ark` or `scores.npz`.  If the `-r` option is used, a report on the statistical score error is generated for each utterance such as
the following:

```
./speech_sample -d GNA_AUTO -bs 2 -i dev93_10.ark -m wsj_dnn5b.xml -o scores.ark -r dev93_scores_10.ark
[ INFO ] InferenceEngine:
        API version ............ <version>
        Build .................. <build>
        Description ....... API
[ INFO ] Parsing input parameters
[ INFO ] No extensions provided
[ INFO ] Loading Inference Engine
[ INFO ] Device info:
        GNA
        GNAPlugin version ......... <version>
        Build ........... <build>

[ INFO ] Loading network files
[ INFO ] Batch size is 2
[ INFO ] Using scale factor of 2175.43 calculated from first utterance.
[ INFO ] Loading model to the device
[ INFO ] Model loading time 124.21 ms
[ INFO ] Number scores per frame : 6850
Utterance 0:
Total time in Infer (HW and SW):        2291.64 ms
Frames in utterance:                    1294 frames
Average Infer time per frame:           1.77097 ms
         max error: 0.705184
         avg error: 0.0448388
     avg rms error: 0.0287733
       stdev error: 0.0371649


End of Utterance 0

[ INFO ] Number scores per frame : 6850
Utterance X:
...
End of Utterance X
[ INFO ] Execution successful

[ INFO ] This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool
```

## Use of Sample in Kaldi* Speech Recognition Pipeline

The Wall Street Journal DNN model used in this example was prepared using the Kaldi s5 recipe and the Kaldi Nnet (nnet1) framework.  It is possible to recognize speech by substituting the `speech_sample` for
Kaldi's nnet-forward command. Since the speech_sample does not yet use pipes, it is necessary to use temporary files for speaker-transformed feature vectors and scores when running the Kaldi speech recognition pipeline. The following operations assume that feature extraction was already performed according to the `s5` recipe and that the working directory within the Kaldi source tree is `egs/wsj/s5`.

1. Prepare a speaker-transformed feature set given the feature transform specified in `final.feature_transform` and the feature files specified in `feats.scp`:

```sh
nnet-forward --use-gpu=no final.feature_transform "ark,s,cs:copy-feats scp:feats.scp ark:- |" ark:feat.ark
```

2. Score the feature set using the `speech_sample`:

```sh
./speech_sample -d GNA_AUTO -bs 8 -i feat.ark -m wsj_dnn5b.xml -o scores.ark
```
Inference Engine Intermediate Representation `wsj_dnn5b.xml` file was generated in the previous [Model Preparation](#model-preparation) section.

3. Run the Kaldi decoder to produce n-best text hypotheses and select most likely text given the WFST (`HCLG.fst`), vocabulary (`words.txt`), and TID/PID mapping (`final.mdl`):

```sh
latgen-faster-mapped --max-active=7000 --max-mem=50000000 --beam=13.0 --lattice-beam=6.0 --acoustic-scale=0.0833 --allow-partial=true --word-symbol-table=words.txt final.mdl HCLG.fst ark:scores.ark ark:-| lattice-scale --inv-acoustic-scale=13 ark:- ark:- | lattice-best-path --word-symbol-table=words.txt ark:- ark,t:-  > out.txt &
```

4. Run the word error rate tool to check accuracy given the vocabulary (`words.txt`) and reference transcript (`test_filt.txt`):

```sh
cat out.txt | utils/int2sym.pl -f 2- words.txt | sed s:\<UNK\>::g | compute-wer --text --mode=present ark:test_filt.txt ark,p:-
```

All of mentioned files can be downloaded from [https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/wsj_dnn5b_smbr)

## See Also

- [Integrate the Inference Engine with Your Application](../../../docs/IE_DG/Integrate_with_customer_application_new_API.md)
- [Using Inference Engine Samples](../../../docs/IE_DG/Samples_Overview.md)
- [Model Downloader](@ref omz_tools_downloader_README)
- [Model Optimizer](../../../docs/MO_DG/Deep_Learning_Model_Optimizer_DevGuide.md)
