# GNA Plugin {#openvino_docs_IE_DG_supported_plugins_GNA}

## Introducing the GNA Plugin

Intel&reg; Gaussian & Neural Accelerator is a low-power neural coprocessor for continuous inference at the edge.

Intel&reg; GNA is not intended to replace classic inference devices such as
CPU, graphics processing unit (GPU), or vision processing unit (VPU) . It is designed for offloading 
continuous inference workloads including but not limited to noise reduction or speech recognition 
to save power and free CPU resources.

The GNA plugin provides a way to run inference on Intel&reg; GNA, as well as in the software execution mode on CPU.

## Devices with Intel&reg; GNA

Devices with Intel&reg; GNA support:

* [Intel&reg; Speech Enabling Developer Kit](https://www.intel.com/content/www/us/en/support/articles/000026156/boards-and-kits/smart-home.html)

* [Amazon Alexa* Premium Far-Field Developer Kit](https://developer.amazon.com/en-US/alexa/alexa-voice-service/dev-kits/amazon-premium-voice)

* [Intel&reg; Pentium&reg; Silver Processors N5xxx, J5xxx and Intel&reg; Celeron&reg; Processors N4xxx, J4xxx](https://ark.intel.com/content/www/us/en/ark/products/codename/83915/gemini-lake.html):
	- Intel&reg; Pentium&reg; Silver J5005 Processor
	- Intel&reg; Pentium&reg; Silver N5000 Processor
	- Intel&reg; Celeron&reg; J4005 Processor
	- Intel&reg; Celeron&reg; J4105 Processor
	- Intel&reg; Celeron&reg; Processor N4100
	- Intel&reg; Celeron&reg; Processor N4000

* [Intel&reg; Core&trade; Processors (formerly codenamed Cannon Lake)](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html):
Intel&reg; Core&trade; i3-8121U Processor

* [10th Generation Intel&reg; Core&trade; Processors (formerly codenamed Ice Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/ice-lake.html):
	- Intel&reg; Core&trade; i7-1065G7 Processor
	- Intel&reg; Core&trade; i7-1060G7 Processor
	- Intel&reg; Core&trade; i5-1035G4 Processor
	- Intel&reg; Core&trade; i5-1035G7 Processor
	- Intel&reg; Core&trade; i5-1035G1 Processor
	- Intel&reg; Core&trade; i5-1030G7 Processor
	- Intel&reg; Core&trade; i5-1030G4 Processor
	- Intel&reg; Core&trade; i3-1005G1 Processor
	- Intel&reg; Core&trade; i3-1000G1 Processor
	- Intel&reg; Core&trade; i3-1000G4 Processor

* All [11th Generation Intel&reg; Core&trade; Processors (formerly codenamed Tiger Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/tiger-lake.html).

> **NOTE**: On platforms where Intel&reg; GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

## Drivers and Dependencies

Intel&reg; GNA hardware requires a driver to be installed on the system.

* Linux\* OS:
[Download Intel&reg; GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.0+)](https://download.01.org/opencv/drivers/gna/)

* Windows\* OS:
Intel&reg; GNA driver for Windows is available through Windows Update\*

## Models and Layers Limitations

Because of specifics of hardware architecture, Intel&reg; GNA supports a limited set of layers, their kinds and combinations.
For example, you should not expect the GNA Plugin to be able to run computer vision models, except those specifically adapted for the GNA Plugin, because the plugin does not fully support
2D convolutions.

The list of supported layers can be found
[here](Supported_Devices.md) (see the GNA column of Supported Layers section).
Limitations include:

- Only 1D convolutions are natively supported in the models converted from:
	- [Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md) framework;
	- [TensorFlow](../../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md) framework; note that for TensorFlow models, the option `--disable_nhwc_to_nchw` must be used when running the Model Optimizer.
- The number of output channels for convolutions must be a multiple of 4
- Permute layer support is limited to the cases where no data reordering is needed, or when reordering is happening for 2 dimensions, at least one of which is not greater than 8

#### Experimental Support for 2D Convolutions

The Intel&reg; GNA hardware natively supports only 1D convolution.

However, 2D convolutions can be mapped to 1D when a convolution kernel moves in a single direction. Such a transformation is performed by the GNA Plugin for Kaldi `nnet1` convolution. From this perspective, the Intel&reg; GNA hardware convolution operation accepts a `NHWC` input and produces `NHWC` output. Because OpenVINO&trade; only supports the `NCHW` layout, it may be necessary to insert `Permute` layers before or after convolutions.

For example, the Kaldi model optimizer inserts such a permute after convolution for the [rm_cnn4a network](https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi/rm_cnn4a_smbr/). This `Permute` layer is automatically removed by the GNA Plugin, because the Intel&reg; GNA hardware convolution layer already produces the required `NHWC` result.

## Operation Precision

Intel&reg; GNA essentially operates in the low-precision mode, which represents a mix of 8-bit (`I8`), 16-bit (`I16`), and 32-bit (`I32`) integer computations, so compared to 32-bit floating point (`FP32`) results – for example, calculated on CPU using Inference Engine [CPU Plugin](CPU.md) – outputs calculated using reduced integer precision are different from the scores calculated using floating point.

Unlike other plugins supporting low-precision execution, the GNA plugin calculates quantization factors at the model loading time, so a model can run without calibration.

## <a name="execution-models">Execution Modes</a>

| Mode | Description |
| :---------------------------------| :---------------------------------------------------------|
| `GNA_AUTO` | Uses Intel&reg; GNA if available, otherwise uses software execution mode on CPU. |
| `GNA_HW` | Uses Intel&reg; GNA if available, otherwise raises an error. |
| `GNA_SW` | *Deprecated*. Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel&reg; GNA, but not in the bit-exact mode. |
| `GNA_SW_EXACT` | Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel&reg; GNA in the bit-exact mode. |
| `GNA_SW_FP32` | Executes the GNA-compiled graph on CPU but substitutes parameters and calculations from low precision to floating point (`FP32`). |

## Supported Configuration Parameters

The plugin supports the configuration parameters listed below.
The parameters are passed as `std::map<std::string, std::string>` on `InferenceEngine::Core::LoadNetwork` or `InferenceEngine::SetConfig`.

The parameter `KEY_GNA_DEVICE_MODE` can also be changed at run time using `InferenceEngine::ExecutableNetwork::SetConfig` (for any values excluding `GNA_SW_FP32`). This allows switching the
execution between software emulation mode and hardware emulation mode after the model is loaded.

The parameter names below correspond to their usage through API keys, such as `GNAConfigParams::KEY_GNA_DEVICE_MODE` or `PluginConfigParams::KEY_PERF_COUNT`.
When specifying key values as raw strings (that is, when using Python API), omit the `KEY_` prefix.

| Parameter Name                    | Parameter Values                                          | Default Value     | Description                                                              |
| :---------------------------------| :---------------------------------------------------------| :-----------| :------------------------------------------------------------------------|
| `KEY_GNA_COMPACT_MODE`            | `YES`/`NO`                                                | `NO`       | Reuse I/O buffers to save space (makes debugging harder)                 |
| `KEY_GNA_SCALE_FACTOR`            | `FP32` number                                             | 1.0         | Scale factor to use for input quantization                               |
| `KEY_GNA_DEVICE_MODE`             | `GNA_AUTO`/`GNA_HW`/`GNA_SW_EXACT`/`GNA_SW_FP32` | `GNA_AUTO`  | One of the modes described <a name="execution-models">Execution Models</a> |
| `KEY_GNA_FIRMWARE_MODEL_IMAGE`    | `std::string`                                             | `""`        | Name for embedded model binary dump file                                 |
| `KEY_GNA_PRECISION`               | `I16`/`I8`                                                | `I16`       | Hint to GNA plugin: preferred integer weight resolution for quantization |
| `KEY_PERF_COUNT`                  | `YES`/`NO`                                                | `NO`        | Turn on performance counters reporting                                   |
| `KEY_GNA_LIB_N_THREADS`           | 1-127 integer number                                      | 1           | Sets the number of GNA accelerator library worker threads used for inference computation in software modes

## How to Interpret Performance Counters

As a result of collecting performance counters using `InferenceEngine::InferRequest::GetPerformanceCounts`, you can find various performance data about execution on GNA.
Returned map stores a counter description as a key, counter value is stored in the `realTime_uSec` field of the `InferenceEngineProfileInfo` structure. Current GNA implementation calculates counters for the whole utterance scoring and does not provide per-layer information. API allows to retrieve counter units in cycles, but they can be converted to seconds as follows:

```
seconds = cycles / frequency
```

Refer to the table below to learn about the frequency of Intel&reg; GNA inside a particular processor.
Processor | Frequency of Intel&reg; GNA
---|---
Intel&reg; Ice Lake processors| 400MHz
Intel&reg; Core&trade; i3-8121U processor| 400MHz
Intel&reg; Gemini Lake  processors | 200MHz

Performance counters provided for the time being:

* Scoring request performance results
	* Number of total cycles spent on scoring in hardware (including compute and memory stall cycles)
	* Number of stall cycles spent in hardware

## Multithreading Support in GNA Plugin

The GNA plugin supports the following configuration parameters for multithreading management:

* `KEY_GNA_LIB_N_THREADS`

	By default, the GNA plugin uses one worker thread for inference computations. This parameter allows you to create up to 127 threads for software modes.

> **NOTE:** Multithreading mode does not guarantee the same computation order as the order of issuing. Additionally, in this case, software modes do not implement any serializations.

## Network Batch Size

Intel&reg; GNA plugin supports the processing of context-windowed speech frames in batches of 1-8 frames in one
input blob using `InferenceEngine::ICNNNetwork::setBatchSize`. Increasing batch size only improves efficiency of `Fully Connected` layers.

> **NOTE**: For networks with `Convolutional`, `LSTM`, or `Memory` layers, the only supported batch size is 1.

## Compatibility with Heterogeneous Plugin

Heterogeneous plugin was tested with the Intel&reg; GNA as a primary device and CPU as a secondary device. To run inference of networks with layers unsupported by the GNA plugin (for example, Softmax), use the Heterogeneous plugin with the `HETERO:GNA,CPU` configuration. For the list of supported networks, see the [Supported Frameworks](#supported-frameworks).

> **NOTE:** Due to limitation of the Intel&reg; GNA backend library, heterogeneous support is limited to cases where in the resulted sliced graph, only one subgraph is scheduled to run on GNA\_HW or GNA\_SW devices.

## Recovery from interruption by high-priority Windows audio processes\*

As noted in the introduction, GNA is designed for real-time workloads such as noise reduction.
For such workloads, processing should be time constrained, otherwise extra delays may cause undesired effects such as
audio "glitches". To make sure that processing can satisfy real time requirements, the GNA driver provides a QoS
(Quality of Service) mechanism which interrupts requests that might cause high-priority Windows audio processes to miss
schedule, thereby causing long running GNA tasks to terminate early.

Applications should be prepared for this situation.
If an inference (in `GNA_HW` mode) cannot be executed because of such an interruption, then `InferRequest::Wait()` will return status code
`StatusCode::INFER_NOT_STARTED` (note that it will be changed to a more meaningful status code in future releases).

Any application working with GNA must properly react if it receives this code. Various strategies are possible.
One of the options is to immediately switch to GNA SW emulation mode:

```cpp
std::map<std::string, Parameter> newConfig;
newConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] = Parameter("GNA_SW_EXACT");
executableNet.SetConfig(newConfig);

```

then resubmit and switch back to GNA_HW after some time hoping that the competing application has finished.

## See Also

* [Supported Devices](Supported_Devices.md)
* [Converting Model](../../MO_DG/prepare_model/convert_model/Converting_Model.md)
* [Convert model from Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
