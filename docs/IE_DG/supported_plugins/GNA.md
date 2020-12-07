# GNA Plugin {#openvino_docs_IE_DG_supported_plugins_GNA}

## Introducing the GNA Plugin

Intel® Gaussian & Neural Accelerator is a low-power neural coprocessor for continuous inference at the edge.

Intel® GNA is not intended to replace classic inference devices such as
CPU, graphics processing unit (GPU), or vision processing unit (VPU). It is designed for offloading 
continuous inference workloads including but not limited to noise reduction or speech recognition 
to save power and free CPU resources.

The GNA plugin provides a way to run inference on Intel® GNA, as well as in the software execution mode on CPU.

## Devices with Intel® GNA

Devices with Intel® GNA support:

* [Intel® Speech Enabling Developer Kit](https://www.intel.com/content/www/us/en/support/articles/000026156/boards-and-kits/smart-home.html)

* [Amazon Alexa\* Premium Far-Field Developer Kit](https://developer.amazon.com/en-US/alexa/alexa-voice-service/dev-kits/amazon-premium-voice)

* [Intel® Pentium® Silver Processors N5xxx, J5xxx and Intel® Celeron® Processors N4xxx, J4xxx](https://ark.intel.com/content/www/us/en/ark/products/codename/83915/gemini-lake.html):
	- Intel® Pentium® Silver J5005 Processor
	- Intel® Pentium® Silver N5000 Processor
	- Intel® Celeron® J4005 Processor
	- Intel® Celeron® J4105 Processor
	- Intel® Celeron® Processor N4100
	- Intel® Celeron® Processor N4000

* [Intel® Core™ Processors (formerly codenamed Cannon Lake)](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html):
Intel® Core™ i3-8121U Processor

* [10th Generation Intel® Core™ Processors (formerly codenamed Ice Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/ice-lake.html):
	- Intel® Core™ i7-1065G7 Processor
	- Intel® Core™ i7-1060G7 Processor
	- Intel® Core™ i5-1035G4 Processor
	- Intel® Core™ i5-1035G7 Processor
	- Intel® Core™ i5-1035G1 Processor
	- Intel® Core™ i5-1030G7 Processor
	- Intel® Core™ i5-1030G4 Processor
	- Intel® Core™ i3-1005G1 Processor
	- Intel® Core™ i3-1000G1 Processor
	- Intel® Core™ i3-1000G4 Processor

* All [11th Generation Intel® Core™ Processors (formerly codenamed Tiger Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/tiger-lake.html).

> **NOTE**: On platforms where Intel® GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

## Drivers and Dependencies

Intel® GNA hardware requires a driver to be installed on the system.

* Linux\* OS:
[Download Intel® GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.0+)](https://download.01.org/opencv/drivers/gna/)

* Windows\* OS:
Intel® GNA driver for Windows is available through Windows Update\*

## Models and Layers Limitations

Because of specifics of hardware architecture, Intel® GNA supports a limited set of layers, their kinds and combinations.
For example, you should not expect the GNA Plugin to be able to run computer vision models, except those specifically adapted 
for the GNA Plugin, because the plugin does not fully support 2D convolutions.

For the list of supported layers, see the **GNA** column of the **Supported Layers** section in [Supported Devices](Supported_Devices.md).

Limitations include:

- Only 1D convolutions are natively supported in the models converted from:
	- [Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md) framework
	- [TensorFlow](../../MO_DG/prepare_model/convert_model/Convert_Model_From_TensorFlow.md) framework. For TensorFlow models, use the `--disable_nhwc_to_nchw` option when running the Model Optimizer.
- The number of output channels for convolutions must be a multiple of 4.
- Permute layer support is limited to the cases where no data reordering is needed or when reordering is happening for two dimensions, at least one of which is not greater than 8.

#### Experimental Support for 2D Convolutions

The Intel® GNA hardware natively supports only 1D convolution.

However, 2D convolutions can be mapped to 1D when a convolution kernel moves in a single direction. GNA Plugin performs such a transformation for Kaldi `nnet1` convolution. From this perspective, the Intel® GNA hardware convolution operation accepts an `NHWC` input and produces an `NHWC` output. Because OpenVINO™ only supports the `NCHW` layout, you may need to insert `Permute` layers before or after convolutions.

For example, the Kaldi model optimizer inserts such a permute after convolution for the [rm_cnn4a network](https://download.01.org/openvinotoolkit/models_contrib/speech/kaldi/rm_cnn4a_smbr/). This `Permute` layer is automatically removed by the GNA Plugin, because the Intel® GNA hardware convolution layer already produces the required `NHWC` result.

## Operation Precision

Intel® GNA essentially operates in the low-precision mode, which represents a mix of 8-bit (`I8`), 16-bit (`I16`), and 32-bit (`I32`) integer computations. Outputs calculated using a reduced integer precision are different from the scores calculated using the floating point format, for example, `FP32` outputs calculated on CPU using the Inference Engine [CPU Plugin](CPU.md).

Unlike other plugins supporting low-precision execution, the GNA plugin calculates quantization factors at the model loading time, so you can run a model without calibration.

## <a name="execution-modes">Execution Modes</a>

| Mode | Description |
| :---------------------------------| :---------------------------------------------------------|
| `GNA_AUTO` | Uses Intel® GNA if available, otherwise uses software execution mode on CPU. |
| `GNA_HW` | Uses Intel® GNA if available, otherwise raises an error. |
| `GNA_SW` | *Deprecated*. Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA, but not in the bit-exact mode. |
| `GNA_SW_EXACT` | Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA in the bit-exact mode. |
| `GNA_SW_FP32` | Executes the GNA-compiled graph on CPU but substitutes parameters and calculations from low precision to floating point (`FP32`). |

## Supported Configuration Parameters

The plugin supports the configuration parameters listed below.
The parameters are passed as `std::map<std::string, std::string>` on `InferenceEngine::Core::LoadNetwork` or `InferenceEngine::SetConfig`.

You can change the `KEY_GNA_DEVICE_MODE` parameter at run time using `InferenceEngine::ExecutableNetwork::SetConfig`, which works for any value excluding `GNA_SW_FP32`. This enables you to switch the
execution between software emulation mode and hardware emulation mode after the model is loaded.

The parameter names below correspond to their usage through API keys, such as `GNAConfigParams::KEY_GNA_DEVICE_MODE` or `PluginConfigParams::KEY_PERF_COUNT`.
When specifying key values as raw strings, that is, when using Python API, omit the `KEY_` prefix.

| Parameter Name                    | Parameter Values                                          | Default Value     | Description                                                              |
| :---------------------------------| :---------------------------------------------------------| :-----------| :------------------------------------------------------------------------|
| `KEY_GNA_COMPACT_MODE`            | `YES`/`NO`                                                | `NO`       | Enables I/O buffers reuse to save space. Makes debugging harder.              |
| `KEY_GNA_SCALE_FACTOR`            | `FP32` number                                             | 1.0         | Sets the scale factor to use for input quantization.                               |
| `KEY_GNA_DEVICE_MODE`             | `GNA_AUTO`/`GNA_HW`/`GNA_SW_EXACT`/`GNA_SW_FP32` | `GNA_AUTO`  |  One of the modes described in <a href="#execution-modes">Execution Modes</a> |
| `KEY_GNA_FIRMWARE_MODEL_IMAGE`    | `std::string`                                             | `""`        | Sets the name for the embedded model binary dump file.                                 |
| `KEY_GNA_PRECISION`               | `I16`/`I8`                                                | `I16`       | Sets the preferred integer weight resolution for quantization. |
| `KEY_PERF_COUNT`                  | `YES`/`NO`                                                | `NO`        | Turns on performance counters reporting.                                   |
| `KEY_GNA_LIB_N_THREADS`           | 1-127 integer number                                      | 1           | Sets the number of GNA accelerator library worker threads used for inference computation in software modes.

## How to Interpret Performance Counters

As a result of collecting performance counters using `InferenceEngine::InferRequest::GetPerformanceCounts`, you can find various performance data about execution on GNA.
Returned map stores a counter description as a key, and a counter value in the `realTime_uSec` field of the `InferenceEngineProfileInfo` structure. Current GNA implementation calculates counters for the whole utterance scoring and does not provide per-layer information. The API enables you to retrieve counter units in cycles, you can convert cycles to seconds as follows:

```
seconds = cycles / frequency
```

Refer to the table below to learn about the frequency of Intel® GNA inside a particular processor.
Processor | Frequency of Intel® GNA
---|---
Intel® Ice Lake processors| 400MHz
Intel® Core™ i3-8121U processor| 400MHz
Intel® Gemini Lake  processors | 200MHz

Performance counters provided for the time being:

* Scoring request performance results
	* Number of total cycles spent on scoring in hardware including compute and memory stall cycles
	* Number of stall cycles spent in hardware

## Multithreading Support in GNA Plugin

The GNA plugin supports the following configuration parameters for multithreading management:

* `KEY_GNA_LIB_N_THREADS`

	By default, the GNA plugin uses one worker thread for inference computations. This parameter allows you to create up to 127 threads for software modes.

> **NOTE:** Multithreading mode does not guarantee the same computation order as the order of issuing. Additionally, in this case, software modes do not implement any serializations.

## Network Batch Size

Intel® GNA plugin supports the processing of context-windowed speech frames in batches of 1-8 frames in one
input blob using `InferenceEngine::ICNNNetwork::setBatchSize`. Increasing batch size only improves efficiency of `Fully Connected` layers.

> **NOTE**: For networks with `Convolutional`, `LSTM`, or `Memory` layers, the only supported batch size is 1.

## Compatibility with Heterogeneous Plugin

Heterogeneous plugin was tested with the Intel® GNA as a primary device and CPU as a secondary device. To run inference of networks with layers unsupported by the GNA plugin, such as Softmax, use the Heterogeneous plugin with the `HETERO:GNA,CPU` configuration.

> **NOTE:** Due to limitation of the Intel® GNA backend library, heterogenous support is limited to cases where in the resulted sliced graph, only one subgraph is scheduled to run on GNA\_HW or GNA\_SW devices.

## Recovery from Interruption by High-Priority Windows Audio Processes\*

GNA is designed for real-time workloads such as noise reduction.
For such workloads, processing should be time constrained, otherwise extra delays may cause undesired effects such as
*audio glitches*. To make sure that processing can satisfy real-time requirements, the GNA driver provides a Quality of Service
(QoS) mechanism, which interrupts requests that might cause high-priority Windows audio processes to miss
the schedule, thereby causing long running GNA tasks to terminate early.

Applications should be prepared for this situation.
If an inference in the `GNA_HW` mode cannot be executed because of such an interruption, then `InferRequest::Wait()` returns status code
`StatusCode::INFER_NOT_STARTED`. In future releases, it will be changed to a more meaningful status code.

Any application working with GNA must properly react to this code.
One of the strategies to adapt an application:

1. Immediately switch to the GNA_SW emulation mode:
```cpp
std::map<std::string, Parameter> newConfig;
newConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] = Parameter("GNA_SW_EXACT");
executableNet.SetConfig(newConfig);

```
2. Resubmit and switch back to GNA_HW expecting that the competing application has finished.

## See Also

* [Supported Devices](Supported_Devices.md)
* [Converting Model](../../MO_DG/prepare_model/convert_model/Converting_Model.md)
* [Convert model from Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)
