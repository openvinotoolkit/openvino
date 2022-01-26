# GNA Plugin {#openvino_docs_IE_DG_supported_plugins_GNA}
## Introducing the GNA Plugin

The Intel® Gaussian & Neural Accelerator is a low-power neural coprocessor for continuous inference at the edge.

Intel® GNA is not intended to replace typical inference devices such as the
CPU, graphics processing unit (GPU), or vision processing unit (VPU). It is designed for offloading
continuous inference workloads including but not limited to noise reduction or speech recognition
to save power and free CPU resources.

The GNA plugin provides a way to run inference on Intel® GNA, as well as in the software execution mode on CPU.

## Devices with Intel® GNA

Devices with Intel® GNA support:

* [Intel® Speech Enabling Developer Kit](https://www.intel.com/content/www/us/en/support/articles/000026156/boards-and-kits/smart-home.html)

* [Amazon Alexa\* Premium Far-Field Developer Kit](https://developer.amazon.com/en-US/alexa/alexa-voice-service/dev-kits/amazon-premium-voice)

* [Intel® Pentium® Silver Processors N5xxx, J5xxx and Intel® Celeron® Processors N4xxx, J4xxx (formerly codenamed Gemini Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/83915/gemini-lake.html):
   - Intel® Pentium® Silver J5005 Processor
   - Intel® Pentium® Silver N5000 Processor
   - Intel® Celeron® J4005 Processor
   - Intel® Celeron® J4105 Processor
   - Intel® Celeron® J4125 Processor
   - Intel® Celeron® Processor N4100
   - Intel® Celeron® Processor N4000

* [Intel® Pentium® Processors N6xxx, J6xxx, Intel® Celeron® Processors N6xxx, J6xxx and Intel Atom® x6xxxxx (formerly codenamed Elkhart Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/128825/products-formerly-elkhart-lake.html)

* [Intel® Core™ Processors (formerly codenamed Cannon Lake)](https://ark.intel.com/content/www/us/en/ark/products/136863/intel-core-i3-8121u-processor-4m-cache-up-to-3-20-ghz.html)

* [10th Generation Intel® Core™ Processors (formerly codenamed Ice Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/74979/ice-lake.html):

* [11th Generation Intel® Core™ Processors (formerly codenamed Tiger Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/88759/tiger-lake.html).

* [12th Generation Intel® Core™ Processors (formerly codenamed Alder Lake)](https://ark.intel.com/content/www/us/en/ark/products/codename/147470/products-formerly-alder-lake.html).

> **NOTE**: On platforms where Intel® GNA is not enabled in the BIOS, the driver cannot be installed, so the GNA plugin uses the software emulation mode only.

## Intel® GNA Generational Differences

The first and second versions of Intel® GNA found in 10th and 11th generation Intel® Core™ Processors may be considered to be functionally equivalent.  Intel® GNA 2.0 provided performance improvement with respect to Intel® GNA 1.0.  Starting with 12th Generation Intel® Core™ Processors (formerly codenamed Alder Lake), support for Intel® GNA 3.0 features is being added.

In the rest of this documentation, "GNA 2.0" refers to Intel® GNA hardware delivered on 10th and 11th generation Intel® Core™ processors, and the term "GNA 3.0" will be used to refer to GNA hardware delivered on 12th generation Intel® Core™ processors.

Initially, a limited subset of Intel® GNA 3.0 features are added to the previous feature set including the following:

* **2D VALID Convolution With Small 2D Kernels:**  Two-dimensional convolutions with the following kernel dimensions [H,W] are supported:  [1,1], [2,2], [3,3], [2,1], [3,1], [4,1], [5,1], [6,1], [7,1], [1,2], or [1,3].  Input tensor dimensions are limited to [1,8,16,16] <= [N,C,H,W] <= [1,120,384,240].  Up to 384 channels C may be used with a subset of kernel sizes (see table below).  Up to 256 kernels (output channels) are supported.  Pooling is limited to pool shapes of [1,1], [2,2], or [3,3].  Not all combinations of kernel shape and input tensor shape are supported (see the tables below for exact limitations).

The tables below show that the exact limitation on the input tensor width W depends on the number of input channels C (indicated as Ci below) and the kernel shape.  There is much more freedom to choose the input tensor height and number of output channels.

## Initially Supported Subset of Intel® GNA 2D Convolutions

The following tables provide a more explicit representation of the Intel(R) GNA 3.0 2D convolution operations initially supported.  The limits depend strongly on number of input tensor channels (Ci) and the input tensor width (W).  Other factors are kernel height (KH), kernel width (KW), pool height (PH), pool width (PW), horizontal pool step (SH), and vertical pool step (PW).  For example, the first table shows that for a 3x3 kernel with max pooling, only square pools are supported, and W is limited to 87 when there are 64 input channels.

**Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters** (Input and Kernel Precision: 2 bytes)

|KH|KW|PH|PW|SH|SW|H|W<br>Ci=8<br>Co=256|W<br>Ci=16<br>Co=256|W<br>Ci=32<br>Co=256|W<br>Ci=64<br>Co=256|W<br>Ci=128<br>Co=256|W<br>Ci=256<br>Co=256|W<br>Ci=384<br>Co=256|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|1|1|1|1|1|1|128|240|240|240|240|240|240|170|
|1|1|1|1|1|1|256|240|240|240|240|240|128|85|
|1|1|1|1|1|1|384|240|240|240|240|170|85|56|
|1|2|1|1|1|1|128|240|240|240|240|   |  |  |
|1|2|1|1|1|1|256|240|240|240|240|   |  |  |
|1|2|1|1|1|1|384|240|240|240|240|   |  |  |
|1|3|1|1|1|1|128|240|240|240|240|   |  |  |
|1|3|1|1|1|1|256|240|240|240|240|   |  |  |
|1|3|1|1|1|1|384|240|240|240|240|   |  |  |
|2|1|1|1|1|1|128|192|192|192|192|192|192|128|
|2|1|1|1|1|1|256|192|192|192|192|192|128|85|
|2|1|1|1|1|1|384|192|192|192|192|170|85|56|
|2|2|1|1|1|1|128|193|193|193|193|   |  |  |
|2|2|1|1|1|1|256|193|193|193|193|   |  |  |
|2|2|1|1|1|1|384|193|193|193|193|   |  |  |
|2|2|2|2|1|1|128|193|193|192|179|   |  |  |
|2|2|2|2|1|1|256|193|193|192|179|   |  |  |
|2|2|2|2|1|1|384|193|193|192|179|   |  |  |
|2|2|2|2|1|2|128|193|193|192|179|   |  |  |
|2|2|2|2|1|2|256|193|193|192|179|   |  |  |
|2|2|2|2|1|2|384|193|193|192|179|   |  |  |
|2|2|2|2|2|1|128|193|193|192|179|   |  |  |
|2|2|2|2|2|1|256|193|193|192|179|   |  |  |
|2|2|2|2|2|1|384|193|193|192|179|   |  |  |
|2|2|2|2|2|2|128|193|193|192|179|   |  |  |
|2|2|2|2|2|2|256|193|193|192|179|   |  |  |
|2|2|2|2|2|2|384|193|193|192|179|   |  |  |
|3|1|1|1|1|1|128|128|128|128|128|128|85|42|
|3|1|1|1|1|1|256|128|128|128|128|128|85|42|
|3|1|1|1|1|1|384|128|128|128|128|128|85|42|
|3|3|1|1|1|1|128|130|130|130|87|   |  |  |
|3|3|1|1|1|1|256|130|130|130|87|   |  |  |
|3|3|1|1|1|1|384|130|130|130|87|   |  |  |
|3|3|2|2|1|1|128|130|130|126|87|   |  |  |
|3|3|2|2|1|1|256|130|130|126|87|   |  |  |
|3|3|2|2|1|1|384|130|130|126|87|   |  |  |
|3|3|2|2|1|2|128|130|130|126|87|   |  |  |
|3|3|2|2|1|2|256|130|130|126|87|   |  |  |
|3|3|2|2|1|2|384|130|130|126|87|   |  |  |
|3|3|2|2|2|1|128|130|130|126|87|   |  |  |
|3|3|2|2|2|1|256|130|130|126|87|   |  |  |
|3|3|2|2|2|1|384|130|130|126|87|   |  |  |
|3|3|2|2|2|2|128|130|130|126|87|   |  |  |
|3|3|2|2|2|2|256|130|130|126|87|   |  |  |
|3|3|2|2|2|2|384|130|130|126|87|   |  |  |
|3|3|3|3|1|1|128|130|128|118|87|   |  |  |
|3|3|3|3|1|1|256|130|128|118|87|   |  |  |
|3|3|3|3|1|1|384|130|128|118|87|   |  |  |
|3|3|3|3|1|2|128|130|128|118|87|   |  |  |
|3|3|3|3|1|2|256|130|128|118|87|   |  |  |
|3|3|3|3|1|2|384|130|128|118|87|   |  |  |
|3|3|3|3|1|3|128|130|128|118|87|   |  |  |
|3|3|3|3|1|3|256|130|128|118|87|   |  |  |
|3|3|3|3|1|3|384|130|128|118|87|   |  |  |
|3|3|3|3|2|1|128|130|128|118|87|   |  |  |
|3|3|3|3|2|1|256|130|128|118|87|   |  |  |
|3|3|3|3|2|1|384|130|128|118|87|   |  |  |
|3|3|3|3|2|2|128|130|128|118|87|   |  |  |
|3|3|3|3|2|2|256|130|128|118|87|   |  |  |
|3|3|3|3|2|2|384|130|128|118|87|   |  |  |
|3|3|3|3|2|3|128|130|128|118|87|   |  |  |
|3|3|3|3|2|3|256|130|128|118|87|   |  |  |
|3|3|3|3|2|3|384|130|128|118|87|   |  |  |
|3|3|3|3|3|1|128|130|128|118|87|   |  |  |
|3|3|3|3|3|1|256|130|128|118|87|   |  |  |
|3|3|3|3|3|1|384|130|128|118|87|   |  |  |
|3|3|3|3|3|2|128|130|128|118|87|   |  |  |
|3|3|3|3|3|2|256|130|128|118|87|   |  |  |
|3|3|3|3|3|2|384|130|128|118|87|   |  |  |
|3|3|3|3|3|3|128|130|128|118|87|   |  |  |
|3|3|3|3|3|3|256|130|128|118|87|   |  |  |
|3|3|3|3|3|3|384|130|128|118|87|   |  |  |
|4|1|1|1|1|1|128|96|96|96|96|96|64|32|
|4|1|1|1|1|1|256|96|96|96|96|96|64|32|
|4|1|1|1|1|1|384|96|96|96|96|96|64|32|
|5|1|1|1|1|1|128|76|76|76|76|51|25|  |
|5|1|1|1|1|1|256|76|76|76|76|51|25|  |
|5|1|1|1|1|1|384|76|76|76|76|51|25|  |
|6|1|1|1|1|1|128|64|64|64|64|42|21|  |
|6|1|1|1|1|1|256|64|64|64|64|42|21|  |
|6|1|1|1|1|1|384|64|64|64|64|42|21|  |
|7|1|1|1|1|1|128|54|54|54|54|36|  |  |
|7|1|1|1|1|1|256|54|54|54|54|36|  |  |
|7|1|1|1|1|1|384|54|54|54|54|36|  |  |

**Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters** (Input and Kernel Precision: 1 bytes)

|KH|KW|PH|PW|SH|SW|H|W<br>Ci=8<br>Co=256|W<br>Ci=16<br>Co=256|W<br>Ci=32<br>Co=256|W<br>Ci=64<br>Co=256|W<br>Ci=128<br>Co=256|W<br>Ci=256<br>Co=256|W<br>Ci=384<br>Co=256|
|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
|1|1|1|1|1|1|128|240|240|240|240|240|240|240|
|1|1|1|1|1|1|256|240|240|240|240|240|240|170|
|1|1|1|1|1|1|384|240|240|240|240|240|170|113|
|1|2|1|1|1|1|128|240|240|240|240|240|240|240|
|1|2|1|1|1|1|256|240|240|240|240|240|240|170|
|1|2|1|1|1|1|384|240|240|240|240|240|170|113|
|1|3|1|1|1|1|128|240|240|240|240|240|   |   |
|1|3|1|1|1|1|256|240|240|240|240|240|   |   |
|1|3|1|1|1|1|384|240|240|240|240|240|   |   |
|2|1|1|1|1|1|128|192|192|192|192|192|192|192|
|2|1|1|1|1|1|256|192|192|192|192|192|192|170|
|2|1|1|1|1|1|384|192|192|192|192|192|170|113|
|2|2|1|1|1|1|128|193|193|193|193|193|193|129|
|2|2|1|1|1|1|256|193|193|193|193|193|193|129|
|2|2|1|1|1|1|384|193|193|193|193|193|170|113|
|3|1|1|1|1|1|128|128|128|128|128|128|128|85|
|3|1|1|1|1|1|256|128|128|128|128|128|128|85|
|3|1|1|1|1|1|384|128|128|128|128|128|128|85|
|3|3|1|1|1|1|128|130|130|130|130|87 |   |  |
|3|3|1|1|1|1|256|130|130|130|130|87 |   |  |
|3|3|1|1|1|1|384|130|130|130|130|87 |   |  |
|4|1|1|1|1|1|128|96|96|96|96|96|96|64|
|4|1|1|1|1|1|256|96|96|96|96|96|96|64|
|4|1|1|1|1|1|384|96|96|96|96|96|96|64|
|5|1|1|1|1|1|128|76|76|76|76|76|51|51|
|5|1|1|1|1|1|256|76|76|76|76|76|51|51|
|5|1|1|1|1|1|384|76|76|76|76|76|51|51|
|6|1|1|1|1|1|128|64|64|64|64|64|42|21|
|6|1|1|1|1|1|256|64|64|64|64|64|42|21|
|6|1|1|1|1|1|384|64|64|64|64|64|42|21|
|7|1|1|1|1|1|128|54|54|54|54|54|36|18|
|7|1|1|1|1|1|256|54|54|54|54|54|36|18|
|7|1|1|1|1|1|384|54|54|54|54|54|36|18|


> **NOTE**:  The above limitations only apply to the new hardware 2D convolution operation.  When possible, the Intel® GNA plugin graph compiler flattens 2D convolutions so that the second generation Intel® GNA 1D convolution operations (without these limitations) may be used.  The plugin will also flatten 2D convolutions regardless of the sizes  if GNA 2.0 compilation target is selected (see below).

## Intel® GNA Forward and Backward Compatibility

In the general case, there is no guarantee that a model compiled for GNA 2.0 will run on GNA 3.0, or vice versa.

However, in most cases, networks compiled for GNA 2.0 will run as expected on GNA 3.0, although the performance may be worse compared to the case when a network is compiled specifically for the latter.  The exception is networks with convolutions with the number of filters greater than 8192 (see the <a href="#models-and-layers-limitations">Models and Layers Limitations</a> section).

Networks compiled for GNA 3.0 should run on GNA 2.0 with incompatible layers emulated on CPU.

You can use the following options `KEY_GNA_EXEC_TARGET` and `KEY_GNA_COMPILE_TARGET` options  to check interoperability (see the <a href="#supported-configuration-parameters">Supported Configuration Parameters</a> section below):

@sphinxdirective
.. tab:: C++

   ``KEY_GNA_EXEC_TARGET``,  ``KEY_GNA_COMPILE_TARGET``

.. tab:: Python

   ``GNA_EXEC_TARGET``,  ``GNA_COMPILE_TARGET``

@endsphinxdirective

## Drivers and Dependencies

Intel® GNA hardware requires a driver to be installed on the system.

* Linux\* OS:
[Download Intel® GNA driver for Ubuntu Linux 18.04.3 LTS (with HWE Kernel version 5.4+)](https://storage.openvinotoolkit.org/drivers/gna/)

* Windows\* OS:
Intel® GNA driver for Windows is available through Windows Update\*

## <a name="models-and-layers-limitations">Models and Layers Limitations</a>

Because of specifics of hardware architecture, Intel® GNA supports a limited set of layers, their kinds and combinations.
For example, you should not expect the GNA Plugin to be able to run computer vision models, except those specifically adapted for the GNA Plugin, because the plugin does not fully support 2D convolutions.

For the list of supported layers, see the **GNA** column of the **Supported Layers** section in [Supported Devices](Supported_Devices.md).

Limitations include:

- Only 1D convolutions are natively supported.
- The number of output channels for convolutions must be a multiple of 4.
- The maximum number of filters is 65532 for GNA 2.0 and 8192 for GNA 3.0.
- Permute layer support is limited to the cases where no data reordering is needed or when reordering is happening for two dimensions, at least one of which is not greater than 8.
- Splits and concatenations are supported for continuous portions of memory (e.g., split of 1,2,3,4 to 1,1,3,4 and 1,1,3,4 or concats of 1,2,3,4 and 1,2,3,5 to 2,2,3,4).
- For Multiply, Add and Subtract layers, auto broadcasting is only supported for constant inputs.

### Support for 2D Convolutions in Previous Generations of GNA Hardware

The Intel® GNA 1.0 and 2.0 hardware natively supports only 1D convolutions.

However, 2D convolutions can be mapped to 1D when a convolution kernel moves in a single direction. GNA Plugin performs such a transformation for Kaldi `nnet1` convolution. From this perspective, the Intel® GNA hardware convolution operation accepts an `NHWC` input and produces an `NHWC` output. Because OpenVINO™ only supports the `NCHW` layout, you may need to insert `Permute` layers before or after convolutions.

For example, the Kaldi model optimizer inserts such a permute after convolution for the [rm_cnn4a network](https://storage.openvinotoolkit.org/models_contrib/speech/2021.2/rm_cnn4a_smbr/). This `Permute` layer is automatically removed by the GNA Plugin, because the Intel® GNA hardware convolution layer already produces the required `NHWC` result.

## Operation Precision

Intel® GNA essentially operates in the low-precision mode, which represents a mix of 8-bit (`I8`), 16-bit (`I16`), and 32-bit (`I32`) integer computations. Outputs calculated using a reduced integer precision are different from the scores calculated using the floating point format, for example, `FP32` outputs calculated on CPU using the Inference Engine [CPU Plugin](CPU.md).

Unlike other plugins supporting low-precision execution, the GNA plugin can calculate quantization factors at the model loading time, so you can run a model without calibration using the [Post-Training Optimization Tool](@ref pot_README).
However, this mode may not provide satisfactory accuracy because the internal quantization algorithm is based on heuristics which may or may not be efficient, depending on the model and dynamic range of input data.

Starting with 2021.4 release of OpenVINO, GNA plugin users are encouraged to use the [POT API Usage sample for GNA](@ref pot_sample_speech_README) to get a model with quantization hints based on statistics for the provided dataset.

## <a name="execution-modes">Execution Modes</a>

@sphinxdirective
.. tab:: C++

   ============================  ==============================================================================================================================================
   Mode                          Description
   ============================  ==============================================================================================================================================
   ``KEY_GNA_AUTO``              Uses Intel® GNA if available, otherwise uses software execution mode on CPU.
   ``KEY_GNA_HW``                Uses Intel® GNA if available, otherwise raises an error.
   ``KEY_GNA_SW``                *Deprecated*. Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA, but not in the bit-exact mode.
   ``KEY_GNA_SW_EXACT``          Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA in the bit-exact mode.
   ``KEY_GNA_HW_WITH_SW_FBACK``  Uses Intel® GNA if available, otherwise raises an error. If the hardware queue is not empty, automatically falls back to CPU in the bit-exact mode.
   ``KEY_GNA_SW_FP32``           Executes the GNA-compiled graph on CPU but substitutes parameters and calculations from low precision to floating point (``FP32``).
   ============================  ==============================================================================================================================================

.. tab:: Python

   ========================  ==============================================================================================================================================
   Mode                      Description
   ========================  ==============================================================================================================================================
   ``GNA_AUTO``              Uses Intel® GNA if available, otherwise uses software execution mode on CPU.
   ``GNA_HW``                Uses Intel® GNA if available, otherwise raises an error.
   ``GNA_SW``                *Deprecated*. Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA, but not in the bit-exact mode.
   ``GNA_SW_EXACT``          Executes the GNA-compiled graph on CPU performing calculations in the same precision as the Intel® GNA in the bit-exact mode.
   ``GNA_HW_WITH_SW_FBACK``  Uses Intel® GNA if available, otherwise raises an error. If the hardware queue is not empty, automatically falls back to CPU in the bit-exact mode.
   ``GNA_SW_FP32``           Executes the GNA-compiled graph on CPU but substitutes parameters and calculations from low precision to floating point (``FP32``).
   ========================  ==============================================================================================================================================

@endsphinxdirective

## <a name="supported-configuration-parameters">Supported Configuration Parameters</a>

The plugin supports the configuration parameters listed below. The parameter names correspond to their usage through API keys, such as ``GNAConfigParams::KEY_GNA_DEVICE_MODE`` or ``PluginConfigParams::KEY_PERF_COUNT`` in C++ and ``GNA_DEVICE_MODE`` or ``PERF_COUNT`` in Python.

@sphinxdirective
.. tab:: C++

   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | Parameter Name                   | Values                  | Default Value | Description                                                     |
   +==================================+=========================+===============+=================================================================+
   | ``KEY_GNA_EXEC_TARGET``          | ``TARGET_2_0``,         | *see below*   | Defines the execution target.                                   |
   |                                  | ``TARGET_3_0``          |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_COMPILE_TARGET``       | ``TARGET_2_0``,         | *see below*   | Defines the compilation target.                                 |
   |                                  | ``TARGET_3_0``          |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_COMPACT_MODE``         | ``YES``, ``NO``         | ``NO``        | Enables I/O buffers reuse to save space.                        |
   |                                  |                         |               | Makes debugging harder.                                         |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_SCALE_FACTOR``         | FP32 number             | 1.0           | Sets the scale factor to use for input quantization.            |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_DEVICE_MODE``          | ``GNA_AUTO``,           | ``GNA_AUTO``  | One of the modes described                                      |
   |                                  | ``GNA_HW``,             |               | in `Execution Modes <#execution-modes>`_.                       |
   |                                  | ``GNA_HW_WITH_SW_FBACK``|               |                                                                 |
   |                                  | ``GNA_SW_EXACT``,       |               |                                                                 |
   |                                  | ``GNA_SW_FP32``         |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_FIRMWARE_MODEL_IMAGE`` | ``std::string``         | ``""``        | Sets the name for the embedded model binary dump file.          |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_PRECISION``            | ``I16``, ``I8``         | ``I16``       | Sets the preferred integer weight resolution for quantization   |
   |                                  |                         |               | (ignored for models produced using POT).                        |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_PERF_COUNT``               | ``YES``, ``NO``         | ``NO``        | Turns on performance counters reporting.                        |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+

   The parameters are passed as ``std::map<std::string, std::string>`` on ``InferenceEngine::Core::LoadNetwork`` or ``InferenceEngine::SetConfig``.

   Normally, you do not need to select the execution target (``KEY_GNA_EXEC_TARGET``) and compilation target (``KEY_GNA_COMPILE_TARGET``). The default value for the execution target corresponds to available hardware, or latest hardware version supported by the plugin (i.e., GNA 3.0) if there is no GNA HW in the system. The compilation target is the same as the execution target by default. However, you may want to change the targets, for example, if you want to check how a model compiled for one generation would behave on the other generation (using the software emulation mode), or if you are willing to export a model for a specific version of GNA HW.

   You can change the ``KEY_GNA_DEVICE_MODE`` parameter at run time using ``InferenceEngine::ExecutableNetwork::SetConfig``, which works for any value excluding ``GNA_SW_FP32``. This enables you to switch the execution between software emulation mode and hardware execution mode after the model is loaded.

.. tab:: Python

   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | Parameter Name                   | Values                  | Default Value | Description                                                     |
   +==================================+=========================+===============+=================================================================+
   | ``GNA_EXEC_TARGET``              | ``TARGET_2_0``,         | _see below_   | Defines the execution target.                                   |
   |                                  | ``TARGET_3_0``          |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``GNA_COMPILE_TARGET``           | ``TARGET_2_0``,         | _see below_   | Defines the compilation target.                                 |
   |                                  | ``TARGET_3_0``          |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``GNA_COMPACT_MODE``             | ``YES``, ``NO``         | ``NO``        | Enables I/O buffers reuse to save space.                        |
   |                                  |                         |               | Makes debugging harder.                                         |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``GNA_SCALE_FACTOR``             | FP32 number             | 1.0           | Sets the scale factor to use for input quantization.            |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``KEY_GNA_DEVICE_MODE``          | ``GNA_AUTO``,           | ``GNA_AUTO``  | One of the modes described                                      |
   |                                  | ``GNA_HW``,             |               | in `Execution Modes <#execution-modes>`_.                       |
   |                                  | ``GNA_HW_WITH_SW_FBACK``|               |                                                                 |
   |                                  | ``GNA_SW_EXACT``,       |               |                                                                 |
   |                                  | ``GNA_SW_FP32``         |               |                                                                 |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``GNA_FIRMWARE_MODEL_IMAGE``     | ``string``              | ``""``        | Sets the name for the embedded model binary dump file.          |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``GNA_PRECISION``                | ``I16``, ``I8``         | ``I16``       | Sets the preferred integer weight resolution for quantization   |
   |                                  |                         |               | (ignored for models produced using POT).                        |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+
   | ``PERF_COUNT``                   | ``YES``, ``NO``         | ``NO``        | Turns on performance counters reporting.                        |
   +----------------------------------+-------------------------+---------------+-----------------------------------------------------------------+

   The parameters are passed as strings to `IECore.load_network <api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.load_network>`_.

   Normally, you do not need to select the execution target (``GNA_EXEC_TARGET``) and compilation target (``GNA_COMPILE_TARGET``). The default value for the execution target corresponds to available hardware, or latest hardware version supported by the plugin (i.e., GNA 3.0) if there is no GNA HW in the system. The compilation target is the same as the execution target by default. However, you may want to change the targets, for example, if you want to check how a model compiled for one generation would behave on the other generation (using the SW emulation mode), or if you are willing to export a model for a specific version of GNA HW.

   You can change the ``GNA_DEVICE_MODE`` parameter at run time by sending a configuration dict to the `IECore.load_network <api/ie_python_api/_autosummary/openvino.inference_engine.IECore.html#openvino.inference_engine.IECore.load_network>`_ call, which works for any value excluding ``GNA_SW_FP32``. This enables you to switch the execution between software emulation mode and hardware execution mode after the model is loaded.

@endsphinxdirective
## How to Interpret Performance Counters

With the following methods, you can collect performance counters that provides various performance data about execution on GNA:

@sphinxdirective
.. tab:: C++

   ``InferenceEngine::InferRequest::GetPerformanceCounts``

   The returned map stores a counter description as a key, and a counter value in the ``realTime_uSec`` field of the ``InferenceEngineProfileInfo`` structure.


.. tab:: Python

   ``openvino.inference_engine.InferRequest.get_perf_counts``

   The returned map stores a counter description as a key, and a counter value in the ``real_time`` field.

@endsphinxdirective

The current GNA implementation calculates counters for the whole utterance scoring and does not provide per-layer information. The API enables you to retrieve counter units in cycles, you can convert cycles to seconds as follows:

```
seconds = cycles / frequency
```

Refer to the table below to learn about the frequency of Intel® GNA inside a particular processor:
Processor | Frequency of Intel® GNA
---|---
Intel® Core™ processors| 400MHz
Intel® processors formerly codenamed Elkhart Lake | 200MHz
Intel® processors formerly codenamed Gemini Lake | 200MHz

Performance counters provided for the time being:

* Scoring request performance results
	* Number of total cycles spent on scoring in hardware including compute and memory stall cycles
	* Number of stall cycles spent in hardware

## Network Batch Size

Intel® GNA plugin supports the processing of context-windowed speech frames in batches of 1-8 frames in one
input blob using the following methods:

@sphinxdirective
.. tab:: C++

   ``InferenceEngine::ICNNNetwork::setBatchSize``

.. tab:: Python

   `IENetwork.batch_size <api/ie_python_api/_autosummary/openvino.inference_engine.IENetwork.html#openvino.inference_engine.IENetwork.batch_size>`_

@endsphinxdirective

Increasing batch size only improves efficiency of `Fully Connected` layers.

> **NOTE**: For networks with `Convolutional`, `LSTM`, or `Memory` layers, the only supported batch size is 1.

## Compatibility with Heterogeneous Plugin

Heterogeneous plugin was tested with the Intel® GNA as a primary device and CPU as a secondary device. To run inference of networks with layers unsupported by the GNA plugin, such as Softmax, use the Heterogeneous plugin with the `HETERO:GNA,CPU` configuration.

> **NOTE**: Due to limitation of the Intel® GNA backend library, heterogenous support is limited to cases where in the resulted sliced graph, only one subgraph is scheduled to run on GNA\_HW or GNA\_SW devices.

## Recovery from Interruption by High-Priority Windows Audio Processes\*

GNA is designed for real-time workloads such as noise reduction.
For such workloads, processing should be time constrained, otherwise extra delays may cause undesired effects such as
*audio glitches*. To make sure that processing can satisfy real-time requirements, the GNA driver provides a Quality of Service
(QoS) mechanism, which interrupts requests that might cause high-priority Windows audio processes to miss
the schedule, thereby causing long running GNA tasks to terminate early.

Applications should be prepared for this situation.

If an inference in the `GNA_HW` mode cannot be executed because of such an interruption, then the `wait` method returns the following status code:

@sphinxdirective
.. tab:: C++

   ``InferRequest::Wait()`` returns status code ``StatusCode::INFER_NOT_STARTED``.

.. tab:: Python

   `InferRequest.wait <api/ie_python_api/_autosummary/openvino.inference_engine.InferRequest.html#openvino.inference_engine.InferRequest.wait>`_ returns status code `INFER_NOT_STARTED`.

@endsphinxdirective

In future releases, it will be changed to a more meaningful status code.

Any application working with GNA must properly react to this code.
One of the strategies to adapt an application:

1. Immediately switch to the GNA_SW_EXACT emulation mode:
@sphinxdirective
.. tab:: C++

   .. code-block:: cpp

      std::map<std::string, Parameter> newConfig;
      newConfig[GNAConfigParams::KEY_GNA_DEVICE_MODE] = Parameter("GNA_SW_EXACT");
      executableNet.SetConfig(newConfig);

.. tab:: Python

   .. code-block:: python

      from openvino.inference_engine import IECore

      ie = IECore()
      new_cfg = {'GNA_DEVICE_MODE' : 'GNA_SW_EXACT'}
      net = ie.read_network(model=path_to_model)
      exec_net = ie.load_network(network=net, device_name="GNA", config=new_cfg)

@endsphinxdirective

2. Resubmit and switch back to GNA_HW expecting that the competing application has finished.

   > **NOTE**: This method is deprecated since a new automatic QoS mode has been introduced in 2021.4.1 release of OpenVINO™ (see below).

## GNA3 Automatic QoS Feature on Windows*

Starting with 2021.4.1 release of OpenVINO and 03.00.00.1363 version of Windows* GNA driver, a new execution mode `GNA_HW_WITH_SW_FBACK` is introduced
to assure that workloads satisfy real-time execution. In this mode, the GNA driver automatically falls back on CPU for a particular infer request
if the HW queue is not empty, so there is no need for explicitly switching between GNA and CPU.

> **NOTE**: Due to the "first come - first served" nature of GNA driver and the QoS feature, this mode may lead to increased CPU consumption
if there are several clients using GNA simultaneously.
Even a lightweight competing infer request which has not been cleared at the time when the user's GNA client process makes its request,
can cause the user's request to be executed on CPU, thereby unnecessarily increasing CPU utilization and power.

## See Also

* [Supported Devices](Supported_Devices.md)
* [Converting Model](../../MO_DG/prepare_model/convert_model/Converting_Model.md)
* [Convert model from Kaldi](../../MO_DG/prepare_model/convert_model/Convert_Model_From_Kaldi.md)