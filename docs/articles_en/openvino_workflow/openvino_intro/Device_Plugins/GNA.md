# GNA Device {#openvino_docs_OV_UG_supported_plugins_GNA}




@sphinxdirective

.. meta::
   :description: The GNA plugin in OpenVINO™ Runtime enables running inference 
                 on Intel® Gaussian & Neural Accelerator (GNA) and in the 
                 software execution mode on CPU.


The Intel® Gaussian & Neural Accelerator (GNA) is a low-power neural coprocessor for continuous inference at the edge.

Intel® GNA is not intended to replace typical inference devices such as the CPU and GPU. It is designed for offloading
continuous inference workloads including but not limited to noise reduction or speech recognition
to save power and free CPU resources. It lets you run inference on Intel® GNA, as well as the CPU, in the software execution mode.
For more details on how to configure a system to use GNA, see the :doc:`GNA configuration page <openvino_docs_install_guides_configurations_for_intel_gna>`.

.. note::

   Intel's GNA is being discontinued and Intel® Core™ Ultra (formerly known as Meteor Lake) 
   will be the last generation of hardware to include it.
   For this reason, the GNA plugin will soon be discontinued.
   Consider Intel's new Neural Processing Unit as a low-power solution for offloading 
   neural network computation, for processors offering the technology.
   


Intel® GNA Generational Differences
###########################################################

The first (1.0) and second (2.0) versions of Intel® GNA found in 10th and 11th generation Intel® Core™ Processors may be considered
functionally equivalent. Intel® GNA 2.0 provided performance improvement with respect to Intel® GNA 1.0.

=======================  ========================
 Intel CPU generation     GNA HW Version
=======================  ========================
10th, 11th                GNA 2.0
12th, 13th                GNA 3.0
14th                      GNA 3.5
=======================  ========================

In this documentation, "GNA 2.0" refers to Intel® GNA hardware delivered on 10th and 11th generation Intel® Core™ processors,
and the term "GNA 3.0" refers to GNA hardware delivered on 12th, 13th generation Intel® Core™ processors, and the term
"GNA 3.5" refers to GNA hardware delivered on 14th generation of Intel® Core™ processors.

Intel® GNA Forward and Backward Compatibility
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

When a model is run, using the GNA plugin, it is compiled internally for the specific hardware target. It is possible to export a compiled model, 
using `Import/Export <#import-export>`__ functionality to use it later. In general, there is no guarantee that a model compiled and 
exported for GNA 2.0 runs on GNA 3.0 or vice versa.

==================  ========================  =======================================================  =======================================================
 Hardware            Compile target 2.0        Compile target 3.0                                       Compile target 3.5
==================  ========================  =======================================================  =======================================================
 GNA 2.0             Supported                 Not supported (incompatible layers emulated on CPU)      Not supported (incompatible layers emulated on CPU)
 GNA 3.0             Partially supported       Supported                                                Not supported (incompatible layers emulated on CPU)
 GNA 3.5             Partially supported       Partially supported                                      Supported
==================  ========================  =======================================================  =======================================================

.. note::

   In most cases, a network compiled for GNA 2.0 runs as expected on GNA 3.0. However, performance may be worse 
   compared to when a network is compiled specifically for the latter. The exception is a network with convolutions 
   with the number of filters greater than 8192 (see the `Model and Operation Limitations <#model-and-operation-limitations>`__ section).


For optimal work with POT quantized models, which include 2D convolutions on GNA 3.0/3.5 hardware, the following requirements should be satisfied:

* Choose a compile target with priority on: cross-platform execution, performance, memory, or power optimization.
* To check interoperability in your application use: ``ov::intel_gna::execution_target`` and ``ov::intel_gna::compile_target``.

:doc:`Speech C++ Sample <openvino_inference_engine_samples_speech_sample_README>` can be used for experiments (see the ``-exec_target`` and ``-compile_target`` command line options).


Software Emulation Mode
###########################################################

Software emulation mode is used by default on platforms without GNA hardware support. Therefore, model runs even if there is no GNA HW within your platform.
GNA plugin enables switching the execution between software emulation mode and hardware execution mode once the model has been loaded.
For details, see a description of the ``ov::intel_gna::execution_mode`` property.

Recovery from Interruption by High-Priority Windows Audio Processes
############################################################################

GNA is designed for real-time workloads i.e., noise reduction. For such workloads, processing should be time constrained. 
Otherwise, extra delays may cause undesired effects such as *audio glitches*. The GNA driver provides a Quality of Service (QoS) 
mechanism to ensure that processing can satisfy real-time requirements. The mechanism interrupts requests that might cause 
high-priority Windows audio processes to miss the schedule. As a result, long running GNA tasks terminate early.

To prepare the applications correctly, use Automatic QoS Feature described below.

Automatic QoS Feature on Windows
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Starting with the 2021.4.1 release of OpenVINO™ and the 03.00.00.1363 version of Windows GNA driver, the execution mode of 
``ov::intel_gna::ExecutionMode::HW_WITH_SW_FBACK`` has been available to ensure that workloads satisfy real-time execution. 
In this mode, the GNA driver automatically falls back on CPU for a particular infer request if the HW queue is not empty. 
Therefore, there is no need for explicitly switching between GNA and CPU.

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gna/configure.py
         :language: py
         :fragment: [import]

      .. doxygensnippet:: docs/snippets/gna/configure.py
         :language: py
         :fragment: [ov_gna_exec_mode_hw_with_sw_fback]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gna/configure.cpp
         :language: cpp
         :fragment: [include]

      .. doxygensnippet:: docs/snippets/gna/configure.cpp
         :language: cpp
         :fragment: [ov_gna_exec_mode_hw_with_sw_fback]


.. note:: 
   
   Due to the "first come - first served" nature of GNA driver and the QoS feature, this mode may lead to increased 
   CPU consumption if there are several clients using GNA simultaneously. Even a lightweight competing infer request, 
   not cleared at the time when the user's GNA client process makes its request, can cause the user's request to be 
   executed on CPU, unnecessarily increasing CPU utilization and power.


Supported Inference Data Types
###########################################################

Intel® GNA essentially operates in the low-precision mode which represents a mix of 8-bit (``i8``), 16-bit (``i16``), and 32-bit (``i32``) 
integer computations. Unlike other OpenVINO devices supporting low-precision execution, it can calculate quantization factors at the 
model loading time. Therefore, a model can be run without calibration. However, this mode may not provide satisfactory accuracy 
because the internal quantization algorithm is based on heuristics, the efficiency of which depends on the model and dynamic range of input data. 
This mode is going to be deprecated soon. GNA supports the ``i16`` and ``i8`` quantized data types as inference precision of internal primitives.

GNA users are encouraged to use the :doc:`Post-Training Optimization Tool <pot_introduction>` to get a model with 
quantization hints based on statistics for the provided dataset. 

:doc:`Hello Query Device C++ Sample <openvino_inference_engine_samples_hello_query_device_README>` can be used to print out supported data types for all detected devices.

:doc:`POT API Usage sample for GNA <pot_example_speech_README>` demonstrates how a model can be quantized for GNA, using POT API in two modes:

* Accuracy (i16 weights)
* Performance (i8 weights)

For POT quantized models, the ``ov::hint::inference_precision`` property has no effect except in cases described in the
`Model and Operation Limitations section <#model-and-operation-limitations>`__.


Supported Features
###########################################################

Model Caching
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Due to import/export functionality support (see below), cache for GNA plugin may be enabled via common ``ov::cache_dir`` property of OpenVINO™.

For more details, see the :doc:`Model caching overview <openvino_docs_OV_UG_Model_caching_overview>`.


Import/Export
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The GNA plugin supports import/export capability, which helps decrease first inference time significantly. 
The model compile target is the same as the execution target by default. If there is no GNA HW in the system, 
the default value for the execution target corresponds to available hardware or latest hardware version, 
supported by the plugin (i.e., GNA 3.0).

To export a model for a specific version of GNA HW, use the ``ov::intel_gna::compile_target`` property and then export the model:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gna/import_export.py
         :language: py
         :fragment: [ov_gna_export]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gna/import_export.cpp
         :language: cpp
         :fragment: [ov_gna_export]


Import model:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gna/import_export.py
         :language: py
         :fragment: [ov_gna_import]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gna/import_export.cpp
         :language: cpp
         :fragment: [ov_gna_import]


To compile a model, use either :ref:`compile Tool <openvino_ecosystem>` or 
:doc:`Speech C++ Sample <openvino_inference_engine_samples_speech_sample_README>`.

Stateful Models
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

GNA plugin natively supports stateful models. For more details on such models, refer to the :doc:`Stateful models <openvino_docs_OV_UG_model_state_intro>`.

.. note:: 

   The GNA is typically used in streaming scenarios when minimizing latency is important. Taking into account that POT does not 
   support the ``TensorIterator`` operation, the recommendation is to use the ``transform`` option of model conversion API 
   to apply ``LowLatency2`` transformation when converting an original model.

Profiling
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The GNA plugin allows turning on profiling, using the ``ov::enable_profiling`` property.
With the following methods, you can collect profiling information with various performance data about execution on GNA:

.. tab-set::

   .. tab-item:: Python
      :sync: py
   
      ``openvino.runtime.InferRequest.get_profiling_info``

   .. tab-item:: C++
      :sync: cpp
   
      ``ov::InferRequest::get_profiling_info``


The current GNA implementation calculates counters for the whole utterance scoring and does not provide per-layer information. 
The API enables you to retrieve counter units in cycles. You can convert cycles to seconds as follows:

.. code-block:: sh

   seconds = cycles / frequency


Refer to the table below for the frequency of Intel® GNA inside particular processors:

==========================================================  ==================================
 Processor                                                   Frequency of Intel® GNA, MHz
==========================================================  ==================================
Intel® Core™ processors                                      400
Intel® processors formerly codenamed Elkhart Lake            200
Intel® processors formerly codenamed Gemini Lake             200
==========================================================  ==================================


Inference request performance counters provided for the time being:

* The number of total cycles spent on scoring in hardware, including compute and memory stall cycles
* The number of stall cycles spent in hardware


Supported Properties
###########################################################

Read-write Properties
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In order to take effect, the following parameters must be set before model compilation or passed as additional arguments to ``ov::Core::compile_model()``:

- ``ov::cache_dir``
- ``ov::enable_profiling``
- ``ov::hint::inference_precision``
- ``ov::hint::num_requests``
- ``ov::intel_gna::compile_target``
- ``ov::intel_gna::firmware_model_image_path``
- ``ov::intel_gna::execution_target``
- ``ov::intel_gna::pwl_design_algorithm``
- ``ov::intel_gna::pwl_max_error_percent``
- ``ov::intel_gna::scale_factors_per_input``

These parameters can be changed after model compilation ``ov::CompiledModel::set_property``:

- ``ov::hint::performance_mode``
- ``ov::intel_gna::execution_mode``
- ``ov::log::level``

Read-only Properties
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

- ``ov::available_devices``
- ``ov::device::capabilities``
- ``ov::device::full_name``
- ``ov::intel_gna::library_full_version``
- ``ov::optimal_number_of_infer_requests``
- ``ov::range_for_async_infer_requests``
- ``ov::supported_properties``

Limitations
###########################################################

Model and Operation Limitations
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Due to the specification of hardware architecture, Intel® GNA supports a limited set of operations (including their kinds and combinations).
For example, GNA Plugin should not be expected to run computer vision models because the plugin does not fully support 2D convolutions. 
The exception are the models specifically adapted for the GNA Plugin.

Limitations include:

- Prior to GNA 3.0, only 1D convolutions are natively supported on the HW; 2D convolutions have specific limitations (see the table below).
- The number of output channels for convolutions must be a multiple of 4.
- The maximum number of filters is 65532 for GNA 2.0 and 8192 for GNA 3.0.
- Starting with Intel® GNA 3.5 the support for Int8 convolution weights has been added. Int8 weights can be used in models quantized by POT.
- *Transpose* layer support is limited to the cases where no data reordering is needed or when reordering is happening for two dimensions, at least one of which is not greater than 8.
- Splits and concatenations are supported for continuous portions of memory (e.g., split of 1,2,3,4 to 1,1,3,4 and 1,1,3,4 or concats of 1,2,3,4 and 1,2,3,5 to 2,2,3,4).
- For *Multiply*, *Add* and *Subtract* layers, auto broadcasting is only supported for constant inputs.


Support for 2D Convolutions up to GNA 3.0
-----------------------------------------------------------

The Intel® GNA 1.0 and 2.0 hardware natively supports only 1D convolutions. However, 2D convolutions can be mapped to 1D when 
a convolution kernel moves in a single direction. Initially, a limited subset of Intel® GNA 3.0 features are added to the 
previous feature set including:

* **2D VALID Convolution With Small 2D Kernels:**  Two-dimensional convolutions with the following kernel dimensions
  [``H``,``W``] are supported: [1,1], [2,2], [3,3], [2,1], [3,1], [4,1], [5,1], [6,1], [7,1], [1,2], or [1,3]. 
  Input tensor dimensions are limited to [1,8,16,16] <= [``N``,``C``,``H``,``W``] <= [1,120,384,240]. Up to 384 ``C`` 
  channels may be used with a subset of kernel sizes (see the table below). Up to 256 kernels (output channels) 
  are supported. Pooling is limited to pool shapes of [1,1], [2,2], or [3,3]. Not all combinations of kernel 
  shape and input tensor shape are supported (see the tables below for exact limitations).

The tables below show that the exact limitation on the input tensor width W depends on the number of input channels 
*C* (indicated as *Ci* below) and the kernel shape.  There is much more freedom to choose the input tensor height and number of output channels.

The following tables provide a more explicit representation of the Intel(R) GNA 3.0 2D convolution operations 
initially supported. The limits depend strongly on number of input tensor channels (*Ci*) and the input tensor width (*W*). 
Other factors are kernel height (*KH*), kernel width (*KW*), pool height (*PH*), pool width (*PW*), horizontal pool step (*SH*), 
and vertical pool step (*PW*). For example, the first table shows that for a 3x3 kernel with max pooling, only square pools are supported, 
and *W* is limited to 87 when there are 64 input channels.


:download:`Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters (Input and Kernel Precision: i16) <../../../docs/OV_Runtime_UG/supported_plugins/files/GNA_Maximum_Input_Tensor_Widths_i16.csv>`

:download:`Table of Maximum Input Tensor Widths (W) vs. Rest of Parameters (Input and Kernel Precision: i8) <../../../docs/OV_Runtime_UG/supported_plugins/files/GNA_Maximum_Input_Tensor_Widths_i8.csv>`


.. note:: 

   The above limitations only apply to the new hardware 2D convolution operation. For GNA 3.0, when possible, the Intel® GNA
   plugin graph compiler flattens 2D convolutions so that the second generation Intel® GNA 1D convolution operations 
   (without these limitations) may be used. The plugin will also flatten 2D convolutions regardless of the sizes if GNA 2.0 
   compilation target is selected (see below).
Support for Convolutions since GNA 3.5
--------------------------------------------------------------------------------------------------------------------------------------

Starting from Intel® GNA 3.5, 1D convolutions are handled in a different way than in GNA 3.0. Convolutions have the following limitations:

============================  =======================  =================
 Limitation                    Convolution 1D           Convolution 2D
============================  =======================  =================
Input height                   1                        1-65535
Input Width                    1-65535                  1-65535
Input channel number           1                        1-1024
Kernel number                  1-8192                   1-8192
Kernel height                  1                        1-255
Kernel width                   1-2048                   1-256
Stride height                  1                        1-255
Stride width                   1-2048                   1-256
Dilation height                1                        1
Dilation width                 1                        1
Pooling window height          1-1                      1-255
Pooling window width           1-255                    1-255
Pooling stride height          1                        1-255
Pooling stride width           1-255                    1-255
============================  =======================  =================


Limitations for GNA 3.5 refers to the specific dimension. The full range of parameters is not always fully supported,
e.g. where Convolution 2D Kernel can have height 255 and width 256, it may not work with Kernel with shape 255x256.

Support for 2D Convolutions using POT
-----------------------------------------------------------

For POT to successfully work with the models including GNA3.0 2D convolutions, the following requirements must be met:

* All convolution parameters are natively supported by HW (see tables above).
* The runtime precision is explicitly set by the ``ov::hint::inference_precision`` property as ``i8`` for the models produced by 
  the ``performance mode`` of POT, and as ``i16`` for the models produced by the ``accuracy mode`` of POT.


Batch Size Limitation
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Intel® GNA plugin supports processing of context-windowed speech frames in batches of 1-8 frames. 
Refer to the :doc:`Layout API overview <openvino_docs_OV_UG_Layout_Overview>` to determine batch dimension.
To set the layout of model inputs in runtime, use the :doc:`Optimize Preprocessing <openvino_docs_OV_UG_Preprocessing_Overview>` guide:


.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gna/set_batch.py
         :language: py
         :fragment: [import]
      
      .. doxygensnippet:: docs/snippets/gna/set_batch.py
         :language: py
         :fragment: [ov_gna_set_nc_layout]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
         :language: cpp
         :fragment: [include]
      
      .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
         :language: cpp
         :fragment: [ov_gna_set_nc_layout]


then set batch size:

.. tab-set::

   .. tab-item:: Python
      :sync: py

      .. doxygensnippet:: docs/snippets/gna/set_batch.py
         :language: py
         :fragment: [ov_gna_set_batch_size]

   .. tab-item:: C++
      :sync: cpp

      .. doxygensnippet:: docs/snippets/gna/set_batch.cpp
         :language: cpp
         :fragment: [ov_gna_set_batch_size]


Increasing batch size only improves efficiency of ``MatMul`` layers.

.. note:: 
   
   For models with ``Convolution``, ``LSTMCell``, ``GRUCell``, or ``ReadValue`` / ``Assign`` operations, the only supported batch size is 1.


Compatibility with Heterogeneous mode
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:doc:`Heterogeneous execution <openvino_docs_OV_UG_Hetero_execution>` is currently not supported by GNA plugin.

See Also
###########################################################

* :doc:`Supported Devices <openvino_docs_OV_UG_supported_plugins_Supported_Devices>`
* :doc:`Converting Model <openvino_docs_MO_DG_prepare_model_convert_model_Converting_Model>`
* :doc:`Convert model from Kaldi <openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_Kaldi>`


@endsphinxdirective


