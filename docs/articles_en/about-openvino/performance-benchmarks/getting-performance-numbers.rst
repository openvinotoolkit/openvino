.. {#openvino_docs_MO_DG_Getting_Performance_Numbers}

Getting Performance Numbers
===========================



This guide explains how to use the benchmark_app to get performance numbers. It also explains how the performance
numbers are reflected through internal inference performance counters and execution graphs. It also includes
information on using ITT and Intel® VTune™ Profiler to get performance insights.


.. raw:: html

   <h2>Test performance with the benchmark_app</h2>



You can run OpenVINO benchmarks in both C++ and Python APIs, yet the experience differs in each case.
The Python one is part of OpenVINO Runtime installation, while C++ is available as a code sample.
For a detailed description, see: :doc:`benchmark_app <../../learn-openvino/openvino-samples/benchmark-tool>`.

Make sure to install the latest release package with support for frameworks of the models you want to test.
For the most reliable performance benchmarks, :doc:`prepare the model for use with OpenVINO <../../openvino-workflow/model-preparation>`.



.. raw:: html

   <h3>Running the benchmark application</h3>


The benchmark_app includes a lot of device-specific options, but the primary usage is as simple as:

.. code-block:: sh

   benchmark_app -m <model> -d <device> -i <input>


Each of the :doc:`OpenVINO supported devices <../compatibility-and-support/supported-devices>` offers
performance settings that contain command-line equivalents in the Benchmark app.

While these settings provide really low-level control for the optimal model performance on the *specific* device,
it is recommended to always start performance evaluation with the :doc:`OpenVINO High-Level Performance Hints <../../openvino-workflow/running-inference/optimize-inference/high-level-performance-hints>` first, like so:

.. code-block:: sh

   # for throughput prioritization
   benchmark_app -hint tput -m <model> -d <device>
   # for latency prioritization
   benchmark_app -hint latency -m <model> -d <device>



.. raw:: html

   <h2>Additional benchmarking considerations</h2>



.. raw:: html

   <h3>1 - Select a Proper Set of Operations to Measure</h3>


When evaluating performance of a model with OpenVINO Runtime, it is required to measure a proper set of operations.

- Avoid including one-time costs such as model loading.
- Track operations that occur outside OpenVINO Runtime (such as video decoding) separately.


.. note::

   Some image pre-processing can be baked into OpenVINO IR and accelerated accordingly. For more information,
   refer to :doc:`Embedding Pre-processing <../../documentation/legacy-features/transition-legacy-conversion-api/legacy-conversion-api/[legacy]-embedding-preprocessing-computation>` and
   :doc:`General Runtime Optimizations <../../openvino-workflow/running-inference/optimize-inference/general-optimizations>`.



.. raw:: html

   <h3>2 - Try to Get Credible Data</h3>

Performance conclusions should be build upon reproducible data. As for the performance measurements, they should
be done with a large number of invocations of the same routine. Since the first iteration is almost always significantly
slower than the subsequent ones, an aggregated value can be used for the execution time for final projections:

- If the warm-up run does not help or execution time still varies, you can try running a large number of iterations
  and then average or find a mean of the results.
- If the time values range too much, consider geomean.
- Be aware of the throttling and other power oddities. A device can exist in one of several different power states.
  When optimizing your model, consider fixing the device frequency for better performance data reproducibility.
  However, the end-to-end (application) benchmarking should also be performed under real operational conditions.



.. raw:: html

   <h3>3 - Compare Performance with Native/Framework Code</h3>

When comparing the OpenVINO Runtime performance with the framework or another reference code, make sure that both versions are as similar as possible:

-	Wrap the exact inference execution (for examples, see :doc:`Benchmark app <../../learn-openvino/openvino-samples/benchmark-tool>`).
-	Do not include model loading time.
-	Ensure that the inputs are identical for OpenVINO Runtime and the framework. For example, watch out for random values that can be used to populate the inputs.
-	In situations when any user-side pre-processing should be tracked separately, consider :doc:`image pre-processing and conversion <../../openvino-workflow/running-inference/optimize-inference/optimize-preprocessing>`.
-  When applicable, leverage the :doc:`Dynamic Shapes support <../../openvino-workflow/running-inference/dynamic-shapes>`.
-	If possible, demand the same accuracy. For example, TensorFlow allows ``FP16`` execution, so when comparing to that, make sure to test the OpenVINO Runtime with the ``FP16`` as well.


.. raw:: html

   <h3>Internal Inference Performance Counters and Execution Graphs</h3>

More detailed insights into inference performance breakdown can be achieved with device-specific performance counters and/or execution graphs.
Both :doc:`C++ and Python <../../learn-openvino/openvino-samples/benchmark-tool>`
versions of the *benchmark_app* support a ``-pc`` command-line parameter that outputs internal execution breakdown.

For example, the table shown below is part of performance counters for quantized
`TensorFlow implementation of ResNet-50 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/resnet-50-tf>`__
model inference on :doc:`CPU Plugin <../../openvino-workflow/running-inference/inference-devices-and-modes/cpu-device>`.
Keep in mind that since the device is CPU, the ``realTime`` wall clock and the ``cpu`` time layers are the same.
Information about layer precision is also stored in the performance counters.


===========================================================  =============  ==============  =====================  =================  ==============
 layerName                                                    execStatus     layerType       execType               realTime (ms)      cpuTime (ms)
===========================================================  =============  ==============  =====================  =================  ==============
 resnet\_model/batch\_normalization\_15/FusedBatchNorm/Add    EXECUTED       Convolution     jit\_avx512\_1x1\_I8   0.377              0.377
 resnet\_model/conv2d\_16/Conv2D/fq\_input\_0                 NOT\_RUN       FakeQuantize    undef                  0                  0
 resnet\_model/batch\_normalization\_16/FusedBatchNorm/Add    EXECUTED       Convolution     jit\_avx512\_I8        0.499              0.499
 resnet\_model/conv2d\_17/Conv2D/fq\_input\_0                 NOT\_RUN       FakeQuantize    undef                  0                  0
 resnet\_model/batch\_normalization\_17/FusedBatchNorm/Add    EXECUTED       Convolution     jit\_avx512\_1x1\_I8   0.399              0.399
 resnet\_model/add\_4/fq\_input\_0                            NOT\_RUN       FakeQuantize    undef                  0                  0
 resnet\_model/add\_4                                         NOT\_RUN       Eltwise         undef                  0                  0
 resnet\_model/add\_5/fq\_input\_1                            NOT\_RUN       FakeQuantize    undef                  0                  0
===========================================================  =============  ==============  =====================  =================  ==============

|   The ``execStatus`` column of the table includes the following possible values:
|     - ``EXECUTED`` - the layer was executed by standalone primitive.
|     - ``NOT_RUN`` - the layer was not executed by standalone primitive or was fused with another operation and executed in another layer primitive.
|
|   The ``execType`` column of the table includes inference primitives with specific suffixes. The layers could have the following marks:
|     - The ``I8`` suffix is for layers that had 8-bit data type input and were computed in 8-bit precision.
|     - The ``FP32`` suffix is for layers computed in 32-bit precision.
|
|   All ``Convolution`` layers are executed in ``int8`` precision. The rest of the layers are fused into Convolutions using post-operation optimization,
    as described in :doc:`CPU Device <../../openvino-workflow/running-inference/inference-devices-and-modes/cpu-device>`. This contains layer names
    (as seen in OpenVINO IR), type of the layer, and execution statistics.


Both *benchmark_app* versions also support the ``exec_graph_path`` command-line option. It requires OpenVINO to output the same execution
statistics per layer, but in the form of plugin-specific `Netron-viewable <https://netron.app/>`__ graph to the specified file.

Especially when performance-debugging the :doc:`latency <../../openvino-workflow/running-inference/optimize-inference/optimizing-latency>`, note that the counters
do not reflect the time spent in the ``plugin/device/driver/etc`` queues. If the sum of the counters is too different from the latency
of an inference request, consider testing with less inference requests. For example, running single
:doc:`OpenVINO stream <../../openvino-workflow/running-inference/optimize-inference/optimizing-throughput>` with multiple requests would produce nearly identical
counters as running a single inference request, while the actual latency can be quite different.

Lastly, the performance statistics with both performance counters and execution graphs are averaged,
so such data for the :doc:`inputs of dynamic shapes <../../openvino-workflow/running-inference/dynamic-shapes>` should be measured carefully,
preferably by isolating the specific shape and executing multiple times in a loop, to gather reliable data.


.. raw:: html

   <h3>Use ITT to Get Performance Insights</h3>

In general, OpenVINO and its individual plugins are heavily instrumented with Intel® Instrumentation and Tracing Technology (ITT).
Therefore, you can also compile OpenVINO from the source code with ITT enabled and use tools like
`Intel® VTune™ Profiler <https://software.intel.com/en-us/vtune>`__ to get detailed inference performance breakdown and additional
insights in the application-level performance on the timeline view.




.. raw:: html

   <link rel="stylesheet" type="text/css" href="../../_static/css/benchmark-banner.css">

.. container:: benchmark-banner

   Results may vary. For more information, see
   :doc:`F.A.Q. <./performance-benchmarks-faq>` and
   :doc:`Platforms, Configurations, Methodology <../performance-benchmarks>`.
   See :doc:`Legal Information <../additional-resources/terms-of-use>`.