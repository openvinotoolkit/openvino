Hello NPU
=========

Working with NPU in OpenVINO™
-----------------------------


**Table of contents:**


-  `Introduction <#introduction>`__

   -  `Install required packages <#install-required-packages>`__

-  `Checking NPU with Query Device <#checking-npu-with-query-device>`__

   -  `List the NPU with
      core.available_devices <#list-the-npu-with-core-available_devices>`__
   -  `Check Properties with
      core.get_property <#check-properties-with-core-get_property>`__
   -  `Brief Descriptions of Key
      Properties <#brief-descriptions-of-key-properties>`__

-  `Compiling a Model on NPU <#compiling-a-model-on-npu>`__

   -  `Download a Model <#download-and-convert-a-model>`__
   -  `Compile with Default
      Configuration <#compile-with-default-configuration>`__
   -  `Reduce Compile Time through Model
      Caching <#reduce-compile-time-through-model-caching>`__

      -  `UMD Model Caching <#umd-model-caching>`__
      -  `OpenVINO Model Caching <#openvino-model-caching>`__

   -  `Throughput and Latency Performance
      Hints <#throughput-and-latency-performance-hints>`__

-  `Performance Comparison with
   benchmark_app <#performance-comparison-with-benchmark_app>`__

   -  `NPU vs CPU with Latency Hint <#npu-vs-cpu-with-latency-hint>`__

      -  `Effects of UMD Model
         Caching <#effects-of-umd-model-caching>`__

   -  `NPU vs CPU with Throughput
      Hint <#npu-vs-cpu-with-throughput-hint>`__

-  `Limitations <#limitations>`__
-  `Conclusion <#conclusion>`__


This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

This tutorial provides a high-level overview of working with the NPU
device **Intel(R) AI Boost** (introduced with the Intel® Core™ Ultra
generation of CPUs) in OpenVINO. It explains some of the key properties
of the NPU and shows how to compile a model on NPU with performance
hints.

This tutorial also shows example commands for benchmark_app that can be
run to compare NPU performance with CPU in different configurations.

Introduction
------------



The Neural Processing Unit (NPU) is a low power hardware solution which
enables you to offload certain neural network computation tasks from
other devices, for more streamlined resource management.

Note that the NPU plugin is included in PIP installation of OpenVINO™
and you need to `install a proper NPU
driver <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-npu.html>`__
to use it successfully.

| **Supported Platforms**:
| Host: Intel® Core™ Ultra
| NPU device: NPU 3720
| OS: Ubuntu 22.04 (with Linux Kernel 6.6+), MS Windows 11 (both 64-bit)

To learn more about the NPU Device, see the
`page <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html>`__.

Install required packages
~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "openvino>=2024.1.0" huggingface_hub

Checking NPU with Query Device
------------------------------



In this section, we will see how to list the available NPU and check its
properties. Some of the key properties will be defined.

List the NPU with core.available_devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO Runtime provides the ``available_devices`` method for checking
which devices are available for inference. The following code will
output a list a compatible OpenVINO devices, in which Intel NPU should
appear (ensure that the driver is installed successfully).

.. code:: ipython3

    import openvino as ov

    core = ov.Core()
    core.available_devices




.. parsed-literal::

    ['CPU', 'GPU', 'NPU']



Check Properties with core.get_property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To get information about the NPU, we can use device properties. In
OpenVINO, devices have properties that describe their characteristics
and configurations. Each property has a name and associated value that
can be queried with the ``get_property`` method.

To get the value of a property, such as the device name, we can use the
``get_property`` method as follows:

.. code:: ipython3

    import openvino.properties as props


    device = "NPU"

    core.get_property(device, props.device.full_name)




.. parsed-literal::

    'Intel(R) AI Boost'



Each device also has a specific property called
``SUPPORTED_PROPERTIES``, that enables viewing all the available
properties in the device. We can check the value for each property by
simply looping through the dictionary returned by
``core.get_property("NPU", props.supported_properties)`` and then
querying for that property.

.. code:: ipython3

    print(f"{device} SUPPORTED_PROPERTIES:\n")
    supported_properties = core.get_property(device, props.supported_properties)
    indent = len(max(supported_properties, key=len))

    for property_key in supported_properties:
        if property_key not in ("SUPPORTED_METRICS", "SUPPORTED_CONFIG_KEYS", "SUPPORTED_PROPERTIES"):
            try:
                property_val = core.get_property(device, property_key)
            except TypeError:
                property_val = "UNSUPPORTED TYPE"
            print(f"{property_key:<{indent}}: {property_val}")

Brief Descriptions of Key Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Each device has several properties as seen in the last command. Some of
the key properties are: - ``FULL_DEVICE_NAME`` - The product name of the
NPU. - ``PERFORMANCE_HINT`` - A high-level way to tune the device for a
specific performance metric, such as latency or throughput, without
worrying about device-specific settings. - ``CACHE_DIR`` - The directory
where the OpenVINO model cache data is stored to speed up the
compilation time. - ``OPTIMIZATION_CAPABILITIES`` - The model data types
(INT8, FP16, FP32, etc) that are supported by this NPU.

To learn more about devices and properties, see the `Query Device
Properties <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
page.

Compiling a Model on NPU
------------------------



Now, we know the NPU present in the system and we have checked its
properties. We can easily use it for compiling and running models with
OpenVINO NPU plugin.

Download a Model
----------------



This tutorial uses the ``resnet50`` model. The ``resnet50`` model is
used for image classification tasks. The model was trained on
`ImageNet <https://www.image-net.org/index.php>`__ dataset which
contains over a million images categorized into 1000 classes. To read
more about resnet50, see the
`paper <https://ieeexplore.ieee.org/document/7780459>`__. As our
tutorial focused on inference part, we skip model conversion step. To
convert this Pytorch model to OpenVINO IR, `Model Conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__
should be used. Please check this
`tutorial <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/pytorch-to-openvino/pytorch-to-openvino.ipynb>`__
for details how to convert pytorch model.

.. code:: ipython3

    from pathlib import Path

    # create a directory for resnet model file
    MODEL_DIRECTORY_PATH = Path("model")
    MODEL_DIRECTORY_PATH.mkdir(exist_ok=True)

    model_name = "resnet50"

.. code:: ipython3

    import huggingface_hub as hf_hub

.. code:: ipython3

    precision = "FP16"

    model_path = MODEL_DIRECTORY_PATH / "ir_model" / f"{model_name}_{precision.lower()}.xml"

    model = None
    if not model_path.exists():
        hf_hub.snapshot_download("katuni4ka/resnet50_fp16", local_dir=model_path.parent)
        print("IR model saved to {}".format(model_path))
        model = core.read_model(model_path)
    else:
        print("Read IR model from {}".format(model_path))
        model = core.read_model(model_path)


.. parsed-literal::

    Read IR model from model\ir_model\resnet50_fp16.xml


**Note:** NPU also supports ``INT8`` quantized models.

Compile with Default Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



When the model is ready, first we need to read it, using the
``read_model`` method. Then, we can use the ``compile_model`` method and
specify the name of the device we want to compile the model on, in this
case, “NPU”.

.. code:: ipython3

    compiled_model = core.compile_model(model, device)

Reduce Compile Time through Model Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Depending on the model used, device-specific optimizations and network
compilations can cause the compile step to be time-consuming, especially
with larger models, which may lead to bad user experience in the
application. To solve this **Model Caching** can be used.

Model Caching helps reduce application startup delays by exporting and
reusing the compiled model automatically. The following two
compilation-related metrics are crucial in this area:

-  **First-Ever Inference Latency (FEIL)**:
   Measures all steps required to compile and execute a model on the
   device for the first time. It includes model compilation time, the
   time required to load and initialize the model on the device and the
   first inference execution.
-  **First Inference Latency (FIL)**:
   Measures the time required to load and initialize the pre-compiled
   model on the device and the first inference execution.

In NPU, UMD model caching is a solution enabled by default by the
driver. It improves time to first inference (FIL) by storing the model
in the cache after compilation (included in FEIL). Learn more about UMD
Caching
`here <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html#umd-dynamic-model-caching>`__.
Due to this caching, it takes lesser time to load the model after first
compilation.

| You can also use OpenVINO Model Caching, which is a common mechanism
  for all OpenVINO device plugins and can be enabled by setting the
  ``cache_dir`` property.
| By enabling OpenVINO Model Caching, the UMD caching is automatically
  bypassed by the NPU plugin, which means the model will only be stored
  in the OpenVINO cache after compilation. When a cache hit occurs for
  subsequent compilation requests, the plugin will import the model
  instead of recompiling it.

UMD Model Caching
^^^^^^^^^^^^^^^^^



To see how UMD caching see the following example:

.. code:: ipython3

    import time
    from pathlib import Path

    start = time.time()
    core = ov.Core()

    # Compile the model as before
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"UMD Caching (first time) - compile time: {time.time() - start}s")


.. parsed-literal::

    UMD Caching (first time) - compile time: 3.2854952812194824s


.. code:: ipython3

    start = time.time()
    core = ov.Core()

    # Compile the model once again to see UMD Caching
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"UMD Caching - compile time: {time.time() - start}s")


.. parsed-literal::

    UMD Caching - compile time: 2.269814968109131s


OpenVINO Model Caching
^^^^^^^^^^^^^^^^^^^^^^



To get an idea of OpenVINO model caching, we can use the OpenVINO cache
as follow

.. code:: ipython3

    # Create cache folder
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)

    start = time.time()
    core = ov.Core()

    # Set cache folder
    core.set_property({props.cache_dir(): cache_folder})

    # Compile the model
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache enabled (first time) - compile time: {time.time() - start}s")

    start = time.time()
    core = ov.Core()

    # Set cache folder
    core.set_property({props.cache_dir(): cache_folder})

    # Compile the model as before
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache enabled (second time) - compile time: {time.time() - start}s")


.. parsed-literal::

    Cache enabled (first time) - compile time: 0.6362860202789307s
    Cache enabled (second time) - compile time: 0.3032548427581787s


And when the OpenVINO cache is disabled:

.. code:: ipython3

    start = time.time()
    core = ov.Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache disabled - compile time: {time.time() - start}s")


.. parsed-literal::

    Cache disabled - compile time: 3.0127954483032227s


The actual time improvements will depend on the environment as well as
the model being used but it is definitely something to consider when
optimizing an application. To read more about this, see the `Model
Caching
docs <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__.

Throughput and Latency Performance Hints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To simplify device and pipeline configuration, OpenVINO provides
high-level performance hints that automatically set the batch size and
number of parallel threads for inference. The “LATENCY” performance hint
optimizes for fast inference times while the “THROUGHPUT” performance
hint optimizes for high overall bandwidth or FPS.

To use the “LATENCY” performance hint, add
``{hints.performance_mode(): hints.PerformanceMode.LATENCY}`` when
compiling the model as shown below. For NPU, this automatically
minimizes the batch size and number of parallel streams such that all of
the compute resources can focus on completing a single inference as fast
as possible.

.. code:: ipython3

    import openvino.properties.hint as hints


    compiled_model = core.compile_model(model, device, {hints.performance_mode(): hints.PerformanceMode.LATENCY})

To use the “THROUGHPUT” performance hint, add
``{hints.performance_mode(): hints.PerformanceMode.THROUGHPUT}`` when
compiling the model. For NPUs, this creates multiple processing streams
to efficiently utilize all the execution cores and optimizes the batch
size to fill the available memory.

.. code:: ipython3

    compiled_model = core.compile_model(model, device, {hints.performance_mode(): hints.PerformanceMode.THROUGHPUT})

Performance Comparison with benchmark_app
-----------------------------------------



Given all the different options available when compiling a model, it may
be difficult to know which settings work best for a certain application.
Thankfully, OpenVINO provides ``benchmark_app`` - a performance
benchmarking tool.

The basic syntax of ``benchmark_app`` is as follows:

``benchmark_app -m PATH_TO_MODEL -d TARGET_DEVICE -hint {throughput,cumulative_throughput,latency,none}``

where ``TARGET_DEVICE`` is any device shown by the ``available_devices``
method as well as the MULTI and AUTO devices we saw previously, and the
value of hint should be one of the values between brackets.

Note that benchmark_app only requires the model path to run but both
device and hint arguments will be useful to us. For more advanced
usages, the tool itself has other options that can be checked by running
``benchmark_app -h`` or reading the
`docs <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
The following example shows us to benchmark a simple model, using a NPU
with latency focus:

``benchmark_app -m {model_path} -d NPU -hint latency``

| For completeness, let us list here some of the comparisons we may want
  to do by varying the device and hint used. Note that the actual
  performance may depend on the hardware used. Generally, we should
  expect NPU to be better than CPU.
| Please refer to the ``benchmark_app`` log entries under
  ``[Step 11/11] Dumping statistics report`` to observe the differences
  in latency and throughput between the CPU and NPU..

NPU vs CPU with Latency Hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    !benchmark_app -m {model_path} -d CPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 14.00 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 143.22 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model2
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   AFFINITY: Affinity.HYBRID_AWARE
    [ INFO ]   INFERENCE_NUM_THREADS: 12
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: LATENCY
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: False
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]   ENABLE_HYPER_THREADING: False
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   LOG_LEVEL: Level.NO
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]   KV_CACHE_PRECISION: <Type: 'float16'>
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 28.95 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            1612 iterations
    [ INFO ] Duration:         60039.72 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        39.99 ms
    [ INFO ]    Average:       37.13 ms
    [ INFO ]    Min:           19.13 ms
    [ INFO ]    Max:           71.94 ms
    [ INFO ] Throughput:   26.85 FPS


.. code:: ipython3

    !benchmark_app -m {model_path} -d NPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] NPU
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 11.51 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2302.40 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   DEVICE_ID:
    [ INFO ]   ENABLE_CPU_PINNING: False
    [ INFO ]   EXECUTION_DEVICES: NPU.3720
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float16'>
    [ INFO ]   INTERNAL_SUPPORTED_PROPERTIES: {'CACHING_PROPERTIES': 'RO'}
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   NETWORK_NAME:
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 1
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.94 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:NPU.3720
    [ INFO ] Count:            17908 iterations
    [ INFO ] Duration:         60004.49 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.29 ms
    [ INFO ]    Average:       3.33 ms
    [ INFO ]    Min:           3.21 ms
    [ INFO ]    Max:           6.90 ms
    [ INFO ] Throughput:   298.44 FPS


Effects of UMD Model Caching
''''''''''''''''''''''''''''



To see the effects of UMD Model caching, we are going to run the
benchmark_app and see the difference in model read time and compilation
time:

.. code:: ipython3

    !benchmark_app -m {model_path} -d NPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] NPU
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 11.00 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2157.58 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   DEVICE_ID:
    [ INFO ]   ENABLE_CPU_PINNING: False
    [ INFO ]   EXECUTION_DEVICES: NPU.3720
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float16'>
    [ INFO ]   INTERNAL_SUPPORTED_PROPERTIES: {'CACHING_PROPERTIES': 'RO'}
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   NETWORK_NAME:
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 1
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.94 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:NPU.3720
    [ INFO ] Count:            17894 iterations
    [ INFO ] Duration:         60004.76 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.29 ms
    [ INFO ]    Average:       3.33 ms
    [ INFO ]    Min:           3.21 ms
    [ INFO ]    Max:           14.38 ms
    [ INFO ] Throughput:   298.21 FPS


As you can see from the log entries ``[Step 4/11] Reading model files``
and ``[Step 7/11] Loading the model to the device``, it takes less time
to read and compile the model after the initial load.

NPU vs CPU with Throughput Hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    !benchmark_app -m {model_path} -d CPU -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 12.00 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 177.18 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model2
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
    [ INFO ]   NUM_STREAMS: 4
    [ INFO ]   AFFINITY: Affinity.HYBRID_AWARE
    [ INFO ]   INFERENCE_NUM_THREADS: 16
    [ INFO ]   PERF_COUNT: NO
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: False
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   LOG_LEVEL: Level.NO
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]   DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]   KV_CACHE_PRECISION: <Type: 'float16'>
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 31.62 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            3212 iterations
    [ INFO ] Duration:         60082.26 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        65.28 ms
    [ INFO ]    Average:       74.60 ms
    [ INFO ]    Min:           35.65 ms
    [ INFO ]    Max:           157.31 ms
    [ INFO ] Throughput:   53.46 FPS


.. code:: ipython3

    !benchmark_app -m {model_path} -d NPU -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] NPU
    [ INFO ] Build ................................. 2024.1.0-14992-621b025bef4
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 11.50 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,224,224]
    [ INFO ] Model outputs:
    [ INFO ]     x.45 (node: aten::linear/Add) : f32 / [...] / [1,1000]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2265.07 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   DEVICE_ID:
    [ INFO ]   ENABLE_CPU_PINNING: False
    [ INFO ]   EXECUTION_DEVICES: NPU.3720
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float16'>
    [ INFO ]   INTERNAL_SUPPORTED_PROPERTIES: {'CACHING_PROPERTIES': 'RO'}
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   NETWORK_NAME:
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 1
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 7.95 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:NPU.3720
    [ INFO ] Count:            19080 iterations
    [ INFO ] Duration:         60024.79 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        12.51 ms
    [ INFO ]    Average:       12.56 ms
    [ INFO ]    Min:           6.92 ms
    [ INFO ]    Max:           25.80 ms
    [ INFO ] Throughput:   317.87 FPS


Limitations
-----------



1. Currently, only the models with static shapes are supported on NPU.
2. If the path to the model file includes non-Unicode symbols, such as
   in Chinese, the model cannot be used for inference on NPU. It will
   return an error.

Conclusion
----------



This tutorial demonstrates how easy it is to use NPU in OpenVINO, check
its properties, and even tailor the model performance through the
different performance hints.

Discover the power of Neural Processing Unit (NPU) with OpenVINO through
these interactive Jupyter notebooks:

- `hello-world <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-world>`__:
  Start your OpenVINO journey by performing inference on an OpenVINO IR
  model.
- `hello-segmentation <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/hello-segmentation>`__:
  Dive into inference with a segmentation model and explore image
  segmentation capabilities.

Model Optimization and Conversion
'''''''''''''''''''''''''''''''''

-  `tflite-to-openvino <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tflite-to-openvino>`__:
   Learn the process of converting TensorFlow Lite models to OpenVINO IR
   format.
-  `yolov7-optimization <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/226-yolov7-optimization>`__:
   Optimize the YOLOv7 model for enhanced performance in OpenVINO.
-  `yolov8-optimization <https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/yolov8-optimization>`__:
   Convert and optimize YOLOv8 models for efficient deployment with
   OpenVINO.

Advanced Computer Vision Techniques
'''''''''''''''''''''''''''''''''''

-  `vision-background-removal <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-background-removal>`__:
   Implement advanced image segmentation and background manipulation
   with U^2-Net.
-  `handwritten-ocr <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/handwritten-ocr>`__:
   Apply optical character recognition to handwritten Chinese and
   Japanese text.
-  `vehicle-detection-and-recognition <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vehicle-detection-and-recognition>`__:
   Use pre-trained models for vehicle detection and recognition in
   images.
-  `vision-image-colorization <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/vision-image-colorization>`__:
   Bring black and white images to life by adding color with neural
   networks.

Real-Time Webcam Applications
'''''''''''''''''''''''''''''

-  `tflite-selfie-segmentation <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/tflite-selfie-segmentation>`__:
   Apply TensorFlow Lite models for selfie segmentation and background
   processing.
-  `object-detection-webcam <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/object-detection-webcam>`__:
   Experience real-time object detection using your webcam and OpenVINO.
-  `pose-estimation-webcam <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pose-estimation-webcam>`__:
   Perform human pose estimation in real-time with webcam integration.
-  `action-recognition-webcam <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/action-recognition-webcam>`__:
   Recognize and classify human actions live with your webcam.
-  `style-transfer-webcam <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/style-transfer-webcam>`__:
   Transform your webcam feed with artistic styles in real-time using
   pre-trained models.
-  `3D-pose-estimation-webcam <https://github.com/openvinotoolkit/openvino_notebooks/tree/latest/notebooks/pose-estimation-webcam>`__:
   Perform 3D multi-person pose estimation with OpenVINO.
