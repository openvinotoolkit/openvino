Working with GPUs in OpenVINO™
==============================


**Table of contents:**


-  `Introduction <#introduction>`__

   -  `Install required packages <#install-required-packages>`__

-  `Checking GPUs with Query
   Device <#checking-gpus-with-query-device>`__

   -  `List GPUs with
      core.available_devices <#list-gpus-with-core-available_devices>`__
   -  `Check Properties with
      core.get_property <#check-properties-with-core-get_property>`__
   -  `Brief Descriptions of Key
      Properties <#brief-descriptions-of-key-properties>`__

-  `Compiling a Model on GPU <#compiling-a-model-on-gpu>`__

   -  `Download and Convert a Model <#download-and-convert-a-model>`__

      -  `Download and unpack the
         Model <#download-and-unpack-the-model>`__
      -  `Convert the Model to OpenVINO IR
         format <#convert-the-model-to-openvino-ir-format>`__

   -  `Compile with Default
      Configuration <#compile-with-default-configuration>`__
   -  `Reduce Compile Time through Model
      Caching <#reduce-compile-time-through-model-caching>`__
   -  `Throughput and Latency Performance
      Hints <#throughput-and-latency-performance-hints>`__
   -  `Using Multiple GPUs with Multi-Device and Cumulative
      Throughput <#using-multiple-gpus-with-multi-device-and-cumulative-throughput>`__

-  `Performance Comparison with
   benchmark_app <#performance-comparison-with-benchmark_app>`__

   -  `CPU vs GPU with Latency Hint <#cpu-vs-gpu-with-latency-hint>`__
   -  `CPU vs GPU with Throughput
      Hint <#cpu-vs-gpu-with-throughput-hint>`__
   -  `Single GPU vs Multiple GPUs <#single-gpu-vs-multiple-gpus>`__

-  `Basic Application Using GPUs <#basic-application-using-gpus>`__

   -  `Import Necessary Packages <#import-necessary-packages>`__
   -  `Compile the Model <#compile-the-model>`__
   -  `Load and Preprocess Video
      Frames <#load-and-preprocess-video-frames>`__
   -  `Define Model Output Classes <#define-model-output-classes>`__
   -  `Set up Asynchronous Pipeline <#set-up-asynchronous-pipeline>`__

      -  `Callback Definition <#callback-definition>`__
      -  `Create Async Pipeline <#create-async-pipeline>`__

   -  `Perform Inference <#perform-inference>`__
   -  `Process Results <#process-results>`__

-  `Conclusion <#conclusion>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

This tutorial provides a high-level overview of working with Intel GPUs
in OpenVINO. It shows how to use Query Device to list system GPUs and
check their properties, and it explains some of the key properties. It
shows how to compile a model on GPU with performance hints and how to
use multiple GPUs using MULTI or CUMULATIVE_THROUGHPUT.

The tutorial also shows example commands for benchmark_app that can be
run to compare GPU performance in different configurations. It also
provides the code for a basic end-to-end application that compiles a
model on GPU and uses it to run inference.

Introduction
------------



Originally, graphic processing units (GPUs) began as specialized chips,
developed to accelerate the rendering of computer graphics. In contrast
to CPUs, which have few but powerful cores, GPUs have many more
specialized cores, making them ideal for workloads that can be
parallelized into simpler tasks. Nowadays, one such workload is deep
learning, where GPUs can easily accelerate inference of neural networks
by splitting operations across multiple cores.

OpenVINO supports inference on Intel integrated GPUs (which are included
with most `Intel® Core™ desktop and mobile
processors <https://www.intel.com/content/www/us/en/products/details/processors/core.html>`__)
or on Intel discrete GPU products like the `Intel® Arc™ A-Series
Graphics
cards <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/arc.html>`__
and `Intel® Data Center GPU Flex
Series <https://www.intel.com/content/www/us/en/products/details/discrete-gpus/data-center-gpu/flex-series.html>`__.
To get started, first `install
OpenVINO <https://docs.openvino.ai/2024/get-started/install-openvino.html>`__
on a system equipped with one or more Intel GPUs. Follow the `GPU
configuration
instructions <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
to configure OpenVINO to work with your GPU. Then, read on to learn how
to accelerate inference with GPUs in OpenVINO!

Install required packages
~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "openvino-dev>=2024.0.0" "opencv-python" "tqdm"
    %pip install -q "tensorflow-macos>=2.5; sys_platform == 'darwin' and platform_machine == 'arm64' and python_version > '3.8'" # macOS M1 and M2
    %pip install -q "tensorflow>=2.5; sys_platform == 'darwin' and platform_machine != 'arm64' and python_version > '3.8'" # macOS x86
    %pip install -q "tensorflow>=2.5; sys_platform != 'darwin' and python_version > '3.8'"

Checking GPUs with Query Device
-------------------------------



In this section, we will see how to list the available GPUs and check
their properties. Some of the key properties will also be defined.

List GPUs with core.available_devices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO Runtime provides the ``available_devices`` method for checking
which devices are available for inference. The following code will
output a list of compatible OpenVINO devices, in which Intel GPUs should
appear.

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    core.available_devices




.. parsed-literal::

    ['CPU', 'GPU']



Note that GPU devices are numbered starting at 0, where the integrated
GPU always takes the id ``0`` if the system has one. For instance, if
the system has a CPU, an integrated and discrete GPU, we should expect
to see a list like this: ``['CPU', 'GPU.0', 'GPU.1']``. To simplify its
use, the “GPU.0” can also be addressed with just “GPU”. For more
details, see the `Device Naming
Convention <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html#device-naming-convention>`__
section.

If the GPUs are installed correctly on the system and still do not
appear in the list, follow the steps described
`here <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
to configure your GPU drivers to work with OpenVINO. Once we have the
GPUs working with OpenVINO, we can proceed with the next sections.

Check Properties with core.get_property
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To get information about the GPUs, we can use device properties. In
OpenVINO, devices have properties that describe their characteristics
and configuration. Each property has a name and associated value that
can be queried with the ``get_property`` method.

To get the value of a property, such as the device name, we can use the
``get_property`` method as follows:

.. code:: ipython3

    import openvino.properties as props
    
    
    device = "GPU"
    
    core.get_property(device, props.device.full_name)




.. parsed-literal::

    'Intel(R) Graphics [0x46a6] (iGPU)'



Each device also has a specific property called
``SUPPORTED_PROPERTIES``, that enables viewing all the available
properties in the device. We can check the value for each property by
simply looping through the dictionary returned by
``core.get_property("GPU", props.supported_properties)`` and then
querying for that property.

.. code:: ipython3

    print(f"{device} SUPPORTED_PROPERTIES:\n")
    supported_properties = core.get_property(device, props.supported_properties)
    indent = len(max(supported_properties, key=len))
    
    for property_key in supported_properties:
        if property_key not in (
            "SUPPORTED_METRICS",
            "SUPPORTED_CONFIG_KEYS",
            "SUPPORTED_PROPERTIES",
        ):
            try:
                property_val = core.get_property(device, property_key)
            except TypeError:
                property_val = "UNSUPPORTED TYPE"
            print(f"{property_key:<{indent}}: {property_val}")


.. parsed-literal::

    GPU SUPPORTED_PROPERTIES:
    
    AVAILABLE_DEVICES             : ['0']
    RANGE_FOR_ASYNC_INFER_REQUESTS: (1, 2, 1)
    RANGE_FOR_STREAMS             : (1, 2)
    OPTIMAL_BATCH_SIZE            : 1
    MAX_BATCH_SIZE                : 1
    CACHING_PROPERTIES            : {'GPU_UARCH_VERSION': 'RO', 'GPU_EXECUTION_UNITS_COUNT': 'RO', 'GPU_DRIVER_VERSION': 'RO', 'GPU_DEVICE_ID': 'RO'}
    DEVICE_ARCHITECTURE           : GPU: v12.0.0
    FULL_DEVICE_NAME              : Intel(R) Graphics [0x46a6] (iGPU)
    DEVICE_UUID                   : UNSUPPORTED TYPE
    DEVICE_TYPE                   : Type.INTEGRATED
    DEVICE_GOPS                   : UNSUPPORTED TYPE
    OPTIMIZATION_CAPABILITIES     : ['FP32', 'BIN', 'FP16', 'INT8']
    GPU_DEVICE_TOTAL_MEM_SIZE     : UNSUPPORTED TYPE
    GPU_UARCH_VERSION             : 12.0.0
    GPU_EXECUTION_UNITS_COUNT     : 96
    GPU_MEMORY_STATISTICS         : UNSUPPORTED TYPE
    PERF_COUNT                    : False
    MODEL_PRIORITY                : Priority.MEDIUM
    GPU_HOST_TASK_PRIORITY        : Priority.MEDIUM
    GPU_QUEUE_PRIORITY            : Priority.MEDIUM
    GPU_QUEUE_THROTTLE            : Priority.MEDIUM
    GPU_ENABLE_LOOP_UNROLLING     : True
    CACHE_DIR                     : 
    PERFORMANCE_HINT              : PerformanceMode.UNDEFINED
    COMPILATION_NUM_THREADS       : 20
    NUM_STREAMS                   : 1
    PERFORMANCE_HINT_NUM_REQUESTS : 0
    INFERENCE_PRECISION_HINT      : <Type: 'undefined'>
    DEVICE_ID                     : 0


Brief Descriptions of Key Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Each device has several properties as seen in the last command. Some of
the key properties are:

-  ``FULL_DEVICE_NAME`` - The product name of the GPU and whether it is
   an integrated or discrete GPU (iGPU or dGPU).
-  ``OPTIMIZATION_CAPABILITIES`` - The model data types (INT8, FP16,
   FP32, etc) that are supported by this GPU.
-  ``GPU_EXECUTION_UNITS_COUNT`` - The execution cores available in the
   GPU’s architecture, which is a relative measure of the GPU’s
   processing power.
-  ``RANGE_FOR_STREAMS`` - The number of processing streams available on
   the GPU that can be used to execute parallel inference requests. When
   compiling a model in LATENCY or THROUGHPUT mode, OpenVINO will
   automatically select the best number of streams for low latency or
   high throughput.
-  ``PERFORMANCE_HINT`` - A high-level way to tune the device for a
   specific performance metric, such as latency or throughput, without
   worrying about device-specific settings.
-  ``CACHE_DIR`` - The directory where the model cache data is stored to
   speed up compilation time.

To learn more about devices and properties, see the `Query Device
Properties <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
page.

Compiling a Model on GPU
------------------------



Now, we know how to list the GPUs in the system and check their
properties. We can easily use one for compiling and running models with
OpenVINO `GPU
plugin <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>`__.

Download and Convert a Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



This tutorial uses the ``ssdlite_mobilenet_v2`` model. The
``ssdlite_mobilenet_v2`` model is used for object detection. The model
was trained on `Common Objects in Context
(COCO) <https://cocodataset.org/#home>`__ dataset version with 91
categories of object. For details, see the
`paper <https://arxiv.org/abs/1801.04381>`__.

Download and unpack the Model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Use the ``download_file`` function from the ``notebook_utils`` to
download an archive with the model. It automatically creates a directory
structure and downloads the selected model. This step is skipped if the
package is already downloaded.

.. code:: ipython3

    import tarfile
    from pathlib import Path
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file
    
    # A directory where the model will be downloaded.
    base_model_dir = Path("./model").expanduser()
    
    model_name = "ssdlite_mobilenet_v2"
    archive_name = Path(f"{model_name}_coco_2018_05_09.tar.gz")
    
    # Download the archive
    downloaded_model_path = base_model_dir / archive_name
    if not downloaded_model_path.exists():
        model_url = f"http://download.tensorflow.org/models/object_detection/{archive_name}"
        download_file(model_url, downloaded_model_path.name, downloaded_model_path.parent)
    
    # Unpack the model
    tf_model_path = base_model_dir / archive_name.with_suffix("").stem / "frozen_inference_graph.pb"
    if not tf_model_path.exists():
        with tarfile.open(downloaded_model_path) as file:
            file.extractall(base_model_dir)



.. parsed-literal::

    model/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz:   0%|          | 0.00/48.7M [00:00<?, ?B/s]


.. parsed-literal::

    IOPub message rate exceeded.
    The notebook server will temporarily stop sending output
    to the client in order to avoid crashing it.
    To change this limit, set the config variable
    `--NotebookApp.iopub_msg_rate_limit`.
    
    Current values:
    NotebookApp.iopub_msg_rate_limit=1000.0 (msgs/sec)
    NotebookApp.rate_limit_window=3.0 (secs)
    


Convert the Model to OpenVINO IR format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



To convert the model to OpenVINO IR with ``FP16`` precision, use model
conversion API. The models are saved to the ``model/ir_model/``
directory. For more details about model conversion, see this
`page <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html>`__.

.. code:: ipython3

    from openvino.tools.mo.front import tf as ov_tf_front
    
    precision = "FP16"
    
    # The output path for the conversion.
    model_path = base_model_dir / "ir_model" / f"{model_name}_{precision.lower()}.xml"
    
    trans_config_path = Path(ov_tf_front.__file__).parent / "ssd_v2_support.json"
    pipeline_config = base_model_dir / archive_name.with_suffix("").stem / "pipeline.config"
    
    model = None
    if not model_path.exists():
        model = ov.tools.mo.convert_model(
            input_model=tf_model_path,
            input_shape=[1, 300, 300, 3],
            layout="NHWC",
            transformations_config=trans_config_path,
            tensorflow_object_detection_api_pipeline_config=pipeline_config,
            reverse_input_channels=True,
        )
        ov.save_model(model, model_path, compress_to_fp16=(precision == "FP16"))
        print("IR model saved to {}".format(model_path))
    else:
        print("Read IR model from {}".format(model_path))
        model = core.read_model(model_path)


.. parsed-literal::

    [ WARNING ]  The Preprocessor block has been removed. Only nodes performing mean value subtraction and scaling (if applicable) are kept.


.. parsed-literal::

    IR model saved to model/ir_model/ssdlite_mobilenet_v2_fp16.xml


Compile with Default Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



When the model is ready, first we need to read it, using the
``read_model`` method. Then, we can use the ``compile_model`` method and
specify the name of the device we want to compile the model on, in this
case, “GPU”.

.. code:: ipython3

    compiled_model = core.compile_model(model, device)

If you have multiple GPUs in the system, you can specify which one to
use by using “GPU.0”, “GPU.1”, etc. Any of the device names returned by
the ``available_devices`` method are valid device specifiers. You may
also use “AUTO”, which will automatically select the best device for
inference (which is often the GPU). To learn more about AUTO plugin,
visit the `Automatic Device
Selection <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
page as well as the `AUTO device
tutorial <auto-device-with-output.html>`__.

Reduce Compile Time through Model Caching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Depending on the model used, device-specific optimizations and network
compilations can cause the compile step to be time-consuming, especially
with larger models, which may lead to bad user experience in the
application, in which they are used. To solve this, OpenVINO can cache
the model once it is compiled on supported devices and reuse it in later
``compile_model`` calls by simply setting a cache folder beforehand. For
instance, to cache the same model we compiled above, we can do the
following:

.. code:: ipython3

    import time
    from pathlib import Path
    
    # Create cache folder
    cache_folder = Path("cache")
    cache_folder.mkdir(exist_ok=True)
    
    start = time.time()
    core = ov.Core()
    
    # Set cache folder
    core.set_property({props.cache_dir(): cache_folder})
    
    # Compile the model as before
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache enabled (first time) - compile time: {time.time() - start}s")


.. parsed-literal::

    Cache enabled (first time) - compile time: 1.692436695098877s


To get an idea of the effect that caching can have, we can measure the
compile times with caching enabled and disabled as follows:

.. code:: ipython3

    start = time.time()
    core = ov.Core()
    core.set_property({props.cache_dir(): "cache"})
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache enabled  - compile time: {time.time() - start}s")
    
    start = time.time()
    core = ov.Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model, device)
    print(f"Cache disabled - compile time: {time.time() - start}s")


.. parsed-literal::

    Cache enabled  - compile time: 0.26888394355773926s
    Cache disabled - compile time: 1.982884168624878s


The actual time improvements will depend on the environment as well as
the model being used but it is definitely something to consider when
optimizing an application. To read more about this, see the `Model
Caching <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__
docs.

Throughput and Latency Performance Hints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To simplify device and pipeline configuration, OpenVINO provides
high-level performance hints that automatically set the batch size and
number of parallel threads to use for inference. The “LATENCY”
performance hint optimizes for fast inference times while the
“THROUGHPUT” performance hint optimizes for high overall bandwidth or
FPS.

To use the “LATENCY” performance hint, add
``{hints.performance_mode(): hints.PerformanceMode.LATENCY}`` when
compiling the model as shown below. For GPUs, this automatically
minimizes the batch size and number of parallel streams such that all of
the compute resources can focus on completing a single inference as fast
as possible.

.. code:: ipython3

    import openvino.properties.hint as hints
    
    
    compiled_model = core.compile_model(model, device, {hints.performance_mode(): hints.PerformanceMode.LATENCY})

To use the “THROUGHPUT” performance hint, add
``{hints.performance_mode(): hints.PerformanceMode.THROUGHPUT}`` when
compiling the model. For GPUs, this creates multiple processing streams
to efficiently utilize all the execution cores and optimizes the batch
size to fill the available memory.

.. code:: ipython3

    compiled_model = core.compile_model(model, device, {hints.performance_mode(): hints.PerformanceMode.THROUGHPUT})

Using Multiple GPUs with Multi-Device and Cumulative Throughput
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The latency and throughput hints mentioned above are great and can make
a difference when used adequately but they usually use just one device,
either due to the `AUTO
plugin <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html#how-auto-works>`__
or by manual specification of the device name as above. When we have
multiple devices, such as an integrated and discrete GPU, we may use
both at the same time to improve the utilization of the resources. In
order to do this, OpenVINO provides a virtual device called
`MULTI <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html>`__,
which is just a combination of the existent devices that knows how to
split inference work between them, leveraging the capabilities of each
device.

As an example, if we want to use both integrated and discrete GPUs and
the CPU at the same time, we can compile the model as follows:

``compiled_model = core.compile_model(model=model, device_name="MULTI:GPU.1,GPU.0,CPU")``

Note that we always need to explicitly specify the device list for MULTI
to work, otherwise MULTI does not know which devices are available for
inference. However, this is not the only way to use multiple devices in
OpenVINO. There is another performance hint called
“CUMULATIVE_THROUGHPUT” that works similar to MULTI, except it uses the
devices automatically selected by AUTO. This way, we do not need to
manually specify devices to use. Below is an example showing how to use
“CUMULATIVE_THROUGHPUT”, equivalent to the MULTI one:

\`

compiled_model = core.compile_model(model=model, device_name=“AUTO”,
config={hints.performance_mode():
hints.PerformanceMode.CUMULATIVE_THROUGHPUT}) \`

   **Important**: **The “THROUGHPUT”, “MULTI”, and
   “CUMULATIVE_THROUGHPUT” modes are only applicable to asynchronous
   inferencing pipelines. The example at the end of this article shows
   how to set up an asynchronous pipeline that takes advantage of
   parallelism to increase throughput.** To learn more, see
   `Asynchronous
   Inferencing <https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request.html>`__
   in OpenVINO as well as the `Asynchronous Inference
   notebook <async-api-with-output.html>`__.

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

Note that benchmark_app only requires the model path to run but both the
device and hint arguments will be useful to us. For more advanced
usages, the tool itself has other options that can be checked by running
``benchmark_app -h`` or reading the
`docs <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.
The following example shows how to benchmark a simple model, using a GPU
with a latency focus:

.. code:: ipython3

    !benchmark_app -m {model_path} -d GPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 14.02 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 1932.50 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: frozen_inference_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_HOST_TASK_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_THROTTLE: Priority.MEDIUM
    [ INFO ]   GPU_ENABLE_LOOP_UNROLLING: True
    [ INFO ]   CACHE_DIR: 
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   COMPILATION_NUM_THREADS: 20
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'undefined'>
    [ INFO ]   DEVICE_ID: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'image_tensor'!. This input will be filled with random values!
    [ INFO ] Fill input 'image_tensor' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 6.17 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            12710 iterations
    [ INFO ] Duration:         60006.58 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.52 ms
    [ INFO ]    Average:       4.57 ms
    [ INFO ]    Min:           3.13 ms
    [ INFO ]    Max:           17.62 ms
    [ INFO ] Throughput:   211.81 FPS


For completeness, let us list here some of the comparisons we may want
to do by varying the device and hint used. Note that the actual
performance may depend on the hardware used. Generally, we should expect
GPU to be better than CPU, whereas multiple GPUs should be better than a
single GPU as long as there is enough work for each of them.

CPU vs GPU with Latency Hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    !benchmark_app -m {model_path} -d CPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 30.38 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 127.72 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: frozen_inference_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 14
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'image_tensor'!. This input will be filled with random values!
    [ INFO ] Fill input 'image_tensor' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 4.42 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            15304 iterations
    [ INFO ] Duration:         60005.72 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        3.87 ms
    [ INFO ]    Average:       3.88 ms
    [ INFO ]    Min:           3.49 ms
    [ INFO ]    Max:           5.95 ms
    [ INFO ] Throughput:   255.04 FPS


.. code:: ipython3

    !benchmark_app -m {model_path} -d GPU -hint latency


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 14.65 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2254.81 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: frozen_inference_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 1
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_HOST_TASK_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_THROTTLE: Priority.MEDIUM
    [ INFO ]   GPU_ENABLE_LOOP_UNROLLING: True
    [ INFO ]   CACHE_DIR: 
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.LATENCY
    [ INFO ]   COMPILATION_NUM_THREADS: 20
    [ INFO ]   NUM_STREAMS: 1
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'undefined'>
    [ INFO ]   DEVICE_ID: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'image_tensor'!. This input will be filled with random values!
    [ INFO ] Fill input 'image_tensor' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 1 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 8.79 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            11354 iterations
    [ INFO ] Duration:         60007.21 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        4.57 ms
    [ INFO ]    Average:       5.16 ms
    [ INFO ]    Min:           3.18 ms
    [ INFO ]    Max:           34.87 ms
    [ INFO ] Throughput:   189.21 FPS


CPU vs GPU with Throughput Hint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    !benchmark_app -m {model_path} -d CPU -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 29.56 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor:0 , image_tensor (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor:0 , image_tensor (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 158.91 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: frozen_inference_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 5
    [ INFO ]   NUM_STREAMS: 5
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 20
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'image_tensor'!. This input will be filled with random values!
    [ INFO ] Fill input 'image_tensor' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 5 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 8.15 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            25240 iterations
    [ INFO ] Duration:         60010.99 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        10.16 ms
    [ INFO ]    Average:       11.84 ms
    [ INFO ]    Min:           7.96 ms
    [ INFO ]    Max:           37.53 ms
    [ INFO ] Throughput:   420.59 FPS


.. code:: ipython3

    !benchmark_app -m {model_path} -d GPU -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 15.45 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2249.04 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: frozen_inference_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 4
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_HOST_TASK_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_PRIORITY: Priority.MEDIUM
    [ INFO ]   GPU_QUEUE_THROTTLE: Priority.MEDIUM
    [ INFO ]   GPU_ENABLE_LOOP_UNROLLING: True
    [ INFO ]   CACHE_DIR: 
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   COMPILATION_NUM_THREADS: 20
    [ INFO ]   NUM_STREAMS: 2
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'undefined'>
    [ INFO ]   DEVICE_ID: 0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'image_tensor'!. This input will be filled with random values!
    [ INFO ] Fill input 'image_tensor' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 4 inference requests, limits: 60000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 9.17 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Count:            19588 iterations
    [ INFO ] Duration:         60023.47 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        11.31 ms
    [ INFO ]    Average:       12.15 ms
    [ INFO ]    Min:           9.26 ms
    [ INFO ]    Max:           36.04 ms
    [ INFO ] Throughput:   326.34 FPS


Single GPU vs Multiple GPUs
^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    !benchmark_app -m {model_path} -d GPU.1 -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Device GPU.1 does not support performance hint property(-hint).
    [ ERROR ] Config for device with 1 ID is not registered in GPU plugin
    Traceback (most recent call last):
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py", line 329, in main
        benchmark.set_config(config)
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/benchmark.py", line 57, in set_config
        self.core.set_property(device, config[device])
    RuntimeError: Config for device with 1 ID is not registered in GPU plugin


.. code:: ipython3

    !benchmark_app -m {model_path} -d AUTO:GPU.1,GPU.0 -hint cumulative_throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Device GPU.1 does not support performance hint property(-hint).
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 26.66 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor , image_tensor:0 (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ ERROR ] Config for device with 1 ID is not registered in GPU plugin
    Traceback (most recent call last):
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py", line 414, in main
        compiled_model = benchmark.core.compile_model(model, benchmark.device)
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/runtime/ie_api.py", line 399, in compile_model
        super().compile_model(model, device_name, {} if config is None else config),
    RuntimeError: Config for device with 1 ID is not registered in GPU plugin


.. code:: ipython3

    !benchmark_app -m {model_path} -d MULTI:GPU.1,GPU.0 -hint throughput


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] GPU
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] MULTI
    [ INFO ] Build ................................. 2022.3.0-9052-9752fafe8eb-releases/2022/3
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Device GPU.1 does not support performance hint property(-hint).
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 14.84 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor:0 , image_tensor (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     image_tensor:0 , image_tensor (node: image_tensor) : u8 / [N,H,W,C] / [1,300,300,3]
    [ INFO ] Model outputs:
    [ INFO ]     detection_boxes:0 (node: DetectionOutput) : f32 / [...] / [1,1,100,7]
    [Step 7/11] Loading the model to the device
    [ ERROR ] Config for device with 1 ID is not registered in GPU plugin
    Traceback (most recent call last):
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/tools/benchmark/main.py", line 414, in main
        compiled_model = benchmark.core.compile_model(model, benchmark.device)
      File "/home/adrian/repos/openvino_notebooks/venv/lib/python3.9/site-packages/openvino/runtime/ie_api.py", line 399, in compile_model
        super().compile_model(model, device_name, {} if config is None else config),
    RuntimeError: Config for device with 1 ID is not registered in GPU plugin


Basic Application Using GPUs
----------------------------



We will now show an end-to-end object detection example using GPUs in
OpenVINO. The application compiles a model on GPU with the “THROUGHPUT”
hint, then loads a video and preprocesses every frame to convert them to
the shape expected by the model. Once the frames are loaded, it sets up
an asynchronous pipeline, performs inference and saves the detections
found in each frame. The detections are then drawn on their
corresponding frame and saved as a video, which is displayed at the end
of the application.

Import Necessary Packages
~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import time
    from pathlib import Path
    
    import cv2
    import numpy as np
    from IPython.display import Video
    import openvino as ov
    
    # Instantiate OpenVINO Runtime
    core = ov.Core()
    core.available_devices




.. parsed-literal::

    ['CPU', 'GPU']



Compile the Model
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Read model and compile it on GPU in THROUGHPUT mode
    model = core.read_model(model=model_path)
    device_name = "GPU"
    compiled_model = core.compile_model(model=model, device_name=device_name, config={hints.performance_mode(): hints.PerformanceMode.THROUGHPUT})
    
    # Get the input and output nodes
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Get the input size
    num, height, width, channels = input_layer.shape
    print("Model input shape:", num, height, width, channels)


.. parsed-literal::

    Model input shape: 1 300 300 3


Load and Preprocess Video Frames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Load video
    video_file = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/Coco%20Walking%20in%20Berkeley.mp4"
    video = cv2.VideoCapture(video_file)
    framebuf = []
    
    # Go through every frame of video and resize it
    print("Loading video...")
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            print("Video loaded!")
            video.release()
            break
    
        # Preprocess frames - convert them to shape expected by model
        input_frame = cv2.resize(src=frame, dsize=(width, height), interpolation=cv2.INTER_AREA)
        input_frame = np.expand_dims(input_frame, axis=0)
    
        # Append frame to framebuffer
        framebuf.append(input_frame)
    
    
    print("Frame shape: ", framebuf[0].shape)
    print("Number of frames: ", len(framebuf))
    
    # Show original video file
    # If the video does not display correctly inside the notebook, please open it with your favorite media player
    Video(video_file)


.. parsed-literal::

    Loading video...
    Video loaded!
    Frame shape:  (1, 300, 300, 3)
    Number of frames:  288


Define Model Output Classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Define the model's labelmap (this model uses COCO classes)
    classes = [
        "background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "street sign",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "hat",
        "backpack",
        "umbrella",
        "shoe",
        "eye glasses",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "plate",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "mirror",
        "dining table",
        "window",
        "desk",
        "toilet",
        "door",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "blender",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
        "hair brush",
    ]

Set up Asynchronous Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Callback Definition
^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    # Define a callback function that runs every time the asynchronous pipeline completes inference on a frame
    def completion_callback(infer_request: ov.InferRequest, frame_id: int) -> None:
        global frame_number
        stop_time = time.time()
        frame_number += 1
    
        predictions = next(iter(infer_request.results.values()))
        results[frame_id] = predictions[:10]  # Grab first 10 predictions for this frame
    
        total_time = stop_time - start_time
        frame_fps[frame_id] = frame_number / total_time

Create Async Pipeline
^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    # Create asynchronous inference queue with optimal number of infer requests
    infer_queue = ov.AsyncInferQueue(compiled_model)
    infer_queue.set_callback(completion_callback)

Perform Inference
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    # Perform inference on every frame in the framebuffer
    results = {}
    frame_fps = {}
    frame_number = 0
    start_time = time.time()
    for i, input_frame in enumerate(framebuf):
        infer_queue.start_async({0: input_frame}, i)
    
    infer_queue.wait_all()  # Wait until all inference requests in the AsyncInferQueue are completed
    stop_time = time.time()
    
    # Calculate total inference time and FPS
    total_time = stop_time - start_time
    fps = len(framebuf) / total_time
    time_per_frame = 1 / fps
    print(f"Total time to infer all frames: {total_time:.3f}s")
    print(f"Time per frame: {time_per_frame:.6f}s ({fps:.3f} FPS)")


.. parsed-literal::

    Total time to infer all frames: 1.366s
    Time per frame: 0.004744s (210.774 FPS)


Process Results
~~~~~~~~~~~~~~~



.. code:: ipython3

    # Set minimum detection threshold
    min_thresh = 0.6
    
    # Load video
    video = cv2.VideoCapture(video_file)
    
    # Get video parameters
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    fourcc = int(video.get(cv2.CAP_PROP_FOURCC))
    
    # Create folder and VideoWriter to save output video
    Path("./output").mkdir(exist_ok=True)
    output = cv2.VideoWriter("output/output.mp4", fourcc, fps, (frame_width, frame_height))
    
    # Draw detection results on every frame of video and save as a new video file
    while video.isOpened():
        current_frame = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = video.read()
        if not ret:
            print("Video loaded!")
            output.release()
            video.release()
            break
    
        # Draw info at the top left such as current fps, the devices and the performance hint being used
        cv2.putText(
            frame,
            f"fps {str(round(frame_fps[current_frame], 2))}",
            (5, 20),
            cv2.FONT_ITALIC,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"device {device_name}",
            (5, 40),
            cv2.FONT_ITALIC,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"hint {compiled_model.get_property(hints.performance_mode)}",
            (5, 60),
            cv2.FONT_ITALIC,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    
        # prediction contains [image_id, label, conf, x_min, y_min, x_max, y_max] according to model
        for prediction in np.squeeze(results[current_frame]):
            if prediction[2] > min_thresh:
                x_min = int(prediction[3] * frame_width)
                y_min = int(prediction[4] * frame_height)
                x_max = int(prediction[5] * frame_width)
                y_max = int(prediction[6] * frame_height)
                label = classes[int(prediction[1])]
    
                # Draw a bounding box with its label above it
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    label,
                    (x_min, y_min - 10),
                    cv2.FONT_ITALIC,
                    1,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
    
        output.write(frame)
    
    # Show output video file
    # If the video does not display correctly inside the notebook, please open it with your favorite media player
    Video("output/output.mp4", width=800, embed=True)


.. parsed-literal::

    Video loaded!




.. raw:: html

    <video controls  width="800" >
     <source src="data:None;base64,output/output.mp4" type="None">
     Your browser does not support the video tag.
     </video>



Conclusion
----------



This tutorial demonstrates how easy it is to use one or more GPUs in
OpenVINO, check their properties, and even tailor the model performance
through the different performance hints. It also provides a walk-through
of a basic object detection application that uses a GPU and displays the
detected bounding boxes.

To read more about any of these topics, feel free to visit their
corresponding documentation:

-  `GPU
   Plugin <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/gpu-device.html>`__
-  `AUTO
   Plugin <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/auto-device-selection.html>`__
-  `Model
   Caching <https://docs.openvino.ai/2024/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html>`__
-  `MULTI Device
   Mode <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/multi-device.html>`__
-  `Query Device
   Properties <https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/query-device-properties.html>`__
-  `Configurations for GPUs with
   OpenVINO <https://docs.openvino.ai/2024/get-started/configurations/configurations-intel-gpu.html>`__
-  `Benchmark Python
   Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
-  `Asynchronous
   Inferencing <https://docs.openvino.ai/2024/documentation/openvino-extensibility/openvino-plugin-library/asynch-inference-request.html>`__
