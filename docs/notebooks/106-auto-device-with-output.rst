Automatic Device Selection with OpenVINO™
=========================================

The `Auto
device <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_AUTO.html>`__
(or AUTO in short) selects the most suitable device for inference by
considering the model precision, power efficiency and processing
capability of the available `compute
devices <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html>`__.
The model precision (such as ``FP32``, ``FP16``, ``INT8``, etc.) is the
first consideration to filter out the devices that cannot run the
network efficiently.

Next, if dedicated accelerators are available, these devices are
preferred (for example, integrated and discrete
`GPU <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU.html#doxid-openvino-docs-o-v-u-g-supported-plugins-g-p-u>`__
or
`VPU <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_VPU.html>`__).
`CPU <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_CPU.html>`__
is used as the default “fallback device”. Keep in mind that AUTO makes
this selection only once, during the loading of a model.

When using accelerator devices such as GPUs, loading models to these
devices may take a long time. To address this challenge for applications
that require fast first inference response, AUTO starts inference
immediately on the CPU and then transparently shifts inference to the
GPU, once it is ready. This dramatically reduces the time to execute
first inference.

.. raw:: html

   <center>

.. raw:: html

   </center>

Download and convert the model
------------------------------

This tutorial uses the
`bvlc_googlenet <https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet>`__
model. The bvlc_googlenet model is the first of the
`Inception <https://github.com/tensorflow/tpu/tree/master/models/experimental/inception>`__
family of models designed to perform image classification. Like other
Inception models, bvlc_googlenet was pre-trained on the
`ImageNet <https://image-net.org/>`__ data set. For more details about
this family of models, see the `research
paper <https://arxiv.org/abs/1512.00567>`__.

.. code:: ipython3

    import sys
    
    from pathlib import Path
    from openvino.tools import mo
    from openvino.runtime import serialize
    from IPython.display import Markdown, display
    
    sys.path.append("../utils")
    
    import notebook_utils as utils
    
    base_model_dir = Path("./model").expanduser()
    
    model_name = "bvlc_googlenet"
    caffemodel_name = f'{model_name}.caffemodel'
    prototxt_name = f'{model_name}.prototxt'
    
    caffemodel_path = base_model_dir / caffemodel_name
    prototxt_path = base_model_dir / prototxt_name
    
    if not caffemodel_path.exists() or not prototxt_path.exists():
        caffemodel_url = "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/googlenet-v1/bvlc_googlenet.caffemodel"
        prototxt_url = "https://raw.githubusercontent.com/BVLC/caffe/88c96189bcbf3853b93e2b65c7b5e4948f9d5f67/models/bvlc_googlenet/deploy.prototxt"
    
        utils.download_file(caffemodel_url, caffemodel_name, base_model_dir)
        utils.download_file(prototxt_url, prototxt_name, base_model_dir)
    else:
        print(f'{caffemodel_name} and {prototxt_name} already downloaded to {base_model_dir}')
    
    # postprocessing of model
    text = prototxt_path.read_text()
    text = text.replace('dim: 10', 'dim: 1')
    res = prototxt_path.write_text(text)



.. parsed-literal::

    model/bvlc_googlenet.caffemodel:   0%|          | 0.00/51.1M [00:00<?, ?B/s]



.. parsed-literal::

    model/bvlc_googlenet.prototxt:   0%|          | 0.00/2.19k [00:00<?, ?B/s]


Import modules and create Core
------------------------------

.. code:: ipython3

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    from openvino.runtime import Core, CompiledModel, AsyncInferQueue, InferRequest
    import sys
    import time
    
    ie = Core()
    
    if "GPU" not in ie.available_devices:
        display(Markdown('<div class="alert alert-block alert-danger"><b>Warning: </b> A GPU device is not available. This notebook requires GPU device to have meaningful results. </div>'))



.. container:: alert alert-block alert-danger

   Warning: A GPU device is not available. This notebook requires GPU
   device to have meaningful results.


Convert the model to OpenVINO IR format
---------------------------------------

Use Model Optimizer to convert the Caffe model to OpenVINO IR with
``FP16`` precision. The models are saved to the ``model/ir_model/``
directory. For more information about Model Optimizer, see the `Model
Optimizer Developer
Guide <https://docs.openvino.ai/latest/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__.

.. code:: ipython3

    ir_model_path = base_model_dir / 'ir_model' / f'{model_name}.xml'
    model = None
    
    if not ir_model_path.exists():
        model = mo.convert_model(input_model=base_model_dir / caffemodel_name,
                                 input_proto=base_model_dir / prototxt_name,
                                 input_shape=[1, 3, 224, 224],
                                 layout="NCHW",
                                 mean_values=[104.0,117.0,123.0],
                                 output="prob",
                                 compress_to_fp16=True)
        serialize(model, str(ir_model_path))
        print("IR model saved to {}".format(ir_model_path))
    else:
        print("Read IR model from {}".format(ir_model_path))
        model = ie.read_model(ir_model_path)


.. parsed-literal::

    /opt/home/k8sworker/cibuilds/ov-notebook/OVNotebookOps-416/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/numpy/lib/function_base.py:959: VisibleDeprecationWarning: Creating an ndarray from ragged nested sequences (which is a list-or-tuple of lists-or-tuples-or ndarrays with different lengths or shapes) is deprecated. If you meant to do this, you must specify 'dtype=object' when creating the ndarray.
      return array(a, order=order, subok=subok, copy=True)


.. parsed-literal::

    IR model saved to model/ir_model/bvlc_googlenet.xml


(1) Simplify selection logic
----------------------------

Default behavior of Core::compile_model API without device_name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``compile_model`` API will select **AUTO** as
``device_name`` if no device is specified.

.. code:: ipython3

    # Set LOG_LEVEL to LOG_INFO.
    ie.set_property("AUTO", {"LOG_LEVEL":"LOG_INFO"})
    
    # Load the model onto the target device.
    compiled_model = ie.compile_model(model=model)
    
    if isinstance(compiled_model, CompiledModel):
        print("Successfully compiled model without a device_name.")   


.. parsed-literal::

    [22:35:25.7084]I[plugin.cpp:402][AUTO] load with CNN network
    [22:35:25.7136]I[plugin.cpp:422][AUTO] device:CPU, config:EXCLUSIVE_ASYNC_REQUESTS=NO
    [22:35:25.7137]I[plugin.cpp:422][AUTO] device:CPU, config:PERFORMANCE_HINT=LATENCY
    [22:35:25.7137]I[plugin.cpp:422][AUTO] device:CPU, config:PERFORMANCE_HINT_NUM_REQUESTS=0
    [22:35:25.7137]I[plugin.cpp:422][AUTO] device:CPU, config:PERF_COUNT=NO
    [22:35:25.7137]I[plugin.cpp:435][AUTO] device:CPU, priority:0
    [22:35:25.7141]I[auto_schedule.cpp:103][AUTO] ExecutableNetwork start
    [22:35:25.7145]I[auto_schedule.cpp:146][AUTO] select device:CPU
    [22:35:25.8945]I[auto_schedule.cpp:188][AUTO] device:CPU loading Network finished
    Successfully compiled model without a device_name.


.. code:: ipython3

    # Deleted model will wait until compiling on the selected device is complete.
    del compiled_model
    print("Deleted compiled_model")


.. parsed-literal::

    [22:35:25.9051]I[auto_schedule.cpp:509][AUTO] ExecutableNetwork end
    [22:35:25.9052]I[multi_schedule.cpp:254][AUTO] CPU:infer:0
    Deleted compiled_model


Explicitly pass AUTO as device_name to Core::compile_model API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It is optional, but passing AUTO explicitly as ``device_name`` may
improve readability of your code.

.. code:: ipython3

    # Set LOG_LEVEL to LOG_NONE.
    ie.set_property("AUTO", {"LOG_LEVEL":"LOG_NONE"})
    
    compiled_model = ie.compile_model(model=model, device_name="AUTO")
    
    if isinstance(compiled_model, CompiledModel):
        print("Successfully compiled model using AUTO.")


.. parsed-literal::

    Successfully compiled model using AUTO.


.. code:: ipython3

    # Deleted model will wait until compiling on the selected device is complete.
    del compiled_model
    print("Deleted compiled_model")


.. parsed-literal::

    Deleted compiled_model


(2) Improve the first inference latency
---------------------------------------

One of the benefits of using AUTO device selection is reducing FIL
(first inference latency). FIL is the model compilation time combined
with the first inference execution time. Using the CPU device explicitly
will produce the shortest first inference latency, as the OpenVINO graph
representation loads quickly on CPU, using just-in-time (JIT)
compilation. The challenge is with GPU devices since OpenCL graph
complication to GPU-optimized kernels takes a few seconds to complete.
This initialization time may be intolerable for some applications. To
avoid this delay, the AUTO uses CPU transparently as the first inference
device until GPU is ready. ### Load an Image

.. code:: ipython3

    # For demonstration purposes, load the model to CPU and get inputs for buffer preparation.
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    
    input_layer_ir = next(iter(compiled_model.inputs))
    
    # Read image in BGR format.
    image = cv2.imread("../data/image/coco.jpg")
    
    # N, C, H, W = batch size, number of channels, height, width.
    N, C, H, W = input_layer_ir.shape
    
    # Resize image to the input size expected by the model.
    resized_image = cv2.resize(image, (W, H))
    
    # Reshape to match the input shape expected by the model.
    input_image = np.expand_dims(resized_image.transpose(2, 0, 1), 0)
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    del compiled_model



.. image:: 106-auto-device-with-output_files/106-auto-device-with-output_14_0.png


Load the model to GPU device and perform inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    if "GPU" not in ie.available_devices:
        print(f"A GPU device is not available. Available devices are: {ie.available_devices}")
    else :       
        # Start time.
        gpu_load_start_time = time.perf_counter()
        compiled_model = ie.compile_model(model=model, device_name="GPU")  # load to GPU
    
        # Get input and output nodes.
        input_layer = compiled_model.input(0)
        output_layer = compiled_model.output(0)
    
        # Execute the first inference.
        results = compiled_model([input_image])[output_layer]
    
        # Measure time to the first inference.
        gpu_fil_end_time = time.perf_counter()
        gpu_fil_span = gpu_fil_end_time - gpu_load_start_time
        print(f"Time to load model on GPU device and get first inference: {gpu_fil_end_time-gpu_load_start_time:.2f} seconds.")
        del compiled_model


.. parsed-literal::

    A GPU device is not available. Available devices are: ['CPU']


Load the model using AUTO device and do inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When GPU is the best available device, the first few inferences will be
executed on CPU until GPU is ready.

.. code:: ipython3

    # Start time.
    auto_load_start_time = time.perf_counter()
    compiled_model = ie.compile_model(model=model)  # The device_name is AUTO by default.
    
    # Get input and output nodes.
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    
    # Execute the first inference.
    results = compiled_model([input_image])[output_layer]
    
    
    # Measure time to the first inference.
    auto_fil_end_time = time.perf_counter()
    auto_fil_span = auto_fil_end_time - auto_load_start_time
    print(f"Time to load model using AUTO device and get first inference: {auto_fil_end_time-auto_load_start_time:.2f} seconds.")


.. parsed-literal::

    Time to load model using AUTO device and get first inference: 0.15 seconds.


.. code:: ipython3

    # Deleted model will wait for compiling on the selected device to complete.
    del compiled_model

(3) Achieve different performance for different targets
-------------------------------------------------------

It is an advantage to define **performance hints** when using Automatic
Device Selection. By specifying a **THROUGHPUT** or **LATENCY** hint,
AUTO optimizes the performance based on the desired metric. The
**THROUGHPUT** hint delivers higher frame per second (FPS) performance
than the **LATENCY** hint, which delivers lower latency. The performance
hints do not require any device-specific settings and they are
completely portable between devices – meaning AUTO can configure the
performance hint on whichever device is being used.

For more information, refer to the `Performance
Hints <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_AUTO.html#performance-hints>`__
section of `Automatic Device
Selection <https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_AUTO.html>`__
article.

Class and callback definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    class PerformanceMetrics:
        """
        Record the latest performance metrics (fps and latency), update the metrics in each @interval seconds
        :member: fps: Frames per second, indicates the average number of inferences executed each second during the last @interval seconds.
        :member: latency: Average latency of inferences executed in the last @interval seconds.
        :member: start_time: Record the start timestamp of onging @interval seconds duration.
        :member: latency_list: Record the latency of each inference execution over @interval seconds duration.
        :member: interval: The metrics will be updated every @interval seconds
        """
        def __init__(self, interval):
            """
            Create and initilize one instance of class PerformanceMetrics.
            :param: interval: The metrics will be updated every @interval seconds
            :returns:
                Instance of PerformanceMetrics
            """
            self.fps = 0
            self.latency = 0
            
            self.start_time = time.perf_counter()
            self.latency_list = []
            self.interval = interval
            
        def update(self, infer_request: InferRequest) -> bool:
            """
            Update the metrics if current ongoing @interval seconds duration is expired. Record the latency only if it is not expired.
            :param: infer_request: InferRequest returned from inference callback, which includes the result of inference request.
            :returns:
                True, if metrics are updated.
                False, if @interval seconds duration is not expired and metrics are not updated.
            """
            self.latency_list.append(infer_request.latency)
            exec_time = time.perf_counter() - self.start_time
            if exec_time >= self.interval:
                # Update the performance metrics.
                self.start_time = time.perf_counter()
                self.fps = len(self.latency_list) / exec_time
                self.latency = sum(self.latency_list) / len(self.latency_list)
                print(f"throughput: {self.fps: .2f}fps, latency: {self.latency: .2f}ms, time interval:{exec_time: .2f}s")
                sys.stdout.flush()
                self.latency_list = []
                return True
            else :
                return False
    
    
    class InferContext:
        """
        Inference context. Record and update peforamnce metrics via @metrics, set @feed_inference to False once @remaining_update_num <=0
        :member: metrics: instance of class PerformanceMetrics 
        :member: remaining_update_num: the remaining times for peforamnce metrics updating.
        :member: feed_inference: if feed inference request is required or not.
        """
        def __init__(self, update_interval, num):
            """
            Create and initilize one instance of class InferContext.
            :param: update_interval: The performance metrics will be updated every @update_interval seconds. This parameter will be passed to class PerformanceMetrics directly.
            :param: num: The number of times performance metrics are updated.
            :returns:
                Instance of InferContext.
            """
            self.metrics = PerformanceMetrics(update_interval)
            self.remaining_update_num = num
            self.feed_inference = True
            
        def update(self, infer_request: InferRequest):
            """
            Update the context. Set @feed_inference to False if the number of remaining performance metric updates (@remaining_update_num) reaches 0
            :param: infer_request: InferRequest returned from inference callback, which includes the result of inference request.
            :returns: None
            """
            if self.remaining_update_num <= 0 :
                self.feed_inference = False
                
            if self.metrics.update(infer_request) :
                self.remaining_update_num = self.remaining_update_num - 1
                if self.remaining_update_num <= 0 :
                    self.feed_inference = False
    
    
    def completion_callback(infer_request: InferRequest, context) -> None:
        """
        callback for the inference request, pass the @infer_request to @context for updating
        :param: infer_request: InferRequest returned for the callback, which includes the result of inference request.
        :param: context: user data which is passed as the second parameter to AsyncInferQueue:start_async()
        :returns: None
        """
        context.update(infer_request)
    
    
    # Performance metrics update interval (seconds) and number of times.
    metrics_update_interval = 10
    metrics_update_num = 6

Inference with THROUGHPUT hint
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loop for inference and update the FPS/Latency every
@metrics_update_interval seconds.

.. code:: ipython3

    THROUGHPUT_hint_context = InferContext(metrics_update_interval, metrics_update_num)
    
    print("Compiling Model for AUTO device with THROUGHPUT hint")
    sys.stdout.flush()
    
    compiled_model = ie.compile_model(model=model, config={"PERFORMANCE_HINT":"THROUGHPUT"})
    
    infer_queue = AsyncInferQueue(compiled_model, 0)  # Setting to 0 will query optimal number by default.
    infer_queue.set_callback(completion_callback)
    
    print(f"Start inference, {metrics_update_num: .0f} groups of FPS/latency will be measured over {metrics_update_interval: .0f}s intervals")
    sys.stdout.flush()
    
    while THROUGHPUT_hint_context.feed_inference:
        infer_queue.start_async({input_layer_ir.any_name: input_image}, THROUGHPUT_hint_context)
        
    infer_queue.wait_all()
    
    # Take the FPS and latency of the latest period.
    THROUGHPUT_hint_fps = THROUGHPUT_hint_context.metrics.fps
    THROUGHPUT_hint_latency = THROUGHPUT_hint_context.metrics.latency
    
    print("Done")
    
    del compiled_model


.. parsed-literal::

    Compiling Model for AUTO device with THROUGHPUT hint
    Start inference,  6 groups of FPS/latency will be measured over  10s intervals
    throughput:  461.74fps, latency:  24.68ms, time interval: 10.01s
    throughput:  470.76fps, latency:  24.89ms, time interval: 10.00s
    throughput:  470.13fps, latency:  24.96ms, time interval: 10.01s
    throughput:  470.19fps, latency:  24.89ms, time interval: 10.00s
    throughput:  471.13fps, latency:  24.87ms, time interval: 10.00s
    throughput:  469.51fps, latency:  24.92ms, time interval: 10.00s
    Done


Inference with LATENCY hint
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loop for inference and update the FPS/Latency for each
@metrics_update_interval seconds

.. code:: ipython3

    LATENCY_hint_context = InferContext(metrics_update_interval, metrics_update_num)
    
    print("Compiling Model for AUTO Device with LATENCY hint")
    sys.stdout.flush()
    
    compiled_model = ie.compile_model(model=model, config={"PERFORMANCE_HINT":"LATENCY"})
    
    # Setting to 0 will query optimal number by default.
    infer_queue = AsyncInferQueue(compiled_model, 0)
    infer_queue.set_callback(completion_callback)
    
    print(f"Start inference, {metrics_update_num: .0f} groups fps/latency will be out with {metrics_update_interval: .0f}s interval")
    sys.stdout.flush()
    
    while LATENCY_hint_context.feed_inference:
        infer_queue.start_async({input_layer_ir.any_name: input_image}, LATENCY_hint_context)
        
    infer_queue.wait_all()
    
    # Take the FPS and latency of the latest period.
    LATENCY_hint_fps = LATENCY_hint_context.metrics.fps
    LATENCY_hint_latency = LATENCY_hint_context.metrics.latency
    
    print("Done")
    
    del compiled_model


.. parsed-literal::

    Compiling Model for AUTO Device with LATENCY hint
    Start inference,  6 groups fps/latency will be out with  10s interval
    throughput:  250.83fps, latency:  3.62ms, time interval: 10.00s
    throughput:  253.12fps, latency:  3.70ms, time interval: 10.00s
    throughput:  250.90fps, latency:  3.73ms, time interval: 10.00s
    throughput:  249.98fps, latency:  3.74ms, time interval: 10.00s
    throughput:  248.29fps, latency:  3.77ms, time interval: 10.00s
    throughput:  255.00fps, latency:  3.67ms, time interval: 10.00s
    Done


Difference in FPS and latency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    TPUT = 0
    LAT = 1
    labels = ["THROUGHPUT hint", "LATENCY hint"]
    
    fig1, ax1 = plt.subplots(1, 1) 
    fig1.patch.set_visible(False)
    ax1.axis('tight') 
    ax1.axis('off') 
    
    cell_text = []
    cell_text.append(['%.2f%s' % (THROUGHPUT_hint_fps," FPS"), '%.2f%s' % (THROUGHPUT_hint_latency, " ms")])
    cell_text.append(['%.2f%s' % (LATENCY_hint_fps," FPS"), '%.2f%s' % (LATENCY_hint_latency, " ms")])
    
    table = ax1.table(cellText=cell_text, colLabels=["FPS (Higher is better)", "Latency (Lower is better)"], rowLabels=labels,  
                      rowColours=["deepskyblue"] * 2, colColours=["deepskyblue"] * 2,
                      cellLoc='center', loc='upper left')
    table.auto_set_font_size(False)
    table.set_fontsize(18)
    table.auto_set_column_width(0)
    table.auto_set_column_width(1)
    table.scale(1, 3)
    
    fig1.tight_layout()
    plt.show()



.. image:: 106-auto-device-with-output_files/106-auto-device-with-output_27_0.png


.. code:: ipython3

    # Output the difference.
    width = 0.4
    fontsize = 14
    
    plt.rc('font', size=fontsize)
    fig, ax = plt.subplots(1,2, figsize=(10, 8))
    
    rects1 = ax[0].bar([0], THROUGHPUT_hint_fps, width, label=labels[TPUT], color='#557f2d')
    rects2 = ax[0].bar([width], LATENCY_hint_fps, width, label=labels[LAT])
    ax[0].set_ylabel("frames per second")
    ax[0].set_xticks([width / 2]) 
    ax[0].set_xticklabels(["FPS"])
    ax[0].set_xlabel("Higher is better")
    
    rects1 = ax[1].bar([0], THROUGHPUT_hint_latency, width, label=labels[TPUT], color='#557f2d')
    rects2 = ax[1].bar([width], LATENCY_hint_latency, width, label=labels[LAT])
    ax[1].set_ylabel("milliseconds")
    ax[1].set_xticks([width / 2])
    ax[1].set_xticklabels(["Latency (ms)"])
    ax[1].set_xlabel("Lower is better")
    
    fig.suptitle('Performance Hints')
    fig.legend(labels, fontsize=fontsize)
    fig.tight_layout()
    
    plt.show()



.. image:: 106-auto-device-with-output_files/106-auto-device-with-output_28_0.png

