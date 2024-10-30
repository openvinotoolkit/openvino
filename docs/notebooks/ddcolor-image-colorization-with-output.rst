Colorize grayscale images using DDColor and OpenVINO
======================================================

Image colorization is the process of adding color to grayscale images.
Initially captured in black and white, these images are transformed into
vibrant, lifelike representations by estimating RGB colors. This
technology enhances both aesthetic appeal and perceptual quality.
Historically, artists manually applied colors to monochromatic
photographs, a painstaking task that could take up to a month for a
single image. However, with advancements in information technology and
the rise of deep neural networks, automated image colorization has
become increasingly important.

DDColor is one of the most progressive methods of image colorization in
our days. It is a novel approach using dual decoders: a pixel decoder
and a query-based color decoder, that stands out in its ability to
produce photo-realistic colorization, particularly in complex scenes
with multiple objects and diverse contexts. |image0|

More details about this approach can be found in original model
`repository <https://github.com/piddnad/DDColor>`__ and
`paper <https://arxiv.org/abs/2212.11613>`__.

In this tutorial we consider how to convert and run DDColor using
OpenVINO. Additionally, we will demonstrate how to optimize this model
using `NNCF <https://github.com/openvinotoolkit/nncf/>`__.

🪄 Let’s start to explore magic of image colorization! #### Table of
contents:

-  `Prerequisites <#prerequisites>`__
-  `Load PyTorch model <#load-pytorch-model>`__
-  `Run PyTorch model inference <#run-pytorch-model-inference>`__
-  `Convert PyTorch model to OpenVINO Intermediate
   Representation <#convert-pytorch-model-to-openvino-intermediate-representation>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__
-  `Optimize OpenVINO model using
   NNCF <#optimize-openvino-model-using-nncf>`__

   -  `Collect quantization dataset <#collect-quantization-dataset>`__
   -  `Perform model quantization <#perform-model-quantization>`__

-  `Run INT8 model inference <#run-int8-model-inference>`__
-  `Compare FP16 and INT8 model
   size <#compare-fp16-and-int8-model-size>`__
-  `Compare inference time of the FP16 and INT8
   models <#compare-inference-time-of-the-fp16-and-int8-models>`__
-  `Interactive inference <#interactive-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |image0| image:: https://github.com/piddnad/DDColor/raw/master/assets/network_arch.jpg

Prerequisites
-------------



.. code:: ipython3

    import platform
    
    %pip install -q "nncf>=2.11.0" "torch>=2.1" "torchvision" "timm" "opencv_python" "pillow" "PyYAML" "scipy" "scikit-image" "datasets" "gradio>=4.19"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -Uq "openvino>=2024.3.0"
    if platform.python_version_tuple()[1] in ["8", "9"]:
        %pip install -q "gradio-imageslider<=0.0.17" "typing-extensions>=4.9.0"
    else:
        %pip install -q "gradio-imageslider"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path
    import requests
    
    repo_dir = Path("DDColor")
    
    if not repo_dir.exists():
        !git clone https://github.com/piddnad/DDColor.git
    
    sys.path.append(str(repo_dir))
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)


.. parsed-literal::

    Cloning into 'DDColor'...
    remote: Enumerating objects: 233, done.[K
    remote: Counting objects: 100% (76/76), done.[K
    remote: Compressing objects: 100% (42/42), done.[K
    remote: Total 233 (delta 54), reused 34 (delta 34), pack-reused 157 (from 1)[K
    Receiving objects: 100% (233/233), 13.34 MiB | 641.00 KiB/s, done.
    Resolving deltas: 100% (80/80), done.




.. parsed-literal::

    24692



.. code:: ipython3

    try:
        from inference.colorization_pipeline_hf import DDColorHF, ImageColorizationPipelineHF
    except Exception:
        from inference.colorization_pipeline_hf import DDColorHF, ImageColorizationPipelineHF


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/timm/models/layers/__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
      warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)


Load PyTorch model
------------------



There are several models from DDColor’s family provided in `model
repository <https://github.com/piddnad/DDColor/blob/master/MODEL_ZOO.md>`__.
We will use DDColor-T, the most lightweight version of ddcolor model,
but demonstrated in the tutorial steps are also applicable to other
models from DDColor family.

.. code:: ipython3

    import torch
    
    model_name = "ddcolor_paper_tiny"
    
    ddcolor_model = DDColorHF.from_pretrained(f"piddnad/{model_name}")
    
    
    colorizer = ImageColorizationPipelineHF(model=ddcolor_model, input_size=512)
    
    ddcolor_model.to("cpu")
    colorizer.device = torch.device("cpu")

Run PyTorch model inference
---------------------------



.. code:: ipython3

    import cv2
    import PIL
    
    IMG_PATH = "DDColor/assets/test_images/Ansel Adams _ Moore Photography.jpeg"
    
    
    img = cv2.imread(IMG_PATH)
    
    PIL.Image.fromarray(img[:, :, ::-1])




.. image:: ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_8_0.png



.. code:: ipython3

    image_out = colorizer.process(img)
    PIL.Image.fromarray(image_out[:, :, ::-1])




.. image:: ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_9_0.png



Convert PyTorch model to OpenVINO Intermediate Representation
-------------------------------------------------------------



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). OpenVINO model conversion API should be used for
these purposes. ``ov.convert_model`` function accepts original PyTorch
model instance and example input for tracing and returns ``ov.Model``
representing this model in OpenVINO framework. Converted model can be
used for saving on disk using ``ov.save_model`` function or directly
loading on device using ``core.complie_model``.

.. code:: ipython3

    import openvino as ov
    import torch
    
    OV_COLORIZER_PATH = Path("ddcolor.xml")
    
    if not OV_COLORIZER_PATH.exists():
        ov_model = ov.convert_model(ddcolor_model, example_input=torch.ones((1, 3, 512, 512)), input=[1, 3, 512, 512])
        ov.save_model(ov_model, OV_COLORIZER_PATH)

Run OpenVINO model inference
----------------------------



Select one of supported devices for inference using dropdown list.

.. code:: ipython3

    from notebook_utils import device_widget
    
    core = ov.Core()
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    compiled_model = core.compile_model(OV_COLORIZER_PATH, device.value)

.. code:: ipython3

    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    
    
    def process(img, compiled_model):
        # Preprocess input image
        height, width = img.shape[:2]
    
        # Normalize to [0, 1] range
        img = (img / 255.0).astype(np.float32)
        orig_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]  # (h, w, 1)
    
        # Resize rgb image -> lab -> get grey -> rgb
        img = cv2.resize(img, (512, 512))
        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)[:, :, :1]
        img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
        img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
        # Transpose HWC -> CHW and add batch dimension
        tensor_gray_rgb = torch.from_numpy(img_gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
    
        # Run model inference
        output_ab = compiled_model(tensor_gray_rgb)[0]
    
        # Postprocess result
        # resize ab -> concat original l -> rgb
        output_ab_resize = F.interpolate(torch.from_numpy(output_ab), size=(height, width))[0].float().numpy().transpose(1, 2, 0)
        output_lab = np.concatenate((orig_l, output_ab_resize), axis=-1)
        output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
    
        output_img = (output_bgr * 255.0).round().astype(np.uint8)
    
        return output_img

.. code:: ipython3

    ov_processed_img = process(img, compiled_model)
    PIL.Image.fromarray(ov_processed_img[:, :, ::-1])




.. image:: ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_16_0.png



Optimize OpenVINO model using NNCF
----------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize()`` to obtain quantized model.
3. Save the ``INT8`` model using ``openvino.save_model()`` function.

Please select below whether you would like to run quantization to
improve model inference speed.

.. code:: ipython3

    from notebook_utils import quantization_widget
    
    to_quantize = quantization_widget()
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    import requests
    
    OV_INT8_COLORIZER_PATH = Path("ddcolor_int8.xml")
    compiled_int8_model = None
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    %load_ext skip_kernel_extension

Collect quantization dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`ummagumm-a/colorization_dataset <https://huggingface.co/datasets/ummagumm-a/colorization_dataset>`__
dataset from Hugging Face as calibration data.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from datasets import load_dataset
    
    subset_size = 300
    calibration_data = []
    
    if not OV_INT8_COLORIZER_PATH.exists():
        dataset = load_dataset("ummagumm-a/colorization_dataset", split="train", streaming=True).shuffle(seed=42).take(subset_size)
        for idx, batch in enumerate(dataset):
            if idx >= subset_size:
                break
            img = np.array(batch["conditioning_image"])
            img = (img / 255.0).astype(np.float32)
            img = cv2.resize(img, (512, 512))
            img_l = cv2.cvtColor(np.stack([img, img, img], axis=2), cv2.COLOR_BGR2Lab)[:, :, :1]
            img_gray_lab = np.concatenate((img_l, np.zeros_like(img_l), np.zeros_like(img_l)), axis=-1)
            img_gray_rgb = cv2.cvtColor(img_gray_lab, cv2.COLOR_LAB2RGB)
    
            image = np.expand_dims(img_gray_rgb.transpose((2, 0, 1)).astype(np.float32), axis=0)
            calibration_data.append(image)

Perform model quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    
    if not OV_INT8_COLORIZER_PATH.exists():
        ov_model = core.read_model(OV_COLORIZER_PATH)
        quantized_model = nncf.quantize(
                model=ov_model,
                subset_size=subset_size,
                calibration_dataset=nncf.Dataset(calibration_data),
            )
        ov.save_model(quantized_model, OV_INT8_COLORIZER_PATH)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2024-10-22 22:45:07.339219: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-22 22:45:07.378241: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-22 22:45:07.784302: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









Run INT8 model inference
------------------------



.. code:: ipython3

    from IPython.display import display
    
    if OV_INT8_COLORIZER_PATH.exists():
        compiled_int8_model = core.compile_model(OV_INT8_COLORIZER_PATH, device.value)
        img = cv2.imread("DDColor/assets/test_images/Ansel Adams _ Moore Photography.jpeg")
        img_out = process(img, compiled_int8_model)
        display(PIL.Image.fromarray(img_out[:, :, ::-1]))



.. image:: ddcolor-image-colorization-with-output_files/ddcolor-image-colorization-with-output_25_0.png


Compare FP16 and INT8 model size
--------------------------------



.. code:: ipython3

    fp16_ir_model_size = OV_COLORIZER_PATH.with_suffix(".bin").stat().st_size / 2**20
    
    print(f"FP16 model size: {fp16_ir_model_size:.2f} MB")
    
    if OV_INT8_COLORIZER_PATH.exists():
        quantized_model_size = OV_INT8_COLORIZER_PATH.with_suffix(".bin").stat().st_size / 2**20
        print(f"INT8 model size: {quantized_model_size:.2f} MB")
        print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 model size: 104.89 MB
    INT8 model size: 52.97 MB
    Model compression rate: 1.980


Compare inference time of the FP16 and INT8 models
--------------------------------------------------



To measure the inference performance of OpenVINO FP16 and INT8 models,
use `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    !benchmark_app  -m $OV_COLORIZER_PATH -d $device.value -api async -shape "[1,3,512,512]" -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 41.84 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.refine_net.0.0/aten::_convolution/Add) : f32 / [...] / [1,2,512,512]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,512,512]
    [ INFO ] Reshape model took 0.04 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.refine_net.0.0/aten::_convolution/Add) : f32 / [...] / [1,2,512,512]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 1322.68 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 6
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 545.38 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            78 iterations
    [ INFO ] Duration:         17636.85 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1348.38 ms
    [ INFO ]    Average:       1347.67 ms
    [ INFO ]    Min:           1203.19 ms
    [ INFO ]    Max:           1415.43 ms
    [ INFO ] Throughput:   4.42 FPS


.. code:: ipython3

    if OV_INT8_COLORIZER_PATH.exists():
        !benchmark_app  -m $OV_INT8_COLORIZER_PATH -d $device.value -api async -shape "[1,3,512,512]" -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16579-c3152d32c9c-releases/2024/4
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 69.74 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.refine_net.0.0/aten::_convolution/Add) : f32 / [...] / [1,2,512,512]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,512,512]
    [ INFO ] Reshape model took 0.04 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,512,512]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.refine_net.0.0/aten::_convolution/Add) : f32 / [...] / [1,2,512,512]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 2263.12 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 24
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 6
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [ INFO ]   PERF_COUNT: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 270.31 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            156 iterations
    [ INFO ] Duration:         15580.45 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        592.37 ms
    [ INFO ]    Average:       595.05 ms
    [ INFO ]    Min:           334.73 ms
    [ INFO ]    Max:           668.12 ms
    [ INFO ] Throughput:   10.01 FPS


Interactive inference
---------------------



.. code:: ipython3

    def generate(image, use_int8=True):
        image_in = cv2.imread(image)
        image_out = process(image_in, compiled_model if not use_int8 else compiled_int8_model)
        image_in_pil = PIL.Image.fromarray(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
        image_out_pil = PIL.Image.fromarray(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB))
        return (image_in_pil, image_out_pil)
    
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/ddcolor-image-colorization/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=generate, quantized=compiled_int8_model is not None)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







