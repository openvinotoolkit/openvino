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

.. |image0| image:: https://github.com/piddnad/DDColor/raw/master/assets/network_arch.jpg

Prerequisites
-------------



.. code:: ipython3

    import platform

    %pip install -q "nncf>=2.11.0" "torch>=2.1" "torchvision" "timm" "opencv_python" "pillow" "PyYAML" "scipy" "scikit-image" "datasets" "gradio>=4.19"  --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -Uq --pre "openvino" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

    if platform.python_version_tuple()[1] in ["8", "9"]:
        %pip install -q "gradio-imageslider<=0.0.17" "typing-extensions>=4.9.0"
    else:
        %pip install -q "gradio-imageslider"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    openvino-dev 2024.2.0 requires openvino==2024.2.0, but you have openvino 2024.4.0.dev20240712 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import sys
    from pathlib import Path

    repo_dir = Path("DDColor")

    if not repo_dir.exists():
        !git clone https://github.com/piddnad/DDColor.git

    sys.path.append(str(repo_dir))


.. parsed-literal::

    Cloning into 'DDColor'...
    remote: Enumerating objects: 230, done.[K
    remote: Counting objects: 100% (76/76), done.[K
    remote: Compressing objects: 100% (39/39), done.[K
    remote: Total 230 (delta 54), reused 40 (delta 36), pack-reused 154[K
    Receiving objects: 100% (230/230), 13.34 MiB | 20.76 MiB/s, done.
    Resolving deltas: 100% (75/75), done.


.. code:: ipython3

    try:
        from inference.colorization_pipeline_hf import DDColorHF, ImageColorizationPipelineHF
    except Exception:
        from inference.colorization_pipeline_hf import DDColorHF, ImageColorizationPipelineHF

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


.. parsed-literal::

    ['x']


Run OpenVINO model inference
----------------------------



Select one of supported devices for inference using dropdown list.

.. code:: ipython3

    import ipywidgets as widgets

    core = ov.Core()

    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value="AUTO",
        description="Device:",
        disabled=False,
    )

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

    to_quantize = widgets.Checkbox(
        value=True,
        description="Quantization",
        disabled=False,
    )

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

    2024-07-12 23:51:48.961005: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-07-12 23:51:49.001135: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-07-12 23:51:49.398556: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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
    [ INFO ] Build ................................. 2024.4.0-16028-fe423b97163
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16028-fe423b97163
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 41.74 ms
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
    [ INFO ] Compile model took 1302.32 ms
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
    [ INFO ] First inference took 537.09 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            72 iterations
    [ INFO ] Duration:         16531.03 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        1375.35 ms
    [ INFO ]    Average:       1368.70 ms
    [ INFO ]    Min:           1259.43 ms
    [ INFO ]    Max:           1453.51 ms
    [ INFO ] Throughput:   4.36 FPS


.. code:: ipython3

    if OV_INT8_COLORIZER_PATH.exists():
        !benchmark_app  -m $OV_INT8_COLORIZER_PATH -d $device.value -api async -shape "[1,3,512,512]" -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.4.0-16028-fe423b97163
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.4.0-16028-fe423b97163
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 68.54 ms
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
    [ INFO ] Compile model took 2180.66 ms
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
    [ INFO ] First inference took 283.96 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            156 iterations
    [ INFO ] Duration:         15915.24 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        608.17 ms
    [ INFO ]    Average:       609.92 ms
    [ INFO ]    Min:           550.02 ms
    [ INFO ]    Max:           718.37 ms
    [ INFO ] Throughput:   9.80 FPS


Interactive inference
---------------------



.. code:: ipython3

    import gradio as gr
    from gradio_imageslider import ImageSlider
    from functools import partial


    def generate(image, use_int8=True):
        image_in = cv2.imread(image)
        image_out = process(image_in, compiled_model if not use_int8 else compiled_int8_model)
        image_in_pil = PIL.Image.fromarray(cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB))
        image_out_pil = PIL.Image.fromarray(cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB))
        return (image_in_pil, image_out_pil)


    with gr.Blocks() as demo:
        with gr.Row(equal_height=False):
            image = gr.Image(type="filepath")
            with gr.Column():
                output_image = ImageSlider(show_label=True, type="filepath", interactive=False, label="FP16 model output")
                button = gr.Button(value="Run{}".format(" FP16 model" if compiled_int8_model is not None else ""))
            with gr.Column(visible=compiled_int8_model is not None):
                output_image_int8 = ImageSlider(show_label=True, type="filepath", interactive=False, label="INT8 model output")
                button_i8 = gr.Button(value="Run INT8 model")
        button.click(fn=partial(generate, use_int8=False), inputs=[image], outputs=[output_image])
        button_i8.click(fn=partial(generate, use_int8=True), inputs=[image], outputs=[output_image_int8])
        examples = gr.Examples(
            [
                "DDColor/assets/test_images/New York Riverfront December 15, 1931.jpg",
                "DDColor/assets/test_images/Audrey Hepburn.jpg",
                "DDColor/assets/test_images/Acrobats Balance On Top Of The Empire State Building, 1934.jpg",
            ],
            inputs=[image],
        )


    if __name__ == "__main__":
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







