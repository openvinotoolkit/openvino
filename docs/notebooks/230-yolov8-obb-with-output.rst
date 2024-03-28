YOLOv8 Oriented Bounding Boxes Object Detection with OpenVINO™
==============================================================

`YOLOv8-OBB <https://docs.ultralytics.com/tasks/obb/>`__ is introduced
by Ultralytics.

Oriented object detection goes a step further than object detection and
introduce an extra angle to locate objects more accurate in an image.

The output of an oriented object detector is a set of rotated bounding
boxes that exactly enclose the objects in the image, along with class
labels and confidence scores for each box. Object detection is a good
choice when you need to identify objects of interest in a scene, but
don’t need to know exactly where the object is or its exact shape.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Get PyTorch model <#get-pytorch-model>`__
-  `Prepare dataset and dataloader <#prepare-dataset-and-dataloader>`__
-  `Run inference <#run-inference>`__
-  `Convert PyTorch model to OpenVINO
   IR <#convert-pytorch-model-to-openvino-ir>`__

   -  `Select inference device <#select-inference-device>`__
   -  `Compile model <#compile-model>`__
   -  `Prepare the model for
      inference <#prepare-the-model-for-inference>`__
   -  `Run inference <#run-inference>`__

-  `Quantization <#quantization>`__
-  `Compare inference time and model
   sizes. <#compare-inference-time-and-model-sizes>`__

Prerequisites
~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "ultralytics==8.1.24" "openvino>=2024.0.0" "nncf>=2.9.0"

Import required utility functions. The lower cell will download the
notebook_utils Python module from GitHub.

.. code:: ipython3

    from pathlib import Path
    
    # Fetch the notebook utils script from the openvino_notebooks repo
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import download_file

Get PyTorch model
~~~~~~~~~~~~~~~~~



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv8 pretrained OBB large model (also known as ``yolov8l-obbn``)
pre-trained on a DOTAv1 dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other YOLOv8 models.

.. code:: ipython3

    from ultralytics import YOLO
    
    model = YOLO('yolov8l-obb.pt')

Prepare dataset and dataloader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



YOLOv8-obb is pre-trained on the DOTA dataset. Also, Ultralytics
provides DOTA8 dataset. It is a small, but versatile oriented object
detection dataset composed of the first 8 images of 8 images of the
split DOTAv1 set, 4 for training and 4 for validation. This dataset is
ideal for testing and debugging object detection models, or for
experimenting with new detection approaches. With 8 images, it is small
enough to be easily manageable, yet diverse enough to test training
pipelines for errors and act as a sanity check before training larger
datasets.

The original model repository uses a Validator wrapper, which represents
the accuracy validation pipeline. It creates dataloader and evaluation
metrics and updates metrics on each data batch produced by the
dataloader. Besides that, it is responsible for data preprocessing and
results postprocessing. For class initialization, the configuration
should be provided. We will use the default setup, but it can be
replaced with some parameters overriding to test on custom data. The
model has connected the task_map, which allows to get a validator class
instance.

.. code:: ipython3

    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils import DEFAULT_CFG, DATASETS_DIR
    
    
    CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/dota8.yaml"
    OUT_DIR = Path('./datasets')
    CFG_PATH = OUT_DIR / "dota8.yaml"
    
    download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
    
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = CFG_PATH
    args.task = model.task
    
    validator = model.task_map[model.task]['validator'](args=args)
    
    validator.stride = 32
    validator.data = check_det_dataset(str(args.data))
    data_loader = validator.get_dataloader(DATASETS_DIR / 'dota8', 1)
    example_image_path = list(data_loader)[1]['im_file'][0]

Run inference
~~~~~~~~~~~~~



.. code:: ipython3

    from PIL import Image
    
    res = model(example_image_path, device='cpu')
    Image.fromarray(res[0].plot()[:, :, ::-1])


.. parsed-literal::

    
    image 1/1 /home/maleksandr/test_notebooks/yolo8obb/openvino_notebooks/notebooks/288-yolov8-obb/datasets/dota8/images/train/P1053__1024__0___90.jpg: 1024x1024 367.6ms
    Speed: 2.6ms preprocess, 367.6ms inference, 2.6ms postprocess per image at shape (1, 3, 1024, 1024)




.. image:: 230-yolov8-obb-with-output_files/230-yolov8-obb-with-output_10_1.png



Convert PyTorch model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



YOLOv8 provides API for convenient model exporting to different formats
including OpenVINO IR. ``model.export`` is responsible for model
conversion. We need to specify the format, and additionally, we can
preserve dynamic shapes in the model.

.. code:: ipython3

    from pathlib import Path
    
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    
    OV_MODEL_NAME = "yolov8l-obb"
    
    
    OV_MODEL_PATH = Path(f"{OV_MODEL_NAME}_openvino_model/{OV_MODEL_NAME}.xml")
    if not OV_MODEL_PATH.exists():
        model.export(format="openvino", dynamic=True, half=True)

Select inference device
^^^^^^^^^^^^^^^^^^^^^^^



Select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Compile model
^^^^^^^^^^^^^



.. code:: ipython3

    ov_config = {}
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    
    ov_model = core.read_model(OV_MODEL_PATH)
    compiled_ov_model = core.compile_model(ov_model, device.value, ov_config)

Prepare the model for inference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



We can reuse the base model pipeline for pre- and postprocessing just
replacing the inference method where we will use the IR model for
inference.

.. code:: ipython3

    import torch
    
    def infer(*args):
        result = compiled_ov_model(args)[0]
        return torch.from_numpy(result)
    
    model.predictor.inference = infer

Run inference
^^^^^^^^^^^^^



.. code:: ipython3

    res = model(example_image_path, device='cpu')
    Image.fromarray(res[0].plot()[:, :, ::-1])


.. parsed-literal::

    
    image 1/1 /home/maleksandr/test_notebooks/yolo8obb/openvino_notebooks/notebooks/288-yolov8-obb/datasets/dota8/images/train/P1053__1024__0___90.jpg: 1024x1024 354.5ms
    Speed: 10.3ms preprocess, 354.5ms inference, 2.5ms postprocess per image at shape (1, 3, 1024, 1024)




.. image:: 230-yolov8-obb-with-output_files/230-yolov8-obb-with-output_20_1.png



Quantization
~~~~~~~~~~~~



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

    import ipywidgets as widgets
    
    INT8_OV_PATH = Path("model/int8_model.xml")
    
    to_quantize = widgets.Checkbox(
        value=True,
        description='Quantization',
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



Let’s load ``skip magic`` extension to skip quantization if
``to_quantize`` is not selected

.. code:: ipython3

    import sys
    
    sys.path.append("../utils")
    
    %load_ext skip_kernel_extension

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from typing import Dict
    
    import nncf
    
    
    def transform_fn(data_item: Dict):
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor
    
    
    quantization_dataset = nncf.Dataset(data_loader, transform_fn)

Create a quantized model from the pre-trained converted OpenVINO model.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

..

   **NOTE**: We use the tiny DOTA8 dataset as a calibration dataset. It
   gives a good enough result for tutorial purpose. For batter results,
   use a bigger dataset. Usually 300 examples are enough.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if INT8_OV_PATH.exists():
        print("Loading quantized model")
        quantized_model = core.read_model(INT8_OV_PATH)
    else:
        quantized_model = nncf.quantize(
            ov_model,
            quantization_dataset,
            preset=nncf.QuantizationPreset.MIXED,
        )
        ov.save_model(quantized_model, INT8_OV_PATH)
    
    model_optimized = core.compile_model(INT8_OV_PATH, device.value)

We can reuse the base model pipeline in the same way as for IR model.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    def infer(*args):
        result = model_optimized(args)[0]
        return torch.from_numpy(result)
    
    model.predictor.inference = infer

Run inference

.. code:: ipython3

    %%skip not $to_quantize.value
    
    res = model(example_image_path, device='cpu')
    Image.fromarray(res[0].plot()[:, :, ::-1])


.. parsed-literal::

    
    image 1/1 /home/maleksandr/test_notebooks/yolo8obb/openvino_notebooks/notebooks/288-yolov8-obb/datasets/dota8/images/train/P1053__1024__0___90.jpg: 1024x1024 262.4ms
    Speed: 6.6ms preprocess, 262.4ms inference, 3.1ms postprocess per image at shape (1, 3, 1024, 1024)




.. image:: 230-yolov8-obb-with-output_files/230-yolov8-obb-with-output_31_1.png



You can see that the result is almost the same but it has a small
difference. One small vehicle was recognized as two vehicles. But one
large car was also identified, unlike the original model.

Compare inference time and model sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_ir_model_size = OV_MODEL_PATH.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = INT8_OV_PATH.with_suffix(".bin").stat().st_size / 1024
    
    print(f"FP16 model size: {fp16_ir_model_size:.2f} KB")
    print(f"INT8 model size: {quantized_model_size:.2f} KB")
    print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 model size: 173697.94 KB
    INT8 model size: 43494.75 KB
    Model compression rate: 3.994


.. code:: ipython3

    # Inference FP32 model (OpenVINO IR)
    !benchmark_app -m $OV_MODEL_PATH -d $device.value -api async -shape "[1,3,640,640]"


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 16.98 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.22/aten::cat/Concat_9) : f32 / [...] / [?,20,16..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 8.93 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.22/aten::cat/Concat_9) : f32 / [...] / [1,20,8400]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 554.80 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 36
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 324.82 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            1728 iterations
    [ INFO ] Duration:         121146.03 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        838.25 ms
    [ INFO ]    Average:       839.50 ms
    [ INFO ]    Min:           603.03 ms
    [ INFO ]    Max:           1051.74 ms
    [ INFO ] Throughput:   14.26 FPS


.. code:: ipython3

    if INT8_OV_PATH.exists():
        # Inference INT8 model (Quantized model)
        !benchmark_app -m $INT8_OV_PATH -d $device.value -api async -shape "[1,3,640,640]" -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.0.0-14509-34caeefd078-releases/2024/0
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 39.15 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.22/aten::cat/Concat_9) : f32 / [...] / [?,20,16..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 18.90 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.22/aten::cat/Concat_9) : f32 / [...] / [1,20,8400]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 1236.04 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: Model0
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]   MULTI_DEVICE_PRIORITIES: CPU
    [ INFO ]   CPU:
    [ INFO ]     AFFINITY: Affinity.CORE
    [ INFO ]     CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]     CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 0
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 36
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'x'!. This input will be filled with random values!
    [ INFO ] Fill input 'x' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 118.47 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            768 iterations
    [ INFO ] Duration:         15304.44 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        237.48 ms
    [ INFO ]    Average:       237.91 ms
    [ INFO ]    Min:           138.26 ms
    [ INFO ]    Max:           266.14 ms
    [ INFO ] Throughput:   50.18 FPS

