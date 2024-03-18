Convert and Optimize YOLOv8 with OpenVINO™
==========================================

The YOLOv8 algorithm developed by Ultralytics is a cutting-edge,
state-of-the-art (SOTA) model that is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object
detection, image segmentation, and image classification tasks. More
details about its realization can be found in the original model
`repository <https://github.com/ultralytics/ultralytics>`__.

This tutorial demonstrates step-by-step instructions on how to run apply
quantization with accuracy control to PyTorch YOLOv8. The advanced
quantization flow allows to apply 8-bit quantization to the model with
control of accuracy metric. This is achieved by keeping the most
impactful operations within the model in the original precision. The
flow is based on the `Basic 8-bit
quantization <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training/basic-quantization-flow.html>`__
and has the following differences:

-  Besides the calibration dataset, a validation dataset is required to
   compute the accuracy metric. Both datasets can refer to the same data
   in the simplest case.
-  Validation function, used to compute accuracy metric is required. It
   can be a function that is already available in the source framework
   or a custom function.
-  Since accuracy validation is run several times during the
   quantization process, quantization with accuracy control can take
   more time than the Basic 8-bit quantization flow.
-  The resulted model can provide smaller performance improvement than
   the Basic 8-bit quantization flow because some of the operations are
   kept in the original precision.

..

   **NOTE**: Currently, 8-bit quantization with accuracy control in NNCF
   is available only for models in OpenVINO representation.

The steps for the quantization with accuracy control are described
below.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Get Pytorch model and OpenVINO IR
   model <#get-pytorch-model-and-openvino-ir-model>`__

   -  `Define validator and data
      loader <#define-validator-and-data-loader>`__
   -  `Prepare calibration and validation
      datasets <#prepare-calibration-and-validation-datasets>`__
   -  `Prepare validation function <#prepare-validation-function>`__

-  `Run quantization with accuracy
   control <#run-quantization-with-accuracy-control>`__
-  `Compare Accuracy and Performance of the Original and Quantized
   Models <#compare-accuracy-and-performance-of-the-original-and-quantized-models>`__

Prerequisites
^^^^^^^^^^^^^



Install necessary packages.

.. code:: ipython3

    %pip install -q "openvino>=2023.1.0"
    %pip install -q "nncf>=2.6.0"
    %pip install -q "ultralytics==8.0.43" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Get Pytorch model and OpenVINO IR model
---------------------------------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv8 nano model (also known as ``yolov8n``) pre-trained on a COCO
dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other YOLOv8 models. Typical steps to obtain a
pre-trained model:

1. Create an instance of a model class.
2. Load a checkpoint state dict, which contains the pre-trained model
   weights.

In this case, the creators of the model provide an API that enables
converting the YOLOv8 model to ONNX and then to OpenVINO IR. Therefore,
we do not need to do these steps manually.

.. code:: ipython3

    import os
    from pathlib import Path
    
    from ultralytics import YOLO
    from ultralytics.yolo.cfg import get_cfg
    from ultralytics.yolo.data.utils import check_det_dataset
    from ultralytics.yolo.engine.validator import BaseValidator as Validator
    from ultralytics.yolo.utils import DEFAULT_CFG
    from ultralytics.yolo.utils import ops
    from ultralytics.yolo.utils.metrics import ConfusionMatrix
    
    ROOT = os.path.abspath('')
    
    MODEL_NAME = "yolov8n-seg"
    
    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"

.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    
    from notebook_utils import download_file

.. code:: ipython3

    from zipfile import ZipFile
    
    DATA_URL = "https://www.ultralytics.com/assets/coco128-seg.zip"
    CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/8ebe94d1e928687feaa1fee6d5668987df5e43be/ultralytics/datasets/coco128-seg.yaml"  # last compatible format with ultralytics 8.0.43
    
    OUT_DIR = Path('./datasets')
    
    DATA_PATH = OUT_DIR / "coco128-seg.zip"
    CFG_PATH = OUT_DIR / "coco128-seg.yaml"
    
    download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
    download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
    
    if not (OUT_DIR / "coco128/labels").exists():
        with ZipFile(DATA_PATH , "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)


.. parsed-literal::

    'datasets/coco128-seg.zip' already exists.



.. parsed-literal::

    datasets/coco128-seg.yaml:   0%|          | 0.00/0.98k [00:00<?, ?B/s]


Load model.

.. code:: ipython3

    import openvino as ov
    
    
    model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
    if not model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)
    
    ov_model = ov.Core().read_model(model_path)

Define validator and data loader
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



The original model repository uses a ``Validator`` wrapper, which
represents the accuracy validation pipeline. It creates dataloader and
evaluation metrics and updates metrics on each data batch produced by
the dataloader. Besides that, it is responsible for data preprocessing
and results postprocessing. For class initialization, the configuration
should be provided. We will use the default setup, but it can be
replaced with some parameters overriding to test on custom data. The
model has connected the ``ValidatorClass`` method, which creates a
validator class instance.

.. code:: ipython3

    validator = model.ValidatorClass(args)
    validator.data = check_det_dataset(args.data)
    data_loader = validator.get_dataloader("datasets/coco128-seg", 1)
    
    validator.is_coco = True
    validator.class_map = ops.coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc
    validator.nm = 32
    validator.process = ops.process_mask
    validator.plot_masks = []


.. parsed-literal::

    val: Scanning datasets/coco128-seg/labels/train2017... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 [00:00<00:00, 964.99it/s]
    val: New cache created: datasets/coco128-seg/labels/train2017.cache


Prepare calibration and validation datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



We can use one dataset as calibration and validation datasets. Name it
``quantization_dataset``.

.. code:: ipython3

    from typing import Dict
    
    import nncf
    
    
    def transform_fn(data_item: Dict):
        input_tensor = validator.preprocess(data_item)["img"].numpy()
        return input_tensor
    
    
    quantization_dataset = nncf.Dataset(data_loader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Prepare validation function
^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    from functools import partial
    
    import torch
    from nncf.quantization.advanced_parameters import AdvancedAccuracyRestorerParameters
    
    
    def validation_ac(
        compiled_model: ov.CompiledModel,
        validation_loader: torch.utils.data.DataLoader,
        validator: Validator,
        num_samples: int = None,
        log=True
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = []
        validator.batch_i = 1
        validator.confusion_matrix = ConfusionMatrix(nc=validator.nc)
        num_outputs = len(compiled_model.outputs)
    
        counter = 0
        for batch_i, batch in enumerate(validation_loader):
            if num_samples is not None and batch_i == num_samples:
                break
            batch = validator.preprocess(batch)
            results = compiled_model(batch["img"])
            if num_outputs == 1:
                preds = torch.from_numpy(results[compiled_model.output(0)])
            else:
                preds = [
                    torch.from_numpy(results[compiled_model.output(0)]),
                    torch.from_numpy(results[compiled_model.output(1)]),
                ]
            preds = validator.postprocess(preds)
            validator.update_metrics(preds, batch)
            counter += 1
        stats = validator.get_stats()
        if num_outputs == 1:
            stats_metrics = stats["metrics/mAP50-95(B)"]
        else:
            stats_metrics = stats["metrics/mAP50-95(M)"]
        if log:
            print(f"Validate: dataset length = {counter}, metric value = {stats_metrics:.3f}")
        
        return stats_metrics
    
    
    validation_fn = partial(validation_ac, validator=validator, log=False)

Run quantization with accuracy control
--------------------------------------



You should provide the calibration dataset and the validation dataset.
It can be the same dataset. - parameter ``max_drop`` defines the
accuracy drop threshold. The quantization process stops when the
degradation of accuracy metric on the validation dataset is less than
the ``max_drop``. The default value is 0.01. NNCF will stop the
quantization and report an error if the ``max_drop`` value can’t be
reached. - ``drop_type`` defines how the accuracy drop will be
calculated: ABSOLUTE (used by default) or RELATIVE. -
``ranking_subset_size`` - size of a subset that is used to rank layers
by their contribution to the accuracy drop. Default value is 300, and
the more samples it has the better ranking, potentially. Here we use the
value 25 to speed up the execution.

   **NOTE**: Execution can take tens of minutes and requires up to 15 GB
   of free memory

.. code:: ipython3

    quantized_model = nncf.quantize_with_accuracy_control(
        ov_model,
        quantization_dataset,
        quantization_dataset,
        validation_fn=validation_fn,
        max_drop=0.01,
        preset=nncf.QuantizationPreset.MIXED,
        subset_size=128,
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(
            ranking_subset_size=25
        ),
    )


.. parsed-literal::

    2024-02-28 13:33:46.187903: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-28 13:33:46.189894: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-02-28 13:33:46.226943: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-02-28 13:33:46.942396: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    INFO:nncf:Validation of initial model was started
    INFO:nncf:Elapsed Time: 00:00:00
    INFO:nncf:Elapsed Time: 00:00:05
    INFO:nncf:Metric of initial model: 0.36611468358574506
    INFO:nncf:Collecting values for each data item using the initial model
    INFO:nncf:Elapsed Time: 00:00:06
    INFO:nncf:Validation of quantized model was started
    INFO:nncf:Elapsed Time: 00:00:00
    INFO:nncf:Elapsed Time: 00:00:05
    INFO:nncf:Metric of quantized model: 0.3406029678292
    INFO:nncf:Collecting values for each data item using the quantized model
    INFO:nncf:Elapsed Time: 00:00:06
    INFO:nncf:Accuracy drop: 0.02551171575654504 (absolute)
    INFO:nncf:Accuracy drop: 0.02551171575654504 (absolute)
    INFO:nncf:Total number of quantized operations in the model: 91
    INFO:nncf:Number of parallel workers to rank quantized operations: 1
    INFO:nncf:ORIGINAL metric is used to rank quantizers



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    INFO:nncf:Elapsed Time: 00:02:25
    INFO:nncf:Changing the scope of quantizer nodes was started
    INFO:nncf:Reverted 2 operations to the floating-point precision: 
    	/model.22/Add_11
    	/model.22/Sub_1
    INFO:nncf:Accuracy drop with the new quantization scope is 0.013524778006655136 (absolute)
    INFO:nncf:Reverted 1 operations to the floating-point precision: 
    	/model.22/Mul_5
    INFO:nncf:Accuracy drop with the new quantization scope is 0.011937545450662279 (absolute)
    INFO:nncf:Reverted 1 operations to the floating-point precision: 
    	/model.2/cv1/conv/Conv/WithoutBiases
    INFO:nncf:Algorithm completed: achieved required accuracy drop 0.00905169821338292 (absolute)
    INFO:nncf:4 out of 91 were reverted back to the floating-point precision:
    	/model.22/Add_11
    	/model.22/Sub_1
    	/model.22/Mul_5
    	/model.2/cv1/conv/Conv/WithoutBiases


Compare Accuracy and Performance of the Original and Quantized Models
---------------------------------------------------------------------



Now we can compare metrics of the Original non-quantized OpenVINO IR
model and Quantized OpenVINO IR model to make sure that the ``max_drop``
is not exceeded.

.. code:: ipython3

    import ipywidgets as widgets
    
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



.. code:: ipython3

    core = ov.Core()
    ov_config = {}
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    quantized_compiled_model = core.compile_model(model=quantized_model, device_name=device.value, ov_config)
    compiled_ov_model = core.compile_model(model=ov_model, device_name=device.value, ov_config)
    
    pt_result = validation_ac(compiled_ov_model, data_loader, validator)
    quantized_result = validation_ac(quantized_compiled_model, data_loader, validator)
    
    
    print(f'[Original OpenVINO]: {pt_result:.4f}')
    print(f'[Quantized OpenVINO]: {quantized_result:.4f}')


.. parsed-literal::

    Validate: dataset length = 128, metric value = 0.368
    Validate: dataset length = 128, metric value = 0.357
    [Original OpenVINO]: 0.3677
    [Quantized OpenVINO]: 0.3570


And compare performance.

.. code:: ipython3

    from pathlib import Path
    # Set model directory
    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(exist_ok=True)
    
    ir_model_path = MODEL_DIR / 'ir_model.xml'
    quantized_model_path = MODEL_DIR / 'quantized_model.xml'
    
    # Save models to use them in the commandline banchmark app
    ov.save_model(ov_model, ir_model_path, compress_to_fp16=False)
    ov.save_model(quantized_model, quantized_model_path, compress_to_fp16=False)

.. code:: ipython3

    # Inference Original model (OpenVINO IR)
    ! benchmark_app -m $ir_model_path -shape "[1,3,640,640]" -d $device.value -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14589-0ef2fab3490
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.1.0-14589-0ef2fab3490
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 17.83 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [?,116,?]
    [ INFO ]     output1 (node: output1) : f32 / [...] / [?,32,8..,8..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 13.50 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,116,8400]
    [ INFO ]     output1 (node: output1) : f32 / [...] / [1,32,160,160]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 320.32 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
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
    [ INFO ]     NETWORK_NAME: torch_jit
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 44.16 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            14820 iterations
    [ INFO ] Duration:         120139.10 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        95.27 ms
    [ INFO ]    Average:       97.10 ms
    [ INFO ]    Min:           72.93 ms
    [ INFO ]    Max:           164.81 ms
    [ INFO ] Throughput:   123.36 FPS


.. code:: ipython3

    # Inference Quantized model (OpenVINO IR)
    ! benchmark_app -m $quantized_model_path -shape "[1,3,640,640]" -d $device.value -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.1.0-14589-0ef2fab3490
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.1.0-14589-0ef2fab3490
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 28.33 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [?,116,?]
    [ INFO ]     output1 (node: output1) : f32 / [...] / [?,32,8..,8..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 17.60 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,116,8400]
    [ INFO ]     output1 (node: output1) : f32 / [...] / [1,32,160,160]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 605.73 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: torch_jit
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
    [ INFO ]     NETWORK_NAME: torch_jit
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
    [ INFO ]     PERFORMANCE_HINT: THROUGHPUT
    [ INFO ]     PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]     PERF_COUNT: NO
    [ INFO ]     SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   MODEL_PRIORITY: Priority.MEDIUM
    [ INFO ]   LOADED_FROM_CACHE: False
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values 
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 22.24 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            33624 iterations
    [ INFO ] Duration:         120057.46 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        41.97 ms
    [ INFO ]    Average:       42.70 ms
    [ INFO ]    Min:           31.11 ms
    [ INFO ]    Max:           86.51 ms
    [ INFO ] Throughput:   280.07 FPS

