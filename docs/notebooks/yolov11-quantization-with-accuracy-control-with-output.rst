Convert and Optimize YOLOv11 with OpenVINO™
===========================================

The YOLOv11 algorithm developed by Ultralytics is a cutting-edge,
state-of-the-art (SOTA) model that is designed to be fast, accurate, and
easy to use, making it an excellent choice for a wide range of object
detection, image segmentation, and image classification tasks. More
details about its realization can be found in the original model
`repository <https://github.com/ultralytics/ultralytics>`__.

This tutorial demonstrates step-by-step instructions on how to run apply
quantization with accuracy control to PyTorch YOLOv11. The advanced
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


**Table of contents:**


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

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
^^^^^^^^^^^^^



Install necessary packages.

.. code:: ipython3

    %pip install -q "openvino>=2024.0.0"
    %pip install -q "nncf>=2.9.0"
    %pip install -q "ultralytics==8.3.0" tqdm --extra-index-url https://download.pytorch.org/whl/cpu

Get Pytorch model and OpenVINO IR model
---------------------------------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv11 nano model (also known as ``yolo11n``) pre-trained on a COCO
dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other YOLOv11 models. Typical steps to obtain a
pre-trained model:

1. Create an instance of a model class.
2. Load a checkpoint state dict, which contains the pre-trained model
   weights.

In this case, the creators of the model provide an API that enables
converting the YOLOv11 model to ONNX and then to OpenVINO IR. Therefore,
we do not need to do these steps manually.

.. code:: ipython3

    import os
    from pathlib import Path

    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.engine.validator import BaseValidator as Validator
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.utils import ops
    from ultralytics.utils.metrics import ConfusionMatrix

    ROOT = os.path.abspath("")

    MODEL_NAME = "yolo11n-seg"

    model = YOLO(f"{ROOT}/{MODEL_NAME}.pt")
    args = get_cfg(cfg=DEFAULT_CFG)
    args.data = "coco128-seg.yaml"

.. code:: ipython3

    # Fetch the notebook utils script from the openvino_notebooks repo
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)

    from notebook_utils import download_file, device_widget

.. code:: ipython3

    from zipfile import ZipFile

    from ultralytics.data.utils import DATASETS_DIR

    DATA_URL = "https://www.ultralytics.com/assets/coco128-seg.zip"
    CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco.yaml"

    OUT_DIR = DATASETS_DIR

    DATA_PATH = OUT_DIR / "coco128-seg.zip"
    CFG_PATH = OUT_DIR / "coco128-seg.yaml"

    download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
    download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)

    if not (OUT_DIR / "coco128/labels").exists():
        with ZipFile(DATA_PATH, "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)

Load model.

.. code:: ipython3

    import openvino as ov


    model_path = Path(f"{ROOT}/{MODEL_NAME}_openvino_model/{MODEL_NAME}.xml")
    if not model_path.exists():
        model.export(format="openvino", dynamic=True, half=False)

    ov_model = ov.Core().read_model(model_path)


.. parsed-literal::

    Ultralytics 8.3.0 🚀 Python-3.10.12 torch-2.5.1+cpu CPU (Intel Core(TM) i9-10980XE 3.00GHz)
    YOLO11n-seg summary (fused): 265 layers, 2,868,664 parameters, 0 gradients, 10.4 GFLOPs

    PyTorch: starting from '/home/maleksandr/test_notebooks/yolo/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control/yolo11n-seg.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) ((1, 116, 8400), (1, 32, 160, 160)) (5.9 MB)

    OpenVINO: starting export with openvino 2024.4.0-16579-c3152d32c9c-releases/2024/4...
    OpenVINO: export success ✅ 2.3s, saved as '/home/maleksandr/test_notebooks/yolo/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control/yolo11n-seg_openvino_model/' (11.3 MB)

    Export complete (2.6s)
    Results saved to /home/maleksandr/test_notebooks/yolo/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control
    Predict:         yolo predict task=segment model=/home/maleksandr/test_notebooks/yolo/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control/yolo11n-seg_openvino_model imgsz=640
    Validate:        yolo val task=segment model=/home/maleksandr/test_notebooks/yolo/openvino_notebooks/notebooks/quantizing-model-with-accuracy-control/yolo11n-seg_openvino_model imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco.yaml
    Visualize:       https://netron.app


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

    from ultralytics.data.converter import coco80_to_coco91_class


    validator = model.task_map[model.task]["validator"](args=args)
    validator.data = check_det_dataset(args.data)
    validator.stride = 3
    data_loader = validator.get_dataloader(OUT_DIR / "coco128-seg", 1)

    validator.is_coco = True
    validator.class_map = coco80_to_coco91_class()
    validator.names = model.model.names
    validator.metrics.names = validator.names
    validator.nc = model.model.model[-1].nc
    validator.nm = 32
    validator.process = ops.process_mask
    validator.plot_masks = []

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

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino


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
        log=True,
    ) -> float:
        validator.seen = 0
        validator.jdict = []
        validator.stats = dict(tp_m=[], tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
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
It can be the same dataset.

- parameter ``max_drop`` defines the
  accuracy drop threshold. The quantization process stops when the
  degradation of accuracy metric on the validation dataset is less than
  the ``max_drop``. The default value is 0.01. NNCF will stop the
  quantization and report an error if the ``max_drop`` value can’t be
  reached.
- ``drop_type`` defines how the accuracy drop will be
  calculated: ABSOLUTE (used by default) or RELATIVE.
- ``ranking_subset_size`` - size of a subset that is used to rank layers
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
        advanced_accuracy_restorer_parameters=AdvancedAccuracyRestorerParameters(ranking_subset_size=25),
    )



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Validation of initial model was started
    INFO:nncf:Elapsed Time: 00:00:00
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Metric of initial model: 0.39269183697877214
    INFO:nncf:Collecting values for each data item using the initial model
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Validation of quantized model was started
    INFO:nncf:Elapsed Time: 00:00:00
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Metric of quantized model: 0.37315412232099016
    INFO:nncf:Collecting values for each data item using the quantized model
    INFO:nncf:Elapsed Time: 00:00:04
    INFO:nncf:Accuracy drop: 0.01953771465778198 (absolute)
    INFO:nncf:Accuracy drop: 0.01953771465778198 (absolute)
    INFO:nncf:Total number of quantized operations in the model: 127
    INFO:nncf:Number of parallel workers to rank quantized operations: 1
    INFO:nncf:ORIGINAL metric is used to rank quantizers



.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Elapsed Time: 00:02:11
    INFO:nncf:Changing the scope of quantizer nodes was started
    INFO:nncf:Reverted 1 operations to the floating-point precision:
    	__module.model.23/aten::mul/Multiply_3
    INFO:nncf:Accuracy drop with the new quantization scope is 0.016468171203124993 (absolute)
    INFO:nncf:Reverted 1 operations to the floating-point precision:
    	__module.model.0.conv/aten::_convolution/Convolution
    INFO:nncf:Accuracy drop with the new quantization scope is 0.01663718104550027 (absolute)
    INFO:nncf:Re-calculating ranking scores for remaining groups



.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Elapsed Time: 00:02:07
    INFO:nncf:Reverted 2 operations to the floating-point precision:
    	__module.model.23/aten::sub/Subtract_1
    	__module.model.23/aten::add/Add_7
    INFO:nncf:Algorithm completed: achieved required accuracy drop 0.006520879829061188 (absolute)
    INFO:nncf:3 out of 127 were reverted back to the floating-point precision:
    	__module.model.23/aten::mul/Multiply_3
    	__module.model.23/aten::sub/Subtract_1
    	__module.model.23/aten::add/Add_7


Compare Accuracy and Performance of the Original and Quantized Models
---------------------------------------------------------------------



Now we can compare metrics of the Original non-quantized OpenVINO IR
model and Quantized OpenVINO IR model to make sure that the ``max_drop``
is not exceeded.

.. code:: ipython3

    device = device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    core = ov.Core()
    ov_config = {}
    if device.value != "CPU":
        quantized_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    quantized_compiled_model = core.compile_model(quantized_model, device.value, ov_config)
    compiled_ov_model = core.compile_model(ov_model, device.value, ov_config)

    pt_result = validation_ac(compiled_ov_model, data_loader, validator)
    quantized_result = validation_ac(quantized_compiled_model, data_loader, validator)


    print(f"[Original OpenVINO]: {pt_result:.4f}")
    print(f"[Quantized OpenVINO]: {quantized_result:.4f}")


.. parsed-literal::

    Validate: dataset length = 128, metric value = 0.393
    Validate: dataset length = 128, metric value = 0.386
    [Original OpenVINO]: 0.3927
    [Quantized OpenVINO]: 0.3862


And compare performance.

.. code:: ipython3

    from pathlib import Path

    # Set model directory
    MODEL_DIR = Path("model")
    MODEL_DIR.mkdir(exist_ok=True)

    ir_model_path = MODEL_DIR / "ir_model.xml"
    quantized_model_path = MODEL_DIR / "quantized_model.xml"

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
    [ INFO ] Read model took 15.23 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_8) : f32 / [...] / [?,116,21..]
    [ INFO ]     input.255 (node: __module.model.23.cv4.2.1.act/aten::silu_/Swish_46) : f32 / [...] / [?,32,8..,8..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 7.93 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_8) : f32 / [...] / [1,116,8400]
    [ INFO ]     input.255 (node: __module.model.23.cv4.2.1.act/aten::silu_/Swish_46) : f32 / [...] / [1,32,160,160]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 389.72 ms
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
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 36
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
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
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 45.30 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            17580 iterations
    [ INFO ] Duration:         120085.24 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        80.74 ms
    [ INFO ]    Average:       81.80 ms
    [ INFO ]    Min:           59.61 ms
    [ INFO ]    Max:           151.88 ms
    [ INFO ] Throughput:   146.40 FPS


.. code:: ipython3

    # Inference Quantized model (OpenVINO IR)
    ! benchmark_app -m $quantized_model_path -shape "[1,3,640,640]" -d $device.value -api async


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
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
    [ INFO ] Read model took 25.39 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_8) : f32 / [...] / [1,116,8400]
    [ INFO ]     input.255 (node: __module.model.23.cv4.2.1.act/aten::silu_/Swish_46) : f32 / [...] / [1,32,160,160]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 0.05 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_8) : f32 / [...] / [1,116,8400]
    [ INFO ]     input.255 (node: __module.model.23.cv4.2.1.act/aten::silu_/Swish_46) : f32 / [...] / [1,32,160,160]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 639.12 ms
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
    [ INFO ]     DYNAMIC_QUANTIZATION_GROUP_SIZE: 32
    [ INFO ]     ENABLE_CPU_PINNING: True
    [ INFO ]     ENABLE_HYPER_THREADING: True
    [ INFO ]     EXECUTION_DEVICES: ['CPU']
    [ INFO ]     EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]     INFERENCE_NUM_THREADS: 36
    [ INFO ]     INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]     KV_CACHE_PRECISION: <Type: 'float16'>
    [ INFO ]     LOG_LEVEL: Level.NO
    [ INFO ]     MODEL_DISTRIBUTION_POLICY: set()
    [ INFO ]     NETWORK_NAME: Model0
    [ INFO ]     NUM_STREAMS: 12
    [ INFO ]     OPTIMAL_NUMBER_OF_INFER_REQUESTS: 12
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
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 33.20 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            38424 iterations
    [ INFO ] Duration:         120039.54 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        37.07 ms
    [ INFO ]    Average:       37.32 ms
    [ INFO ]    Min:           22.57 ms
    [ INFO ]    Max:           72.28 ms
    [ INFO ] Throughput:   320.09 FPS

