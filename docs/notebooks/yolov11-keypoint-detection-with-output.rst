Convert and Optimize YOLOv11 keypoint detection model with OpenVINO™
====================================================================

Keypoint detection/Pose is a task that involves detecting specific
points in an image or video frame. These points are referred to as
keypoints and are used to track movement or pose estimation. YOLOv11 can
detect keypoints in an image or video frame with high accuracy and
speed.

This tutorial demonstrates step-by-step instructions on how to run and
optimize `PyTorch YOLOv11 Pose
model <https://docs.ultralytics.com/tasks/pose/>`__ with OpenVINO. We
consider the steps required for keypoint detection scenario. You can
find more details about model on `model
page <https://docs.ultralytics.com/models/yolo11/>`__ in Ultralytics
documentation.

The tutorial consists of the following steps: - Prepare the PyTorch
model. - Download and prepare a dataset. - Validate the original model.
- Convert the PyTorch model to OpenVINO IR. - Validate the converted
model. - Prepare and run optimization pipeline. - Compare performance of
the FP32 and quantized models. - Compare accuracy of the FP32 and
quantized models. - Live demo


**Table of contents:**


-  `Get PyTorch model <#get-pytorch-model>`__

   -  `Prerequisites <#prerequisites>`__

-  `Instantiate model <#instantiate-model>`__

   -  `Convert model to OpenVINO IR <#convert-model-to-openvino-ir>`__
   -  `Verify model inference <#verify-model-inference>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Test on single image <#test-on-single-image>`__

-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__

   -  `Validate Quantized model
      inference <#validate-quantized-model-inference>`__

-  `Compare the Original and Quantized
   Models <#compare-the-original-and-quantized-models>`__

   -  `Compare performance of the Original and Quantized
      Models <#compare-performance-of-the-original-and-quantized-models>`__

-  `Other ways to optimize model <#other-ways-to-optimize-model>`__
-  `Live demo <#live-demo>`__

   -  `Run Keypoint Detection on
      video <#run-keypoint-detection-on-video>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Get PyTorch model
-----------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the YOLOv11 nano model (also known as ``yolo11n-pose``) pre-trained on a
COCO dataset, which is available in this
`repo <https://github.com/ultralytics/ultralytics>`__. Similar steps are
also applicable to other Ultralytics models. Typical steps to obtain a
pre-trained model: 1. Create an instance of a model class. 2. Load a
checkpoint state dict, which contains the pre-trained model weights. 3.
Turn the model to evaluation for switching some operations to inference
mode.

In this case, the creators of the model provide an API that enables
converting the YOLOv11 model to OpenVINO IR. Therefore, we do not need
to do these steps manually.

Prerequisites
^^^^^^^^^^^^^



Install necessary packages.

.. code:: ipython3

    %pip install -q "openvino>=2024.0.0" "nncf>=2.9.0"
    %pip install -q "protobuf==3.20.*" "torch>=2.1" "torchvision>=0.16" "ultralytics==8.3.0" tqdm opencv-python --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


Import required utility functions. The lower cell will download the
``notebook_utils`` Python module from GitHub.

.. code:: ipython3

    from pathlib import Path
    
    # Fetch `notebook_utils` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    
    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, VideoPlayer, device_widget

.. code:: ipython3

    # Download a test sample
    IMAGE_PATH = Path("./data/intel_rnb.jpg")
    download_file(
        url="https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/intel_rnb.jpg",
        filename=IMAGE_PATH.name,
        directory=IMAGE_PATH.parent,
    )



.. parsed-literal::

    data/intel_rnb.jpg:   0%|          | 0.00/288k [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov11-optimization/data/intel_rnb.jpg')



Instantiate model
-----------------



For loading the model, required to specify a path to the model
checkpoint. It can be some local path or name available on models hub
(in this case model checkpoint will be downloaded automatically).

Making prediction, the model accepts a path to input image and returns
list with Results class object. Results contains boxes and key points.
Also it contains utilities for processing results, for example,
``plot()`` method for drawing.

Let us consider the examples:

.. code:: ipython3

    import ipywidgets as widgets
    
    model_id = [
        "yolo11n-pose",
        "yolo11s-pose",
        "yolo11m-pose",
        "yolo11l-pose",
        "yolo11x-pose",
        "yolov8n-pose",
        "yolov8s-pose",
        "yolov8m-pose",
        "yolov8l-pose",
        "yolov8x-pose",
    ]
    
    model_name = widgets.Dropdown(options=model_id, value=model_id[0], description="Model")
    
    model_name




.. parsed-literal::

    Dropdown(description='Model', options=('yolo11n-pose', 'yolo11s-pose', 'yolo11m-pose', 'yolo11l-pose', 'yolo11…



.. code:: ipython3

    from PIL import Image
    from ultralytics import YOLO
    
    POSE_MODEL_NAME = model_name.value
    
    pose_model = YOLO(f"{POSE_MODEL_NAME}.pt")
    label_map = pose_model.model.names
    
    res = pose_model(IMAGE_PATH)
    Image.fromarray(res[0].plot()[:, :, ::-1])


.. parsed-literal::

    Downloading https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt to 'yolo11n-pose.pt'...


.. parsed-literal::

    100%|██████████| 5.97M/5.97M [00:00<00:00, 25.3MB/s]


.. parsed-literal::

    
    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov11-optimization/data/intel_rnb.jpg: 480x640 1 person, 59.6ms
    Speed: 2.1ms preprocess, 59.6ms inference, 0.9ms postprocess per image at shape (1, 3, 480, 640)




.. image:: yolov11-keypoint-detection-with-output_files/yolov11-keypoint-detection-with-output_9_3.png



Convert model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Ultralytics provides API for convenient model exporting to different
formats including OpenVINO IR. ``model.export`` is responsible for model
conversion. We need to specify the format, and additionally, we can
preserve dynamic shapes in the model.

.. code:: ipython3

    # object detection model
    pose_model_path = Path(f"{POSE_MODEL_NAME}_openvino_model/{POSE_MODEL_NAME}.xml")
    if not pose_model_path.exists():
        pose_model.export(format="openvino", dynamic=True, half=True)


.. parsed-literal::

    Ultralytics 8.3.0 🚀 Python-3.8.10 torch-2.4.1+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)
    
    PyTorch: starting from 'yolo11n-pose.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 56, 8400) (6.0 MB)
    
    OpenVINO: starting export with openvino 2024.5.0-16993-9c432a3641a...
    OpenVINO: export success ✅ 2.0s, saved as 'yolo11n-pose_openvino_model/' (6.0 MB)
    
    Export complete (2.1s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov11-optimization
    Predict:         yolo predict task=pose model=yolo11n-pose_openvino_model imgsz=640 half 
    Validate:        yolo val task=pose model=yolo11n-pose_openvino_model imgsz=640 data=/ultralytics/ultralytics/cfg/datasets/coco-pose.yaml half 
    Visualize:       https://netron.app


Verify model inference
~~~~~~~~~~~~~~~~~~~~~~



We can reuse the base model pipeline for pre- and postprocessing just
replacing the inference method where we will use the IR model for
inference.

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



Select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Test on single image
~~~~~~~~~~~~~~~~~~~~



Now, once we have defined preprocessing and postprocessing steps, we are
ready to check model prediction.

.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    pose_ov_model = core.read_model(pose_model_path)
    
    ov_config = {}
    if device.value != "CPU":
        pose_ov_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    pose_compiled_model = core.compile_model(pose_ov_model, device.value, ov_config)
    
    pose_model = YOLO(pose_model_path.parent, task="pose")
    
    if pose_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**pose_model.overrides, **custom}
        pose_model.predictor = pose_model._smart_load("predictor")(overrides=args, _callbacks=pose_model.callbacks)
        pose_model.predictor.setup_model(model=pose_model.model)
    
    pose_model.predictor.model.ov_compiled_model = pose_compiled_model
    
    
    res = pose_model(IMAGE_PATH)
    Image.fromarray(res[0].plot()[:, :, ::-1])


.. parsed-literal::

    Ultralytics 8.3.0 🚀 Python-3.8.10 torch-2.4.1+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)
    Loading yolo11n-pose_openvino_model for OpenVINO inference...
    Using OpenVINO LATENCY mode for batch=1 inference...
    
    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov11-optimization/data/intel_rnb.jpg: 640x640 1 person, 19.9ms
    Speed: 2.3ms preprocess, 19.9ms inference, 0.9ms postprocess per image at shape (1, 3, 640, 640)




.. image:: yolov11-keypoint-detection-with-output_files/yolov11-keypoint-detection-with-output_16_1.png



Great! The result is the same, as produced by original models.

Check model accuracy on the dataset
-----------------------------------



For comparing the optimized model result with the original, it is good
to know some measurable results in terms of model accuracy on the
validation dataset.

Download the validation dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



YOLOv8 is pre-trained on the COCO dataset, so to evaluate the model
accuracy we need to download it. According to the instructions provided
in the YOLOv8 repo, we also need to download annotations in the format
used by the author of the model, for use with the original model
evaluation function.

   **Note**: The initial dataset download may take a few minutes to
   complete. The download speed will vary depending on the quality of
   your internet connection.

Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
YOLOv8.

The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize OpenVINO IR model, using the ``openvino.runtime.serialize``
   function.

Please select below whether you would like to run quantization to
improve model inference speed.

.. code:: ipython3

    import ipywidgets as widgets
    
    int8_model_pose_path = Path(f"{POSE_MODEL_NAME}_openvino_int8_model/{POSE_MODEL_NAME}.xml")
    quantized_pose_model = None
    
    to_quantize = widgets.Checkbox(
        value=True,
        description="Quantization",
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



Let’s load ``skip magic`` extension to skip quantization if
``to_quantize`` is not selected

.. code:: ipython3

    # Fetch skip_kernel_extension module
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    %load_ext skip_kernel_extension

Reuse validation dataloader in accuracy testing for quantization. For
that, it should be wrapped into the ``nncf.Dataset`` object and define a
transformation function for getting only input tensors.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    from typing import Dict
    
    from zipfile import ZipFile
    
    from ultralytics.data.utils import DATASETS_DIR
    from ultralytics.utils import DEFAULT_CFG
    from ultralytics.cfg import get_cfg
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.models.yolo.pose import PoseValidator
    from ultralytics.utils.metrics import OKS_SIGMA
    
    if not int8_model_pose_path.exists():
    
        DATA_URL = "https://ultralytics.com/assets/coco8-pose.zip"
        CFG_URL = "https://raw.githubusercontent.com/ultralytics/ultralytics/v8.1.0/ultralytics/cfg/datasets/coco8-pose.yaml"
    
        OUT_DIR = DATASETS_DIR
    
        DATA_PATH = OUT_DIR / "val2017.zip"
        CFG_PATH = OUT_DIR / "coco8-pose.yaml"
    
        download_file(DATA_URL, DATA_PATH.name, DATA_PATH.parent)
        download_file(CFG_URL, CFG_PATH.name, CFG_PATH.parent)
    
        if not (OUT_DIR / "coco8-pose/labels").exists():
            with ZipFile(DATA_PATH, "r") as zip_ref:
                zip_ref.extractall(OUT_DIR)
    
        args = get_cfg(cfg=DEFAULT_CFG)
        args.data = "coco8-pose.yaml"
    
        pose_validator = PoseValidator(args=args)
        pose_validator.data = check_det_dataset(args.data)
        pose_validator.stride = 32
        pose_data_loader = pose_validator.get_dataloader(OUT_DIR / "coco8-pose", 1)
    
        pose_validator.is_coco = True
        pose_validator.names = label_map
        pose_validator.metrics.names = pose_validator.names
        pose_validator.nc = 1
        pose_validator.sigma = OKS_SIGMA
    
    
        def transform_fn(data_item:Dict):
            """
            Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
            Parameters:
               data_item: Dict with data item produced by DataLoader during iteration
            Returns:
                input_tensor: Input data for quantization
            """
            input_tensor = pose_validator.preprocess(data_item)['img'].numpy()
            return input_tensor
    
    
        quantization_dataset = nncf.Dataset(pose_data_loader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino



.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-785/.workspace/scm/datasets/val2017.zip:   0%|   …



.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-785/.workspace/scm/datasets/coco8-pose.yaml:   0%…


.. parsed-literal::

    val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-785/.workspace/scm/datasets/coco8-pose/labels/train.cache... 8 images, 0 backgrounds, 0 corrupt: 100%|██████████| 8/8 [00:00<?, ?it/s]


The ``nncf.quantize`` function provides an interface for model
quantization. It requires an instance of the OpenVINO Model and
quantization dataset. Optionally, some additional parameters for the
configuration quantization process (number of samples for quantization,
preset, ignored scope, etc.) can be provided. YOLOv8 model contains
non-ReLU activation functions, which require asymmetric quantization of
activations. To achieve a better result, we will use a ``mixed``
quantization preset. It provides symmetric quantization of weights and
asymmetric quantization of activations. For more accurate results, we
should keep the operation in the postprocessing subgraph in floating
point precision, using the ``ignored_scope`` parameter.

   **Note**: Model post-training quantization is time-consuming process.
   Be patient, it can take several minutes depending on your hardware.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if not int8_model_pose_path.exists():
    
        ignored_scope = nncf.IgnoredScope(  # post-processing
            subgraphs=[
                nncf.Subgraph(inputs=[f"__module.model.{22 if 'v8' in POSE_MODEL_NAME else 23}/aten::cat/Concat",
                                      f"__module.model.{22 if 'v8' in POSE_MODEL_NAME else 23}/aten::cat/Concat_1",
                                      f"__module.model.{22 if 'v8' in POSE_MODEL_NAME else 23}/aten::cat/Concat_2",
                                     f"__module.model.{22 if 'v8' in POSE_MODEL_NAME else 23}/aten::cat/Concat_7"],
                              outputs=[f"__module.model.{22 if 'v8' in POSE_MODEL_NAME else 23}/aten::cat/Concat_9"])
            ]
        )
    
        # Detection model
        quantized_pose_model = nncf.quantize(
            pose_ov_model,
            quantization_dataset,
            preset=nncf.QuantizationPreset.MIXED,
            ignored_scope=ignored_scope
        )
        print(f"Quantized keypoint detection model will be saved to {int8_model_pose_path}")
        ov.save_model(quantized_pose_model, str(int8_model_pose_path))


.. parsed-literal::

    INFO:nncf:116 ignored nodes were found by subgraphs in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 140 __module.model.23/aten::cat/Concat
    INFO:nncf:Not adding activation input quantizer for operation: 149 __module.model.23/aten::view/Reshape_3
    INFO:nncf:Not adding activation input quantizer for operation: 297 __module.model.23/aten::cat/Concat_1
    INFO:nncf:Not adding activation input quantizer for operation: 311 __module.model.23/aten::view/Reshape_4
    INFO:nncf:Not adding activation input quantizer for operation: 418 __module.model.23/aten::cat/Concat_2
    INFO:nncf:Not adding activation input quantizer for operation: 422 __module.model.23/aten::view/Reshape_5
    INFO:nncf:Not adding activation input quantizer for operation: 151 __module.model.23/aten::cat/Concat_7
    INFO:nncf:Not adding activation input quantizer for operation: 163 __module.model.23/aten::view/Reshape_9
    INFO:nncf:Not adding activation input quantizer for operation: 177 __module.model.23/aten::slice/Slice_2
    INFO:nncf:Not adding activation input quantizer for operation: 178 __module.model.23/aten::slice/Slice_5
    INFO:nncf:Not adding activation input quantizer for operation: 195 __module.model.23/aten::mul/Multiply_4
    215 __module.model.23/aten::add/Add_8
    
    INFO:nncf:Not adding activation input quantizer for operation: 196 __module.model.23/aten::sigmoid/Sigmoid_1
    INFO:nncf:Not adding activation input quantizer for operation: 232 __module.model.23/aten::mul/Multiply_5
    INFO:nncf:Not adding activation input quantizer for operation: 216 __module.model.23/aten::cat/Concat_8
    INFO:nncf:Not adding activation input quantizer for operation: 233 __module.model.23/aten::view/Reshape_10
    INFO:nncf:Not adding activation input quantizer for operation: 161 __module.model.23/aten::cat/Concat_4
    INFO:nncf:Not adding activation input quantizer for operation: 175 __module.model.23/prim::ListUnpack
    INFO:nncf:Not adding activation input quantizer for operation: 192 __module.model.23.dfl/aten::view/Reshape
    INFO:nncf:Not adding activation input quantizer for operation: 193 __module.model.23/aten::sigmoid/Sigmoid
    INFO:nncf:Not adding activation input quantizer for operation: 212 __module.model.23.dfl/aten::transpose/Transpose
    INFO:nncf:Not adding activation input quantizer for operation: 230 __module.model.23.dfl/aten::softmax/Softmax
    INFO:nncf:Not adding activation input quantizer for operation: 245 __module.model.23.dfl.conv/aten::_convolution/Convolution
    INFO:nncf:Not adding activation input quantizer for operation: 256 __module.model.23.dfl/aten::view/Reshape_1
    INFO:nncf:Not adding activation input quantizer for operation: 268 __module.model.23/prim::ListUnpack/VariadicSplit
    INFO:nncf:Not adding activation input quantizer for operation: 279 __module.model.23/aten::sub/Subtract
    INFO:nncf:Not adding activation input quantizer for operation: 280 __module.model.23/aten::add/Add_6
    INFO:nncf:Not adding activation input quantizer for operation: 291 __module.model.23/aten::add/Add_7
    304 __module.model.23/aten::div/Divide
    
    INFO:nncf:Not adding activation input quantizer for operation: 292 __module.model.23/aten::sub/Subtract_1
    INFO:nncf:Not adding activation input quantizer for operation: 305 __module.model.23/aten::cat/Concat_5
    INFO:nncf:Not adding activation input quantizer for operation: 265 __module.model.23/aten::mul/Multiply_3
    INFO:nncf:Not adding activation input quantizer for operation: 213 __module.model.23/aten::cat/Concat_9



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









.. parsed-literal::

    Quantized keypoint detection model will be saved to yolo11n-pose_openvino_int8_model/yolo11n-pose.xml


Validate Quantized model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



``nncf.quantize`` returns the OpenVINO Model class instance, which is
suitable for loading on a device for making predictions. ``INT8`` model
input data and output result formats have no difference from the
floating point model representation. Therefore, we can reuse the same
``detect`` function defined above for getting the ``INT8`` model result
on the image.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    device

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if quantized_pose_model is None:
        quantized_pose_model = core.read_model()
    
    ov_config = {}
    if device.value != "CPU":
        quantized_pose_model.reshape({0: [1, 3, 640, 640]})
    if "GPU" in device.value or ("AUTO" in device.value and "GPU" in core.available_devices):
        ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
    quantized_pose_compiled_model = core.compile_model(quantized_pose_model, device.value, ov_config)
    
    pose_model = YOLO(pose_model_path.parent, task="pose")
    
    if pose_model.predictor is None:
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
        args = {**pose_model.overrides, **custom}
        pose_model.predictor = pose_model._smart_load("predictor")(overrides=args, _callbacks=pose_model.callbacks)
        pose_model.predictor.setup_model(model=pose_model.model)
    
    pose_model.predictor.model.ov_compiled_model = pose_compiled_model
    
    res = pose_model(IMAGE_PATH)
    display(Image.fromarray(res[0].plot()[:, :, ::-1]))


.. parsed-literal::

    Ultralytics 8.3.0 🚀 Python-3.8.10 torch-2.4.1+cpu CPU (Intel Core(TM) i9-10920X 3.50GHz)
    Loading yolo11n-pose_openvino_model for OpenVINO inference...
    Using OpenVINO LATENCY mode for batch=1 inference...
    
    image 1/1 /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov11-optimization/data/intel_rnb.jpg: 640x640 1 person, 28.0ms
    Speed: 2.0ms preprocess, 28.0ms inference, 1.1ms postprocess per image at shape (1, 3, 640, 640)



.. image:: yolov11-keypoint-detection-with-output_files/yolov11-keypoint-detection-with-output_30_1.png


Compare the Original and Quantized Models
-----------------------------------------



Compare performance of the Original and Quantized Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, use the OpenVINO
`Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
to measure the inference performance of the ``FP32`` and ``INT8``
models.

   **Note**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run
   ``benchmark_app -m <model_path> -d CPU -shape "<input_shape>"`` to
   benchmark async inference on CPU on specific input data shape for one
   minute. Change ``CPU`` to ``GPU`` to benchmark on GPU. Run
   ``benchmark_app --help`` to see an overview of all command-line
   options.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    device

.. code:: ipython3

    if int8_model_pose_path.exists():
        # Inference FP32 model (OpenVINO IR)
        !benchmark_app -m $pose_model_path -d $device.value -api async -shape "[1,3,640,640]"


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ WARNING ] Default duration 120 seconds is used for unknown device AUTO
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 19.55 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_9) : f32 / [...] / [?,56,21..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 8.54 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_9) : f32 / [...] / [1,56,8400]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 329.35 ms
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
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 120000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 33.67 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            17280 iterations
    [ INFO ] Duration:         120049.75 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        40.74 ms
    [ INFO ]    Average:       41.55 ms
    [ INFO ]    Min:           24.20 ms
    [ INFO ]    Max:           98.48 ms
    [ INFO ] Throughput:   143.94 FPS


.. code:: ipython3

    if int8_model_pose_path.exists():
        # Inference INT8 model (OpenVINO IR)
        !benchmark_app -m $int8_model_pose_path -d $device.value -api async -shape "[1,3,640,640]" -t 15


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] Device info:
    [ INFO ] AUTO
    [ INFO ] Build ................................. 2024.5.0-16993-9c432a3641a
    [ INFO ] 
    [ INFO ] 
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(AUTO) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files
    [ INFO ] Read model took 28.39 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_9) : f32 / [...] / [1,56,8400]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'x': [1,3,640,640]
    [ INFO ] Reshape model took 0.04 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     x (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     ***NO_NAME*** (node: __module.model.23/aten::cat/Concat_9) : f32 / [...] / [1,56,8400]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 556.59 ms
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
    [ INFO ]     INFERENCE_NUM_THREADS: 24
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
    [Step 10/11] Measuring performance (Start inference asynchronously, 12 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 30.44 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            5160 iterations
    [ INFO ] Duration:         15031.11 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        34.62 ms
    [ INFO ]    Average:       34.76 ms
    [ INFO ]    Min:           25.46 ms
    [ INFO ]    Max:           51.89 ms
    [ INFO ] Throughput:   343.29 FPS


Compare accuracy of the Original and Quantized Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



As we can see, there is no significant difference between ``INT8`` and
float model result in a single image test. To understand how
quantization influences model prediction precision, we can compare model
accuracy on a dataset.

Other ways to optimize model
----------------------------



The performance could be also improved by another OpenVINO method such
as async inference pipeline or preprocessing API.

Async Inference pipeline help to utilize the device more optimal. The
key advantage of the Async API is that when a device is busy with
inference, the application can perform other tasks in parallel (for
example, populating inputs or scheduling other requests) rather than
wait for the current inference to complete first. To understand how to
perform async inference using openvino, refer to `Async API
tutorial <async-api-with-output.html>`__

Preprocessing API enables making preprocessing a part of the model
reducing application code and dependency on additional image processing
libraries. The main advantage of Preprocessing API is that preprocessing
steps will be integrated into the execution graph and will be performed
on a selected device (CPU/GPU etc.) rather than always being executed on
CPU as part of an application. This will also improve selected device
utilization. For more information, refer to the overview of
`Preprocessing API
tutorial <optimize-preprocessing-with-output.html>`__. To
see, how it could be used with YOLOV8 object detection model , please,
see `Convert and Optimize YOLOv8 real-time object detection with
OpenVINO tutorial <yolov8-object-detection-with-output.html>`__

Live demo
---------



The following code runs model inference on a video:

.. code:: ipython3

    import collections
    import time
    from IPython import display
    import cv2
    import numpy as np
    
    
    def run_keypoint_detection(
        source=0,
        flip=False,
        use_popup=False,
        skip_first_frames=0,
        model=pose_model,
        device=device.value,
    ):
        player = None
    
        ov_config = {}
        if device != "CPU":
            model.reshape({0: [1, 3, 640, 640]})
        if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
            ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
        compiled_model = core.compile_model(model, device, ov_config)
    
        if pose_model.predictor is None:
            custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}  # method defaults
            args = {**seg_model.overrides, **custom}
            pose_model.predictor = pose_model._smart_load("predictor")(overrides=args, _callbacks=pose_model.callbacks)
            pose_model.predictor.setup_model(model=pose_model.model)
    
        pose_model.predictor.model.ov_compiled_model = compiled_model
    
        try:
            # Create a video player to play with target fps.
            player = VideoPlayer(source=source, flip=flip, fps=30, skip_first_frames=skip_first_frames)
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(winname=title, flags=cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
            processing_times = collections.deque()
            while True:
                # Grab the frame.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
                # If the frame is larger than full HD, reduce size to improve the performance.
                scale = 1280 / max(frame.shape)
                if scale < 1:
                    frame = cv2.resize(
                        src=frame,
                        dsize=None,
                        fx=scale,
                        fy=scale,
                        interpolation=cv2.INTER_AREA,
                    )
                # Get the results
                input_image = np.array(frame)
    
                start_time = time.time()
    
                detections = pose_model(input_image)
                stop_time = time.time()
                frame = detections[0].plot()
    
                processing_times.append(stop_time - start_time)
                # Use processing times from last 200 frames.
                if len(processing_times) > 200:
                    processing_times.popleft()
    
                _, f_width = frame.shape[:2]
                # Mean processing time [ms].
                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time
                cv2.putText(
                    img=frame,
                    text=f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    org=(20, 40),
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=f_width / 1000,
                    color=(0, 0, 255),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )
                # Use this workaround if there is flickering.
                if use_popup:
                    cv2.imshow(winname=title, mat=frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(ext=".jpg", img=frame, params=[cv2.IMWRITE_JPEG_QUALITY, 100])
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

Run Keypoint Detection on video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    VIDEO_SOURCE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    run_keypoint_detection(
        source=VIDEO_SOURCE,
        flip=True,
        use_popup=False,
        model=pose_ov_model,
        device=device.value,
    )



.. image:: yolov11-keypoint-detection-with-output_files/yolov11-keypoint-detection-with-output_43_0.png


.. parsed-literal::

    Source ended

