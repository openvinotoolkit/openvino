Convert and Optimize YOLOv9 with OpenVINO™
==========================================

YOLOv9 marks a significant advancement in real-time object detection,
introducing groundbreaking techniques such as Programmable Gradient
Information (PGI) and the Generalized Efficient Layer Aggregation
Network (GELAN). This model demonstrates remarkable improvements in
efficiency, accuracy, and adaptability, setting new benchmarks on the MS
COCO dataset. More details about model can be found in
`paper <https://arxiv.org/abs/2402.13616>`__ and `original
repository <https://github.com/WongKinYiu/yolov9>`__ This tutorial
demonstrates step-by-step instructions on how to run and optimize
PyTorch YOLO V9 with OpenVINO.

The tutorial consists of the following steps:

-  Prepare PyTorch model
-  Convert PyTorch model to OpenVINO IR
-  Run model inference with OpenVINO
-  Prepare and run optimization pipeline
-  Compare performance of the FP32 and quantized models.
-  Run optimized model inference on video


**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Get PyTorch model <#get-pytorch-model>`__
-  `Convert PyTorch model to OpenVINO
   IR <#convert-pytorch-model-to-openvino-ir>`__
-  `Verify model inference <#verify-model-inference>`__

   -  `Preprocessing <#preprocessing>`__
   -  `Postprocessing <#postprocessing>`__
   -  `Select inference device <#select-inference-device>`__

-  `Optimize model using NNCF Post-training Quantization
   API <#optimize-model-using-nncf-post-training-quantization-api>`__

   -  `Prepare dataset <#prepare-dataset>`__
   -  `Perform model quantization <#perform-model-quantization>`__

-  `Run quantized model inference <#run-quantized-model-inference>`__
-  `Compare Performance of the Original and Quantized
   Models <#compare-performance-of-the-original-and-quantized-models>`__
-  `Run Live Object Detection <#run-live-object-detection>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------


.. code:: ipython3

    %pip install -q "openvino>=2023.3.0" "nncf>=2.8.1" "opencv-python" "matplotlib>=3.4" "seaborn" "pandas" "scikit-learn" "torch" "torchvision" "tqdm"  --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)
    from notebook_utils import download_file, VideoPlayer, device_widget

    if not Path("yolov9").exists():
        !git clone https://github.com/WongKinYiu/yolov9
    %cd yolov9


.. parsed-literal::

    Cloning into 'yolov9'...
    remote: Enumerating objects: 781, done.[K
    remote: Total 781 (delta 0), reused 0 (delta 0), pack-reused 781 (from 1)[K
    Receiving objects: 100% (781/781), 3.27 MiB | 19.93 MiB/s, done.
    Resolving deltas: 100% (331/331), done.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9


Get PyTorch model
-----------------



Generally, PyTorch models represent an instance of the
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__
class, initialized by a state dictionary with model weights. We will use
the ``gelan-c`` (light-weight version of yolov9) model pre-trained on a
COCO dataset, which is available in this
`repo <https://github.com/WongKinYiu/yolov9>`__, but the same steps are
applicable for other models from YOLO V9 family.

.. code:: ipython3

    # Download pre-trained model weights
    MODEL_LINK = "https://github.com/WongKinYiu/yolov9/releases/download/v0.1/gelan-c.pt"
    DATA_DIR = Path("data/")
    MODEL_DIR = Path("model/")
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)

    download_file(MODEL_LINK, directory=MODEL_DIR, show_progress=True)



.. parsed-literal::

    model/gelan-c.pt:   0%|          | 0.00/49.1M [00:00<?, ?B/s]




.. parsed-literal::

    PosixPath('/opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9/model/gelan-c.pt')



Convert PyTorch model to OpenVINO IR
------------------------------------



OpenVINO supports PyTorch model conversion via Model Conversion API.
``ov.convert_model`` function accepts model object and example input for
tracing the model and returns an instance of ``ov.Model``, representing
this model in OpenVINO format. The Obtained model is ready for loading
on specific devices or can be saved on disk for the next deployment
using ``ov.save_model``.

.. code:: ipython3

    from models.experimental import attempt_load
    import torch
    import openvino as ov
    from models.yolo import Detect, DualDDetect
    from utils.general import yaml_save, yaml_load

    weights = MODEL_DIR / "gelan-c.pt"
    ov_model_path = MODEL_DIR / weights.name.replace(".pt", "_openvino_model") / weights.name.replace(".pt", ".xml")

    if not ov_model_path.exists():
        model = attempt_load(weights, device="cpu", inplace=True, fuse=True)
        metadata = {"stride": int(max(model.stride)), "names": model.names}

        model.eval()
        for k, m in model.named_modules():
            if isinstance(m, (Detect, DualDDetect)):
                m.inplace = False
                m.dynamic = True
                m.export = True

        example_input = torch.zeros((1, 3, 640, 640))
        model(example_input)

        ov_model = ov.convert_model(model, example_input=example_input)

        # specify input and output names for compatibility with yolov9 repo interface
        ov_model.outputs[0].get_tensor().set_names({"output0"})
        ov_model.inputs[0].get_tensor().set_names({"images"})
        ov.save_model(ov_model, ov_model_path)
        # save metadata
        yaml_save(ov_model_path.parent / weights.name.replace(".pt", ".yaml"), metadata)
    else:
        metadata = yaml_load(ov_model_path.parent / weights.name.replace(".pt", ".yaml"))


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9/models/experimental.py:243: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
    Fusing layers...
    Model summary: 387 layers, 25288768 parameters, 0 gradients, 102.1 GFLOPs
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/yolov9-optimization/yolov9/models/yolo.py:108: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      elif self.dynamic or self.shape != shape:


Verify model inference
----------------------



To test model work, we create inference pipeline similar to
``detect.py``. The pipeline consists of preprocessing step, inference of
OpenVINO model, and results post-processing to get bounding boxes.

Preprocessing
~~~~~~~~~~~~~



Model input is a tensor with the ``[1, 3, 640, 640]`` shape in
``N, C, H, W`` format, where

-  ``N`` - number of images in batch (batch size)
-  ``C`` - image channels
-  ``H`` - image height
-  ``W`` - image width

Model expects images in RGB channels format and normalized in [0, 1]
range. To resize images to fit model size ``letterbox`` resize approach
is used where the aspect ratio of width and height is preserved. It is
defined in yolov9 repository.

To keep specific shape, preprocessing automatically enables padding.

.. code:: ipython3

    import numpy as np
    import torch
    from PIL import Image
    from utils.augmentations import letterbox

    image_url = "https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/7b6af406-4ccb-4ded-a13d-62b7c0e42e96"
    download_file(image_url, directory=DATA_DIR, filename="test_image.jpg", show_progress=True)


    def preprocess_image(img0: np.ndarray):
        """
        Preprocess image according to YOLOv9 input requirements.
        Takes image in np.array format, resizes it to specific size using letterbox resize, converts color space from BGR (default in OpenCV) to RGB and changes data layout from HWC to CHW.

        Parameters:
          img0 (np.ndarray): image for preprocessing
        Returns:
          img (np.ndarray): image after preprocessing
          img0 (np.ndarray): original image
        """
        # resize
        img = letterbox(img0, auto=False)[0]

        # Convert
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        return img, img0


    def prepare_input_tensor(image: np.ndarray):
        """
        Converts preprocessed image to tensor format according to YOLOv9 input requirements.
        Takes image in np.array format with unit8 data in [0, 255] range and converts it to torch.Tensor object with float data in [0, 1] range

        Parameters:
          image (np.ndarray): image for conversion to tensor
        Returns:
          input_tensor (torch.Tensor): float tensor ready to use for YOLOv9 inference
        """
        input_tensor = image.astype(np.float32)  # uint8 to fp16/32
        input_tensor /= 255.0  # 0 - 255 to 0.0 - 1.0

        if input_tensor.ndim == 3:
            input_tensor = np.expand_dims(input_tensor, 0)
        return input_tensor


    NAMES = metadata["names"]



.. parsed-literal::

    data/test_image.jpg:   0%|          | 0.00/101k [00:00<?, ?B/s]


Postprocessing
~~~~~~~~~~~~~~



Model output contains detection boxes candidates. It is a tensor with
the ``[1,25200,85]`` shape in the ``B, N, 85`` format, where:

-  ``B`` - batch size
-  ``N`` - number of detection boxes

Detection box has the [``x``, ``y``, ``h``, ``w``, ``box_score``,
``class_no_1``, …, ``class_no_80``] format, where:

-  (``x``, ``y``) - raw coordinates of box center
-  ``h``, ``w`` - raw height and width of box
-  ``box_score`` - confidence of detection box
-  ``class_no_1``, …, ``class_no_80`` - probability distribution over
   the classes.

For getting final prediction, we need to apply non maximum suppression
algorithm and rescale boxes coordinates to original image size.

.. code:: ipython3

    from utils.plots import Annotator, colors

    from typing import List, Tuple
    from utils.general import scale_boxes, non_max_suppression


    def detect(
        model: ov.Model,
        image_path: Path,
        conf_thres: float = 0.25,
        iou_thres: float = 0.45,
        classes: List[int] = None,
        agnostic_nms: bool = False,
    ):
        """
        OpenVINO YOLOv9 model inference function. Reads image, preprocess it, runs model inference and postprocess results using NMS.
        Parameters:
            model (Model): OpenVINO compiled model.
            image_path (Path): input image path.
            conf_thres (float, *optional*, 0.25): minimal accepted confidence for object filtering
            iou_thres (float, *optional*, 0.45): minimal overlap score for removing objects duplicates in NMS
            classes (List[int], *optional*, None): labels for prediction filtering, if not provided all predicted labels will be used
            agnostic_nms (bool, *optional*, False): apply class agnostic NMS approach or not
        Returns:
           pred (List): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
           orig_img (np.ndarray): image before preprocessing, can be used for results visualization
           inpjut_shape (Tuple[int]): shape of model input tensor, can be used for output rescaling
        """
        if isinstance(image_path, np.ndarray):
            img = image_path
        else:
            img = np.array(Image.open(image_path))
        preprocessed_img, orig_img = preprocess_image(img)
        input_tensor = prepare_input_tensor(preprocessed_img)
        predictions = torch.from_numpy(model(input_tensor)[0])
        pred = non_max_suppression(predictions, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
        return pred, orig_img, input_tensor.shape


    def draw_boxes(
        predictions: np.ndarray,
        input_shape: Tuple[int],
        image: np.ndarray,
        names: List[str],
    ):
        """
        Utility function for drawing predicted bounding boxes on image
        Parameters:
            predictions (np.ndarray): list of detections with (n,6) shape, where n - number of detected boxes in format [x1, y1, x2, y2, score, label]
            image (np.ndarray): image for boxes visualization
            names (List[str]): list of names for each class in dataset
            colors (Dict[str, int]): mapping between class name and drawing color
        Returns:
            image (np.ndarray): box visualization result
        """
        if not len(predictions):
            return image

        annotator = Annotator(image, line_width=1, example=str(names))
        # Rescale boxes from input size to original image size
        predictions[:, :4] = scale_boxes(input_shape[2:], predictions[:, :4], image.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(predictions):
            label = f"{names[int(cls)]} {conf:.2f}"
            annotator.box_label(xyxy, label, color=colors(int(cls), True))
        return image

.. code:: ipython3

    core = ov.Core()
    # read converted model
    ov_model = core.read_model(ov_model_path)

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device = device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    # load model on selected device
    if device.value != "CPU":
        ov_model.reshape({0: [1, 3, 640, 640]})
    compiled_model = core.compile_model(ov_model, device.value)

.. code:: ipython3

    boxes, image, input_shape = detect(compiled_model, DATA_DIR / "test_image.jpg")
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES)
    # visualize results
    Image.fromarray(image_with_boxes)




.. image:: yolov9-optimization-with-output_files/yolov9-optimization-with-output_16_0.png



Optimize model using NNCF Post-training Quantization API
--------------------------------------------------------



`NNCF <https://github.com/openvinotoolkit/nncf>`__ provides a suite of
advanced algorithms for Neural Networks inference optimization in
OpenVINO with minimal accuracy drop. We will use 8-bit quantization in
post-training mode (without the fine-tuning pipeline) to optimize
YOLOv9. The optimization process contains the following steps:

1. Create a Dataset for quantization.
2. Run ``nncf.quantize`` for getting an optimized model.
3. Serialize an OpenVINO IR model, using the ``ov.save_model`` function.

Prepare dataset
~~~~~~~~~~~~~~~



The code below downloads COCO dataset and prepares a dataloader that is
used to evaluate the yolov9 model accuracy. We reuse its subset for
quantization.

.. code:: ipython3

    from zipfile import ZipFile


    DATA_URL = "http://images.cocodataset.org/zips/val2017.zip"
    LABELS_URL = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels-segments.zip"

    OUT_DIR = Path(".")

    download_file(DATA_URL, directory=OUT_DIR, show_progress=True)
    download_file(LABELS_URL, directory=OUT_DIR, show_progress=True)

    if not (OUT_DIR / "coco/labels").exists():
        with ZipFile("coco2017labels-segments.zip", "r") as zip_ref:
            zip_ref.extractall(OUT_DIR)
        with ZipFile("val2017.zip", "r") as zip_ref:
            zip_ref.extractall(OUT_DIR / "coco/images")



.. parsed-literal::

    val2017.zip:   0%|          | 0.00/778M [00:00<?, ?B/s]



.. parsed-literal::

    coco2017labels-segments.zip:   0%|          | 0.00/169M [00:00<?, ?B/s]


.. code:: ipython3

    from collections import namedtuple
    import yaml
    from utils.dataloaders import create_dataloader
    from utils.general import colorstr

    # read dataset config
    DATA_CONFIG = "data/coco.yaml"
    with open(DATA_CONFIG) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Dataloader
    TASK = "val"  # path to train/val/test images
    Option = namedtuple("Options", ["single_cls"])  # imitation of commandline provided options for single class evaluation
    opt = Option(False)
    dataloader = create_dataloader(
        str(Path("coco") / data[TASK]),
        640,
        1,
        32,
        opt,
        pad=0.5,
        prefix=colorstr(f"{TASK}: "),
    )[0]


.. parsed-literal::

    val: Scanning coco/val2017... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 00:00
    val: New cache created: coco/val2017.cache


NNCF provides ``nncf.Dataset`` wrapper for using native framework
dataloaders in quantization pipeline. Additionally, we specify transform
function that will be responsible for preparing input data in model
expected format.

.. code:: ipython3

    import nncf


    def transform_fn(data_item):
        """
        Quantization transform function. Extracts and preprocess input data from dataloader item for quantization.
        Parameters:
           data_item: Tuple with data item produced by DataLoader during iteration
        Returns:
            input_tensor: Input data for quantization
        """
        img = data_item[0].numpy()
        input_tensor = prepare_input_tensor(img)
        return input_tensor


    quantization_dataset = nncf.Dataset(dataloader, transform_fn)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Perform model quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~



The ``nncf.quantize`` function provides an interface for model
quantization. It requires an instance of the OpenVINO Model and
quantization dataset. Optionally, some additional parameters for the
configuration quantization process (number of samples for quantization,
preset, ignored scope etc.) can be provided. YOLOv9 model contains
non-ReLU activation functions, which require asymmetric quantization of
activations. To achieve better results, we will use a ``mixed``
quantization preset. It provides symmetric quantization of weights and
asymmetric quantization of activations.

.. code:: ipython3

    ov_int8_model_path = MODEL_DIR / weights.name.replace(".pt", "_int8_openvino_model") / weights.name.replace(".pt", "_int8.xml")

    if not ov_int8_model_path.exists():
        quantized_model = nncf.quantize(ov_model, quantization_dataset, preset=nncf.QuantizationPreset.MIXED)

        ov.save_model(quantized_model, ov_int8_model_path)
        yaml_save(ov_int8_model_path.parent / weights.name.replace(".pt", "_int8.yaml"), metadata)


.. parsed-literal::

    2024-10-23 05:32:14.632848: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 05:32:14.668324: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 05:32:15.276616: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Output()










.. parsed-literal::

    Output()









Run quantized model inference
-----------------------------



There are no changes in model usage after applying quantization. Let’s
check the model work on the previously used image.

.. code:: ipython3

    quantized_model = core.read_model(ov_int8_model_path)

    if device.value != "CPU":
        quantized_model.reshape({0: [1, 3, 640, 640]})

    compiled_model = core.compile_model(quantized_model, device.value)

.. code:: ipython3

    boxes, image, input_shape = detect(compiled_model, DATA_DIR / "test_image.jpg")
    image_with_boxes = draw_boxes(boxes[0], input_shape, image, NAMES)
    # visualize results
    Image.fromarray(image_with_boxes)




.. image:: yolov9-optimization-with-output_files/yolov9-optimization-with-output_27_0.png



Compare Performance of the Original and Quantized Models
--------------------------------------------------------



We use the OpenVINO `Benchmark
Tool <https://docs.openvino.ai/2024/learn-openvino/openvino-samples/benchmark-tool.html>`__
to measure the inference performance of the ``FP32`` and ``INT8``
models.

   **NOTE**: For more accurate performance, it is recommended to run
   ``benchmark_app`` in a terminal/command prompt after closing other
   applications. Run ``benchmark_app -m model.xml -d CPU`` to benchmark
   async inference on CPU for one minute. Change ``CPU`` to ``GPU`` to
   benchmark on GPU. Run ``benchmark_app --help`` to see an overview of
   all command-line options.

.. code:: ipython3

    !benchmark_app -m $ov_model_path -shape "[1,3,640,640]" -d $device.value -api async -t 15


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
    [ INFO ] Read model took 26.16 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : f32 / [...] / [?,3,?,?]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [?,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [?,144,4..,4..]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [?,144,2..,2..]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [?,144,1..,1..]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 7.97 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 475.37 ms
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
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 182.45 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            228 iterations
    [ INFO ] Duration:         15670.39 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        412.82 ms
    [ INFO ]    Average:       410.86 ms
    [ INFO ]    Min:           309.65 ms
    [ INFO ]    Max:           431.51 ms
    [ INFO ] Throughput:   14.55 FPS


.. code:: ipython3

    !benchmark_app -m $ov_int8_model_path -shape "[1,3,640,640]" -d $device.value -api async -t 15


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
    [ INFO ] Read model took 41.07 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [ INFO ] Reshaping model: 'images': [1,3,640,640]
    [ INFO ] Reshape model took 0.05 ms
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: x) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: __module.model.22/aten::cat/Concat_5) : f32 / [...] / [1,84,8400]
    [ INFO ]     xi.1 (node: __module.model.22/aten::cat/Concat_2) : f32 / [...] / [1,144,80,80]
    [ INFO ]     xi.3 (node: __module.model.22/aten::cat/Concat_1) : f32 / [...] / [1,144,40,40]
    [ INFO ]     xi (node: __module.model.22/aten::cat/Concat) : f32 / [...] / [1,144,20,20]
    [Step 7/11] Loading the model to the device
    [ INFO ] Compile model took 943.07 ms
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
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).
    [ INFO ] First inference took 64.84 ms
    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            720 iterations
    [ INFO ] Duration:         15158.98 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        119.92 ms
    [ INFO ]    Average:       125.90 ms
    [ INFO ]    Min:           80.89 ms
    [ INFO ]    Max:           277.84 ms
    [ INFO ] Throughput:   47.50 FPS


Run Live Object Detection
-------------------------



.. code:: ipython3

    import collections
    import time
    from IPython import display
    import cv2


    # Main processing function to run object detection.
    def run_object_detection(
        source=0,
        flip=False,
        use_popup=False,
        skip_first_frames=0,
        model=ov_model,
        device=device.value,
    ):
        player = None
        compiled_model = core.compile_model(model, device)
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
                # Get the results.
                input_image = np.array(frame)

                start_time = time.time()
                # model expects RGB image, while video capturing in BGR
                detections, _, input_shape = detect(compiled_model, input_image[:, :, ::-1])
                stop_time = time.time()

                image_with_boxes = draw_boxes(detections[0], input_shape, input_image, NAMES)
                frame = image_with_boxes

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
                    # Create an IPython image.⬆️
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

Use a webcam as the video input. By default, the primary webcam is set
with \ ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set \ ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering,
set \ ``use_popup=True``.

   **NOTE**: To use this notebook with a webcam, you need to run the
   notebook on a computer with a webcam. If you run the notebook on a
   remote server (for example, in Binder or Google Colab service), the
   webcam will not work. By default, the lower cell will run model
   inference on a video file. If you want to try live inference on your
   webcam set ``WEBCAM_INFERENCE = True``

Run the object detection:

.. code:: ipython3

    WEBCAM_INFERENCE = False

    if WEBCAM_INFERENCE:
        VIDEO_SOURCE = 0  # Webcam
    else:
        VIDEO_SOURCE = "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/video/people.mp4"

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    quantized_model = core.read_model(ov_int8_model_path)

    run_object_detection(
        source=VIDEO_SOURCE,
        flip=True,
        use_popup=False,
        model=quantized_model,
        device=device.value,
    )



.. image:: yolov9-optimization-with-output_files/yolov9-optimization-with-output_36_0.png


.. parsed-literal::

    Source ended

