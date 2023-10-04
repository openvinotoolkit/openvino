Migrate quantization from POT API to NNCF API
=============================================

This tutorial demonstrates how to migrate quantization pipeline written
using the OpenVINO `Post-Training Optimization Tool
(POT) <https://docs.openvino.ai/2023.0/pot_introduction.html>`__ to
`NNCF Post-Training Quantization
API <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__.
This tutorial is based on `Ultralytics
YOLOv5 <https://github.com/ultralytics/yolov5>`__ model and additionally
it compares model accuracy between the FP32 precision and quantized INT8
precision models and runs a demo of model inference based on sample code
from `Ultralytics YOLOv5 <https://github.com/ultralytics/yolov5>`__ with
the OpenVINO backend.

The tutorial consists from the following parts:

1. Convert YOLOv5 model to OpenVINO IR.
2. Prepare dataset for quantization.
3. Configure quantization pipeline.
4. Perform model optimization.
5. Compare accuracy FP32 and INT8 models
6. Run model inference demo
7. Compare performance FP32 and INT8 models

**Table of contents:**

- `Preparation <#preparation>`__

  - `Download the YOLOv5 model <#download-the-yolov5-model>`__
  - `Conversion of the YOLOv5 model to OpenVINO <#conversion-of-the-yolov5-model-to-openvino>`__
  - `Imports <#imports>`__

- `Prepare dataset for quantization <#prepare-dataset-for-quantization>`__

  - `Create YOLOv5 DataLoader class for POT <#create-yolov5-dataloader-class-for-pot>`__
  - `Create NNCF Dataset <#create-nncf-dataset>`__

- `Configure quantization pipeline <#configure-quantization-pipeline>`__

  - `Prepare config and pipeline for POT <#prepare-config-and-pipeline-for-pot>`__
  - `Prepare configuration parameters for NNCF <#prepare-configuration-parameters-for-nncf>`__

- `Perform model optimization <#perform-model-optimization>`__

  - `Run quantization using POT <#run-quantization-using-pot>`__
  - `Run quantization using NNCF <#run-quantization-using-nncf>`__

- `Compare accuracy FP32 and INT8 models <#compare-accuracy-fp32-and-int8-models>`__
- `Inference Demo Performance Comparison <#inference-demo-performance-comparison>`__
- `Benchmark <#benchmark>`__
- `References <#references>`__

Preparation
###############################################################################################################################

Download the YOLOv5 model
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    !pip install -q "openvino-dev==2023.1.0.dev20230811" "nncf>=2.5.0"
    !pip install -q psutil "seaborn>=0.11.0" matplotlib numpy onnx

.. code:: ipython3

    import sys
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    from IPython.display import Markdown, display
    
    if not Path("./yolov5/").exists():
        command_download = (
            f'{"git clone https://github.com/ultralytics/yolov5.git -b v7.0"}'
        )
        command_download = " ".join(command_download.split())
        print("Download Ultralytics Yolov5 project source:")
        display(Markdown(f"`{command_download}`"))
        download_res = %sx $command_download
    else:
        print("Ultralytics Yolov5 repo already exists.")


.. parsed-literal::

    Download Ultralytics Yolov5 project source:



``git clone https://github.com/ultralytics/yolov5.git -b v7.0``


Conversion of the YOLOv5 model to OpenVINO
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

There are three variables provided for easy run through all the notebook
cells.

-  ``IMAGE_SIZE`` - the image size for model input.
-  ``MODEL_NAME`` - the model you want to use. It can be either yolov5s,
   yolov5m or yolov5l and so on.
-  ``MODEL_PATH`` - to the path of the model directory in the YOLOv5
   repository.

YOLOv5 ``export.py`` scripts support multiple model formats for
conversion. ONNX is also represented among supported formats. We need to
specify ``--include ONNX`` parameter for exporting. As the result,
directory with the ``{MODEL_NAME}`` name will be created with the
following content:

-  ``{MODEL_NAME}.pt`` - the downloaded pre-trained weight.
-  ``{MODEL_NAME}.onnx`` - the Open Neural Network Exchange (ONNX) is an
   open format, built to represent machine learning models.

.. code:: ipython3

    IMAGE_SIZE = 640
    MODEL_NAME = "yolov5m"
    MODEL_PATH = f"yolov5/{MODEL_NAME}"

.. code:: ipython3

    print("Convert PyTorch model to OpenVINO Model:")
    command_export = f"cd yolov5 && python export.py --weights {MODEL_NAME}/{MODEL_NAME}.pt --imgsz {IMAGE_SIZE} --batch-size 1 --include ONNX"
    display(Markdown(f"`{command_export}`"))
    ! $command_export


.. parsed-literal::

    Convert PyTorch model to OpenVINO Model:



``cd yolov5 && python export.py --weights yolov5m/yolov5m.pt --imgsz 640 --batch-size 1 --include ONNX``


.. parsed-literal::

    export: data=data/coco128.yaml, weights=['yolov5m/yolov5m.pt'], imgsz=[640], batch_size=1, device=cpu, half=False, inplace=False, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=12, verbose=False, workspace=4, nms=False, agnostic_nms=False, topk_per_class=100, topk_all=100, iou_thres=0.45, conf_thres=0.25, include=['ONNX']
    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU
    
    Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt to yolov5m/yolov5m.pt...
    100%|██████████████████████████████████████| 40.8M/40.8M [00:10<00:00, 4.11MB/s]
    
    Fusing layers... 
    YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients
    
    PyTorch: starting from yolov5m/yolov5m.pt with output shape (1, 25200, 85) (40.8 MB)
    
    ONNX: starting export with onnx 1.14.1...
    ONNX: export success ✅ 1.3s, saved as yolov5m/yolov5m.onnx (81.2 MB)
    
    Export complete (13.3s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/yolov5m
    Detect:          python detect.py --weights yolov5m/yolov5m.onnx 
    Validate:        python val.py --weights yolov5m/yolov5m.onnx 
    PyTorch Hub:     model = torch.hub.load('ultralytics/yolov5', 'custom', 'yolov5m/yolov5m.onnx')  
    Visualize:       https://netron.app


Convert the ONNX model to OpenVINO Intermediate Representation (IR)
model generated by `OpenVINO model conversion
API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__.
We will use the ``ov.convert_model`` function of model conversion Python
API to convert ONNX model to OpenVINO Model, then it can be serialized
using ``ov.save_model``. As the result, directory with the
``{MODEL_DIR}`` name will be created with the following content: \*
``{MODEL_NAME}_fp32.xml``, ``{MODEL_NAME}_fp32.bin`` - OpenVINO
Intermediate Representation (IR) model generated by `OpenVINO Model
Converter <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__,
saved with FP32 precision. \* ``{MODEL_NAME}_fp16.xml``,
``{MODEL_NAME}_fp16.bin`` - OpenVINO Intermediate Representation (IR)
model generated by `OpenVINO Model
Converter <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__,
saved with FP16 precision.

.. code:: ipython3

    import openvino as ov
    
    onnx_path = f"{MODEL_PATH}/{MODEL_NAME}.onnx"
    
    # fp32 IR model
    fp32_path = f"{MODEL_PATH}/FP32_openvino_model/{MODEL_NAME}_fp32.xml"
    
    print(f"Export ONNX to OpenVINO FP32 IR to: {fp32_path}")
    model = ov.convert_model(onnx_path)
    ov.save_model(model, fp32_path, compress_to_fp16=False)
    
    # fp16 IR model
    fp16_path = f"{MODEL_PATH}/FP16_openvino_model/{MODEL_NAME}_fp16.xml"
    
    print(f"Export ONNX to OpenVINO FP16 IR to: {fp16_path}")
    model = ov.convert_model(onnx_path)
    ov.save_model(model, fp16_path, compress_to_fp16=True)


.. parsed-literal::

    Export ONNX to OpenVINO FP32 IR to: yolov5/yolov5m/FP32_openvino_model/yolov5m_fp32.xml
    Export ONNX to OpenVINO FP16 IR to: yolov5/yolov5m/FP16_openvino_model/yolov5m_fp16.xml


Imports
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code:: ipython3

    sys.path.append("./yolov5")
    
    from yolov5.utils.dataloaders import create_dataloader
    from yolov5.utils.general import check_dataset

Prepare dataset for quantization
###############################################################################################################################

Before starting quantization, we should prepare dataset, which will be
used for quantization. Ultralytics YOLOv5 provides data loader for
iteration over dataset during training and validation. Let’s create it
first.

.. code:: ipython3

    from yolov5.utils.general import download
    
    DATASET_CONFIG = "./yolov5/data/coco128.yaml"
    
    
    def create_data_source():
        """
        Creates COCO 2017 validation data loader. The method downloads COCO 2017
        dataset if it does not exist.
        """
        if not Path("datasets/coco128").exists():
            urls = ["https://ultralytics.com/assets/coco128.zip"]
            download(urls, dir="datasets")
    
        data = check_dataset(DATASET_CONFIG)
        val_dataloader = create_dataloader(
            data["val"], imgsz=640, batch_size=1, stride=32, pad=0.5, workers=1
        )[0]
    
        return val_dataloader
    
    
    data_source = create_data_source()


.. parsed-literal::

    Downloading https://ultralytics.com/assets/coco128.zip to datasets/coco128.zip...



.. parsed-literal::

      0%|          | 0.00/6.66M [00:00<?, ?B/s]


.. parsed-literal::

    Unzipping datasets/coco128.zip...
    Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00
    New cache created: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache


Create YOLOv5 DataLoader class for POT
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Create a class for loading the YOLOv5 dataset and annotation which
inherits from POT API class DataLoader.
``openvino.tools.pot.DataLoader`` interface allows acquiring data from a
dataset and applying model-specific pre-processing providing access by
index. Any implementation should override the following methods:

-  The ``__len__()``, returns the size of the dataset.

-  The ``__getitem__()``, provides access to the data by index in range
   of 0 to ``len(self)``. It can also encapsulate the logic of
   model-specific pre-processing. This method should return data in the
   (data, annotation) format, in which:

   -  The ``data`` is the input that is passed to the model at inference
      so that it should be properly preprocessed. It can be either the
      ``numpy.array`` object or a dictionary, where the key is the name
      of the model input and value is ``numpy.array`` which corresponds
      to this input.

   -  The ``annotation`` is not used by the Default Quantization method.
      Therefore, this object can be None in this case.

.. code:: ipython3

    from openvino.tools.pot.api import DataLoader
    
    class YOLOv5POTDataLoader(DataLoader):
        """Inherit from DataLoader function and implement for YOLOv5."""
    
        def __init__(self, data_source):
            super().__init__({})
            self._data_loader = data_source
            self._data_iter = iter(self._data_loader)
    
        def __len__(self):
            return len(self._data_loader.dataset)
    
        def __getitem__(self, item):
            try:
                batch_data = next(self._data_iter)
            except StopIteration:
                self._data_iter = iter(self._data_loader)
                batch_data = next(self._data_iter)
    
            im, target, path, shape = batch_data
    
            im = im.float()
            im /= 255
            nb, _, height, width = im.shape
            img = im.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
    
            annotation = dict()
            annotation["image_path"] = path
            annotation["target"] = target
            annotation["batch_size"] = nb
            annotation["shape"] = shape
            annotation["width"] = width
            annotation["height"] = height
            annotation["img"] = img
    
            return (item, annotation), img
    
    pot_data_loader = YOLOv5POTDataLoader(data_source)


.. parsed-literal::

    [ DEBUG ] Creating converter from 7 to 5
    [ DEBUG ] Creating converter from 5 to 7
    [ DEBUG ] Creating converter from 7 to 5
    [ DEBUG ] Creating converter from 5 to 7
    [ WARNING ] /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/accuracy_checker/preprocessor/launcher_preprocessing/ie_preprocessor.py:21: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
      from openvino.inference_engine import ResizeAlgorithm, PreProcessInfo, ColorFormat, MeanVariant  # pylint: disable=import-outside-toplevel,package-absolute-imports
    
    [ WARNING ] /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/accuracy_checker/launcher/dlsdk_launcher.py:60: FutureWarning: OpenVINO nGraph Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
      import ngraph as ng
    


.. parsed-literal::

    Post-training Optimization Tool is deprecated and will be removed in the future. Please use Neural Network Compression Framework instead: https://github.com/openvinotoolkit/nncf
    Nevergrad package could not be imported. If you are planning to use any hyperparameter optimization algo, consider installing it using pip. This implies advanced usage of the tool. Note that nevergrad is compatible only with Python 3.7+


Create NNCF Dataset
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

For preparing quantization dataset for NNCF, we should wrap
framework-specific data source into ``nncf.Dataset`` instance.
Additionally, to transform data into model expected format we can define
transformation function, which accept data item for single dataset
iteration and transform it for feeding into model (e.g. in simplest
case, if data item contains input tensor and annotation, we should
extract only input data from it and convert it into model expected
format).

.. code:: ipython3

    import nncf
    
    # Define the transformation method. This method should take a data item returned
    # per iteration through the `data_source` object and transform it into the model's
    # expected input that can be used for the model inference.
    def transform_fn(data_item):
        # unpack input images tensor
        images = data_item[0]
        # convert input tensor into float format
        images = images.float()
        # scale input
        images = images / 255
        # convert torch tensor to numpy array
        images = images.cpu().detach().numpy()
        return images
    
    # Wrap framework-specific data source into the `nncf.Dataset` object.
    nncf_calibration_dataset = nncf.Dataset(data_source, transform_fn)


.. parsed-literal::

    2023-09-08 22:55:10.459989: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-08 22:55:10.495368: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-08 22:55:11.043305: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Configure quantization pipeline
###############################################################################################################################

Next, we should define quantization algorithm parameters.

Prepare config and pipeline for POT
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

in POT, all quantization parameters should be defined using
configuration dictionary. Config consists of 3 sections: ``algorithms``
for description quantization algorithm parameters, ``engine`` for
description inference pipeline parameters (if required) and ``model``
contains path to floating point model.

.. code:: ipython3

    algorithms_config = [
        {
            "name": "DefaultQuantization",
            "params": {
                "preset": "mixed",
                "stat_subset_size": 300,
                "target_device": "CPU"
            },
        }
    ]
    
    engine_config = {"device": "CPU"}
    
    model_config = {
        "model_name": f"{MODEL_NAME}",
        "model": fp32_path,
        "weights": fp32_path.replace(".xml", ".bin"),
    }

When we define configs, we should create quantization engine class (in
our case, default ``IEEngine`` will be enough) and build quantization
pipeline using ``create_pipeline`` function.

.. code:: ipython3

    from openvino.tools.pot.engines.ie_engine import IEEngine
    from openvino.tools.pot.graph import load_model
    from openvino.tools.pot.pipeline.initializer import create_pipeline
    
    #  Load model as POT model representation
    pot_model = load_model(model_config)
    
    #  Initialize the engine for metric calculation and statistics collection.
    engine = IEEngine(config=engine_config, data_loader=pot_data_loader)
    
    # Step 5: Create a pipeline of compression algorithms.
    pipeline = create_pipeline(algorithms_config, engine)

Prepare configuration parameters for NNCF
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Post-training quantization pipeline in NNCF represented by
``nncf.quantize`` function for Default Quantization Algorithm and
``nncf.quantize_with_accuracy_control`` for Accuracy Aware Quantization.
Quantization parameters ``preset``, ``model_type``, ``subset_size``,
``fast_bias_correction``, ``ignored_scope`` are arguments of function.
More details about supported parameters and formats can be found in NNCF
Post-Training Quantization
`documentation <https://docs.openvino.ai/2023.0/basic_qauntization_flow.html#tune-quantization-parameters>`__.
NNCF also expect providing model object in inference framework format,
in our case ``ov.Model`` instance created using ``core.read_model`` or
``ov.convert_model``.

.. code:: ipython3

    subset_size = 300
    preset = nncf.QuantizationPreset.MIXED

Perform model optimization
###############################################################################################################################

Run quantization using POT
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To start model quantization using POT API, we should call
``pipeline.run(pot_model)`` method. As the result, we got quantized
model representation from POT, which can be saved on disk using
``openvino.tools.pot.graph.save_model`` function. Optionally, we can
compress model weights to quantized precision in order to reduce the
size of final .bin file.

.. code:: ipython3

    from openvino.tools.pot.graph.model_utils import compress_model_weights
    from openvino.tools.pot.graph import load_model, save_model
    
    compressed_model = pipeline.run(pot_model)
    compress_model_weights(compressed_model)
    optimized_save_dir = Path(f"{MODEL_PATH}/POT_INT8_openvino_model/")
    save_model(compressed_model, optimized_save_dir, model_config["model_name"] + "_int8")
    pot_int8_path = f"{optimized_save_dir}/{MODEL_NAME}_int8.xml"

Run quantization using NNCF
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

To run NNCF quantization, we should call ``nncf.quantize`` function. As
the result, the function returns quantized model in the same format like
input model, so it means that quantized model ready to be compiled on
device for inference and can be saved on disk using
``openvino.save_model``.

.. code:: ipython3

    core = ov.Core()
    ov_model = core.read_model(fp32_path)
    quantized_model = nncf.quantize(
        ov_model, nncf_calibration_dataset, preset=preset, subset_size=subset_size
    )
    nncf_int8_path = f"{MODEL_PATH}/NNCF_INT8_openvino_model/{MODEL_NAME}_int8.xml"
    ov.save_model(quantized_model, nncf_int8_path, compress_to_fp16=False)


.. parsed-literal::

    Statistics collection:  43%|████▎     | 128/300 [00:31<00:42,  4.03it/s]
    Biases correction: 100%|██████████| 82/82 [00:10<00:00,  7.60it/s]


Compare accuracy FP32 and INT8 models
###############################################################################################################################

For getting accuracy results, we will use ``yolov5.val.run`` function
which already supports OpenVINO backend. For making int8 model is
compatible with Ultralytics provided validation pipeline, we also should
provide metadata with information about supported class names in the
same directory, where model located.

.. code:: ipython3

    from yolov5.export import attempt_load, yaml_save
    from yolov5.val import run as validation_fn
    
    
    model = attempt_load(
        f"{MODEL_PATH}/{MODEL_NAME}.pt", device="cpu", inplace=True, fuse=True
    ) 
    metadata = {"stride": int(max(model.stride)), "names": model.names}  # model metadata
    yaml_save(Path(nncf_int8_path).with_suffix(".yaml"), metadata)
    yaml_save(Path(pot_int8_path).with_suffix(".yaml"), metadata)
    yaml_save(Path(fp32_path).with_suffix(".yaml"), metadata)


.. parsed-literal::

    Fusing layers... 
    YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients


.. code:: ipython3

    print("Checking the accuracy of the original model:")
    fp32_metrics = validation_fn(
        data=DATASET_CONFIG,
        weights=Path(fp32_path).parent,
        batch_size=1,
        workers=1,
        plots=False,
        device="cpu",
        iou_thres=0.65,
    )
    
    fp32_ap5 = fp32_metrics[0][2]
    fp32_ap_full = fp32_metrics[0][3]
    print(f"mAP@.5 = {fp32_ap5}")
    print(f"mAP@.5:.95 = {fp32_ap_full}")


.. parsed-literal::

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU
    
    Loading yolov5/yolov5m/FP32_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the original model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models
    val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:05
                       all        128        929      0.726      0.687      0.769      0.554
    Speed: 0.2ms pre-process, 35.3ms inference, 3.1ms NMS per image at shape (1, 3, 640, 640)
    Results saved to yolov5/runs/val/exp


.. parsed-literal::

    mAP@.5 = 0.7686009694748247
    mAP@.5:.95 = 0.5541065589219657


.. code:: ipython3

    print("Checking the accuracy of the POT int8 model:")
    int8_metrics = validation_fn(
        data=DATASET_CONFIG,
        weights=Path(pot_int8_path).parent,
        batch_size=1,
        workers=1,
        plots=False,
        device="cpu",
        iou_thres=0.65,
    )
    
    pot_int8_ap5 = int8_metrics[0][2]
    pot_int8_ap_full = int8_metrics[0][3]
    print(f"mAP@.5 = {pot_int8_ap5}")
    print(f"mAP@.5:.95 = {pot_int8_ap_full}")


.. parsed-literal::

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU
    
    Loading yolov5/yolov5m/POT_INT8_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the POT int8 model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models
    val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:03
                       all        128        929      0.761      0.677      0.773      0.548
    Speed: 0.2ms pre-process, 16.9ms inference, 3.2ms NMS per image at shape (1, 3, 640, 640)
    Results saved to yolov5/runs/val/exp2


.. parsed-literal::

    mAP@.5 = 0.7726143212109754
    mAP@.5:.95 = 0.5482902837946336


.. code:: ipython3

    print("Checking the accuracy of the NNCF int8 model:")
    int8_metrics = validation_fn(
        data=DATASET_CONFIG,
        weights=Path(nncf_int8_path).parent,
        batch_size=1,
        workers=1,
        plots=False,
        device="cpu",
        iou_thres=0.65,
    )
    
    nncf_int8_ap5 = int8_metrics[0][2]
    nncf_int8_ap_full = int8_metrics[0][3]
    print(f"mAP@.5 = {nncf_int8_ap5}")
    print(f"mAP@.5:.95 = {nncf_int8_ap_full}")


.. parsed-literal::

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU
    
    Loading yolov5/yolov5m/NNCF_INT8_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the NNCF int8 model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models
    val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00
                     Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:03
                       all        128        929      0.742      0.684      0.766      0.546
    Speed: 0.2ms pre-process, 17.1ms inference, 3.3ms NMS per image at shape (1, 3, 640, 640)
    Results saved to yolov5/runs/val/exp3


.. parsed-literal::

    mAP@.5 = 0.7660170261123679
    mAP@.5:.95 = 0.5460759842467641


Compare Average Precision of quantized INT8 model with original FP32
model.

.. code:: ipython3

    %matplotlib inline
    plt.style.use("seaborn-deep")
    fp32_acc = np.array([fp32_ap5, fp32_ap_full])
    pot_int8_acc = np.array([pot_int8_ap5, pot_int8_ap_full])
    nncf_int8_acc = np.array([nncf_int8_ap5, nncf_int8_ap_full])
    x_data = ("AP@0.5", "AP@0.5:0.95")
    x_axis = np.arange(len(x_data))
    fig = plt.figure()
    fig.patch.set_facecolor("#FFFFFF")
    fig.patch.set_alpha(0.7)
    ax = fig.add_subplot(111)
    plt.bar(x_axis - 0.2, fp32_acc, 0.3, label="FP32")
    for i in range(0, len(x_axis)):
        plt.text(
            i - 0.3,
            round(fp32_acc[i], 3) + 0.01,
            str(round(fp32_acc[i], 3)),
            fontweight="bold",
        )
    plt.bar(x_axis + 0.15, pot_int8_acc, 0.3, label="POT INT8")
    for i in range(0, len(x_axis)):
        plt.text(
            i + 0.05,
            round(pot_int8_acc[i], 3) + 0.01,
            str(round(pot_int8_acc[i], 3)),
            fontweight="bold",
        )
    
    plt.bar(x_axis + 0.5, nncf_int8_acc, 0.3, label="NNCF INT8")
    for i in range(0, len(x_axis)):
        plt.text(
            i + 0.4,
            round(nncf_int8_acc[i], 3) + 0.01,
            str(round(nncf_int8_acc[i], 3)),
            fontweight="bold",
        )
    plt.xticks(x_axis, x_data)
    plt.xlabel("Average Precision")
    plt.title("Compare Yolov5 FP32 and INT8 model average precision")
    
    plt.legend()
    plt.show()



.. image:: 111-yolov5-quantization-migration-with-output_files/111-yolov5-quantization-migration-with-output_34_0.png


Inference Demo Performance Comparison
###############################################################################################################################

This part shows how to use the Ultralytics model detection code
```detect.py`` <https://github.com/ultralytics/yolov5/blob/master/detect.py>`__
to run synchronous inference, using the OpenVINO Python API on two
images.

.. code:: ipython3

    from yolov5.utils.general import increment_path
    
    fp32_save_dir = increment_path(Path('./yolov5/runs/detect/exp'))

.. code:: ipython3

    command_detect = "cd yolov5 && python detect.py --weights ./yolov5m/FP32_openvino_model"
    display(Markdown(f"`{command_detect}`"))
    %sx $command_detect



``cd yolov5 && python detect.py --weights ./yolov5m/FP32_openvino_model``




.. parsed-literal::

    ["\x1b[34m\x1b[1mdetect: \x1b[0mweights=['./yolov5m/FP32_openvino_model'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1",
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU',
     '',
     'Loading yolov5m/FP32_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 56.6ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 2 ties, 45.9ms',
     'Speed: 1.5ms pre-process, 51.2ms inference, 1.3ms NMS per image at shape (1, 3, 640, 640)',
     'Results saved to \x1b[1mruns/detect/exp\x1b[0m']



.. code:: ipython3

    pot_save_dir = increment_path(Path('./yolov5/runs/detect/exp'))
    command_detect = "cd yolov5 && python detect.py --weights ./yolov5m/POT_INT8_openvino_model"
    display(Markdown(f"`{command_detect}`"))
    %sx $command_detect



``cd yolov5 && python detect.py --weights ./yolov5m/POT_INT8_openvino_model``




.. parsed-literal::

    ["\x1b[34m\x1b[1mdetect: \x1b[0mweights=['./yolov5m/POT_INT8_openvino_model'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1",
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU',
     '',
     'Loading yolov5m/POT_INT8_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 35.4ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 1 tie, 33.8ms',
     'Speed: 1.6ms pre-process, 34.6ms inference, 1.4ms NMS per image at shape (1, 3, 640, 640)',
     'Results saved to \x1b[1mruns/detect/exp2\x1b[0m']



.. code:: ipython3

    nncf_save_dir = increment_path(Path('./yolov5/runs/detect/exp'))
    command_detect = "cd yolov5 && python detect.py --weights ./yolov5m/NNCF_INT8_openvino_model"
    display(Markdown(f"`{command_detect}`"))
    %sx $command_detect



``cd yolov5 && python detect.py --weights ./yolov5m/NNCF_INT8_openvino_model``




.. parsed-literal::

    ["\x1b[34m\x1b[1mdetect: \x1b[0mweights=['./yolov5m/NNCF_INT8_openvino_model'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1",
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-1.13.1+cpu CPU',
     '',
     'Loading yolov5m/NNCF_INT8_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 37.1ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-499/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 2 ties, 30.5ms',
     'Speed: 1.6ms pre-process, 33.8ms inference, 1.4ms NMS per image at shape (1, 3, 640, 640)',
     'Results saved to \x1b[1mruns/detect/exp3\x1b[0m']



.. code:: ipython3

    %matplotlib inline
    import matplotlib.image as mpimg
    
    fig2, axs = plt.subplots(1, 4, figsize=(20, 20))
    fig2.patch.set_facecolor("#FFFFFF")
    fig2.patch.set_alpha(0.7)
    ori = mpimg.imread("./yolov5/data/images/bus.jpg")
    fp32_result = mpimg.imread(fp32_save_dir / "bus.jpg")
    pot_result = mpimg.imread(pot_save_dir / "bus.jpg")
    nncf_result = mpimg.imread(nncf_save_dir / "bus.jpg")
    titles = ["Original", "FP32", "POT INT8", "NNCF INT8"]
    imgs = [ori, fp32_result, pot_result, nncf_result]
    for ax, img, title in zip(axs, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])



.. image:: 111-yolov5-quantization-migration-with-output_files/111-yolov5-quantization-migration-with-output_40_0.png


Benchmark
###############################################################################################################################

.. code:: ipython3

    gpu_available = "GPU" in core.available_devices
    
    print("Inference FP32 model (OpenVINO IR) on CPU")
    !benchmark_app -m  {fp32_path} -d CPU -api async -t 15
    
    if gpu_available:
        print("Inference FP32 model (OpenVINO IR) on GPU")
        !benchmark_app -m  {fp32_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference FP32 model (OpenVINO IR) on CPU
    /bin/bash: benchmark_app: command not found


.. code:: ipython3

    print("Inference FP16 model (OpenVINO IR) on CPU")
    !benchmark_app -m {fp16_path} -d CPU -api async -t 15
    
    if gpu_available:
        print("Inference FP16 model (OpenVINO IR) on GPU")
        !benchmark_app -m {fp16_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference FP16 model (OpenVINO IR) on CPU
    /bin/bash: benchmark_app: command not found


.. code:: ipython3

    print("Inference POT INT8 model (OpenVINO IR) on CPU")
    !benchmark_app -m {pot_int8_path} -d CPU -api async -t 15
    
    if gpu_available:
        print("Inference POT INT8 model (OpenVINO IR) on GPU")
        !benchmark_app -m {pot_int8_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference POT INT8 model (OpenVINO IR) on CPU
    /bin/bash: benchmark_app: command not found


.. code:: ipython3

    print("Inference NNCF INT8 model (OpenVINO IR) on CPU")
    !benchmark_app -m {nncf_int8_path} -d CPU -api async -t 15
    
    if gpu_available:
        print("Inference NNCF INT8 model (OpenVINO IR) on GPU")
        !benchmark_app -m {nncf_int8_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference NNCF INT8 model (OpenVINO IR) on CPU
    /bin/bash: benchmark_app: command not found


References
###############################################################################################################################

-  `Ultralytics YOLOv5 <https://github.com/ultralytics/yolov5>`__
-  `OpenVINO Post-training Optimization
   Tool <https://docs.openvino.ai/2023.0/pot_introduction.html>`__
-  `NNCF Post-training
   quantization <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__
-  `Model Conversion
   API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
