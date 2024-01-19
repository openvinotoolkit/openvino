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

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Preparation <#preparation>`__

   -  `Download the YOLOv5 model <#download-the-yolov-model>`__
   -  `Conversion of the YOLOv5 model to
      OpenVINO <#conversion-of-the-yolov-model-to-openvino>`__
   -  `Imports <#imports>`__

-  `Prepare dataset for
   quantization <#prepare-dataset-for-quantization>`__

   -  `Create YOLOv5 DataLoader class for
      POT <#create-yolov-dataloader-class-for-pot>`__
   -  `Create NNCF Dataset <#create-nncf-dataset>`__

-  `Configure quantization
   pipeline <#configure-quantization-pipeline>`__

   -  `Prepare config and pipeline for
      POT <#prepare-config-and-pipeline-for-pot>`__
   -  `Prepare configuration parameters for
      NNCF <#prepare-configuration-parameters-for-nncf>`__

-  `Perform model optimization <#perform-model-optimization>`__

   -  `Run quantization using POT <#run-quantization-using-pot>`__
   -  `Run quantization using NNCF <#run-quantization-using-nncf>`__

-  `Compare accuracy FP32 and INT8
   models <#compare-accuracy-fp-and-int-models>`__
-  `Inference Demo Performance
   Comparison <#inference-demo-performance-comparison>`__
-  `Benchmark <#benchmark>`__
-  `References <#references>`__

Preparation
-----------



Download the YOLOv5 model
~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %pip install -q "openvino-dev>=2023.1.0" "nncf>=2.5.0"
    %pip install -q psutil "seaborn>=0.11.0" matplotlib numpy onnx


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU



.. parsed-literal::

    Downloading https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt to yolov5m/yolov5m.pt...


.. parsed-literal::


  0%|                                               | 0.00/40.8M [00:00<?, ?B/s]

.. parsed-literal::


  1%|▏                                      | 240k/40.8M [00:00<00:18, 2.31MB/s]

.. parsed-literal::


  1%|▌                                      | 624k/40.8M [00:00<00:13, 3.07MB/s]

.. parsed-literal::


  2%|▉                                     | 0.98M/40.8M [00:00<00:12, 3.28MB/s]

.. parsed-literal::


  3%|█▎                                    | 1.36M/40.8M [00:00<00:12, 3.39MB/s]

.. parsed-literal::


  4%|█▌                                    | 1.73M/40.8M [00:00<00:11, 3.44MB/s]

.. parsed-literal::


  5%|█▉                                    | 2.10M/40.8M [00:00<00:11, 3.49MB/s]

.. parsed-literal::


  6%|██▎                                   | 2.48M/40.8M [00:00<00:11, 3.52MB/s]

.. parsed-literal::


  7%|██▋                                   | 2.84M/40.8M [00:00<00:11, 3.51MB/s]

.. parsed-literal::


  8%|██▉                                   | 3.22M/40.8M [00:00<00:11, 3.52MB/s]

.. parsed-literal::


  9%|███▎                                  | 3.59M/40.8M [00:01<00:11, 3.53MB/s]

.. parsed-literal::


 10%|███▋                                  | 3.97M/40.8M [00:01<00:10, 3.53MB/s]

.. parsed-literal::


 11%|████                                  | 4.34M/40.8M [00:01<00:10, 3.54MB/s]

.. parsed-literal::


 12%|████▍                                 | 4.72M/40.8M [00:01<00:10, 3.50MB/s]

.. parsed-literal::


 12%|████▋                                 | 5.08M/40.8M [00:01<00:10, 3.52MB/s]

.. parsed-literal::


 13%|█████                                 | 5.45M/40.8M [00:01<00:10, 3.51MB/s]

.. parsed-literal::


 14%|█████▍                                | 5.83M/40.8M [00:01<00:10, 3.51MB/s]

.. parsed-literal::


 15%|█████▊                                | 6.20M/40.8M [00:01<00:10, 3.49MB/s]

.. parsed-literal::


 16%|██████                                | 6.58M/40.8M [00:01<00:10, 3.50MB/s]

.. parsed-literal::


 17%|██████▍                               | 6.95M/40.8M [00:02<00:09, 3.56MB/s]

.. parsed-literal::


 18%|██████▊                               | 7.31M/40.8M [00:02<00:09, 3.63MB/s]

.. parsed-literal::


 19%|███████▏                              | 7.66M/40.8M [00:02<00:09, 3.57MB/s]

.. parsed-literal::


 20%|███████▍                              | 8.01M/40.8M [00:02<00:09, 3.57MB/s]

.. parsed-literal::


 20%|███████▊                              | 8.35M/40.8M [00:02<00:09, 3.46MB/s]

.. parsed-literal::


 21%|████████                              | 8.69M/40.8M [00:02<00:09, 3.45MB/s]

.. parsed-literal::


 22%|████████▍                             | 9.06M/40.8M [00:02<00:09, 3.42MB/s]

.. parsed-literal::


 23%|████████▊                             | 9.42M/40.8M [00:02<00:09, 3.49MB/s]

.. parsed-literal::


 24%|█████████                             | 9.80M/40.8M [00:02<00:09, 3.46MB/s]

.. parsed-literal::


 25%|█████████▍                            | 10.2M/40.8M [00:03<00:09, 3.47MB/s]

.. parsed-literal::


 26%|█████████▊                            | 10.5M/40.8M [00:03<00:09, 3.50MB/s]

.. parsed-literal::


 27%|██████████▏                           | 10.9M/40.8M [00:03<00:08, 3.50MB/s]

.. parsed-literal::


 28%|██████████▌                           | 11.3M/40.8M [00:03<00:08, 3.49MB/s]

.. parsed-literal::


 29%|██████████▊                           | 11.7M/40.8M [00:03<00:08, 3.50MB/s]

.. parsed-literal::


 29%|███████████▏                          | 12.0M/40.8M [00:03<00:08, 3.49MB/s]

.. parsed-literal::


 30%|███████████▌                          | 12.4M/40.8M [00:03<00:08, 3.51MB/s]

.. parsed-literal::


 31%|███████████▉                          | 12.8M/40.8M [00:03<00:08, 3.50MB/s]

.. parsed-literal::


 32%|████████████▏                         | 13.2M/40.8M [00:03<00:08, 3.49MB/s]

.. parsed-literal::


 33%|████████████▌                         | 13.5M/40.8M [00:04<00:08, 3.50MB/s]

.. parsed-literal::


 34%|████████████▉                         | 13.9M/40.8M [00:04<00:07, 3.53MB/s]

.. parsed-literal::


 35%|█████████████▎                        | 14.3M/40.8M [00:04<00:07, 3.54MB/s]

.. parsed-literal::


 36%|█████████████▌                        | 14.6M/40.8M [00:04<00:07, 3.60MB/s]

.. parsed-literal::


 37%|█████████████▉                        | 15.0M/40.8M [00:04<00:07, 3.52MB/s]

.. parsed-literal::


 38%|██████████████▎                       | 15.4M/40.8M [00:04<00:07, 3.53MB/s]

.. parsed-literal::


 39%|██████████████▋                       | 15.7M/40.8M [00:04<00:07, 3.45MB/s]

.. parsed-literal::


 39%|██████████████▉                       | 16.1M/40.8M [00:04<00:07, 3.47MB/s]

.. parsed-literal::


 40%|███████████████▎                      | 16.5M/40.8M [00:04<00:07, 3.45MB/s]

.. parsed-literal::


 41%|███████████████▋                      | 16.8M/40.8M [00:05<00:07, 3.46MB/s]

.. parsed-literal::


 42%|███████████████▉                      | 17.1M/40.8M [00:05<00:07, 3.38MB/s]

.. parsed-literal::


 43%|████████████████▎                     | 17.5M/40.8M [00:05<00:07, 3.32MB/s]

.. parsed-literal::


 44%|████████████████▌                     | 17.8M/40.8M [00:05<00:07, 3.35MB/s]

.. parsed-literal::


 45%|████████████████▉                     | 18.2M/40.8M [00:05<00:07, 3.37MB/s]

.. parsed-literal::


 46%|█████████████████▎                    | 18.6M/40.8M [00:05<00:06, 3.36MB/s]

.. parsed-literal::


 46%|█████████████████▋                    | 19.0M/40.8M [00:05<00:06, 3.31MB/s]

.. parsed-literal::


 47%|█████████████████▉                    | 19.3M/40.8M [00:05<00:06, 3.34MB/s]

.. parsed-literal::


 48%|██████████████████▎                   | 19.7M/40.8M [00:05<00:06, 3.41MB/s]

.. parsed-literal::


 49%|██████████████████▌                   | 20.0M/40.8M [00:06<00:06, 3.28MB/s]

.. parsed-literal::


 50%|██████████████████▉                   | 20.3M/40.8M [00:06<00:06, 3.37MB/s]

.. parsed-literal::


 51%|███████████████████▏                  | 20.7M/40.8M [00:06<00:07, 2.73MB/s]

.. parsed-literal::


 51%|███████████████████▌                  | 21.0M/40.8M [00:06<00:07, 2.72MB/s]

.. parsed-literal::


 52%|███████████████████▊                  | 21.3M/40.8M [00:06<00:07, 2.85MB/s]

.. parsed-literal::


 53%|████████████████████▏                 | 21.7M/40.8M [00:06<00:06, 3.05MB/s]

.. parsed-literal::


 54%|████████████████████▌                 | 22.0M/40.8M [00:06<00:06, 3.18MB/s]

.. parsed-literal::


 55%|████████████████████▊                 | 22.4M/40.8M [00:06<00:05, 3.27MB/s]

.. parsed-literal::


 56%|█████████████████████▏                | 22.8M/40.8M [00:07<00:05, 3.35MB/s]

.. parsed-literal::


 57%|█████████████████████▌                | 23.2M/40.8M [00:07<00:05, 3.37MB/s]

.. parsed-literal::


 58%|█████████████████████▉                | 23.5M/40.8M [00:07<00:05, 3.42MB/s]

.. parsed-literal::


 59%|██████████████████████▏               | 23.9M/40.8M [00:07<00:05, 3.40MB/s]

.. parsed-literal::


 59%|██████████████████████▌               | 24.3M/40.8M [00:07<00:05, 3.41MB/s]

.. parsed-literal::


 60%|██████████████████████▉               | 24.6M/40.8M [00:07<00:04, 3.45MB/s]

.. parsed-literal::


 61%|███████████████████████▎              | 25.0M/40.8M [00:07<00:04, 3.42MB/s]

.. parsed-literal::


 62%|███████████████████████▋              | 25.4M/40.8M [00:07<00:04, 3.45MB/s]

.. parsed-literal::


 63%|███████████████████████▉              | 25.8M/40.8M [00:07<00:05, 3.08MB/s]

.. parsed-literal::


 64%|████████████████████████▎             | 26.1M/40.8M [00:08<00:05, 3.02MB/s]

.. parsed-literal::


 65%|████████████████████████▌             | 26.4M/40.8M [00:08<00:04, 3.10MB/s]

.. parsed-literal::


 66%|████████████████████████▉             | 26.8M/40.8M [00:08<00:04, 3.25MB/s]

.. parsed-literal::


 67%|█████████████████████████▎            | 27.2M/40.8M [00:08<00:04, 3.31MB/s]

.. parsed-literal::


 67%|█████████████████████████▋            | 27.5M/40.8M [00:08<00:04, 3.33MB/s]

.. parsed-literal::


 68%|█████████████████████████▉            | 27.9M/40.8M [00:08<00:04, 3.38MB/s]

.. parsed-literal::


 69%|██████████████████████████▎           | 28.3M/40.8M [00:08<00:03, 3.39MB/s]

.. parsed-literal::


 70%|██████████████████████████▋           | 28.7M/40.8M [00:08<00:03, 3.42MB/s]

.. parsed-literal::


 71%|███████████████████████████           | 29.0M/40.8M [00:08<00:03, 3.45MB/s]

.. parsed-literal::


 72%|███████████████████████████▎          | 29.4M/40.8M [00:09<00:03, 3.49MB/s]

.. parsed-literal::


 73%|███████████████████████████▋          | 29.8M/40.8M [00:09<00:03, 3.50MB/s]

.. parsed-literal::


 74%|████████████████████████████          | 30.2M/40.8M [00:09<00:03, 3.51MB/s]

.. parsed-literal::


 75%|████████████████████████████▍         | 30.5M/40.8M [00:09<00:03, 3.49MB/s]

.. parsed-literal::


 76%|████████████████████████████▊         | 30.9M/40.8M [00:09<00:02, 3.50MB/s]

.. parsed-literal::


 77%|█████████████████████████████         | 31.3M/40.8M [00:09<00:02, 3.55MB/s]

.. parsed-literal::


 78%|█████████████████████████████▍        | 31.6M/40.8M [00:09<00:02, 3.48MB/s]

.. parsed-literal::


 78%|█████████████████████████████▊        | 32.0M/40.8M [00:09<00:02, 3.49MB/s]

.. parsed-literal::


 79%|██████████████████████████████▏       | 32.4M/40.8M [00:09<00:02, 3.48MB/s]

.. parsed-literal::


 80%|██████████████████████████████▍       | 32.8M/40.8M [00:10<00:02, 3.49MB/s]

.. parsed-literal::


 81%|██████████████████████████████▊       | 33.1M/40.8M [00:10<00:02, 3.52MB/s]

.. parsed-literal::


 82%|███████████████████████████████▏      | 33.5M/40.8M [00:10<00:02, 3.53MB/s]

.. parsed-literal::


 83%|███████████████████████████████▌      | 33.9M/40.8M [00:10<00:02, 3.55MB/s]

.. parsed-literal::


 84%|███████████████████████████████▉      | 34.2M/40.8M [00:10<00:01, 3.52MB/s]

.. parsed-literal::


 85%|████████████████████████████████▏     | 34.6M/40.8M [00:10<00:01, 3.52MB/s]

.. parsed-literal::


 86%|████████████████████████████████▌     | 35.0M/40.8M [00:10<00:01, 3.53MB/s]

.. parsed-literal::


 87%|████████████████████████████████▉     | 35.4M/40.8M [00:10<00:01, 3.53MB/s]

.. parsed-literal::


 88%|█████████████████████████████████▎    | 35.8M/40.8M [00:10<00:01, 3.51MB/s]

.. parsed-literal::


 88%|█████████████████████████████████▋    | 36.1M/40.8M [00:11<00:01, 3.50MB/s]

.. parsed-literal::


 89%|█████████████████████████████████▉    | 36.5M/40.8M [00:11<00:01, 3.52MB/s]

.. parsed-literal::


 90%|██████████████████████████████████▎   | 36.9M/40.8M [00:11<00:01, 3.50MB/s]

.. parsed-literal::


 91%|██████████████████████████████████▋   | 37.2M/40.8M [00:11<00:01, 3.44MB/s]

.. parsed-literal::


 92%|██████████████████████████████████▉   | 37.5M/40.8M [00:11<00:01, 3.40MB/s]

.. parsed-literal::


 93%|███████████████████████████████████▎  | 37.9M/40.8M [00:11<00:00, 3.41MB/s]

.. parsed-literal::


 94%|███████████████████████████████████▋  | 38.3M/40.8M [00:11<00:00, 3.43MB/s]

.. parsed-literal::


 95%|███████████████████████████████████▉  | 38.7M/40.8M [00:11<00:00, 3.44MB/s]

.. parsed-literal::


 96%|████████████████████████████████████▎ | 39.0M/40.8M [00:11<00:00, 3.47MB/s]

.. parsed-literal::


 97%|████████████████████████████████████▋ | 39.4M/40.8M [00:12<00:00, 3.48MB/s]

.. parsed-literal::


   97%|█████████████████████████████████████ | 39.8M/40.8M [00:12<00:00, 3.51MB/s]

.. parsed-literal::


   98%|█████████████████████████████████████▎| 40.1M/40.8M [00:12<00:00, 3.48MB/s]

.. parsed-literal::


   99%|█████████████████████████████████████▋| 40.5M/40.8M [00:12<00:00, 3.50MB/s]

.. parsed-literal::


   100%|██████████████████████████████████████| 40.8M/40.8M [00:12<00:00, 3.42MB/s]



.. parsed-literal::

    Fusing layers...


.. parsed-literal::

    YOLOv5m summary: 290 layers, 21172173 parameters, 0 gradients


.. parsed-literal::


    PyTorch: starting from yolov5m/yolov5m.pt with output shape (1, 25200, 85) (40.8 MB)

    ONNX: starting export with onnx 1.15.0...


.. parsed-literal::

    ONNX: export success ✅ 1.3s, saved as yolov5m/yolov5m.onnx (81.2 MB)

    Export complete (15.3s)
    Results saved to /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/yolov5m
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


.. parsed-literal::

    Export ONNX to OpenVINO FP16 IR to: yolov5/yolov5m/FP16_openvino_model/yolov5m_fp16.xml


Imports
~~~~~~~



.. code:: ipython3

    sys.path.append("./yolov5")

    from yolov5.utils.dataloaders import create_dataloader
    from yolov5.utils.general import check_dataset

Prepare dataset for quantization
--------------------------------



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


  3%|▎         | 224k/6.66M [00:00<00:03, 2.03MB/s]

.. parsed-literal::


  9%|▊         | 592k/6.66M [00:00<00:02, 2.86MB/s]

.. parsed-literal::


 14%|█▍        | 960k/6.66M [00:00<00:01, 3.18MB/s]

.. parsed-literal::


 19%|█▉        | 1.25M/6.66M [00:00<00:01, 3.12MB/s]

.. parsed-literal::


 24%|██▎       | 1.58M/6.66M [00:00<00:01, 3.15MB/s]

.. parsed-literal::


 28%|██▊       | 1.89M/6.66M [00:00<00:01, 3.16MB/s]

.. parsed-literal::


 33%|███▎      | 2.20M/6.66M [00:00<00:01, 3.11MB/s]

.. parsed-literal::


 38%|███▊      | 2.51M/6.66M [00:00<00:01, 3.13MB/s]

.. parsed-literal::


 42%|████▏     | 2.83M/6.66M [00:00<00:01, 3.17MB/s]

.. parsed-literal::


 47%|████▋     | 3.14M/6.66M [00:01<00:01, 3.19MB/s]

.. parsed-literal::


 52%|█████▏    | 3.45M/6.66M [00:01<00:01, 3.17MB/s]

.. parsed-literal::


 57%|█████▋    | 3.77M/6.66M [00:01<00:00, 3.16MB/s]

.. parsed-literal::


 61%|██████    | 4.08M/6.66M [00:01<00:00, 3.17MB/s]

.. parsed-literal::


 66%|██████▌   | 4.38M/6.66M [00:01<00:00, 3.14MB/s]

.. parsed-literal::


 70%|███████   | 4.69M/6.66M [00:01<00:00, 3.14MB/s]

.. parsed-literal::


 75%|███████▌  | 5.00M/6.66M [00:01<00:00, 2.78MB/s]

.. parsed-literal::


 79%|███████▉  | 5.27M/6.66M [00:01<00:00, 2.79MB/s]

.. parsed-literal::


   84%|████████▍ | 5.62M/6.66M [00:01<00:00, 2.97MB/s]

.. parsed-literal::


   90%|████████▉ | 5.98M/6.66M [00:02<00:00, 3.16MB/s]

.. parsed-literal::


   95%|█████████▍| 6.31M/6.66M [00:02<00:00, 3.23MB/s]

.. parsed-literal::


   100%|█████████▉| 6.64M/6.66M [00:02<00:00, 3.18MB/s]

.. parsed-literal::


   100%|██████████| 6.66M/6.66M [00:02<00:00, 3.10MB/s]

.. parsed-literal::


   Unzipping datasets/coco128.zip...


.. parsed-literal::


   Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017...:   0%|          | 0/128 00:00

.. parsed-literal::


   Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00



.. parsed-literal::

   New cache created: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache


Create YOLOv5 DataLoader class for POT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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


.. parsed-literal::

    [ DEBUG ] Creating converter from 5 to 7


.. parsed-literal::

    [ DEBUG ] Creating converter from 7 to 5


.. parsed-literal::

    [ DEBUG ] Creating converter from 5 to 7


.. parsed-literal::

    [ WARNING ] /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/accuracy_checker/preprocessor/launcher_preprocessing/ie_preprocessor.py:21: FutureWarning: OpenVINO Inference Engine Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
      from openvino.inference_engine import ResizeAlgorithm, PreProcessInfo, ColorFormat, MeanVariant  # pylint: disable=import-outside-toplevel,package-absolute-imports



.. parsed-literal::

    [ WARNING ] /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/tools/accuracy_checker/launcher/dlsdk_launcher.py:60: FutureWarning: OpenVINO nGraph Python API is deprecated and will be removed in 2024.0 release.For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html
      import ngraph as ng



.. parsed-literal::

    Post-training Optimization Tool is deprecated and will be removed in the future. Please use Neural Network Compression Framework instead: https://github.com/openvinotoolkit/nncf


.. parsed-literal::

    Nevergrad package could not be imported. If you are planning to use any hyperparameter optimization algo, consider installing it using pip. This implies advanced usage of the tool. Note that nevergrad is compatible only with Python 3.8+


Create NNCF Dataset
~~~~~~~~~~~~~~~~~~~



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

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Configure quantization pipeline
-------------------------------



Next, we should define quantization algorithm parameters.

Prepare config and pipeline for POT
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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
--------------------------



Run quantization using POT
~~~~~~~~~~~~~~~~~~~~~~~~~~



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
~~~~~~~~~~~~~~~~~~~~~~~~~~~



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

    2024-01-18 22:54:13.958443: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-01-18 22:54:13.989952: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-01-18 22:54:14.387248: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



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



Compare accuracy FP32 and INT8 models
-------------------------------------



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


.. parsed-literal::

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

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU



.. parsed-literal::

    Loading yolov5/yolov5m/FP32_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the original model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models


.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00

.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00


.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   2%|▏         | 2/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   4%|▍         | 5/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   6%|▋         | 8/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   9%|▊         | 11/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  11%|█         | 14/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  13%|█▎        | 17/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  16%|█▌        | 20/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  18%|█▊        | 23/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  20%|██        | 26/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  23%|██▎       | 29/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  25%|██▌       | 32/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  27%|██▋       | 35/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  30%|██▉       | 38/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  32%|███▏      | 41/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  34%|███▍      | 44/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  37%|███▋      | 47/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  39%|███▉      | 50/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  41%|████▏     | 53/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  44%|████▍     | 56/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  46%|████▌     | 59/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  48%|████▊     | 62/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  51%|█████     | 65/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  53%|█████▎    | 68/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  55%|█████▌    | 71/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  58%|█████▊    | 74/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  60%|██████    | 77/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  62%|██████▎   | 80/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  65%|██████▍   | 83/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|██████▋   | 86/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  70%|██████▉   | 89/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  72%|███████▏  | 92/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  74%|███████▍  | 95/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  77%|███████▋  | 98/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  79%|███████▉  | 101/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  81%|████████▏ | 104/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  84%|████████▎ | 107/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  86%|████████▌ | 110/128 00:04

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  88%|████████▊ | 113/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  91%|█████████ | 116/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  93%|█████████▎| 119/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  95%|█████████▌| 122/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  98%|█████████▊| 125/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:05

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:05

.. parsed-literal::


                       all        128        929      0.726      0.687      0.769      0.554


.. parsed-literal::

    Speed: 0.2ms pre-process, 35.2ms inference, 3.9ms NMS per image at shape (1, 3, 640, 640)


.. parsed-literal::

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

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU



.. parsed-literal::

    Loading yolov5/yolov5m/POT_INT8_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the POT int8 model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models


.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00

.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00


.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   3%|▎         | 4/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   6%|▋         | 8/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   9%|▉         | 12/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  13%|█▎        | 17/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  17%|█▋        | 22/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  20%|██        | 26/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  23%|██▎       | 30/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  27%|██▋       | 34/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  30%|██▉       | 38/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  33%|███▎      | 42/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  36%|███▌      | 46/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  39%|███▉      | 50/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  42%|████▏     | 54/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  45%|████▌     | 58/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  49%|████▉     | 63/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  52%|█████▏    | 67/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  55%|█████▌    | 71/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  59%|█████▉    | 76/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  63%|██████▎   | 81/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  67%|██████▋   | 86/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  71%|███████   | 91/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  75%|███████▌  | 96/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  79%|███████▉  | 101/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  83%|████████▎ | 106/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  87%|████████▋ | 111/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  91%|█████████ | 116/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  95%|█████████▍| 121/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  98%|█████████▊| 126/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:03



.. parsed-literal::

                       all        128        929      0.761      0.677      0.773      0.548


.. parsed-literal::

    Speed: 0.2ms pre-process, 17.1ms inference, 4.0ms NMS per image at shape (1, 3, 640, 640)


.. parsed-literal::

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

    YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU



.. parsed-literal::

    Loading yolov5/yolov5m/NNCF_INT8_openvino_model for OpenVINO inference...


.. parsed-literal::

    Checking the accuracy of the NNCF int8 model:


.. parsed-literal::

    Forcing --batch-size 1 square inference (1,3,640,640) for non-PyTorch models


.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00

.. parsed-literal::


   val: Scanning /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/datasets/coco128/labels/train2017.cache... 126 images, 2 backgrounds, 0 corrupt: 100%|██████████| 128/128 00:00


.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   0%|          | 0/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   3%|▎         | 4/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:   7%|▋         | 9/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  11%|█         | 14/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  15%|█▍        | 19/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  19%|█▉        | 24/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  22%|██▏       | 28/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  26%|██▌       | 33/128 00:00

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  29%|██▉       | 37/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  32%|███▏      | 41/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  35%|███▌      | 45/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  38%|███▊      | 49/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  41%|████▏     | 53/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  45%|████▍     | 57/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  48%|████▊     | 61/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  51%|█████     | 65/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  55%|█████▍    | 70/128 00:01

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  58%|█████▊    | 74/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  62%|██████▏   | 79/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  65%|██████▍   | 83/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  69%|██████▉   | 88/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  73%|███████▎  | 93/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  77%|███████▋  | 98/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  80%|████████  | 103/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  84%|████████▎ | 107/128 00:02

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  87%|████████▋ | 111/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  91%|█████████ | 116/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  95%|█████████▍| 121/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95:  98%|█████████▊| 126/128 00:03

.. parsed-literal::


                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 128/128 00:03


.. parsed-literal::

                       all        128        929      0.738      0.682      0.768      0.549


.. parsed-literal::

    Speed: 0.2ms pre-process, 17.0ms inference, 3.9ms NMS per image at shape (1, 3, 640, 640)


.. parsed-literal::

    Results saved to yolov5/runs/val/exp3


.. parsed-literal::

    mAP@.5 = 0.7684598204433661
    mAP@.5:.95 = 0.5487198807173201


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
-------------------------------------



This part shows how to use the Ultralytics model detection code
`detect.py <https://github.com/ultralytics/yolov5/blob/master/detect.py>`__
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
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU',
     '',
     'Loading yolov5m/FP32_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 52.6ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 2 ties, 45.3ms',
     'Speed: 1.4ms pre-process, 48.9ms inference, 1.3ms NMS per image at shape (1, 3, 640, 640)',
     'Results saved to \x1b[1mruns/detect/exp\x1b[0m']



.. code:: ipython3

    pot_save_dir = increment_path(Path('./yolov5/runs/detect/exp'))
    command_detect = "cd yolov5 && python detect.py --weights ./yolov5m/POT_INT8_openvino_model"
    display(Markdown(f"`{command_detect}`"))
    %sx $command_detect



``cd yolov5 && python detect.py --weights ./yolov5m/POT_INT8_openvino_model``




.. parsed-literal::

    ["\x1b[34m\x1b[1mdetect: \x1b[0mweights=['./yolov5m/POT_INT8_openvino_model'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1",
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU',
     '',
     'Loading yolov5m/POT_INT8_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 32.8ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 1 tie, 28.3ms',
     'Speed: 1.5ms pre-process, 30.5ms inference, 1.6ms NMS per image at shape (1, 3, 640, 640)',
     'Results saved to \x1b[1mruns/detect/exp2\x1b[0m']



.. code:: ipython3

    nncf_save_dir = increment_path(Path('./yolov5/runs/detect/exp'))
    command_detect = "cd yolov5 && python detect.py --weights ./yolov5m/NNCF_INT8_openvino_model"
    display(Markdown(f"`{command_detect}`"))
    %sx $command_detect



``cd yolov5 && python detect.py --weights ./yolov5m/NNCF_INT8_openvino_model``




.. parsed-literal::

    ["\x1b[34m\x1b[1mdetect: \x1b[0mweights=['./yolov5m/NNCF_INT8_openvino_model'], source=data/images, data=data/coco128.yaml, imgsz=[640, 640], conf_thres=0.25, iou_thres=0.45, max_det=1000, device=, view_img=False, save_txt=False, save_conf=False, save_crop=False, nosave=False, classes=None, agnostic_nms=False, augment=False, visualize=False, update=False, project=runs/detect, name=exp, exist_ok=False, line_thickness=3, hide_labels=False, hide_conf=False, half=False, dnn=False, vid_stride=1",
     'YOLOv5 🚀 v7.0-0-g915bbf2 Python-3.8.10 torch-2.1.0+cpu CPU',
     '',
     'Loading yolov5m/NNCF_INT8_openvino_model for OpenVINO inference...',
     'image 1/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/bus.jpg: 640x640 4 persons, 1 bus, 32.6ms',
     'image 2/2 /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-593/.workspace/scm/ov-notebook/notebooks/111-yolov5-quantization-migration/yolov5/data/images/zidane.jpg: 640x640 3 persons, 2 ties, 28.4ms',
     'Speed: 1.5ms pre-process, 30.5ms inference, 1.5ms NMS per image at shape (1, 3, 640, 640)',
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
---------



.. code:: ipython3

    gpu_available = "GPU" in core.available_devices

    print("Inference FP32 model (OpenVINO IR) on CPU")
    !benchmark_app -m  {fp32_path} -d CPU -api async -t 15

    if gpu_available:
        print("Inference FP32 model (OpenVINO IR) on GPU")
        !benchmark_app -m  {fp32_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference FP32 model (OpenVINO IR) on CPU


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 36.77 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 327.63 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values


.. parsed-literal::

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 101.18 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            444 iterations
    [ INFO ] Duration:         15317.96 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        207.70 ms
    [ INFO ]    Average:       205.87 ms
    [ INFO ]    Min:           92.11 ms
    [ INFO ]    Max:           224.15 ms
    [ INFO ] Throughput:   28.99 FPS


.. code:: ipython3

    print("Inference FP16 model (OpenVINO IR) on CPU")
    !benchmark_app -m {fp16_path} -d CPU -api async -t 15

    if gpu_available:
        print("Inference FP16 model (OpenVINO IR) on GPU")
        !benchmark_app -m {fp16_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference FP16 model (OpenVINO IR) on CPU


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:


.. parsed-literal::

    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 38.22 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 360.85 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 103.32 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            456 iterations
    [ INFO ] Duration:         15116.05 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        198.86 ms
    [ INFO ]    Average:       198.12 ms
    [ INFO ]    Min:           91.59 ms
    [ INFO ]    Max:           217.56 ms
    [ INFO ] Throughput:   30.17 FPS


.. code:: ipython3

    print("Inference POT INT8 model (OpenVINO IR) on CPU")
    !benchmark_app -m {pot_int8_path} -d CPU -api async -t 15

    if gpu_available:
        print("Inference POT INT8 model (OpenVINO IR) on GPU")
        !benchmark_app -m {pot_int8_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference POT INT8 model (OpenVINO IR) on CPU


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 54.76 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:


.. parsed-literal::

    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 708.63 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:


.. parsed-literal::

    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values
    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 50.02 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            1422 iterations
    [ INFO ] Duration:         15055.10 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        63.41 ms
    [ INFO ]    Average:       63.35 ms
    [ INFO ]    Min:           45.05 ms
    [ INFO ]    Max:           84.41 ms
    [ INFO ] Throughput:   94.45 FPS


.. code:: ipython3

    print("Inference NNCF INT8 model (OpenVINO IR) on CPU")
    !benchmark_app -m {nncf_int8_path} -d CPU -api async -t 15

    if gpu_available:
        print("Inference NNCF INT8 model (OpenVINO IR) on GPU")
        !benchmark_app -m {nncf_int8_path} -d GPU -api async -t 15


.. parsed-literal::

    Inference NNCF INT8 model (OpenVINO IR) on CPU


.. parsed-literal::

    [Step 1/11] Parsing and validating input arguments
    [ INFO ] Parsing input parameters
    [Step 2/11] Loading OpenVINO Runtime
    [ INFO ] OpenVINO:
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ] Device info:
    [ INFO ] CPU
    [ INFO ] Build ................................. 2023.2.0-13089-cfd42bd2cb0-HEAD
    [ INFO ]
    [ INFO ]
    [Step 3/11] Setting device configuration
    [ WARNING ] Performance hint was not explicitly specified in command line. Device(CPU) performance hint will be set to PerformanceMode.THROUGHPUT.
    [Step 4/11] Reading model files
    [ INFO ] Loading model files


.. parsed-literal::

    [ INFO ] Read model took 51.56 ms
    [ INFO ] Original model I/O parameters:
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : f32 / [...] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 5/11] Resizing model to match image sizes and given batch
    [ INFO ] Model batch size: 1
    [Step 6/11] Configuring input of the model
    [ INFO ] Model inputs:
    [ INFO ]     images (node: images) : u8 / [N,C,H,W] / [1,3,640,640]
    [ INFO ] Model outputs:
    [ INFO ]     output0 (node: output0) : f32 / [...] / [1,25200,85]
    [Step 7/11] Loading the model to the device


.. parsed-literal::

    [ INFO ] Compile model took 694.77 ms
    [Step 8/11] Querying optimal runtime parameters
    [ INFO ] Model:
    [ INFO ]   NETWORK_NAME: main_graph
    [ INFO ]   OPTIMAL_NUMBER_OF_INFER_REQUESTS: 6
    [ INFO ]   NUM_STREAMS: 6
    [ INFO ]   AFFINITY: Affinity.CORE
    [ INFO ]   INFERENCE_NUM_THREADS: 24
    [ INFO ]   PERF_COUNT: False
    [ INFO ]   INFERENCE_PRECISION_HINT: <Type: 'float32'>
    [ INFO ]   PERFORMANCE_HINT: PerformanceMode.THROUGHPUT
    [ INFO ]   EXECUTION_MODE_HINT: ExecutionMode.PERFORMANCE
    [ INFO ]   PERFORMANCE_HINT_NUM_REQUESTS: 0
    [ INFO ]   ENABLE_CPU_PINNING: True
    [ INFO ]   SCHEDULING_CORE_TYPE: SchedulingCoreType.ANY_CORE
    [ INFO ]   ENABLE_HYPER_THREADING: True
    [ INFO ]   EXECUTION_DEVICES: ['CPU']
    [ INFO ]   CPU_DENORMALS_OPTIMIZATION: False
    [ INFO ]   CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE: 1.0
    [Step 9/11] Creating infer requests and preparing input tensors
    [ WARNING ] No input files were given for input 'images'!. This input will be filled with random values!
    [ INFO ] Fill input 'images' with random values


.. parsed-literal::

    [Step 10/11] Measuring performance (Start inference asynchronously, 6 inference requests, limits: 15000 ms duration)
    [ INFO ] Benchmarking in inference only mode (inputs filling are not included in measurement loop).


.. parsed-literal::

    [ INFO ] First inference took 50.67 ms


.. parsed-literal::

    [Step 11/11] Dumping statistics report
    [ INFO ] Execution Devices:['CPU']
    [ INFO ] Count:            1428 iterations
    [ INFO ] Duration:         15105.51 ms
    [ INFO ] Latency:
    [ INFO ]    Median:        63.45 ms
    [ INFO ]    Average:       63.31 ms
    [ INFO ]    Min:           46.12 ms
    [ INFO ]    Max:           85.85 ms
    [ INFO ] Throughput:   94.54 FPS


References
----------



-  `Ultralytics YOLOv5 <https://github.com/ultralytics/yolov5>`__
-  `OpenVINO Post-training Optimization
   Tool <https://docs.openvino.ai/2023.0/pot_introduction.html>`__
-  `NNCF Post-training
   quantization <https://docs.openvino.ai/nightly/basic_quantization_flow.html>`__
-  `Model Conversion
   API <https://docs.openvino.ai/2023.0/openvino_docs_model_processing_introduction.html>`__
