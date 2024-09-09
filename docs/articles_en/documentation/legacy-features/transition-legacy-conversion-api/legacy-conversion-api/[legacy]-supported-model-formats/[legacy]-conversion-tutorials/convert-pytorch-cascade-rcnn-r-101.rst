Converting a PyTorch Cascade RCNN R-101 Model
=============================================


.. meta::
   :description: Learn how to convert a Cascade RCNN R-101
                 model from PyTorch to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

The goal of this article is to present a step-by-step guide on how to convert a PyTorch Cascade RCNN R-101 model to OpenVINO IR. First, you need to download the model and convert it to ONNX.

Downloading and Converting Model to ONNX
########################################

* Clone the `repository <https://github.com/open-mmlab/mmdetection>`__ :

  .. code-block:: sh

     git clone https://github.com/open-mmlab/mmdetection
     cd mmdetection


  .. note::

     To set up an environment, refer to the `instructions <https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md#installation>`__.

* Download the pre-trained `model <https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco/cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth>`__. The model is also available `here <https://github.com/open-mmlab/mmdetection/blob/master/configs/cascade_rcnn/README.md>`__.

* To convert the model to ONNX format, use this `script <https://github.com/open-mmlab/mmdetection/blob/master/tools/deployment/pytorch2onnx.py>`__.

  .. code-block:: sh

     python3 tools/deployment/pytorch2onnx.py configs/cascade_rcnn/cascade_rcnn_r101_fpn_1x_coco.py cascade_rcnn_r101_fpn_1x_coco_20200317-0b6a2fbf.pth --output-file    cascade_rcnn_r101_fpn_1x_coco.onnx


The script generates ONNX model file ``cascade_rcnn_r101_fpn_1x_coco.onnx`` in the directory ``tools/deployment/``. If required, specify the model name or output directory, using ``--output-file <path-to-dir>/<model-name>.onnx``.

Converting an ONNX Cascade RCNN R-101 Model to OpenVINO IR
##########################################################

.. code-block:: sh

   mo --input_model cascade_rcnn_r101_fpn_1x_coco.onnx --mean_values [123.675,116.28,103.53] --scale_values [58.395,57.12,57.375]


