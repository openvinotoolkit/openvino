Converting a PyTorch RCAN Model
===============================


.. meta::
   :description: Learn how to convert a RCAN model
                 from PyTorch to the OpenVINO Intermediate Representation.

.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

`RCAN <https://github.com/yulunzhang/RCAN>`__ : Image Super-Resolution Using Very Deep Residual Channel Attention Networks

Downloading and Converting the Model to ONNX
############################################

To download the pre-trained model or train the model yourself, refer to the `instruction <https://github.com/yulunzhang/RCAN/blob/master/README.md>`__ in the RCAN model repository. First, convert the model to ONNX format. Create and run the script with the following content in the root
directory of the model repository:

.. code-block:: py
   :force:

   from argparse import Namespace

   import torch

   from RCAN_TestCode.code.model.rcan import RCAN

   config = Namespace(n_feats=64, n_resblocks=4, n_resgroups=2, reduction=16, scale=[2], data_train='DIV2K', res_scale=1,
                      n_colors=3, rgb_range=255)
   net = RCAN(config)
   net.eval()
   dummy_input = torch.randn(1, 3, 360, 640)
   torch.onnx.export(net, dummy_input, 'RCAN.onnx')


The script generates the ONNX model file ``RCAN.onnx``. More information about model parameters (``n_resblocks``, ``n_resgroups``, and others) and their different values can be found in the model repository. The model conversion was tested with the commit-SHA: ``3339ebc59519c3bb2b5719b87dd36515ec7f3ba7``.

Converting an ONNX RCAN Model to IR
###################################

.. code-block:: sh

   mo --input_model RCAN.onnx


