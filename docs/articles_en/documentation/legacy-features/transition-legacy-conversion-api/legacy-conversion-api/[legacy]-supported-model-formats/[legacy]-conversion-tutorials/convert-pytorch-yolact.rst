Converting a PyTorch YOLACT Model
=================================


.. meta::
   :description: Learn how to convert a YOLACT model
                 from PyTorch to the OpenVINO Intermediate Representation.


.. danger::

   The code described here has been **deprecated!** Do not use it to avoid working with a legacy solution. It will be kept for some time to ensure backwards compatibility, but **you should not use** it in contemporary applications.

   This guide describes a deprecated conversion method. The guide on the new and recommended method can be found in the :doc:`Python tutorials <../../../../../../learn-openvino/interactive-tutorials-python>`.

You Only Look At CoefficienTs (YOLACT) is a simple, fully convolutional model for real-time instance segmentation.
The PyTorch implementation is publicly available in `this GitHub repository <https://github.com/dbolya/yolact>`__.
The YOLACT++ model is not supported, because it uses deformable convolutional layers that cannot be represented in ONNX format.

.. _patch-file-yolact:

Creating a Patch File
#####################

Before converting the model, create a patch file for the repository.
The patch modifies the framework code by adding a special command-line argument to the framework options. The argument enables inference graph dumping:

1. Go to a writable directory and create a ``YOLACT_onnx_export.patch`` file.
2. Copy the following diff code to the file:

   .. code-block:: console

      From 76deb67d4f09f29feda1a633358caa18335d9e9f Mon Sep 17 00:00:00 2001
      From: "OpenVINO" <openvino@intel.com>
      Date: Fri, 12 Mar 2021 00:27:35 +0300
      Subject: [PATCH] Add export to ONNX

      ---
       eval.py                |  5 ++++-
       utils/augmentations.py |  7 +++++--
       yolact.py              | 29 +++++++++++++++++++----------
       3 files changed, 28 insertions(+), 13 deletions(-)

      diff --git a/eval.py b/eval.py
      index 547bc0a..bde0680 100644
      --- a/eval.py
      +++ b/eval.py
      @@ -593,9 +593,12 @@ def badhash(x):
           return x

       def evalimage(net:Yolact, path:str, save_path:str=None):
      -    frame = torch.from_numpy(cv2.imread(path)).cuda().float()
      +    frame = torch.from_numpy(cv2.imread(path)).float()
      +    if torch.cuda.is_available():
      +        frame = frame.cuda()
           batch = FastBaseTransform()(frame.unsqueeze(0))
           preds = net(batch)
      +    torch.onnx.export(net, batch, "yolact.onnx", opset_version=11)

           img_numpy = prep_display(preds, frame, None, None, undo_transform=False)

      diff --git a/utils/augmentations.py b/utils/augmentations.py
      index cc7a73a..2420603 100644
      --- a/utils/augmentations.py
      +++ b/utils/augmentations.py
      @@ -623,8 +623,11 @@ class FastBaseTransform(torch.nn.Module):
           def __init__(self):
               super().__init__()

      -        self.mean = torch.Tensor(MEANS).float().cuda()[None, :, None, None]
      -        self.std  = torch.Tensor( STD ).float().cuda()[None, :, None, None]
      +        self.mean = torch.Tensor(MEANS).float()[None, :, None, None]
      +        self.std  = torch.Tensor( STD ).float()[None, :, None, None]
      +        if torch.cuda.is_available():
      +            self.mean.cuda()
      +            self.std.cuda()
               self.transform = cfg.backbone.transform

           def forward(self, img):
      diff --git a/yolact.py b/yolact.py
      index d83703b..f8c787c 100644
      --- a/yolact.py
      +++ b/yolact.py
      @@ -17,19 +17,22 @@ import torch.backends.cudnn as cudnn
       from utils import timer
       from utils.functions import MovingAverage, make_net

      -# This is required for Pytorch 1.0.1 on Windows to initialize Cuda on some driver versions.
      -# See the bug report here: https://github.com/pytorch/pytorch/issues/17108
      -torch.cuda.current_device()
      -
      -# As of March 10, 2019, Pytorch DataParallel still doesn't support JIT Script Modules
      -use_jit = torch.cuda.device_count() <= 1
      -if not use_jit:
      -    print('Multiple GPUs detected! Turning off JIT.')
      +use_jit = False

       ScriptModuleWrapper = torch.jit.ScriptModule if use_jit else nn.Module
       script_method_wrapper = torch.jit.script_method if use_jit else lambda fn, _rcn=None: fn


      +def decode(loc, priors):
      +    variances = [0.1, 0.2]
      +    boxes = torch.cat((priors[:, :2] + loc[:, :, :2] * variances[0] * priors[:, 2:], priors[:, 2:] * torch.exp(loc[:, :, 2:] * variances[1])), 2)
      +
      +    boxes_result1 = boxes[:, :, :2] - boxes[:, :, 2:] / 2
      +    boxes_result2 = boxes[:, :, 2:] + boxes_result1
      +    boxes_result = torch.cat((boxes_result1, boxes_result2), 2)
      +
      +    return boxes_result
      +

       class Concat(nn.Module):
           def __init__(self, nets, extra_params):
      @@ -476,7 +479,10 @@ class Yolact(nn.Module):

           def load_weights(self, path):
               """ Loads weights from a compressed save file. """
      -        state_dict = torch.load(path)
      +        if torch.cuda.is_available():
      +            state_dict = torch.load(path)
      +        else:
      +            state_dict = torch.load(path, map_location=torch.device('cpu'))

               # For backward compatibility, remove these (the new variable is called layers)
               for key in list(state_dict.keys()):
      @@ -673,8 +679,11 @@ class Yolact(nn.Module):
                       else:
                           pred_outs['conf'] = F.softmax(pred_outs['conf'], -1)

      -            return self.detect(pred_outs, self)
      +            pred_outs['boxes'] = decode(pred_outs['loc'], pred_outs['priors']) # decode output boxes

      +            pred_outs.pop('priors') # remove unused in postprocessing layers
      +            pred_outs.pop('loc') # remove unused in postprocessing layers
      +            return pred_outs



      --


3. Save and close the file.

Converting a YOLACT Model to the OpenVINO IR format
###################################################

**Step 1**. Clone the GitHub repository and check out the commit:

1. Clone the YOLACT repository:

   .. code-block:: sh

      git clone https://github.com/dbolya/yolact


2. Check out the necessary commit:

   .. code-block:: sh

      git checkout 57b8f2d95e62e2e649b382f516ab41f949b57239


3. Set up the environment as described in ``README.md``.

**Step 2**. Download a pre-trained model from the list attached in the ``Evaluation`` section of ``README.md`` document, for example ``yolact_base_54_800000.pth``.

**Step 3**. Export the model to ONNX format.

1. Apply the `YOLACT_onnx_export.patch` patch to the repository. Refer to the :ref:`Create a Patch File <patch-file-yolact>` instructions if you do not have it:

   .. code-block:: sh

      git apply /path/to/patch/YOLACT_onnx_export.patch


2. Evaluate the YOLACT model to export it to ONNX format:

   .. code-block:: sh

      python3 eval.py \
          --trained_model=/path/to/yolact_base_54_800000.pth \
          --score_threshold=0.3 \
          --top_k=10 \
          --image=/path/to/image.jpg \
          --cuda=False


3. The script may fail, but you should get ``yolact.onnx`` file.

**Step 4**. Convert the model to the IR:

.. code-block:: sh

   mo --input_model /path/to/yolact.onnx


**Step 5**. Embed input preprocessing into the IR:

To get performance gain by offloading to the OpenVINO application of mean/scale values and RGB->BGR conversion, use the following model conversion API parameters:

* If the backbone of the model is Resnet50-FPN or Resnet101-FPN, use the following MO command line:

  .. code-block:: sh

     mo \
         --input_model /path/to/yolact.onnx \
         --reverse_input_channels \
         --mean_values "[123.68, 116.78, 103.94]" \
         --scale_values "[58.40, 57.12, 57.38]"


* If the backbone of the model is Darknet53-FPN, use the following MO command line:

  .. code-block:: sh

     mo \
         --input_model /path/to/yolact.onnx \
         --reverse_input_channels \
         --scale 255


