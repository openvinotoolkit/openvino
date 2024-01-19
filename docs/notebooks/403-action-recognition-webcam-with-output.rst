Human Action Recognition with OpenVINO™
=======================================

This notebook demonstrates live human action recognition with OpenVINO,
using the `Action Recognition
Models <https://docs.openvino.ai/2020.2/usergroup13.html>`__ from `Open
Model Zoo <https://github.com/openvinotoolkit/open_model_zoo>`__,
specifically an
`Encoder <https://docs.openvino.ai/2020.2/_models_intel_action_recognition_0001_encoder_description_action_recognition_0001_encoder.html>`__
and a
`Decoder <https://docs.openvino.ai/2020.2/_models_intel_action_recognition_0001_decoder_description_action_recognition_0001_decoder.html>`__.
Both models create a sequence to sequence (``"seq2seq"``) [1] system to
identify the human activities for `Kinetics-400
dataset <https://deepmind.com/research/open-source/kinetics>`__. The
models use the Video Transformer approach with ResNet34 encoder [2]. The
notebook shows how to create the following pipeline:

Final part of this notebook shows live inference results from a webcam.
Additionally, you can also upload a video file.

**NOTE**: To use a webcam, you must run this Jupyter notebook on a
computer with a webcam. If you run on a server, the webcam will not
work. However, you can still do inference on a video in the final step.

--------------

[1] seq2seq: Deep learning models that take a sequence of items to the
input and output. In this case, input: video frames, output: actions
sequence. This ``"seq2seq"`` is composed of an encoder and a decoder.
The encoder captures ``"context"`` of the inputs to be analyzed by the
decoder, and finally gets the human action and confidence.

[2] `Video
Transformer <https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)>`__
and
`ResNet34 <https://pytorch.org/vision/main/models/generated/torchvision.models.resnet34.html>`__.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `The models <#the-models>`__

   -  `Download the models <#download-the-models>`__
   -  `Load your labels <#load-your-labels>`__
   -  `Load the models <#load-the-models>`__

      -  `Model Initialization
         function <#model-initialization-function>`__
      -  `Initialization for Encoder and
         Decoder <#initialization-for-encoder-and-decoder>`__

   -  `Helper functions <#helper-functions>`__
   -  `AI Functions <#ai-functions>`__
   -  `Main Processing Function <#main-processing-function>`__
   -  `Run Action Recognition <#run-action-recognition>`__

.. code:: ipython3

    %pip install -q "openvino-dev>=2023.1.0"


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    

.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import collections
    import os
    import time
    from typing import Tuple, List
    
    import cv2
    import numpy as np
    from IPython import display
    import openvino as ov
    from openvino.runtime.ie_api import CompiledModel
    
    # Fetch `notebook_utils` module
    import urllib.request
    urllib.request.urlretrieve(
        url='https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/main/notebooks/utils/notebook_utils.py',
        filename='notebook_utils.py'
    )
    import notebook_utils as utils

The models
----------



Download the models
~~~~~~~~~~~~~~~~~~~



Use ``omz_downloader``, which is a command-line tool from the
``openvino-dev`` package. It automatically creates a directory structure
and downloads the selected model.

In this case you can use ``"action-recognition-0001"`` as a model name,
and the system automatically downloads the two models
``"action-recognition-0001-encoder"`` and
``"action-recognition-0001-decoder"``

   **NOTE**: If you want to download another model, such as
   ``"driver-action-recognition-adas-0002"``
   (``"driver-action-recognition-adas-0002-encoder"`` +
   ``"driver-action-recognition-adas-0002-decoder"``), replace the name
   of the model in the code below. Using a model outside the list can
   require different pre- and post-processing.

.. code:: ipython3

    # A directory where the model will be downloaded.
    base_model_dir = "model"
    # The name of the model from Open Model Zoo.
    model_name = "action-recognition-0001"
    # Selected precision (FP32, FP16, FP16-INT8).
    precision = "FP16"
    model_path_decoder = (
        f"model/intel/{model_name}/{model_name}-decoder/{precision}/{model_name}-decoder.xml"
    )
    model_path_encoder = (
        f"model/intel/{model_name}/{model_name}-encoder/{precision}/{model_name}-encoder.xml"
    )
    if not os.path.exists(model_path_decoder) or not os.path.exists(model_path_encoder):
        download_command = f"omz_downloader " \
                           f"--name {model_name} " \
                           f"--precision {precision} " \
                           f"--output_dir {base_model_dir}"
        ! $download_command


.. parsed-literal::

    ################|| Downloading action-recognition-0001-encoder ||################
    
    ========== Downloading model/intel/action-recognition-0001/action-recognition-0001-encoder/FP16/action-recognition-0001-encoder.xml


.. parsed-literal::

    ... 22%, 32 KB, 1261 KB/s, 0 seconds passed

.. parsed-literal::

    ... 45%, 64 KB, 1310 KB/s, 0 seconds passed

.. parsed-literal::

    ... 68%, 96 KB, 1099 KB/s, 0 seconds passed
... 91%, 128 KB, 1458 KB/s, 0 seconds passed
... 100%, 140 KB, 1592 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/action-recognition-0001/action-recognition-0001-encoder/FP16/action-recognition-0001-encoder.bin


.. parsed-literal::

    ... 0%, 32 KB, 1259 KB/s, 0 seconds passed
... 0%, 64 KB, 1244 KB/s, 0 seconds passed
... 0%, 96 KB, 1841 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 2072 KB/s, 0 seconds passed
... 0%, 160 KB, 2075 KB/s, 0 seconds passed
... 0%, 192 KB, 2179 KB/s, 0 seconds passed
... 0%, 224 KB, 2524 KB/s, 0 seconds passed
... 0%, 256 KB, 2824 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 288 KB, 2534 KB/s, 0 seconds passed
... 0%, 320 KB, 2562 KB/s, 0 seconds passed
... 0%, 352 KB, 2796 KB/s, 0 seconds passed
... 0%, 384 KB, 2845 KB/s, 0 seconds passed
... 1%, 416 KB, 2760 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 448 KB, 2770 KB/s, 0 seconds passed
... 1%, 480 KB, 2953 KB/s, 0 seconds passed
... 1%, 512 KB, 2736 KB/s, 0 seconds passed
... 1%, 544 KB, 2898 KB/s, 0 seconds passed
... 1%, 576 KB, 2906 KB/s, 0 seconds passed
... 1%, 608 KB, 3044 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 640 KB, 2860 KB/s, 0 seconds passed
... 1%, 672 KB, 2996 KB/s, 0 seconds passed
... 1%, 704 KB, 2998 KB/s, 0 seconds passed
... 1%, 736 KB, 3127 KB/s, 0 seconds passed
... 1%, 768 KB, 3121 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 800 KB, 3069 KB/s, 0 seconds passed
... 2%, 832 KB, 3065 KB/s, 0 seconds passed
... 2%, 864 KB, 3169 KB/s, 0 seconds passed
... 2%, 896 KB, 3166 KB/s, 0 seconds passed
... 2%, 928 KB, 3119 KB/s, 0 seconds passed
... 2%, 960 KB, 3116 KB/s, 0 seconds passed
... 2%, 992 KB, 3207 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 1024 KB, 3205 KB/s, 0 seconds passed
... 2%, 1056 KB, 3163 KB/s, 0 seconds passed
... 2%, 1088 KB, 3157 KB/s, 0 seconds passed
... 2%, 1120 KB, 3237 KB/s, 0 seconds passed
... 2%, 1152 KB, 3235 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 1184 KB, 3195 KB/s, 0 seconds passed
... 2%, 1216 KB, 3191 KB/s, 0 seconds passed
... 3%, 1248 KB, 3258 KB/s, 0 seconds passed
... 3%, 1280 KB, 3147 KB/s, 0 seconds passed
... 3%, 1312 KB, 3220 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 1344 KB, 3214 KB/s, 0 seconds passed
... 3%, 1376 KB, 3278 KB/s, 0 seconds passed
... 3%, 1408 KB, 3273 KB/s, 0 seconds passed
... 3%, 1440 KB, 3241 KB/s, 0 seconds passed
... 3%, 1472 KB, 3236 KB/s, 0 seconds passed
... 3%, 1504 KB, 3291 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 1536 KB, 3190 KB/s, 0 seconds passed
... 3%, 1568 KB, 3253 KB/s, 0 seconds passed
... 3%, 1600 KB, 3252 KB/s, 0 seconds passed
... 3%, 1632 KB, 3305 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 1664 KB, 3208 KB/s, 0 seconds passed
... 4%, 1696 KB, 3266 KB/s, 0 seconds passed
... 4%, 1728 KB, 3269 KB/s, 0 seconds passed
... 4%, 1760 KB, 3315 KB/s, 0 seconds passed
... 4%, 1792 KB, 3225 KB/s, 0 seconds passed
... 4%, 1824 KB, 3278 KB/s, 0 seconds passed
... 4%, 1856 KB, 3281 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 1888 KB, 3328 KB/s, 0 seconds passed
... 4%, 1920 KB, 3240 KB/s, 0 seconds passed
... 4%, 1952 KB, 3252 KB/s, 0 seconds passed
... 4%, 1984 KB, 3295 KB/s, 0 seconds passed
... 4%, 2016 KB, 3337 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 2048 KB, 3255 KB/s, 0 seconds passed
... 5%, 2080 KB, 3265 KB/s, 0 seconds passed
... 5%, 2112 KB, 3303 KB/s, 0 seconds passed
... 5%, 2144 KB, 3343 KB/s, 0 seconds passed
... 5%, 2176 KB, 3269 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 2208 KB, 3273 KB/s, 0 seconds passed
... 5%, 2240 KB, 3312 KB/s, 0 seconds passed
... 5%, 2272 KB, 3352 KB/s, 0 seconds passed
... 5%, 2304 KB, 3282 KB/s, 0 seconds passed
... 5%, 2336 KB, 3322 KB/s, 0 seconds passed
... 5%, 2368 KB, 3322 KB/s, 0 seconds passed
... 5%, 2400 KB, 3363 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 2432 KB, 3292 KB/s, 0 seconds passed
... 5%, 2464 KB, 3299 KB/s, 0 seconds passed
... 6%, 2496 KB, 3330 KB/s, 0 seconds passed
... 6%, 2528 KB, 3369 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 2560 KB, 3302 KB/s, 0 seconds passed
... 6%, 2592 KB, 3337 KB/s, 0 seconds passed
... 6%, 2624 KB, 3338 KB/s, 0 seconds passed
... 6%, 2656 KB, 3374 KB/s, 0 seconds passed
... 6%, 2688 KB, 3311 KB/s, 0 seconds passed
... 6%, 2720 KB, 3343 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 2752 KB, 3341 KB/s, 0 seconds passed
... 6%, 2784 KB, 3377 KB/s, 0 seconds passed
... 6%, 2816 KB, 3318 KB/s, 0 seconds passed
... 6%, 2848 KB, 3320 KB/s, 0 seconds passed
... 6%, 2880 KB, 3346 KB/s, 0 seconds passed
... 7%, 2912 KB, 3381 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 2944 KB, 3327 KB/s, 0 seconds passed
... 7%, 2976 KB, 3331 KB/s, 0 seconds passed
... 7%, 3008 KB, 3350 KB/s, 0 seconds passed
... 7%, 3040 KB, 3384 KB/s, 0 seconds passed
... 7%, 3072 KB, 3331 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 3104 KB, 3336 KB/s, 0 seconds passed
... 7%, 3136 KB, 3361 KB/s, 0 seconds passed
... 7%, 3168 KB, 3387 KB/s, 0 seconds passed
... 7%, 3200 KB, 3337 KB/s, 0 seconds passed
... 7%, 3232 KB, 3334 KB/s, 0 seconds passed
... 7%, 3264 KB, 3360 KB/s, 0 seconds passed
... 7%, 3296 KB, 3391 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 3328 KB, 3340 KB/s, 0 seconds passed
... 8%, 3360 KB, 3334 KB/s, 1 seconds passed
... 8%, 3392 KB, 3363 KB/s, 1 seconds passed
... 8%, 3424 KB, 3392 KB/s, 1 seconds passed

.. parsed-literal::

    ... 8%, 3456 KB, 3347 KB/s, 1 seconds passed
... 8%, 3488 KB, 3340 KB/s, 1 seconds passed
... 8%, 3520 KB, 3366 KB/s, 1 seconds passed
... 8%, 3552 KB, 3394 KB/s, 1 seconds passed
... 8%, 3584 KB, 3349 KB/s, 1 seconds passed

.. parsed-literal::

    ... 8%, 3616 KB, 3343 KB/s, 1 seconds passed
... 8%, 3648 KB, 3369 KB/s, 1 seconds passed
... 8%, 3680 KB, 3367 KB/s, 1 seconds passed
... 8%, 3712 KB, 3352 KB/s, 1 seconds passed
... 9%, 3744 KB, 3348 KB/s, 1 seconds passed
... 9%, 3776 KB, 3374 KB/s, 1 seconds passed
... 9%, 3808 KB, 3400 KB/s, 1 seconds passed

.. parsed-literal::

    ... 9%, 3840 KB, 3357 KB/s, 1 seconds passed
... 9%, 3872 KB, 3353 KB/s, 1 seconds passed
... 9%, 3904 KB, 3377 KB/s, 1 seconds passed
... 9%, 3936 KB, 3401 KB/s, 1 seconds passed
... 9%, 3968 KB, 3361 KB/s, 1 seconds passed

.. parsed-literal::

    ... 9%, 4000 KB, 3357 KB/s, 1 seconds passed
... 9%, 4032 KB, 3381 KB/s, 1 seconds passed
... 9%, 4064 KB, 3405 KB/s, 1 seconds passed
... 9%, 4096 KB, 3365 KB/s, 1 seconds passed
... 9%, 4128 KB, 3361 KB/s, 1 seconds passed
... 10%, 4160 KB, 3385 KB/s, 1 seconds passed
... 10%, 4192 KB, 3407 KB/s, 1 seconds passed

.. parsed-literal::

    ... 10%, 4224 KB, 3368 KB/s, 1 seconds passed
... 10%, 4256 KB, 3365 KB/s, 1 seconds passed
... 10%, 4288 KB, 3387 KB/s, 1 seconds passed
... 10%, 4320 KB, 3409 KB/s, 1 seconds passed

.. parsed-literal::

    ... 10%, 4352 KB, 3371 KB/s, 1 seconds passed
... 10%, 4384 KB, 3369 KB/s, 1 seconds passed
... 10%, 4416 KB, 3390 KB/s, 1 seconds passed
... 10%, 4448 KB, 3411 KB/s, 1 seconds passed
... 10%, 4480 KB, 3376 KB/s, 1 seconds passed

.. parsed-literal::

    ... 10%, 4512 KB, 3371 KB/s, 1 seconds passed
... 10%, 4544 KB, 3391 KB/s, 1 seconds passed
... 11%, 4576 KB, 3392 KB/s, 1 seconds passed
... 11%, 4608 KB, 3370 KB/s, 1 seconds passed
... 11%, 4640 KB, 3371 KB/s, 1 seconds passed
... 11%, 4672 KB, 3393 KB/s, 1 seconds passed
... 11%, 4704 KB, 3413 KB/s, 1 seconds passed

.. parsed-literal::

    ... 11%, 4736 KB, 3372 KB/s, 1 seconds passed
... 11%, 4768 KB, 3375 KB/s, 1 seconds passed
... 11%, 4800 KB, 3395 KB/s, 1 seconds passed
... 11%, 4832 KB, 3396 KB/s, 1 seconds passed

.. parsed-literal::

    ... 11%, 4864 KB, 3373 KB/s, 1 seconds passed
... 11%, 4896 KB, 3377 KB/s, 1 seconds passed
... 11%, 4928 KB, 3398 KB/s, 1 seconds passed
... 11%, 4960 KB, 3395 KB/s, 1 seconds passed
... 12%, 4992 KB, 3376 KB/s, 1 seconds passed
... 12%, 5024 KB, 3379 KB/s, 1 seconds passed
... 12%, 5056 KB, 3399 KB/s, 1 seconds passed

.. parsed-literal::

    ... 12%, 5088 KB, 3397 KB/s, 1 seconds passed
... 12%, 5120 KB, 3378 KB/s, 1 seconds passed
... 12%, 5152 KB, 3381 KB/s, 1 seconds passed
... 12%, 5184 KB, 3401 KB/s, 1 seconds passed
... 12%, 5216 KB, 3398 KB/s, 1 seconds passed

.. parsed-literal::

    ... 12%, 5248 KB, 3380 KB/s, 1 seconds passed
... 12%, 5280 KB, 3384 KB/s, 1 seconds passed
... 12%, 5312 KB, 3403 KB/s, 1 seconds passed
... 12%, 5344 KB, 3397 KB/s, 1 seconds passed
... 12%, 5376 KB, 3382 KB/s, 1 seconds passed

.. parsed-literal::

    ... 13%, 5408 KB, 3385 KB/s, 1 seconds passed
... 13%, 5440 KB, 3404 KB/s, 1 seconds passed
... 13%, 5472 KB, 3400 KB/s, 1 seconds passed
... 13%, 5504 KB, 3385 KB/s, 1 seconds passed
... 13%, 5536 KB, 3388 KB/s, 1 seconds passed
... 13%, 5568 KB, 3406 KB/s, 1 seconds passed

.. parsed-literal::

    ... 13%, 5600 KB, 3382 KB/s, 1 seconds passed
... 13%, 5632 KB, 3386 KB/s, 1 seconds passed
... 13%, 5664 KB, 3390 KB/s, 1 seconds passed
... 13%, 5696 KB, 3408 KB/s, 1 seconds passed
... 13%, 5728 KB, 3384 KB/s, 1 seconds passed

.. parsed-literal::

    ... 13%, 5760 KB, 3389 KB/s, 1 seconds passed
... 13%, 5792 KB, 3392 KB/s, 1 seconds passed
... 14%, 5824 KB, 3409 KB/s, 1 seconds passed
... 14%, 5856 KB, 3386 KB/s, 1 seconds passed
... 14%, 5888 KB, 3389 KB/s, 1 seconds passed
... 14%, 5920 KB, 3392 KB/s, 1 seconds passed

.. parsed-literal::

    ... 14%, 5952 KB, 3410 KB/s, 1 seconds passed
... 14%, 5984 KB, 3388 KB/s, 1 seconds passed
... 14%, 6016 KB, 3393 KB/s, 1 seconds passed
... 14%, 6048 KB, 3395 KB/s, 1 seconds passed
... 14%, 6080 KB, 3411 KB/s, 1 seconds passed

.. parsed-literal::

    ... 14%, 6112 KB, 3391 KB/s, 1 seconds passed
... 14%, 6144 KB, 3394 KB/s, 1 seconds passed
... 14%, 6176 KB, 3394 KB/s, 1 seconds passed
... 14%, 6208 KB, 3411 KB/s, 1 seconds passed
... 15%, 6240 KB, 3392 KB/s, 1 seconds passed
... 15%, 6272 KB, 3394 KB/s, 1 seconds passed

.. parsed-literal::

    ... 15%, 6304 KB, 3395 KB/s, 1 seconds passed
... 15%, 6336 KB, 3412 KB/s, 1 seconds passed
... 15%, 6368 KB, 3409 KB/s, 1 seconds passed
... 15%, 6400 KB, 3397 KB/s, 1 seconds passed
... 15%, 6432 KB, 3399 KB/s, 1 seconds passed
... 15%, 6464 KB, 3412 KB/s, 1 seconds passed

.. parsed-literal::

    ... 15%, 6496 KB, 3395 KB/s, 1 seconds passed
... 15%, 6528 KB, 3397 KB/s, 1 seconds passed
... 15%, 6560 KB, 3398 KB/s, 1 seconds passed
... 15%, 6592 KB, 3413 KB/s, 1 seconds passed
... 15%, 6624 KB, 3396 KB/s, 1 seconds passed

.. parsed-literal::

    ... 16%, 6656 KB, 3398 KB/s, 1 seconds passed
... 16%, 6688 KB, 3396 KB/s, 1 seconds passed
... 16%, 6720 KB, 3411 KB/s, 1 seconds passed
... 16%, 6752 KB, 3397 KB/s, 1 seconds passed
... 16%, 6784 KB, 3393 KB/s, 1 seconds passed

.. parsed-literal::

    ... 16%, 6816 KB, 3396 KB/s, 2 seconds passed
... 16%, 6848 KB, 3411 KB/s, 2 seconds passed
... 16%, 6880 KB, 3395 KB/s, 2 seconds passed
... 16%, 6912 KB, 3395 KB/s, 2 seconds passed
... 16%, 6944 KB, 3398 KB/s, 2 seconds passed
... 16%, 6976 KB, 3412 KB/s, 2 seconds passed

.. parsed-literal::

    ... 16%, 7008 KB, 3401 KB/s, 2 seconds passed
... 16%, 7040 KB, 3403 KB/s, 2 seconds passed
... 17%, 7072 KB, 3401 KB/s, 2 seconds passed
... 17%, 7104 KB, 3414 KB/s, 2 seconds passed
... 17%, 7136 KB, 3401 KB/s, 2 seconds passed

.. parsed-literal::

    ... 17%, 7168 KB, 3404 KB/s, 2 seconds passed
... 17%, 7200 KB, 3401 KB/s, 2 seconds passed
... 17%, 7232 KB, 3414 KB/s, 2 seconds passed
... 17%, 7264 KB, 3400 KB/s, 2 seconds passed
... 17%, 7296 KB, 3399 KB/s, 2 seconds passed
... 17%, 7328 KB, 3402 KB/s, 2 seconds passed
... 17%, 7360 KB, 3416 KB/s, 2 seconds passed

.. parsed-literal::

    ... 17%, 7392 KB, 3402 KB/s, 2 seconds passed
... 17%, 7424 KB, 3406 KB/s, 2 seconds passed
... 17%, 7456 KB, 3404 KB/s, 2 seconds passed
... 18%, 7488 KB, 3417 KB/s, 2 seconds passed

.. parsed-literal::

    ... 18%, 7520 KB, 3405 KB/s, 2 seconds passed
... 18%, 7552 KB, 3407 KB/s, 2 seconds passed
... 18%, 7584 KB, 3405 KB/s, 2 seconds passed
... 18%, 7616 KB, 3418 KB/s, 2 seconds passed
... 18%, 7648 KB, 3407 KB/s, 2 seconds passed
... 18%, 7680 KB, 3410 KB/s, 2 seconds passed

.. parsed-literal::

    ... 18%, 7712 KB, 3407 KB/s, 2 seconds passed
... 18%, 7744 KB, 3420 KB/s, 2 seconds passed
... 18%, 7776 KB, 3408 KB/s, 2 seconds passed
... 18%, 7808 KB, 3412 KB/s, 2 seconds passed
... 18%, 7840 KB, 3408 KB/s, 2 seconds passed
... 18%, 7872 KB, 3421 KB/s, 2 seconds passed

.. parsed-literal::

    ... 19%, 7904 KB, 3410 KB/s, 2 seconds passed
... 19%, 7936 KB, 3405 KB/s, 2 seconds passed
... 19%, 7968 KB, 3408 KB/s, 2 seconds passed
... 19%, 8000 KB, 3421 KB/s, 2 seconds passed
... 19%, 8032 KB, 3408 KB/s, 2 seconds passed

.. parsed-literal::

    ... 19%, 8064 KB, 3407 KB/s, 2 seconds passed
... 19%, 8096 KB, 3410 KB/s, 2 seconds passed
... 19%, 8128 KB, 3422 KB/s, 2 seconds passed
... 19%, 8160 KB, 3409 KB/s, 2 seconds passed
... 19%, 8192 KB, 3407 KB/s, 2 seconds passed
... 19%, 8224 KB, 3411 KB/s, 2 seconds passed
... 19%, 8256 KB, 3423 KB/s, 2 seconds passed

.. parsed-literal::

    ... 19%, 8288 KB, 3405 KB/s, 2 seconds passed
... 20%, 8320 KB, 3408 KB/s, 2 seconds passed
... 20%, 8352 KB, 3412 KB/s, 2 seconds passed
... 20%, 8384 KB, 3424 KB/s, 2 seconds passed

.. parsed-literal::

    ... 20%, 8416 KB, 3406 KB/s, 2 seconds passed
... 20%, 8448 KB, 3409 KB/s, 2 seconds passed
... 20%, 8480 KB, 3413 KB/s, 2 seconds passed
... 20%, 8512 KB, 3424 KB/s, 2 seconds passed
... 20%, 8544 KB, 3407 KB/s, 2 seconds passed

.. parsed-literal::

    ... 20%, 8576 KB, 3409 KB/s, 2 seconds passed
... 20%, 8608 KB, 3413 KB/s, 2 seconds passed
... 20%, 8640 KB, 3425 KB/s, 2 seconds passed
... 20%, 8672 KB, 3405 KB/s, 2 seconds passed
... 20%, 8704 KB, 3410 KB/s, 2 seconds passed
... 21%, 8736 KB, 3412 KB/s, 2 seconds passed
... 21%, 8768 KB, 3424 KB/s, 2 seconds passed

.. parsed-literal::

    ... 21%, 8800 KB, 3406 KB/s, 2 seconds passed
... 21%, 8832 KB, 3410 KB/s, 2 seconds passed
... 21%, 8864 KB, 3413 KB/s, 2 seconds passed
... 21%, 8896 KB, 3424 KB/s, 2 seconds passed

.. parsed-literal::

    ... 21%, 8928 KB, 3410 KB/s, 2 seconds passed
... 21%, 8960 KB, 3412 KB/s, 2 seconds passed
... 21%, 8992 KB, 3415 KB/s, 2 seconds passed
... 21%, 9024 KB, 3425 KB/s, 2 seconds passed
... 21%, 9056 KB, 3408 KB/s, 2 seconds passed
... 21%, 9088 KB, 3407 KB/s, 2 seconds passed

.. parsed-literal::

    ... 21%, 9120 KB, 3414 KB/s, 2 seconds passed
... 22%, 9152 KB, 3425 KB/s, 2 seconds passed
... 22%, 9184 KB, 3409 KB/s, 2 seconds passed
... 22%, 9216 KB, 3410 KB/s, 2 seconds passed
... 22%, 9248 KB, 3415 KB/s, 2 seconds passed
... 22%, 9280 KB, 3425 KB/s, 2 seconds passed

.. parsed-literal::

    ... 22%, 9312 KB, 3409 KB/s, 2 seconds passed
... 22%, 9344 KB, 3411 KB/s, 2 seconds passed
... 22%, 9376 KB, 3415 KB/s, 2 seconds passed
... 22%, 9408 KB, 3425 KB/s, 2 seconds passed
... 22%, 9440 KB, 3411 KB/s, 2 seconds passed

.. parsed-literal::

    ... 22%, 9472 KB, 3416 KB/s, 2 seconds passed
... 22%, 9504 KB, 3418 KB/s, 2 seconds passed
... 22%, 9536 KB, 3426 KB/s, 2 seconds passed
... 23%, 9568 KB, 3416 KB/s, 2 seconds passed
... 23%, 9600 KB, 3417 KB/s, 2 seconds passed
... 23%, 9632 KB, 3419 KB/s, 2 seconds passed
... 23%, 9664 KB, 3427 KB/s, 2 seconds passed

.. parsed-literal::

    ... 23%, 9696 KB, 3417 KB/s, 2 seconds passed
... 23%, 9728 KB, 3411 KB/s, 2 seconds passed
... 23%, 9760 KB, 3417 KB/s, 2 seconds passed
... 23%, 9792 KB, 3425 KB/s, 2 seconds passed

.. parsed-literal::

    ... 23%, 9824 KB, 3413 KB/s, 2 seconds passed
... 23%, 9856 KB, 3412 KB/s, 2 seconds passed
... 23%, 9888 KB, 3418 KB/s, 2 seconds passed
... 23%, 9920 KB, 3426 KB/s, 2 seconds passed
... 23%, 9952 KB, 3414 KB/s, 2 seconds passed
... 24%, 9984 KB, 3413 KB/s, 2 seconds passed

.. parsed-literal::

    ... 24%, 10016 KB, 3418 KB/s, 2 seconds passed
... 24%, 10048 KB, 3428 KB/s, 2 seconds passed
... 24%, 10080 KB, 3415 KB/s, 2 seconds passed
... 24%, 10112 KB, 3414 KB/s, 2 seconds passed
... 24%, 10144 KB, 3419 KB/s, 2 seconds passed
... 24%, 10176 KB, 3429 KB/s, 2 seconds passed

.. parsed-literal::

    ... 24%, 10208 KB, 3417 KB/s, 2 seconds passed
... 24%, 10240 KB, 3417 KB/s, 2 seconds passed
... 24%, 10272 KB, 3420 KB/s, 3 seconds passed
... 24%, 10304 KB, 3430 KB/s, 3 seconds passed
... 24%, 10336 KB, 3418 KB/s, 3 seconds passed

.. parsed-literal::

    ... 24%, 10368 KB, 3416 KB/s, 3 seconds passed
... 25%, 10400 KB, 3421 KB/s, 3 seconds passed
... 25%, 10432 KB, 3430 KB/s, 3 seconds passed
... 25%, 10464 KB, 3417 KB/s, 3 seconds passed
... 25%, 10496 KB, 3417 KB/s, 3 seconds passed
... 25%, 10528 KB, 3421 KB/s, 3 seconds passed
... 25%, 10560 KB, 3431 KB/s, 3 seconds passed

.. parsed-literal::

    ... 25%, 10592 KB, 3418 KB/s, 3 seconds passed
... 25%, 10624 KB, 3418 KB/s, 3 seconds passed
... 25%, 10656 KB, 3422 KB/s, 3 seconds passed
... 25%, 10688 KB, 3431 KB/s, 3 seconds passed

.. parsed-literal::

    ... 25%, 10720 KB, 3419 KB/s, 3 seconds passed
... 25%, 10752 KB, 3418 KB/s, 3 seconds passed
... 25%, 10784 KB, 3422 KB/s, 3 seconds passed
... 26%, 10816 KB, 3431 KB/s, 3 seconds passed
... 26%, 10848 KB, 3419 KB/s, 3 seconds passed

.. parsed-literal::

    ... 26%, 10880 KB, 3418 KB/s, 3 seconds passed
... 26%, 10912 KB, 3421 KB/s, 3 seconds passed
... 26%, 10944 KB, 3431 KB/s, 3 seconds passed
... 26%, 10976 KB, 3420 KB/s, 3 seconds passed
... 26%, 11008 KB, 3416 KB/s, 3 seconds passed
... 26%, 11040 KB, 3422 KB/s, 3 seconds passed
... 26%, 11072 KB, 3431 KB/s, 3 seconds passed

.. parsed-literal::

    ... 26%, 11104 KB, 3418 KB/s, 3 seconds passed
... 26%, 11136 KB, 3417 KB/s, 3 seconds passed
... 26%, 11168 KB, 3423 KB/s, 3 seconds passed
... 26%, 11200 KB, 3431 KB/s, 3 seconds passed

.. parsed-literal::

    ... 27%, 11232 KB, 3419 KB/s, 3 seconds passed
... 27%, 11264 KB, 3418 KB/s, 3 seconds passed
... 27%, 11296 KB, 3423 KB/s, 3 seconds passed
... 27%, 11328 KB, 3432 KB/s, 3 seconds passed
... 27%, 11360 KB, 3420 KB/s, 3 seconds passed
... 27%, 11392 KB, 3418 KB/s, 3 seconds passed

.. parsed-literal::

    ... 27%, 11424 KB, 3423 KB/s, 3 seconds passed
... 27%, 11456 KB, 3421 KB/s, 3 seconds passed
... 27%, 11488 KB, 3419 KB/s, 3 seconds passed
... 27%, 11520 KB, 3418 KB/s, 3 seconds passed
... 27%, 11552 KB, 3424 KB/s, 3 seconds passed
... 27%, 11584 KB, 3432 KB/s, 3 seconds passed

.. parsed-literal::

    ... 27%, 11616 KB, 3419 KB/s, 3 seconds passed
... 28%, 11648 KB, 3419 KB/s, 3 seconds passed
... 28%, 11680 KB, 3422 KB/s, 3 seconds passed
... 28%, 11712 KB, 3422 KB/s, 3 seconds passed
... 28%, 11744 KB, 3419 KB/s, 3 seconds passed

.. parsed-literal::

    ... 28%, 11776 KB, 3419 KB/s, 3 seconds passed
... 28%, 11808 KB, 3423 KB/s, 3 seconds passed
... 28%, 11840 KB, 3423 KB/s, 3 seconds passed
... 28%, 11872 KB, 3420 KB/s, 3 seconds passed
... 28%, 11904 KB, 3419 KB/s, 3 seconds passed
... 28%, 11936 KB, 3424 KB/s, 3 seconds passed

.. parsed-literal::

    ... 28%, 11968 KB, 3424 KB/s, 3 seconds passed
... 28%, 12000 KB, 3421 KB/s, 3 seconds passed
... 28%, 12032 KB, 3421 KB/s, 3 seconds passed
... 29%, 12064 KB, 3426 KB/s, 3 seconds passed
... 29%, 12096 KB, 3433 KB/s, 3 seconds passed

.. parsed-literal::

    ... 29%, 12128 KB, 3421 KB/s, 3 seconds passed
... 29%, 12160 KB, 3421 KB/s, 3 seconds passed
... 29%, 12192 KB, 3426 KB/s, 3 seconds passed
... 29%, 12224 KB, 3433 KB/s, 3 seconds passed
... 29%, 12256 KB, 3422 KB/s, 3 seconds passed
... 29%, 12288 KB, 3421 KB/s, 3 seconds passed

.. parsed-literal::

    ... 29%, 12320 KB, 3425 KB/s, 3 seconds passed
... 29%, 12352 KB, 3423 KB/s, 3 seconds passed
... 29%, 12384 KB, 3422 KB/s, 3 seconds passed
... 29%, 12416 KB, 3422 KB/s, 3 seconds passed
... 29%, 12448 KB, 3426 KB/s, 3 seconds passed

.. parsed-literal::

    ... 30%, 12480 KB, 3423 KB/s, 3 seconds passed
... 30%, 12512 KB, 3423 KB/s, 3 seconds passed
... 30%, 12544 KB, 3422 KB/s, 3 seconds passed
... 30%, 12576 KB, 3427 KB/s, 3 seconds passed
... 30%, 12608 KB, 3424 KB/s, 3 seconds passed
... 30%, 12640 KB, 3423 KB/s, 3 seconds passed

.. parsed-literal::

    ... 30%, 12672 KB, 3423 KB/s, 3 seconds passed
... 30%, 12704 KB, 3427 KB/s, 3 seconds passed
... 30%, 12736 KB, 3425 KB/s, 3 seconds passed
... 30%, 12768 KB, 3424 KB/s, 3 seconds passed
... 30%, 12800 KB, 3424 KB/s, 3 seconds passed
... 30%, 12832 KB, 3428 KB/s, 3 seconds passed

.. parsed-literal::

    ... 30%, 12864 KB, 3423 KB/s, 3 seconds passed
... 31%, 12896 KB, 3425 KB/s, 3 seconds passed
... 31%, 12928 KB, 3423 KB/s, 3 seconds passed
... 31%, 12960 KB, 3428 KB/s, 3 seconds passed
... 31%, 12992 KB, 3424 KB/s, 3 seconds passed

.. parsed-literal::

    ... 31%, 13024 KB, 3425 KB/s, 3 seconds passed
... 31%, 13056 KB, 3423 KB/s, 3 seconds passed
... 31%, 13088 KB, 3428 KB/s, 3 seconds passed
... 31%, 13120 KB, 3424 KB/s, 3 seconds passed
... 31%, 13152 KB, 3422 KB/s, 3 seconds passed

.. parsed-literal::

    ... 31%, 13184 KB, 3424 KB/s, 3 seconds passed
... 31%, 13216 KB, 3428 KB/s, 3 seconds passed
... 31%, 13248 KB, 3425 KB/s, 3 seconds passed
... 31%, 13280 KB, 3422 KB/s, 3 seconds passed
... 32%, 13312 KB, 3424 KB/s, 3 seconds passed
... 32%, 13344 KB, 3429 KB/s, 3 seconds passed

.. parsed-literal::

    ... 32%, 13376 KB, 3423 KB/s, 3 seconds passed
... 32%, 13408 KB, 3423 KB/s, 3 seconds passed
... 32%, 13440 KB, 3426 KB/s, 3 seconds passed
... 32%, 13472 KB, 3430 KB/s, 3 seconds passed
... 32%, 13504 KB, 3426 KB/s, 3 seconds passed
... 32%, 13536 KB, 3428 KB/s, 3 seconds passed

.. parsed-literal::

    ... 32%, 13568 KB, 3426 KB/s, 3 seconds passed
... 32%, 13600 KB, 3429 KB/s, 3 seconds passed
... 32%, 13632 KB, 3422 KB/s, 3 seconds passed
... 32%, 13664 KB, 3420 KB/s, 3 seconds passed
... 32%, 13696 KB, 3425 KB/s, 3 seconds passed
... 33%, 13728 KB, 3430 KB/s, 4 seconds passed

.. parsed-literal::

    ... 33%, 13760 KB, 3422 KB/s, 4 seconds passed
... 33%, 13792 KB, 3421 KB/s, 4 seconds passed
... 33%, 13824 KB, 3425 KB/s, 4 seconds passed
... 33%, 13856 KB, 3431 KB/s, 4 seconds passed
... 33%, 13888 KB, 3427 KB/s, 4 seconds passed

.. parsed-literal::

    ... 33%, 13920 KB, 3425 KB/s, 4 seconds passed
... 33%, 13952 KB, 3426 KB/s, 4 seconds passed
... 33%, 13984 KB, 3430 KB/s, 4 seconds passed
... 33%, 14016 KB, 3426 KB/s, 4 seconds passed
... 33%, 14048 KB, 3422 KB/s, 4 seconds passed

.. parsed-literal::

    ... 33%, 14080 KB, 3426 KB/s, 4 seconds passed
... 33%, 14112 KB, 3432 KB/s, 4 seconds passed
... 34%, 14144 KB, 3429 KB/s, 4 seconds passed
... 34%, 14176 KB, 3431 KB/s, 4 seconds passed
... 34%, 14208 KB, 3428 KB/s, 4 seconds passed
... 34%, 14240 KB, 3432 KB/s, 4 seconds passed

.. parsed-literal::

    ... 34%, 14272 KB, 3429 KB/s, 4 seconds passed
... 34%, 14304 KB, 3422 KB/s, 4 seconds passed
... 34%, 14336 KB, 3426 KB/s, 4 seconds passed
... 34%, 14368 KB, 3430 KB/s, 4 seconds passed
... 34%, 14400 KB, 3426 KB/s, 4 seconds passed

.. parsed-literal::

    ... 34%, 14432 KB, 3422 KB/s, 4 seconds passed
... 34%, 14464 KB, 3427 KB/s, 4 seconds passed
... 34%, 14496 KB, 3431 KB/s, 4 seconds passed
... 34%, 14528 KB, 3427 KB/s, 4 seconds passed
... 35%, 14560 KB, 3422 KB/s, 4 seconds passed
... 35%, 14592 KB, 3427 KB/s, 4 seconds passed

.. parsed-literal::

    ... 35%, 14624 KB, 3431 KB/s, 4 seconds passed
... 35%, 14656 KB, 3426 KB/s, 4 seconds passed
... 35%, 14688 KB, 3421 KB/s, 4 seconds passed
... 35%, 14720 KB, 3428 KB/s, 4 seconds passed
... 35%, 14752 KB, 3432 KB/s, 4 seconds passed

.. parsed-literal::

    ... 35%, 14784 KB, 3426 KB/s, 4 seconds passed
... 35%, 14816 KB, 3422 KB/s, 4 seconds passed
... 35%, 14848 KB, 3428 KB/s, 4 seconds passed
... 35%, 14880 KB, 3432 KB/s, 4 seconds passed
... 35%, 14912 KB, 3425 KB/s, 4 seconds passed

.. parsed-literal::

    ... 35%, 14944 KB, 3422 KB/s, 4 seconds passed
... 36%, 14976 KB, 3428 KB/s, 4 seconds passed
... 36%, 15008 KB, 3432 KB/s, 4 seconds passed
... 36%, 15040 KB, 3426 KB/s, 4 seconds passed
... 36%, 15072 KB, 3422 KB/s, 4 seconds passed
... 36%, 15104 KB, 3428 KB/s, 4 seconds passed
... 36%, 15136 KB, 3432 KB/s, 4 seconds passed

.. parsed-literal::

    ... 36%, 15168 KB, 3426 KB/s, 4 seconds passed
... 36%, 15200 KB, 3422 KB/s, 4 seconds passed
... 36%, 15232 KB, 3428 KB/s, 4 seconds passed
... 36%, 15264 KB, 3432 KB/s, 4 seconds passed
... 36%, 15296 KB, 3426 KB/s, 4 seconds passed

.. parsed-literal::

    ... 36%, 15328 KB, 3422 KB/s, 4 seconds passed
... 36%, 15360 KB, 3428 KB/s, 4 seconds passed
... 37%, 15392 KB, 3431 KB/s, 4 seconds passed
... 37%, 15424 KB, 3424 KB/s, 4 seconds passed

.. parsed-literal::

    ... 37%, 15456 KB, 3422 KB/s, 4 seconds passed
... 37%, 15488 KB, 3428 KB/s, 4 seconds passed
... 37%, 15520 KB, 3431 KB/s, 4 seconds passed
... 37%, 15552 KB, 3425 KB/s, 4 seconds passed
... 37%, 15584 KB, 3423 KB/s, 4 seconds passed
... 37%, 15616 KB, 3428 KB/s, 4 seconds passed
... 37%, 15648 KB, 3431 KB/s, 4 seconds passed

.. parsed-literal::

    ... 37%, 15680 KB, 3423 KB/s, 4 seconds passed
... 37%, 15712 KB, 3422 KB/s, 4 seconds passed
... 37%, 15744 KB, 3426 KB/s, 4 seconds passed
... 37%, 15776 KB, 3429 KB/s, 4 seconds passed

.. parsed-literal::

    ... 38%, 15808 KB, 3423 KB/s, 4 seconds passed
... 38%, 15840 KB, 3420 KB/s, 4 seconds passed
... 38%, 15872 KB, 3426 KB/s, 4 seconds passed
... 38%, 15904 KB, 3424 KB/s, 4 seconds passed
... 38%, 15936 KB, 3423 KB/s, 4 seconds passed
... 38%, 15968 KB, 3424 KB/s, 4 seconds passed
... 38%, 16000 KB, 3427 KB/s, 4 seconds passed

.. parsed-literal::

    ... 38%, 16032 KB, 3430 KB/s, 4 seconds passed
... 38%, 16064 KB, 3424 KB/s, 4 seconds passed
... 38%, 16096 KB, 3424 KB/s, 4 seconds passed
... 38%, 16128 KB, 3428 KB/s, 4 seconds passed
... 38%, 16160 KB, 3430 KB/s, 4 seconds passed

.. parsed-literal::

    ... 38%, 16192 KB, 3425 KB/s, 4 seconds passed
... 39%, 16224 KB, 3424 KB/s, 4 seconds passed
... 39%, 16256 KB, 3428 KB/s, 4 seconds passed
... 39%, 16288 KB, 3431 KB/s, 4 seconds passed
... 39%, 16320 KB, 3426 KB/s, 4 seconds passed

.. parsed-literal::

    ... 39%, 16352 KB, 3422 KB/s, 4 seconds passed
... 39%, 16384 KB, 3427 KB/s, 4 seconds passed
... 39%, 16416 KB, 3424 KB/s, 4 seconds passed
... 39%, 16448 KB, 3425 KB/s, 4 seconds passed
... 39%, 16480 KB, 3422 KB/s, 4 seconds passed
... 39%, 16512 KB, 3428 KB/s, 4 seconds passed

.. parsed-literal::

    ... 39%, 16544 KB, 3425 KB/s, 4 seconds passed
... 39%, 16576 KB, 3425 KB/s, 4 seconds passed
... 39%, 16608 KB, 3423 KB/s, 4 seconds passed
... 40%, 16640 KB, 3428 KB/s, 4 seconds passed
... 40%, 16672 KB, 3431 KB/s, 4 seconds passed

.. parsed-literal::

    ... 40%, 16704 KB, 3426 KB/s, 4 seconds passed
... 40%, 16736 KB, 3424 KB/s, 4 seconds passed
... 40%, 16768 KB, 3429 KB/s, 4 seconds passed
... 40%, 16800 KB, 3432 KB/s, 4 seconds passed
... 40%, 16832 KB, 3426 KB/s, 4 seconds passed
... 40%, 16864 KB, 3425 KB/s, 4 seconds passed

.. parsed-literal::

    ... 40%, 16896 KB, 3428 KB/s, 4 seconds passed
... 40%, 16928 KB, 3432 KB/s, 4 seconds passed
... 40%, 16960 KB, 3427 KB/s, 4 seconds passed
... 40%, 16992 KB, 3423 KB/s, 4 seconds passed
... 40%, 17024 KB, 3427 KB/s, 4 seconds passed
... 41%, 17056 KB, 3432 KB/s, 4 seconds passed

.. parsed-literal::

    ... 41%, 17088 KB, 3427 KB/s, 4 seconds passed
... 41%, 17120 KB, 3423 KB/s, 5 seconds passed
... 41%, 17152 KB, 3427 KB/s, 5 seconds passed
... 41%, 17184 KB, 3432 KB/s, 5 seconds passed
... 41%, 17216 KB, 3427 KB/s, 5 seconds passed

.. parsed-literal::

    ... 41%, 17248 KB, 3423 KB/s, 5 seconds passed
... 41%, 17280 KB, 3428 KB/s, 5 seconds passed
... 41%, 17312 KB, 3427 KB/s, 5 seconds passed
... 41%, 17344 KB, 3427 KB/s, 5 seconds passed
... 41%, 17376 KB, 3424 KB/s, 5 seconds passed
... 41%, 17408 KB, 3428 KB/s, 5 seconds passed

.. parsed-literal::

    ... 41%, 17440 KB, 3427 KB/s, 5 seconds passed
... 42%, 17472 KB, 3427 KB/s, 5 seconds passed
... 42%, 17504 KB, 3424 KB/s, 5 seconds passed
... 42%, 17536 KB, 3429 KB/s, 5 seconds passed
... 42%, 17568 KB, 3428 KB/s, 5 seconds passed

.. parsed-literal::

    ... 42%, 17600 KB, 3427 KB/s, 5 seconds passed
... 42%, 17632 KB, 3424 KB/s, 5 seconds passed
... 42%, 17664 KB, 3429 KB/s, 5 seconds passed
... 42%, 17696 KB, 3428 KB/s, 5 seconds passed
... 42%, 17728 KB, 3427 KB/s, 5 seconds passed

.. parsed-literal::

    ... 42%, 17760 KB, 3424 KB/s, 5 seconds passed
... 42%, 17792 KB, 3429 KB/s, 5 seconds passed
... 42%, 17824 KB, 3428 KB/s, 5 seconds passed
... 42%, 17856 KB, 3428 KB/s, 5 seconds passed
... 43%, 17888 KB, 3425 KB/s, 5 seconds passed
... 43%, 17920 KB, 3430 KB/s, 5 seconds passed

.. parsed-literal::

    ... 43%, 17952 KB, 3428 KB/s, 5 seconds passed
... 43%, 17984 KB, 3427 KB/s, 5 seconds passed
... 43%, 18016 KB, 3425 KB/s, 5 seconds passed
... 43%, 18048 KB, 3430 KB/s, 5 seconds passed
... 43%, 18080 KB, 3428 KB/s, 5 seconds passed

.. parsed-literal::

    ... 43%, 18112 KB, 3422 KB/s, 5 seconds passed
... 43%, 18144 KB, 3425 KB/s, 5 seconds passed
... 43%, 18176 KB, 3430 KB/s, 5 seconds passed
... 43%, 18208 KB, 3429 KB/s, 5 seconds passed
... 43%, 18240 KB, 3429 KB/s, 5 seconds passed
... 43%, 18272 KB, 3426 KB/s, 5 seconds passed
... 44%, 18304 KB, 3431 KB/s, 5 seconds passed

.. parsed-literal::

    ... 44%, 18336 KB, 3429 KB/s, 5 seconds passed
... 44%, 18368 KB, 3429 KB/s, 5 seconds passed
... 44%, 18400 KB, 3425 KB/s, 5 seconds passed
... 44%, 18432 KB, 3431 KB/s, 5 seconds passed
... 44%, 18464 KB, 3429 KB/s, 5 seconds passed

.. parsed-literal::

    ... 44%, 18496 KB, 3430 KB/s, 5 seconds passed
... 44%, 18528 KB, 3427 KB/s, 5 seconds passed
... 44%, 18560 KB, 3430 KB/s, 5 seconds passed
... 44%, 18592 KB, 3429 KB/s, 5 seconds passed
... 44%, 18624 KB, 3430 KB/s, 5 seconds passed

.. parsed-literal::

    ... 44%, 18656 KB, 3427 KB/s, 5 seconds passed
... 44%, 18688 KB, 3430 KB/s, 5 seconds passed
... 45%, 18720 KB, 3429 KB/s, 5 seconds passed
... 45%, 18752 KB, 3430 KB/s, 5 seconds passed
... 45%, 18784 KB, 3428 KB/s, 5 seconds passed
... 45%, 18816 KB, 3430 KB/s, 5 seconds passed

.. parsed-literal::

    ... 45%, 18848 KB, 3429 KB/s, 5 seconds passed
... 45%, 18880 KB, 3424 KB/s, 5 seconds passed
... 45%, 18912 KB, 3426 KB/s, 5 seconds passed
... 45%, 18944 KB, 3430 KB/s, 5 seconds passed
... 45%, 18976 KB, 3429 KB/s, 5 seconds passed

.. parsed-literal::

    ... 45%, 19008 KB, 3424 KB/s, 5 seconds passed
... 45%, 19040 KB, 3426 KB/s, 5 seconds passed
... 45%, 19072 KB, 3430 KB/s, 5 seconds passed
... 45%, 19104 KB, 3426 KB/s, 5 seconds passed
... 46%, 19136 KB, 3425 KB/s, 5 seconds passed

.. parsed-literal::

    ... 46%, 19168 KB, 3426 KB/s, 5 seconds passed
... 46%, 19200 KB, 3430 KB/s, 5 seconds passed
... 46%, 19232 KB, 3426 KB/s, 5 seconds passed
... 46%, 19264 KB, 3425 KB/s, 5 seconds passed
... 46%, 19296 KB, 3427 KB/s, 5 seconds passed
... 46%, 19328 KB, 3431 KB/s, 5 seconds passed

.. parsed-literal::

    ... 46%, 19360 KB, 3427 KB/s, 5 seconds passed
... 46%, 19392 KB, 3425 KB/s, 5 seconds passed
... 46%, 19424 KB, 3427 KB/s, 5 seconds passed
... 46%, 19456 KB, 3431 KB/s, 5 seconds passed
... 46%, 19488 KB, 3427 KB/s, 5 seconds passed

.. parsed-literal::

    ... 46%, 19520 KB, 3426 KB/s, 5 seconds passed
... 47%, 19552 KB, 3427 KB/s, 5 seconds passed
... 47%, 19584 KB, 3431 KB/s, 5 seconds passed
... 47%, 19616 KB, 3427 KB/s, 5 seconds passed
... 47%, 19648 KB, 3425 KB/s, 5 seconds passed
... 47%, 19680 KB, 3428 KB/s, 5 seconds passed
... 47%, 19712 KB, 3432 KB/s, 5 seconds passed

.. parsed-literal::

    ... 47%, 19744 KB, 3428 KB/s, 5 seconds passed
... 47%, 19776 KB, 3426 KB/s, 5 seconds passed
... 47%, 19808 KB, 3428 KB/s, 5 seconds passed
... 47%, 19840 KB, 3432 KB/s, 5 seconds passed
... 47%, 19872 KB, 3428 KB/s, 5 seconds passed

.. parsed-literal::

    ... 47%, 19904 KB, 3426 KB/s, 5 seconds passed
... 47%, 19936 KB, 3428 KB/s, 5 seconds passed
... 48%, 19968 KB, 3432 KB/s, 5 seconds passed
... 48%, 20000 KB, 3428 KB/s, 5 seconds passed
... 48%, 20032 KB, 3426 KB/s, 5 seconds passed

.. parsed-literal::

    ... 48%, 20064 KB, 3428 KB/s, 5 seconds passed
... 48%, 20096 KB, 3432 KB/s, 5 seconds passed
... 48%, 20128 KB, 3429 KB/s, 5 seconds passed
... 48%, 20160 KB, 3426 KB/s, 5 seconds passed
... 48%, 20192 KB, 3428 KB/s, 5 seconds passed
... 48%, 20224 KB, 3432 KB/s, 5 seconds passed

.. parsed-literal::

    ... 48%, 20256 KB, 3429 KB/s, 5 seconds passed
... 48%, 20288 KB, 3427 KB/s, 5 seconds passed
... 48%, 20320 KB, 3429 KB/s, 5 seconds passed
... 48%, 20352 KB, 3433 KB/s, 5 seconds passed
... 49%, 20384 KB, 3432 KB/s, 5 seconds passed

.. parsed-literal::

    ... 49%, 20416 KB, 3427 KB/s, 5 seconds passed
... 49%, 20448 KB, 3429 KB/s, 5 seconds passed
... 49%, 20480 KB, 3434 KB/s, 5 seconds passed
... 49%, 20512 KB, 3432 KB/s, 5 seconds passed
... 49%, 20544 KB, 3428 KB/s, 5 seconds passed
... 49%, 20576 KB, 3429 KB/s, 5 seconds passed
... 49%, 20608 KB, 3433 KB/s, 6 seconds passed

.. parsed-literal::

    ... 49%, 20640 KB, 3429 KB/s, 6 seconds passed
... 49%, 20672 KB, 3428 KB/s, 6 seconds passed
... 49%, 20704 KB, 3430 KB/s, 6 seconds passed
... 49%, 20736 KB, 3428 KB/s, 6 seconds passed

.. parsed-literal::

    ... 49%, 20768 KB, 3428 KB/s, 6 seconds passed
... 50%, 20800 KB, 3427 KB/s, 6 seconds passed
... 50%, 20832 KB, 3429 KB/s, 6 seconds passed
... 50%, 20864 KB, 3434 KB/s, 6 seconds passed
... 50%, 20896 KB, 3428 KB/s, 6 seconds passed
... 50%, 20928 KB, 3428 KB/s, 6 seconds passed

.. parsed-literal::

    ... 50%, 20960 KB, 3430 KB/s, 6 seconds passed
... 50%, 20992 KB, 3435 KB/s, 6 seconds passed
... 50%, 21024 KB, 3429 KB/s, 6 seconds passed
... 50%, 21056 KB, 3428 KB/s, 6 seconds passed
... 50%, 21088 KB, 3430 KB/s, 6 seconds passed
... 50%, 21120 KB, 3435 KB/s, 6 seconds passed

.. parsed-literal::

    ... 50%, 21152 KB, 3429 KB/s, 6 seconds passed
... 50%, 21184 KB, 3428 KB/s, 6 seconds passed
... 51%, 21216 KB, 3431 KB/s, 6 seconds passed
... 51%, 21248 KB, 3435 KB/s, 6 seconds passed
... 51%, 21280 KB, 3429 KB/s, 6 seconds passed

.. parsed-literal::

    ... 51%, 21312 KB, 3428 KB/s, 6 seconds passed
... 51%, 21344 KB, 3431 KB/s, 6 seconds passed
... 51%, 21376 KB, 3435 KB/s, 6 seconds passed
... 51%, 21408 KB, 3429 KB/s, 6 seconds passed
... 51%, 21440 KB, 3428 KB/s, 6 seconds passed
... 51%, 21472 KB, 3431 KB/s, 6 seconds passed
... 51%, 21504 KB, 3436 KB/s, 6 seconds passed

.. parsed-literal::

    ... 51%, 21536 KB, 3429 KB/s, 6 seconds passed
... 51%, 21568 KB, 3429 KB/s, 6 seconds passed
... 51%, 21600 KB, 3431 KB/s, 6 seconds passed
... 52%, 21632 KB, 3434 KB/s, 6 seconds passed

.. parsed-literal::

    ... 52%, 21664 KB, 3430 KB/s, 6 seconds passed
... 52%, 21696 KB, 3429 KB/s, 6 seconds passed
... 52%, 21728 KB, 3432 KB/s, 6 seconds passed
... 52%, 21760 KB, 3435 KB/s, 6 seconds passed
... 52%, 21792 KB, 3430 KB/s, 6 seconds passed

.. parsed-literal::

    ... 52%, 21824 KB, 3429 KB/s, 6 seconds passed
... 52%, 21856 KB, 3432 KB/s, 6 seconds passed
... 52%, 21888 KB, 3430 KB/s, 6 seconds passed
... 52%, 21920 KB, 3430 KB/s, 6 seconds passed
... 52%, 21952 KB, 3430 KB/s, 6 seconds passed
... 52%, 21984 KB, 3432 KB/s, 6 seconds passed
... 52%, 22016 KB, 3437 KB/s, 6 seconds passed

.. parsed-literal::

    ... 53%, 22048 KB, 3431 KB/s, 6 seconds passed
... 53%, 22080 KB, 3430 KB/s, 6 seconds passed
... 53%, 22112 KB, 3432 KB/s, 6 seconds passed
... 53%, 22144 KB, 3430 KB/s, 6 seconds passed
... 53%, 22176 KB, 3431 KB/s, 6 seconds passed

.. parsed-literal::

    ... 53%, 22208 KB, 3430 KB/s, 6 seconds passed
... 53%, 22240 KB, 3432 KB/s, 6 seconds passed
... 53%, 22272 KB, 3431 KB/s, 6 seconds passed
... 53%, 22304 KB, 3431 KB/s, 6 seconds passed
... 53%, 22336 KB, 3431 KB/s, 6 seconds passed
... 53%, 22368 KB, 3433 KB/s, 6 seconds passed

.. parsed-literal::

    ... 53%, 22400 KB, 3431 KB/s, 6 seconds passed
... 53%, 22432 KB, 3432 KB/s, 6 seconds passed
... 54%, 22464 KB, 3431 KB/s, 6 seconds passed
... 54%, 22496 KB, 3433 KB/s, 6 seconds passed
... 54%, 22528 KB, 3432 KB/s, 6 seconds passed

.. parsed-literal::

    ... 54%, 22560 KB, 3432 KB/s, 6 seconds passed
... 54%, 22592 KB, 3432 KB/s, 6 seconds passed
... 54%, 22624 KB, 3433 KB/s, 6 seconds passed
... 54%, 22656 KB, 3431 KB/s, 6 seconds passed
... 54%, 22688 KB, 3432 KB/s, 6 seconds passed

.. parsed-literal::

    ... 54%, 22720 KB, 3431 KB/s, 6 seconds passed
... 54%, 22752 KB, 3433 KB/s, 6 seconds passed
... 54%, 22784 KB, 3430 KB/s, 6 seconds passed
... 54%, 22816 KB, 3432 KB/s, 6 seconds passed
... 54%, 22848 KB, 3432 KB/s, 6 seconds passed
... 55%, 22880 KB, 3434 KB/s, 6 seconds passed

.. parsed-literal::

    ... 55%, 22912 KB, 3431 KB/s, 6 seconds passed
... 55%, 22944 KB, 3433 KB/s, 6 seconds passed
... 55%, 22976 KB, 3432 KB/s, 6 seconds passed
... 55%, 23008 KB, 3434 KB/s, 6 seconds passed
... 55%, 23040 KB, 3430 KB/s, 6 seconds passed
... 55%, 23072 KB, 3433 KB/s, 6 seconds passed

.. parsed-literal::

    ... 55%, 23104 KB, 3433 KB/s, 6 seconds passed
... 55%, 23136 KB, 3434 KB/s, 6 seconds passed
... 55%, 23168 KB, 3431 KB/s, 6 seconds passed
... 55%, 23200 KB, 3433 KB/s, 6 seconds passed
... 55%, 23232 KB, 3433 KB/s, 6 seconds passed

.. parsed-literal::

    ... 55%, 23264 KB, 3434 KB/s, 6 seconds passed
... 56%, 23296 KB, 3431 KB/s, 6 seconds passed
... 56%, 23328 KB, 3433 KB/s, 6 seconds passed
... 56%, 23360 KB, 3433 KB/s, 6 seconds passed
... 56%, 23392 KB, 3434 KB/s, 6 seconds passed

.. parsed-literal::

    ... 56%, 23424 KB, 3431 KB/s, 6 seconds passed
... 56%, 23456 KB, 3433 KB/s, 6 seconds passed
... 56%, 23488 KB, 3433 KB/s, 6 seconds passed
... 56%, 23520 KB, 3434 KB/s, 6 seconds passed
... 56%, 23552 KB, 3431 KB/s, 6 seconds passed
... 56%, 23584 KB, 3433 KB/s, 6 seconds passed

.. parsed-literal::

    ... 56%, 23616 KB, 3433 KB/s, 6 seconds passed
... 56%, 23648 KB, 3434 KB/s, 6 seconds passed
... 56%, 23680 KB, 3431 KB/s, 6 seconds passed
... 57%, 23712 KB, 3433 KB/s, 6 seconds passed
... 57%, 23744 KB, 3433 KB/s, 6 seconds passed
... 57%, 23776 KB, 3434 KB/s, 6 seconds passed

.. parsed-literal::

    ... 57%, 23808 KB, 3431 KB/s, 6 seconds passed
... 57%, 23840 KB, 3432 KB/s, 6 seconds passed
... 57%, 23872 KB, 3433 KB/s, 6 seconds passed
... 57%, 23904 KB, 3434 KB/s, 6 seconds passed
... 57%, 23936 KB, 3431 KB/s, 6 seconds passed

.. parsed-literal::

    ... 57%, 23968 KB, 3433 KB/s, 6 seconds passed
... 57%, 24000 KB, 3433 KB/s, 6 seconds passed
... 57%, 24032 KB, 3434 KB/s, 6 seconds passed
... 57%, 24064 KB, 3432 KB/s, 7 seconds passed
... 57%, 24096 KB, 3433 KB/s, 7 seconds passed
... 58%, 24128 KB, 3434 KB/s, 7 seconds passed

.. parsed-literal::

    ... 58%, 24160 KB, 3435 KB/s, 7 seconds passed
... 58%, 24192 KB, 3432 KB/s, 7 seconds passed
... 58%, 24224 KB, 3433 KB/s, 7 seconds passed
... 58%, 24256 KB, 3434 KB/s, 7 seconds passed
... 58%, 24288 KB, 3435 KB/s, 7 seconds passed

.. parsed-literal::

    ... 58%, 24320 KB, 3432 KB/s, 7 seconds passed
... 58%, 24352 KB, 3431 KB/s, 7 seconds passed
... 58%, 24384 KB, 3434 KB/s, 7 seconds passed
... 58%, 24416 KB, 3435 KB/s, 7 seconds passed
... 58%, 24448 KB, 3430 KB/s, 7 seconds passed

.. parsed-literal::

    ... 58%, 24480 KB, 3432 KB/s, 7 seconds passed
... 58%, 24512 KB, 3434 KB/s, 7 seconds passed
... 59%, 24544 KB, 3435 KB/s, 7 seconds passed
... 59%, 24576 KB, 3432 KB/s, 7 seconds passed
... 59%, 24608 KB, 3434 KB/s, 7 seconds passed
... 59%, 24640 KB, 3434 KB/s, 7 seconds passed
... 59%, 24672 KB, 3435 KB/s, 7 seconds passed

.. parsed-literal::

    ... 59%, 24704 KB, 3430 KB/s, 7 seconds passed
... 59%, 24736 KB, 3432 KB/s, 7 seconds passed
... 59%, 24768 KB, 3435 KB/s, 7 seconds passed
... 59%, 24800 KB, 3435 KB/s, 7 seconds passed

.. parsed-literal::

    ... 59%, 24832 KB, 3430 KB/s, 7 seconds passed
... 59%, 24864 KB, 3432 KB/s, 7 seconds passed
... 59%, 24896 KB, 3431 KB/s, 7 seconds passed
... 59%, 24928 KB, 3435 KB/s, 7 seconds passed
... 60%, 24960 KB, 3430 KB/s, 7 seconds passed
... 60%, 24992 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 60%, 25024 KB, 3432 KB/s, 7 seconds passed
... 60%, 25056 KB, 3435 KB/s, 7 seconds passed
... 60%, 25088 KB, 3430 KB/s, 7 seconds passed
... 60%, 25120 KB, 3432 KB/s, 7 seconds passed
... 60%, 25152 KB, 3432 KB/s, 7 seconds passed
... 60%, 25184 KB, 3436 KB/s, 7 seconds passed

.. parsed-literal::

    ... 60%, 25216 KB, 3433 KB/s, 7 seconds passed
... 60%, 25248 KB, 3432 KB/s, 7 seconds passed
... 60%, 25280 KB, 3432 KB/s, 7 seconds passed
... 60%, 25312 KB, 3435 KB/s, 7 seconds passed
... 60%, 25344 KB, 3430 KB/s, 7 seconds passed

.. parsed-literal::

    ... 61%, 25376 KB, 3432 KB/s, 7 seconds passed
... 61%, 25408 KB, 3432 KB/s, 7 seconds passed
... 61%, 25440 KB, 3432 KB/s, 7 seconds passed
... 61%, 25472 KB, 3430 KB/s, 7 seconds passed
... 61%, 25504 KB, 3432 KB/s, 7 seconds passed
... 61%, 25536 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 61%, 25568 KB, 3431 KB/s, 7 seconds passed
... 61%, 25600 KB, 3430 KB/s, 7 seconds passed
... 61%, 25632 KB, 3433 KB/s, 7 seconds passed
... 61%, 25664 KB, 3432 KB/s, 7 seconds passed
... 61%, 25696 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 61%, 25728 KB, 3431 KB/s, 7 seconds passed
... 61%, 25760 KB, 3433 KB/s, 7 seconds passed
... 62%, 25792 KB, 3432 KB/s, 7 seconds passed
... 62%, 25824 KB, 3432 KB/s, 7 seconds passed
... 62%, 25856 KB, 3431 KB/s, 7 seconds passed
... 62%, 25888 KB, 3433 KB/s, 7 seconds passed

.. parsed-literal::

    ... 62%, 25920 KB, 3432 KB/s, 7 seconds passed
... 62%, 25952 KB, 3432 KB/s, 7 seconds passed
... 62%, 25984 KB, 3431 KB/s, 7 seconds passed
... 62%, 26016 KB, 3433 KB/s, 7 seconds passed
... 62%, 26048 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 62%, 26080 KB, 3432 KB/s, 7 seconds passed
... 62%, 26112 KB, 3431 KB/s, 7 seconds passed
... 62%, 26144 KB, 3434 KB/s, 7 seconds passed
... 62%, 26176 KB, 3432 KB/s, 7 seconds passed
... 63%, 26208 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 63%, 26240 KB, 3431 KB/s, 7 seconds passed
... 63%, 26272 KB, 3434 KB/s, 7 seconds passed
... 63%, 26304 KB, 3432 KB/s, 7 seconds passed
... 63%, 26336 KB, 3434 KB/s, 7 seconds passed
... 63%, 26368 KB, 3433 KB/s, 7 seconds passed
... 63%, 26400 KB, 3434 KB/s, 7 seconds passed

.. parsed-literal::

    ... 63%, 26432 KB, 3433 KB/s, 7 seconds passed
... 63%, 26464 KB, 3434 KB/s, 7 seconds passed
... 63%, 26496 KB, 3433 KB/s, 7 seconds passed
... 63%, 26528 KB, 3434 KB/s, 7 seconds passed
... 63%, 26560 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 63%, 26592 KB, 3432 KB/s, 7 seconds passed
... 64%, 26624 KB, 3432 KB/s, 7 seconds passed
... 64%, 26656 KB, 3434 KB/s, 7 seconds passed
... 64%, 26688 KB, 3433 KB/s, 7 seconds passed
... 64%, 26720 KB, 3433 KB/s, 7 seconds passed
... 64%, 26752 KB, 3432 KB/s, 7 seconds passed

.. parsed-literal::

    ... 64%, 26784 KB, 3434 KB/s, 7 seconds passed
... 64%, 26816 KB, 3432 KB/s, 7 seconds passed
... 64%, 26848 KB, 3433 KB/s, 7 seconds passed
... 64%, 26880 KB, 3432 KB/s, 7 seconds passed
... 64%, 26912 KB, 3434 KB/s, 7 seconds passed
... 64%, 26944 KB, 3433 KB/s, 7 seconds passed

.. parsed-literal::

    ... 64%, 26976 KB, 3434 KB/s, 7 seconds passed
... 64%, 27008 KB, 3433 KB/s, 7 seconds passed
... 65%, 27040 KB, 3434 KB/s, 7 seconds passed
... 65%, 27072 KB, 3433 KB/s, 7 seconds passed
... 65%, 27104 KB, 3434 KB/s, 7 seconds passed

.. parsed-literal::

    ... 65%, 27136 KB, 3433 KB/s, 7 seconds passed
... 65%, 27168 KB, 3434 KB/s, 7 seconds passed
... 65%, 27200 KB, 3433 KB/s, 7 seconds passed
... 65%, 27232 KB, 3432 KB/s, 7 seconds passed
... 65%, 27264 KB, 3432 KB/s, 7 seconds passed
... 65%, 27296 KB, 3434 KB/s, 7 seconds passed

.. parsed-literal::

    ... 65%, 27328 KB, 3433 KB/s, 7 seconds passed
... 65%, 27360 KB, 3433 KB/s, 7 seconds passed
... 65%, 27392 KB, 3433 KB/s, 7 seconds passed
... 65%, 27424 KB, 3434 KB/s, 7 seconds passed
... 66%, 27456 KB, 3433 KB/s, 7 seconds passed

.. parsed-literal::

    ... 66%, 27488 KB, 3433 KB/s, 8 seconds passed
... 66%, 27520 KB, 3433 KB/s, 8 seconds passed
... 66%, 27552 KB, 3435 KB/s, 8 seconds passed
... 66%, 27584 KB, 3434 KB/s, 8 seconds passed
... 66%, 27616 KB, 3433 KB/s, 8 seconds passed
... 66%, 27648 KB, 3433 KB/s, 8 seconds passed

.. parsed-literal::

    ... 66%, 27680 KB, 3435 KB/s, 8 seconds passed
... 66%, 27712 KB, 3433 KB/s, 8 seconds passed
... 66%, 27744 KB, 3432 KB/s, 8 seconds passed
... 66%, 27776 KB, 3433 KB/s, 8 seconds passed
... 66%, 27808 KB, 3435 KB/s, 8 seconds passed

.. parsed-literal::

    ... 66%, 27840 KB, 3432 KB/s, 8 seconds passed
... 67%, 27872 KB, 3432 KB/s, 8 seconds passed
... 67%, 27904 KB, 3433 KB/s, 8 seconds passed
... 67%, 27936 KB, 3435 KB/s, 8 seconds passed
... 67%, 27968 KB, 3432 KB/s, 8 seconds passed
... 67%, 28000 KB, 3432 KB/s, 8 seconds passed

.. parsed-literal::

    ... 67%, 28032 KB, 3433 KB/s, 8 seconds passed
... 67%, 28064 KB, 3435 KB/s, 8 seconds passed
... 67%, 28096 KB, 3433 KB/s, 8 seconds passed
... 67%, 28128 KB, 3432 KB/s, 8 seconds passed
... 67%, 28160 KB, 3433 KB/s, 8 seconds passed
... 67%, 28192 KB, 3435 KB/s, 8 seconds passed

.. parsed-literal::

    ... 67%, 28224 KB, 3433 KB/s, 8 seconds passed
... 67%, 28256 KB, 3432 KB/s, 8 seconds passed
... 68%, 28288 KB, 3433 KB/s, 8 seconds passed
... 68%, 28320 KB, 3430 KB/s, 8 seconds passed
... 68%, 28352 KB, 3433 KB/s, 8 seconds passed

.. parsed-literal::

    ... 68%, 28384 KB, 3433 KB/s, 8 seconds passed
... 68%, 28416 KB, 3433 KB/s, 8 seconds passed
... 68%, 28448 KB, 3430 KB/s, 8 seconds passed
... 68%, 28480 KB, 3433 KB/s, 8 seconds passed
... 68%, 28512 KB, 3432 KB/s, 8 seconds passed

.. parsed-literal::

    ... 68%, 28544 KB, 3433 KB/s, 8 seconds passed
... 68%, 28576 KB, 3430 KB/s, 8 seconds passed
... 68%, 28608 KB, 3433 KB/s, 8 seconds passed
... 68%, 28640 KB, 3432 KB/s, 8 seconds passed
... 68%, 28672 KB, 3433 KB/s, 8 seconds passed

.. parsed-literal::

    ... 69%, 28704 KB, 3430 KB/s, 8 seconds passed
... 69%, 28736 KB, 3433 KB/s, 8 seconds passed
... 69%, 28768 KB, 3432 KB/s, 8 seconds passed
... 69%, 28800 KB, 3433 KB/s, 8 seconds passed
... 69%, 28832 KB, 3430 KB/s, 8 seconds passed
... 69%, 28864 KB, 3434 KB/s, 8 seconds passed
... 69%, 28896 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 69%, 28928 KB, 3434 KB/s, 8 seconds passed
... 69%, 28960 KB, 3435 KB/s, 8 seconds passed
... 69%, 28992 KB, 3433 KB/s, 8 seconds passed
... 69%, 29024 KB, 3434 KB/s, 8 seconds passed
... 69%, 29056 KB, 3433 KB/s, 8 seconds passed
... 69%, 29088 KB, 3436 KB/s, 8 seconds passed

.. parsed-literal::

    ... 70%, 29120 KB, 3434 KB/s, 8 seconds passed
... 70%, 29152 KB, 3434 KB/s, 8 seconds passed
... 70%, 29184 KB, 3434 KB/s, 8 seconds passed
... 70%, 29216 KB, 3437 KB/s, 8 seconds passed
... 70%, 29248 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 70%, 29280 KB, 3434 KB/s, 8 seconds passed
... 70%, 29312 KB, 3434 KB/s, 8 seconds passed
... 70%, 29344 KB, 3431 KB/s, 8 seconds passed
... 70%, 29376 KB, 3434 KB/s, 8 seconds passed
... 70%, 29408 KB, 3433 KB/s, 8 seconds passed

.. parsed-literal::

    ... 70%, 29440 KB, 3434 KB/s, 8 seconds passed
... 70%, 29472 KB, 3431 KB/s, 8 seconds passed
... 70%, 29504 KB, 3434 KB/s, 8 seconds passed
... 71%, 29536 KB, 3433 KB/s, 8 seconds passed
... 71%, 29568 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 71%, 29600 KB, 3431 KB/s, 8 seconds passed
... 71%, 29632 KB, 3434 KB/s, 8 seconds passed
... 71%, 29664 KB, 3433 KB/s, 8 seconds passed
... 71%, 29696 KB, 3434 KB/s, 8 seconds passed
... 71%, 29728 KB, 3431 KB/s, 8 seconds passed
... 71%, 29760 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 71%, 29792 KB, 3434 KB/s, 8 seconds passed
... 71%, 29824 KB, 3434 KB/s, 8 seconds passed
... 71%, 29856 KB, 3432 KB/s, 8 seconds passed
... 71%, 29888 KB, 3435 KB/s, 8 seconds passed
... 72%, 29920 KB, 3434 KB/s, 8 seconds passed
... 72%, 29952 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 72%, 29984 KB, 3432 KB/s, 8 seconds passed
... 72%, 30016 KB, 3435 KB/s, 8 seconds passed
... 72%, 30048 KB, 3434 KB/s, 8 seconds passed
... 72%, 30080 KB, 3434 KB/s, 8 seconds passed
... 72%, 30112 KB, 3432 KB/s, 8 seconds passed

.. parsed-literal::

    ... 72%, 30144 KB, 3435 KB/s, 8 seconds passed
... 72%, 30176 KB, 3434 KB/s, 8 seconds passed
... 72%, 30208 KB, 3434 KB/s, 8 seconds passed
... 72%, 30240 KB, 3432 KB/s, 8 seconds passed
... 72%, 30272 KB, 3435 KB/s, 8 seconds passed

.. parsed-literal::

    ... 72%, 30304 KB, 3433 KB/s, 8 seconds passed
... 73%, 30336 KB, 3433 KB/s, 8 seconds passed
... 73%, 30368 KB, 3431 KB/s, 8 seconds passed
... 73%, 30400 KB, 3434 KB/s, 8 seconds passed
... 73%, 30432 KB, 3433 KB/s, 8 seconds passed
... 73%, 30464 KB, 3433 KB/s, 8 seconds passed

.. parsed-literal::

    ... 73%, 30496 KB, 3431 KB/s, 8 seconds passed
... 73%, 30528 KB, 3434 KB/s, 8 seconds passed
... 73%, 30560 KB, 3434 KB/s, 8 seconds passed
... 73%, 30592 KB, 3433 KB/s, 8 seconds passed
... 73%, 30624 KB, 3431 KB/s, 8 seconds passed
... 73%, 30656 KB, 3435 KB/s, 8 seconds passed

.. parsed-literal::

    ... 73%, 30688 KB, 3434 KB/s, 8 seconds passed
... 73%, 30720 KB, 3433 KB/s, 8 seconds passed
... 74%, 30752 KB, 3431 KB/s, 8 seconds passed
... 74%, 30784 KB, 3432 KB/s, 8 seconds passed
... 74%, 30816 KB, 3434 KB/s, 8 seconds passed

.. parsed-literal::

    ... 74%, 30848 KB, 3434 KB/s, 8 seconds passed
... 74%, 30880 KB, 3430 KB/s, 9 seconds passed
... 74%, 30912 KB, 3433 KB/s, 9 seconds passed
... 74%, 30944 KB, 3434 KB/s, 9 seconds passed
... 74%, 30976 KB, 3433 KB/s, 9 seconds passed

.. parsed-literal::

    ... 74%, 31008 KB, 3430 KB/s, 9 seconds passed
... 74%, 31040 KB, 3433 KB/s, 9 seconds passed
... 74%, 31072 KB, 3435 KB/s, 9 seconds passed
... 74%, 31104 KB, 3434 KB/s, 9 seconds passed
... 74%, 31136 KB, 3430 KB/s, 9 seconds passed
... 75%, 31168 KB, 3434 KB/s, 9 seconds passed
... 75%, 31200 KB, 3436 KB/s, 9 seconds passed

.. parsed-literal::

    ... 75%, 31232 KB, 3433 KB/s, 9 seconds passed
... 75%, 31264 KB, 3430 KB/s, 9 seconds passed
... 75%, 31296 KB, 3433 KB/s, 9 seconds passed
... 75%, 31328 KB, 3433 KB/s, 9 seconds passed

.. parsed-literal::

    ... 75%, 31360 KB, 3433 KB/s, 9 seconds passed
... 75%, 31392 KB, 3430 KB/s, 9 seconds passed
... 75%, 31424 KB, 3433 KB/s, 9 seconds passed
... 75%, 31456 KB, 3433 KB/s, 9 seconds passed
... 75%, 31488 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 75%, 31520 KB, 3431 KB/s, 9 seconds passed
... 75%, 31552 KB, 3433 KB/s, 9 seconds passed
... 76%, 31584 KB, 3434 KB/s, 9 seconds passed
... 76%, 31616 KB, 3434 KB/s, 9 seconds passed
... 76%, 31648 KB, 3431 KB/s, 9 seconds passed
... 76%, 31680 KB, 3433 KB/s, 9 seconds passed
... 76%, 31712 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 76%, 31744 KB, 3434 KB/s, 9 seconds passed
... 76%, 31776 KB, 3431 KB/s, 9 seconds passed
... 76%, 31808 KB, 3434 KB/s, 9 seconds passed
... 76%, 31840 KB, 3434 KB/s, 9 seconds passed
... 76%, 31872 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 76%, 31904 KB, 3431 KB/s, 9 seconds passed
... 76%, 31936 KB, 3434 KB/s, 9 seconds passed
... 76%, 31968 KB, 3434 KB/s, 9 seconds passed
... 77%, 32000 KB, 3435 KB/s, 9 seconds passed
... 77%, 32032 KB, 3432 KB/s, 9 seconds passed
... 77%, 32064 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 77%, 32096 KB, 3435 KB/s, 9 seconds passed
... 77%, 32128 KB, 3434 KB/s, 9 seconds passed
... 77%, 32160 KB, 3432 KB/s, 9 seconds passed
... 77%, 32192 KB, 3434 KB/s, 9 seconds passed
... 77%, 32224 KB, 3435 KB/s, 9 seconds passed

.. parsed-literal::

    ... 77%, 32256 KB, 3435 KB/s, 9 seconds passed
... 77%, 32288 KB, 3432 KB/s, 9 seconds passed
... 77%, 32320 KB, 3434 KB/s, 9 seconds passed
... 77%, 32352 KB, 3435 KB/s, 9 seconds passed
... 77%, 32384 KB, 3435 KB/s, 9 seconds passed

.. parsed-literal::

    ... 78%, 32416 KB, 3432 KB/s, 9 seconds passed
... 78%, 32448 KB, 3434 KB/s, 9 seconds passed
... 78%, 32480 KB, 3434 KB/s, 9 seconds passed
... 78%, 32512 KB, 3435 KB/s, 9 seconds passed
... 78%, 32544 KB, 3432 KB/s, 9 seconds passed
... 78%, 32576 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 78%, 32608 KB, 3434 KB/s, 9 seconds passed
... 78%, 32640 KB, 3435 KB/s, 9 seconds passed
... 78%, 32672 KB, 3432 KB/s, 9 seconds passed
... 78%, 32704 KB, 3434 KB/s, 9 seconds passed
... 78%, 32736 KB, 3435 KB/s, 9 seconds passed
... 78%, 32768 KB, 3436 KB/s, 9 seconds passed

.. parsed-literal::

    ... 78%, 32800 KB, 3433 KB/s, 9 seconds passed
... 79%, 32832 KB, 3435 KB/s, 9 seconds passed
... 79%, 32864 KB, 3436 KB/s, 9 seconds passed
... 79%, 32896 KB, 3436 KB/s, 9 seconds passed
... 79%, 32928 KB, 3433 KB/s, 9 seconds passed
... 79%, 32960 KB, 3435 KB/s, 9 seconds passed

.. parsed-literal::

    ... 79%, 32992 KB, 3436 KB/s, 9 seconds passed
... 79%, 33024 KB, 3430 KB/s, 9 seconds passed
... 79%, 33056 KB, 3432 KB/s, 9 seconds passed
... 79%, 33088 KB, 3435 KB/s, 9 seconds passed
... 79%, 33120 KB, 3435 KB/s, 9 seconds passed

.. parsed-literal::

    ... 79%, 33152 KB, 3430 KB/s, 9 seconds passed
... 79%, 33184 KB, 3432 KB/s, 9 seconds passed
... 79%, 33216 KB, 3435 KB/s, 9 seconds passed
... 80%, 33248 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 80%, 33280 KB, 3430 KB/s, 9 seconds passed
... 80%, 33312 KB, 3433 KB/s, 9 seconds passed
... 80%, 33344 KB, 3434 KB/s, 9 seconds passed
... 80%, 33376 KB, 3435 KB/s, 9 seconds passed
... 80%, 33408 KB, 3436 KB/s, 9 seconds passed
... 80%, 33440 KB, 3431 KB/s, 9 seconds passed
... 80%, 33472 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 80%, 33504 KB, 3435 KB/s, 9 seconds passed
... 80%, 33536 KB, 3436 KB/s, 9 seconds passed
... 80%, 33568 KB, 3432 KB/s, 9 seconds passed
... 80%, 33600 KB, 3433 KB/s, 9 seconds passed
... 80%, 33632 KB, 3435 KB/s, 9 seconds passed
... 81%, 33664 KB, 3436 KB/s, 9 seconds passed

.. parsed-literal::

    ... 81%, 33696 KB, 3433 KB/s, 9 seconds passed
... 81%, 33728 KB, 3433 KB/s, 9 seconds passed
... 81%, 33760 KB, 3435 KB/s, 9 seconds passed
... 81%, 33792 KB, 3436 KB/s, 9 seconds passed

.. parsed-literal::

    ... 81%, 33824 KB, 3433 KB/s, 9 seconds passed
... 81%, 33856 KB, 3434 KB/s, 9 seconds passed
... 81%, 33888 KB, 3434 KB/s, 9 seconds passed
... 81%, 33920 KB, 3430 KB/s, 9 seconds passed
... 81%, 33952 KB, 3431 KB/s, 9 seconds passed
... 81%, 33984 KB, 3434 KB/s, 9 seconds passed

.. parsed-literal::

    ... 81%, 34016 KB, 3435 KB/s, 9 seconds passed
... 81%, 34048 KB, 3430 KB/s, 9 seconds passed
... 82%, 34080 KB, 3431 KB/s, 9 seconds passed
... 82%, 34112 KB, 3434 KB/s, 9 seconds passed
... 82%, 34144 KB, 3435 KB/s, 9 seconds passed

.. parsed-literal::

    ... 82%, 34176 KB, 3430 KB/s, 9 seconds passed
... 82%, 34208 KB, 3431 KB/s, 9 seconds passed
... 82%, 34240 KB, 3434 KB/s, 9 seconds passed
... 82%, 34272 KB, 3435 KB/s, 9 seconds passed
... 82%, 34304 KB, 3430 KB/s, 9 seconds passed

.. parsed-literal::

    ... 82%, 34336 KB, 3431 KB/s, 10 seconds passed
... 82%, 34368 KB, 3434 KB/s, 10 seconds passed
... 82%, 34400 KB, 3435 KB/s, 10 seconds passed
... 82%, 34432 KB, 3430 KB/s, 10 seconds passed
... 82%, 34464 KB, 3431 KB/s, 10 seconds passed
... 83%, 34496 KB, 3434 KB/s, 10 seconds passed
... 83%, 34528 KB, 3435 KB/s, 10 seconds passed

.. parsed-literal::

    ... 83%, 34560 KB, 3430 KB/s, 10 seconds passed
... 83%, 34592 KB, 3431 KB/s, 10 seconds passed
... 83%, 34624 KB, 3434 KB/s, 10 seconds passed
... 83%, 34656 KB, 3435 KB/s, 10 seconds passed

.. parsed-literal::

    ... 83%, 34688 KB, 3431 KB/s, 10 seconds passed
... 83%, 34720 KB, 3432 KB/s, 10 seconds passed
... 83%, 34752 KB, 3435 KB/s, 10 seconds passed
... 83%, 34784 KB, 3435 KB/s, 10 seconds passed
... 83%, 34816 KB, 3437 KB/s, 10 seconds passed
... 83%, 34848 KB, 3432 KB/s, 10 seconds passed
... 83%, 34880 KB, 3435 KB/s, 10 seconds passed

.. parsed-literal::

    ... 84%, 34912 KB, 3436 KB/s, 10 seconds passed
... 84%, 34944 KB, 3438 KB/s, 10 seconds passed
... 84%, 34976 KB, 3432 KB/s, 10 seconds passed
... 84%, 35008 KB, 3435 KB/s, 10 seconds passed
... 84%, 35040 KB, 3437 KB/s, 10 seconds passed
... 84%, 35072 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 84%, 35104 KB, 3433 KB/s, 10 seconds passed
... 84%, 35136 KB, 3435 KB/s, 10 seconds passed
... 84%, 35168 KB, 3436 KB/s, 10 seconds passed
... 84%, 35200 KB, 3431 KB/s, 10 seconds passed

.. parsed-literal::

    ... 84%, 35232 KB, 3433 KB/s, 10 seconds passed
... 84%, 35264 KB, 3435 KB/s, 10 seconds passed
... 84%, 35296 KB, 3436 KB/s, 10 seconds passed
... 85%, 35328 KB, 3432 KB/s, 10 seconds passed
... 85%, 35360 KB, 3432 KB/s, 10 seconds passed
... 85%, 35392 KB, 3435 KB/s, 10 seconds passed
... 85%, 35424 KB, 3437 KB/s, 10 seconds passed

.. parsed-literal::

    ... 85%, 35456 KB, 3432 KB/s, 10 seconds passed
... 85%, 35488 KB, 3432 KB/s, 10 seconds passed
... 85%, 35520 KB, 3435 KB/s, 10 seconds passed
... 85%, 35552 KB, 3437 KB/s, 10 seconds passed

.. parsed-literal::

    ... 85%, 35584 KB, 3433 KB/s, 10 seconds passed
... 85%, 35616 KB, 3433 KB/s, 10 seconds passed
... 85%, 35648 KB, 3435 KB/s, 10 seconds passed
... 85%, 35680 KB, 3437 KB/s, 10 seconds passed
... 85%, 35712 KB, 3432 KB/s, 10 seconds passed
... 86%, 35744 KB, 3433 KB/s, 10 seconds passed
... 86%, 35776 KB, 3435 KB/s, 10 seconds passed

.. parsed-literal::

    ... 86%, 35808 KB, 3437 KB/s, 10 seconds passed
... 86%, 35840 KB, 3432 KB/s, 10 seconds passed
... 86%, 35872 KB, 3433 KB/s, 10 seconds passed
... 86%, 35904 KB, 3436 KB/s, 10 seconds passed
... 86%, 35936 KB, 3437 KB/s, 10 seconds passed

.. parsed-literal::

    ... 86%, 35968 KB, 3433 KB/s, 10 seconds passed
... 86%, 36000 KB, 3433 KB/s, 10 seconds passed
... 86%, 36032 KB, 3436 KB/s, 10 seconds passed
... 86%, 36064 KB, 3437 KB/s, 10 seconds passed
... 86%, 36096 KB, 3433 KB/s, 10 seconds passed

.. parsed-literal::

    ... 86%, 36128 KB, 3433 KB/s, 10 seconds passed
... 87%, 36160 KB, 3436 KB/s, 10 seconds passed
... 87%, 36192 KB, 3437 KB/s, 10 seconds passed
... 87%, 36224 KB, 3433 KB/s, 10 seconds passed
... 87%, 36256 KB, 3433 KB/s, 10 seconds passed
... 87%, 36288 KB, 3436 KB/s, 10 seconds passed
... 87%, 36320 KB, 3437 KB/s, 10 seconds passed

.. parsed-literal::

    ... 87%, 36352 KB, 3433 KB/s, 10 seconds passed
... 87%, 36384 KB, 3434 KB/s, 10 seconds passed
... 87%, 36416 KB, 3436 KB/s, 10 seconds passed
... 87%, 36448 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 87%, 36480 KB, 3433 KB/s, 10 seconds passed
... 87%, 36512 KB, 3434 KB/s, 10 seconds passed
... 87%, 36544 KB, 3437 KB/s, 10 seconds passed
... 88%, 36576 KB, 3438 KB/s, 10 seconds passed
... 88%, 36608 KB, 3433 KB/s, 10 seconds passed
... 88%, 36640 KB, 3434 KB/s, 10 seconds passed
... 88%, 36672 KB, 3437 KB/s, 10 seconds passed

.. parsed-literal::

    ... 88%, 36704 KB, 3438 KB/s, 10 seconds passed
... 88%, 36736 KB, 3434 KB/s, 10 seconds passed
... 88%, 36768 KB, 3434 KB/s, 10 seconds passed
... 88%, 36800 KB, 3437 KB/s, 10 seconds passed
... 88%, 36832 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 88%, 36864 KB, 3434 KB/s, 10 seconds passed
... 88%, 36896 KB, 3434 KB/s, 10 seconds passed
... 88%, 36928 KB, 3437 KB/s, 10 seconds passed
... 88%, 36960 KB, 3438 KB/s, 10 seconds passed
... 89%, 36992 KB, 3434 KB/s, 10 seconds passed

.. parsed-literal::

    ... 89%, 37024 KB, 3434 KB/s, 10 seconds passed
... 89%, 37056 KB, 3437 KB/s, 10 seconds passed
... 89%, 37088 KB, 3438 KB/s, 10 seconds passed
... 89%, 37120 KB, 3434 KB/s, 10 seconds passed
... 89%, 37152 KB, 3434 KB/s, 10 seconds passed
... 89%, 37184 KB, 3437 KB/s, 10 seconds passed
... 89%, 37216 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 89%, 37248 KB, 3432 KB/s, 10 seconds passed
... 89%, 37280 KB, 3434 KB/s, 10 seconds passed
... 89%, 37312 KB, 3437 KB/s, 10 seconds passed
... 89%, 37344 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 89%, 37376 KB, 3434 KB/s, 10 seconds passed
... 90%, 37408 KB, 3434 KB/s, 10 seconds passed
... 90%, 37440 KB, 3437 KB/s, 10 seconds passed
... 90%, 37472 KB, 3438 KB/s, 10 seconds passed
... 90%, 37504 KB, 3434 KB/s, 10 seconds passed
... 90%, 37536 KB, 3435 KB/s, 10 seconds passed

.. parsed-literal::

    ... 90%, 37568 KB, 3437 KB/s, 10 seconds passed
... 90%, 37600 KB, 3438 KB/s, 10 seconds passed
... 90%, 37632 KB, 3433 KB/s, 10 seconds passed
... 90%, 37664 KB, 3434 KB/s, 10 seconds passed
... 90%, 37696 KB, 3437 KB/s, 10 seconds passed
... 90%, 37728 KB, 3438 KB/s, 10 seconds passed

.. parsed-literal::

    ... 90%, 37760 KB, 3433 KB/s, 10 seconds passed
... 90%, 37792 KB, 3435 KB/s, 11 seconds passed
... 91%, 37824 KB, 3437 KB/s, 11 seconds passed
... 91%, 37856 KB, 3438 KB/s, 11 seconds passed

.. parsed-literal::

    ... 91%, 37888 KB, 3433 KB/s, 11 seconds passed
... 91%, 37920 KB, 3435 KB/s, 11 seconds passed
... 91%, 37952 KB, 3437 KB/s, 11 seconds passed
... 91%, 37984 KB, 3438 KB/s, 11 seconds passed
... 91%, 38016 KB, 3433 KB/s, 11 seconds passed
... 91%, 38048 KB, 3435 KB/s, 11 seconds passed
... 91%, 38080 KB, 3438 KB/s, 11 seconds passed

.. parsed-literal::

    ... 91%, 38112 KB, 3438 KB/s, 11 seconds passed
... 91%, 38144 KB, 3435 KB/s, 11 seconds passed
... 91%, 38176 KB, 3435 KB/s, 11 seconds passed
... 91%, 38208 KB, 3438 KB/s, 11 seconds passed
... 92%, 38240 KB, 3439 KB/s, 11 seconds passed

.. parsed-literal::

    ... 92%, 38272 KB, 3433 KB/s, 11 seconds passed
... 92%, 38304 KB, 3435 KB/s, 11 seconds passed
... 92%, 38336 KB, 3438 KB/s, 11 seconds passed
... 92%, 38368 KB, 3438 KB/s, 11 seconds passed
... 92%, 38400 KB, 3433 KB/s, 11 seconds passed

.. parsed-literal::

    ... 92%, 38432 KB, 3435 KB/s, 11 seconds passed
... 92%, 38464 KB, 3438 KB/s, 11 seconds passed
... 92%, 38496 KB, 3439 KB/s, 11 seconds passed
... 92%, 38528 KB, 3433 KB/s, 11 seconds passed
... 92%, 38560 KB, 3435 KB/s, 11 seconds passed
... 92%, 38592 KB, 3438 KB/s, 11 seconds passed
... 92%, 38624 KB, 3439 KB/s, 11 seconds passed

.. parsed-literal::

    ... 93%, 38656 KB, 3433 KB/s, 11 seconds passed
... 93%, 38688 KB, 3435 KB/s, 11 seconds passed
... 93%, 38720 KB, 3438 KB/s, 11 seconds passed
... 93%, 38752 KB, 3439 KB/s, 11 seconds passed

.. parsed-literal::

    ... 93%, 38784 KB, 3432 KB/s, 11 seconds passed
... 93%, 38816 KB, 3434 KB/s, 11 seconds passed
... 93%, 38848 KB, 3437 KB/s, 11 seconds passed
... 93%, 38880 KB, 3439 KB/s, 11 seconds passed
... 93%, 38912 KB, 3432 KB/s, 11 seconds passed
... 93%, 38944 KB, 3434 KB/s, 11 seconds passed

.. parsed-literal::

    ... 93%, 38976 KB, 3437 KB/s, 11 seconds passed
... 93%, 39008 KB, 3433 KB/s, 11 seconds passed
... 93%, 39040 KB, 3432 KB/s, 11 seconds passed
... 94%, 39072 KB, 3434 KB/s, 11 seconds passed
... 94%, 39104 KB, 3436 KB/s, 11 seconds passed

.. parsed-literal::

    ... 94%, 39136 KB, 3434 KB/s, 11 seconds passed
... 94%, 39168 KB, 3432 KB/s, 11 seconds passed
... 94%, 39200 KB, 3434 KB/s, 11 seconds passed
... 94%, 39232 KB, 3436 KB/s, 11 seconds passed
... 94%, 39264 KB, 3434 KB/s, 11 seconds passed

.. parsed-literal::

    ... 94%, 39296 KB, 3432 KB/s, 11 seconds passed
... 94%, 39328 KB, 3434 KB/s, 11 seconds passed
... 94%, 39360 KB, 3436 KB/s, 11 seconds passed
... 94%, 39392 KB, 3434 KB/s, 11 seconds passed
... 94%, 39424 KB, 3432 KB/s, 11 seconds passed
... 94%, 39456 KB, 3434 KB/s, 11 seconds passed
... 95%, 39488 KB, 3437 KB/s, 11 seconds passed
... 95%, 39520 KB, 3439 KB/s, 11 seconds passed

.. parsed-literal::

    ... 95%, 39552 KB, 3433 KB/s, 11 seconds passed
... 95%, 39584 KB, 3434 KB/s, 11 seconds passed
... 95%, 39616 KB, 3437 KB/s, 11 seconds passed
... 95%, 39648 KB, 3434 KB/s, 11 seconds passed

.. parsed-literal::

    ... 95%, 39680 KB, 3432 KB/s, 11 seconds passed
... 95%, 39712 KB, 3434 KB/s, 11 seconds passed
... 95%, 39744 KB, 3437 KB/s, 11 seconds passed
... 95%, 39776 KB, 3439 KB/s, 11 seconds passed
... 95%, 39808 KB, 3434 KB/s, 11 seconds passed

.. parsed-literal::

    ... 95%, 39840 KB, 3435 KB/s, 11 seconds passed
... 95%, 39872 KB, 3437 KB/s, 11 seconds passed
... 96%, 39904 KB, 3439 KB/s, 11 seconds passed
... 96%, 39936 KB, 3434 KB/s, 11 seconds passed
... 96%, 39968 KB, 3435 KB/s, 11 seconds passed
... 96%, 40000 KB, 3437 KB/s, 11 seconds passed
... 96%, 40032 KB, 3439 KB/s, 11 seconds passed

.. parsed-literal::

    ... 96%, 40064 KB, 3434 KB/s, 11 seconds passed
... 96%, 40096 KB, 3435 KB/s, 11 seconds passed
... 96%, 40128 KB, 3437 KB/s, 11 seconds passed
... 96%, 40160 KB, 3440 KB/s, 11 seconds passed

.. parsed-literal::

    ... 96%, 40192 KB, 3434 KB/s, 11 seconds passed
... 96%, 40224 KB, 3435 KB/s, 11 seconds passed
... 96%, 40256 KB, 3437 KB/s, 11 seconds passed
... 96%, 40288 KB, 3440 KB/s, 11 seconds passed
... 97%, 40320 KB, 3434 KB/s, 11 seconds passed
... 97%, 40352 KB, 3435 KB/s, 11 seconds passed
... 97%, 40384 KB, 3438 KB/s, 11 seconds passed
... 97%, 40416 KB, 3440 KB/s, 11 seconds passed

.. parsed-literal::

    ... 97%, 40448 KB, 3433 KB/s, 11 seconds passed
... 97%, 40480 KB, 3435 KB/s, 11 seconds passed
... 97%, 40512 KB, 3438 KB/s, 11 seconds passed

.. parsed-literal::

    ... 97%, 40544 KB, 3435 KB/s, 11 seconds passed
... 97%, 40576 KB, 3433 KB/s, 11 seconds passed
... 97%, 40608 KB, 3435 KB/s, 11 seconds passed
... 97%, 40640 KB, 3438 KB/s, 11 seconds passed
... 97%, 40672 KB, 3435 KB/s, 11 seconds passed

.. parsed-literal::

    ... 97%, 40704 KB, 3433 KB/s, 11 seconds passed
... 98%, 40736 KB, 3435 KB/s, 11 seconds passed
... 98%, 40768 KB, 3438 KB/s, 11 seconds passed
... 98%, 40800 KB, 3435 KB/s, 11 seconds passed
... 98%, 40832 KB, 3434 KB/s, 11 seconds passed
... 98%, 40864 KB, 3435 KB/s, 11 seconds passed
... 98%, 40896 KB, 3438 KB/s, 11 seconds passed

.. parsed-literal::

    ... 98%, 40928 KB, 3435 KB/s, 11 seconds passed
... 98%, 40960 KB, 3434 KB/s, 11 seconds passed
... 98%, 40992 KB, 3435 KB/s, 11 seconds passed
... 98%, 41024 KB, 3438 KB/s, 11 seconds passed
... 98%, 41056 KB, 3435 KB/s, 11 seconds passed

.. parsed-literal::

    ... 98%, 41088 KB, 3433 KB/s, 11 seconds passed
... 98%, 41120 KB, 3435 KB/s, 11 seconds passed
... 99%, 41152 KB, 3438 KB/s, 11 seconds passed
... 99%, 41184 KB, 3435 KB/s, 11 seconds passed
... 99%, 41216 KB, 3434 KB/s, 12 seconds passed
... 99%, 41248 KB, 3436 KB/s, 12 seconds passed

.. parsed-literal::

    ... 99%, 41280 KB, 3438 KB/s, 12 seconds passed
... 99%, 41312 KB, 3436 KB/s, 12 seconds passed
... 99%, 41344 KB, 3434 KB/s, 12 seconds passed
... 99%, 41376 KB, 3436 KB/s, 12 seconds passed
... 99%, 41408 KB, 3438 KB/s, 12 seconds passed

.. parsed-literal::

    ... 99%, 41440 KB, 3435 KB/s, 12 seconds passed
... 99%, 41472 KB, 3434 KB/s, 12 seconds passed
... 99%, 41504 KB, 3436 KB/s, 12 seconds passed
... 99%, 41536 KB, 3438 KB/s, 12 seconds passed
... 100%, 41555 KB, 3439 KB/s, 12 seconds passed



.. parsed-literal::

    
    ################|| Downloading action-recognition-0001-decoder ||################
    
    ========== Downloading model/intel/action-recognition-0001/action-recognition-0001-decoder/FP16/action-recognition-0001-decoder.xml


.. parsed-literal::

    ... 17%, 32 KB, 1113 KB/s, 0 seconds passed

.. parsed-literal::

    ... 34%, 64 KB, 1222 KB/s, 0 seconds passed
... 51%, 96 KB, 1626 KB/s, 0 seconds passed
... 68%, 128 KB, 1767 KB/s, 0 seconds passed
... 85%, 160 KB, 2142 KB/s, 0 seconds passed
... 100%, 186 KB, 2182 KB/s, 0 seconds passed

    
    ========== Downloading model/intel/action-recognition-0001/action-recognition-0001-decoder/FP16/action-recognition-0001-decoder.bin


.. parsed-literal::

    ... 0%, 32 KB, 1109 KB/s, 0 seconds passed
... 0%, 64 KB, 1259 KB/s, 0 seconds passed
... 0%, 96 KB, 1714 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1575 KB/s, 0 seconds passed
... 1%, 160 KB, 1827 KB/s, 0 seconds passed
... 1%, 192 KB, 2073 KB/s, 0 seconds passed
... 1%, 224 KB, 2298 KB/s, 0 seconds passed
... 1%, 256 KB, 2165 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 288 KB, 2302 KB/s, 0 seconds passed
... 2%, 320 KB, 2434 KB/s, 0 seconds passed
... 2%, 352 KB, 2607 KB/s, 0 seconds passed
... 2%, 384 KB, 2459 KB/s, 0 seconds passed
... 2%, 416 KB, 2563 KB/s, 0 seconds passed
... 3%, 448 KB, 2686 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 480 KB, 2797 KB/s, 0 seconds passed
... 3%, 512 KB, 2652 KB/s, 0 seconds passed
... 3%, 544 KB, 2728 KB/s, 0 seconds passed
... 3%, 576 KB, 2839 KB/s, 0 seconds passed
... 4%, 608 KB, 2918 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 640 KB, 2795 KB/s, 0 seconds passed
... 4%, 672 KB, 2851 KB/s, 0 seconds passed
... 4%, 704 KB, 2939 KB/s, 0 seconds passed
... 4%, 736 KB, 3000 KB/s, 0 seconds passed
... 5%, 768 KB, 2876 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 800 KB, 2915 KB/s, 0 seconds passed
... 5%, 832 KB, 2985 KB/s, 0 seconds passed
... 5%, 864 KB, 3048 KB/s, 0 seconds passed
... 6%, 896 KB, 2943 KB/s, 0 seconds passed
... 6%, 928 KB, 2991 KB/s, 0 seconds passed
... 6%, 960 KB, 3054 KB/s, 0 seconds passed
... 6%, 992 KB, 3094 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 1024 KB, 2990 KB/s, 0 seconds passed
... 7%, 1056 KB, 3020 KB/s, 0 seconds passed
... 7%, 1088 KB, 3077 KB/s, 0 seconds passed
... 7%, 1120 KB, 3128 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 1152 KB, 3032 KB/s, 0 seconds passed
... 8%, 1184 KB, 3068 KB/s, 0 seconds passed
... 8%, 1216 KB, 3121 KB/s, 0 seconds passed
... 8%, 1248 KB, 3152 KB/s, 0 seconds passed
... 8%, 1280 KB, 3070 KB/s, 0 seconds passed
... 8%, 1312 KB, 3101 KB/s, 0 seconds passed
... 9%, 1344 KB, 3145 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 1376 KB, 3030 KB/s, 0 seconds passed
... 9%, 1408 KB, 3090 KB/s, 0 seconds passed
... 9%, 1440 KB, 3121 KB/s, 0 seconds passed
... 9%, 1472 KB, 3153 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 1504 KB, 3060 KB/s, 0 seconds passed
... 10%, 1536 KB, 3113 KB/s, 0 seconds passed
... 10%, 1568 KB, 3149 KB/s, 0 seconds passed
... 10%, 1600 KB, 3189 KB/s, 0 seconds passed
... 11%, 1632 KB, 3211 KB/s, 0 seconds passed
... 11%, 1664 KB, 3137 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 1696 KB, 3169 KB/s, 0 seconds passed
... 11%, 1728 KB, 3201 KB/s, 0 seconds passed
... 11%, 1760 KB, 3224 KB/s, 0 seconds passed
... 12%, 1792 KB, 3140 KB/s, 0 seconds passed
... 12%, 1824 KB, 3182 KB/s, 0 seconds passed
... 12%, 1856 KB, 3227 KB/s, 0 seconds passed
... 12%, 1888 KB, 3246 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 1920 KB, 3167 KB/s, 0 seconds passed
... 13%, 1952 KB, 3203 KB/s, 0 seconds passed
... 13%, 1984 KB, 3243 KB/s, 0 seconds passed
... 13%, 2016 KB, 3250 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 2048 KB, 3176 KB/s, 0 seconds passed
... 14%, 2080 KB, 3213 KB/s, 0 seconds passed
... 14%, 2112 KB, 3247 KB/s, 0 seconds passed
... 14%, 2144 KB, 3264 KB/s, 0 seconds passed
... 14%, 2176 KB, 3193 KB/s, 0 seconds passed
... 14%, 2208 KB, 3227 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 2240 KB, 3257 KB/s, 0 seconds passed
... 15%, 2272 KB, 3177 KB/s, 0 seconds passed
... 15%, 2304 KB, 3212 KB/s, 0 seconds passed
... 15%, 2336 KB, 3246 KB/s, 0 seconds passed
... 16%, 2368 KB, 3277 KB/s, 0 seconds passed

.. parsed-literal::

    ... 16%, 2400 KB, 3191 KB/s, 0 seconds passed
... 16%, 2432 KB, 3221 KB/s, 0 seconds passed
... 16%, 2464 KB, 3249 KB/s, 0 seconds passed
... 16%, 2496 KB, 3273 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 2528 KB, 3203 KB/s, 0 seconds passed
... 17%, 2560 KB, 3232 KB/s, 0 seconds passed
... 17%, 2592 KB, 3260 KB/s, 0 seconds passed
... 17%, 2624 KB, 3282 KB/s, 0 seconds passed
... 17%, 2656 KB, 3217 KB/s, 0 seconds passed
... 18%, 2688 KB, 3243 KB/s, 0 seconds passed
... 18%, 2720 KB, 3266 KB/s, 0 seconds passed
... 18%, 2752 KB, 3288 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 2784 KB, 3227 KB/s, 0 seconds passed
... 19%, 2816 KB, 3254 KB/s, 0 seconds passed
... 19%, 2848 KB, 3273 KB/s, 0 seconds passed
... 19%, 2880 KB, 3295 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 2912 KB, 3237 KB/s, 0 seconds passed
... 19%, 2944 KB, 3262 KB/s, 0 seconds passed
... 20%, 2976 KB, 3282 KB/s, 0 seconds passed
... 20%, 3008 KB, 3299 KB/s, 0 seconds passed
... 20%, 3040 KB, 3240 KB/s, 0 seconds passed
... 20%, 3072 KB, 3268 KB/s, 0 seconds passed

.. parsed-literal::

    ... 21%, 3104 KB, 3291 KB/s, 0 seconds passed
... 21%, 3136 KB, 3304 KB/s, 0 seconds passed
... 21%, 3168 KB, 3248 KB/s, 0 seconds passed
... 21%, 3200 KB, 3276 KB/s, 0 seconds passed
... 21%, 3232 KB, 3296 KB/s, 0 seconds passed
... 22%, 3264 KB, 3311 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 3296 KB, 3254 KB/s, 1 seconds passed
... 22%, 3328 KB, 3281 KB/s, 1 seconds passed
... 22%, 3360 KB, 3303 KB/s, 1 seconds passed
... 22%, 3392 KB, 3317 KB/s, 1 seconds passed

.. parsed-literal::

    ... 23%, 3424 KB, 3261 KB/s, 1 seconds passed
... 23%, 3456 KB, 3287 KB/s, 1 seconds passed
... 23%, 3488 KB, 3307 KB/s, 1 seconds passed
... 23%, 3520 KB, 3320 KB/s, 1 seconds passed
... 24%, 3552 KB, 3267 KB/s, 1 seconds passed
... 24%, 3584 KB, 3293 KB/s, 1 seconds passed
... 24%, 3616 KB, 3313 KB/s, 1 seconds passed

.. parsed-literal::

    ... 24%, 3648 KB, 3326 KB/s, 1 seconds passed
... 24%, 3680 KB, 3276 KB/s, 1 seconds passed
... 25%, 3712 KB, 3295 KB/s, 1 seconds passed
... 25%, 3744 KB, 3313 KB/s, 1 seconds passed
... 25%, 3776 KB, 3331 KB/s, 1 seconds passed

.. parsed-literal::

    ... 25%, 3808 KB, 3278 KB/s, 1 seconds passed
... 26%, 3840 KB, 3302 KB/s, 1 seconds passed
... 26%, 3872 KB, 3318 KB/s, 1 seconds passed
... 26%, 3904 KB, 3336 KB/s, 1 seconds passed
... 26%, 3936 KB, 3291 KB/s, 1 seconds passed

.. parsed-literal::

    ... 26%, 3968 KB, 3308 KB/s, 1 seconds passed
... 27%, 4000 KB, 3322 KB/s, 1 seconds passed
... 27%, 4032 KB, 3336 KB/s, 1 seconds passed
... 27%, 4064 KB, 3287 KB/s, 1 seconds passed
... 27%, 4096 KB, 3309 KB/s, 1 seconds passed
... 27%, 4128 KB, 3326 KB/s, 1 seconds passed
... 28%, 4160 KB, 3338 KB/s, 1 seconds passed

.. parsed-literal::

    ... 28%, 4192 KB, 3292 KB/s, 1 seconds passed
... 28%, 4224 KB, 3313 KB/s, 1 seconds passed
... 28%, 4256 KB, 3329 KB/s, 1 seconds passed
... 29%, 4288 KB, 3342 KB/s, 1 seconds passed

.. parsed-literal::

    ... 29%, 4320 KB, 3294 KB/s, 1 seconds passed
... 29%, 4352 KB, 3316 KB/s, 1 seconds passed
... 29%, 4384 KB, 3331 KB/s, 1 seconds passed
... 29%, 4416 KB, 3343 KB/s, 1 seconds passed
... 30%, 4448 KB, 3298 KB/s, 1 seconds passed
... 30%, 4480 KB, 3320 KB/s, 1 seconds passed

.. parsed-literal::

    ... 30%, 4512 KB, 3334 KB/s, 1 seconds passed
... 30%, 4544 KB, 3346 KB/s, 1 seconds passed
... 30%, 4576 KB, 3302 KB/s, 1 seconds passed
... 31%, 4608 KB, 3323 KB/s, 1 seconds passed
... 31%, 4640 KB, 3338 KB/s, 1 seconds passed
... 31%, 4672 KB, 3345 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 4704 KB, 3305 KB/s, 1 seconds passed
... 32%, 4736 KB, 3326 KB/s, 1 seconds passed
... 32%, 4768 KB, 3342 KB/s, 1 seconds passed
... 32%, 4800 KB, 3347 KB/s, 1 seconds passed

.. parsed-literal::

    ... 32%, 4832 KB, 3310 KB/s, 1 seconds passed
... 32%, 4864 KB, 3330 KB/s, 1 seconds passed
... 33%, 4896 KB, 3345 KB/s, 1 seconds passed
... 33%, 4928 KB, 3350 KB/s, 1 seconds passed
... 33%, 4960 KB, 3314 KB/s, 1 seconds passed
... 33%, 4992 KB, 3334 KB/s, 1 seconds passed
... 34%, 5024 KB, 3348 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 5056 KB, 3352 KB/s, 1 seconds passed
... 34%, 5088 KB, 3318 KB/s, 1 seconds passed
... 34%, 5120 KB, 3337 KB/s, 1 seconds passed
... 34%, 5152 KB, 3351 KB/s, 1 seconds passed

.. parsed-literal::

    ... 35%, 5184 KB, 3313 KB/s, 1 seconds passed
... 35%, 5216 KB, 3321 KB/s, 1 seconds passed
... 35%, 5248 KB, 3340 KB/s, 1 seconds passed
... 35%, 5280 KB, 3353 KB/s, 1 seconds passed
... 35%, 5312 KB, 3317 KB/s, 1 seconds passed
... 36%, 5344 KB, 3324 KB/s, 1 seconds passed
... 36%, 5376 KB, 3343 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 5408 KB, 3356 KB/s, 1 seconds passed
... 36%, 5440 KB, 3320 KB/s, 1 seconds passed
... 37%, 5472 KB, 3328 KB/s, 1 seconds passed
... 37%, 5504 KB, 3347 KB/s, 1 seconds passed
... 37%, 5536 KB, 3358 KB/s, 1 seconds passed

.. parsed-literal::

    ... 37%, 5568 KB, 3323 KB/s, 1 seconds passed
... 37%, 5600 KB, 3330 KB/s, 1 seconds passed
... 38%, 5632 KB, 3348 KB/s, 1 seconds passed
... 38%, 5664 KB, 3360 KB/s, 1 seconds passed

.. parsed-literal::

    ... 38%, 5696 KB, 3325 KB/s, 1 seconds passed
... 38%, 5728 KB, 3334 KB/s, 1 seconds passed
... 39%, 5760 KB, 3351 KB/s, 1 seconds passed
... 39%, 5792 KB, 3363 KB/s, 1 seconds passed
... 39%, 5824 KB, 3328 KB/s, 1 seconds passed
... 39%, 5856 KB, 3336 KB/s, 1 seconds passed
... 39%, 5888 KB, 3353 KB/s, 1 seconds passed
... 40%, 5920 KB, 3365 KB/s, 1 seconds passed

.. parsed-literal::

    ... 40%, 5952 KB, 3331 KB/s, 1 seconds passed
... 40%, 5984 KB, 3339 KB/s, 1 seconds passed
... 40%, 6016 KB, 3356 KB/s, 1 seconds passed
... 40%, 6048 KB, 3369 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 6080 KB, 3335 KB/s, 1 seconds passed
... 41%, 6112 KB, 3343 KB/s, 1 seconds passed
... 41%, 6144 KB, 3358 KB/s, 1 seconds passed
... 41%, 6176 KB, 3372 KB/s, 1 seconds passed
... 42%, 6208 KB, 3338 KB/s, 1 seconds passed
... 42%, 6240 KB, 3347 KB/s, 1 seconds passed

.. parsed-literal::

    ... 42%, 6272 KB, 3360 KB/s, 1 seconds passed
... 42%, 6304 KB, 3372 KB/s, 1 seconds passed
... 42%, 6336 KB, 3339 KB/s, 1 seconds passed
... 43%, 6368 KB, 3346 KB/s, 1 seconds passed
... 43%, 6400 KB, 3361 KB/s, 1 seconds passed
... 43%, 6432 KB, 3374 KB/s, 1 seconds passed

.. parsed-literal::

    ... 43%, 6464 KB, 3341 KB/s, 1 seconds passed
... 43%, 6496 KB, 3348 KB/s, 1 seconds passed
... 44%, 6528 KB, 3363 KB/s, 1 seconds passed
... 44%, 6560 KB, 3376 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 6592 KB, 3342 KB/s, 1 seconds passed
... 44%, 6624 KB, 3351 KB/s, 1 seconds passed
... 45%, 6656 KB, 3366 KB/s, 1 seconds passed
... 45%, 6688 KB, 3378 KB/s, 1 seconds passed
... 45%, 6720 KB, 3344 KB/s, 2 seconds passed
... 45%, 6752 KB, 3353 KB/s, 2 seconds passed
... 45%, 6784 KB, 3363 KB/s, 2 seconds passed
... 46%, 6816 KB, 3379 KB/s, 2 seconds passed

.. parsed-literal::

    ... 46%, 6848 KB, 3347 KB/s, 2 seconds passed
... 46%, 6880 KB, 3350 KB/s, 2 seconds passed
... 46%, 6912 KB, 3365 KB/s, 2 seconds passed
... 47%, 6944 KB, 3380 KB/s, 2 seconds passed

.. parsed-literal::

    ... 47%, 6976 KB, 3344 KB/s, 2 seconds passed
... 47%, 7008 KB, 3351 KB/s, 2 seconds passed
... 47%, 7040 KB, 3366 KB/s, 2 seconds passed
... 47%, 7072 KB, 3380 KB/s, 2 seconds passed

.. parsed-literal::

    ... 48%, 7104 KB, 3346 KB/s, 2 seconds passed
... 48%, 7136 KB, 3353 KB/s, 2 seconds passed
... 48%, 7168 KB, 3367 KB/s, 2 seconds passed
... 48%, 7200 KB, 3381 KB/s, 2 seconds passed
... 48%, 7232 KB, 3348 KB/s, 2 seconds passed
... 49%, 7264 KB, 3354 KB/s, 2 seconds passed
... 49%, 7296 KB, 3369 KB/s, 2 seconds passed
... 49%, 7328 KB, 3383 KB/s, 2 seconds passed

.. parsed-literal::

    ... 49%, 7360 KB, 3349 KB/s, 2 seconds passed
... 50%, 7392 KB, 3355 KB/s, 2 seconds passed
... 50%, 7424 KB, 3369 KB/s, 2 seconds passed
... 50%, 7456 KB, 3383 KB/s, 2 seconds passed

.. parsed-literal::

    ... 50%, 7488 KB, 3351 KB/s, 2 seconds passed
... 50%, 7520 KB, 3356 KB/s, 2 seconds passed
... 51%, 7552 KB, 3370 KB/s, 2 seconds passed
... 51%, 7584 KB, 3384 KB/s, 2 seconds passed
... 51%, 7616 KB, 3353 KB/s, 2 seconds passed

.. parsed-literal::

    ... 51%, 7648 KB, 3358 KB/s, 2 seconds passed
... 52%, 7680 KB, 3371 KB/s, 2 seconds passed
... 52%, 7712 KB, 3384 KB/s, 2 seconds passed
... 52%, 7744 KB, 3354 KB/s, 2 seconds passed
... 52%, 7776 KB, 3356 KB/s, 2 seconds passed
... 52%, 7808 KB, 3369 KB/s, 2 seconds passed
... 53%, 7840 KB, 3382 KB/s, 2 seconds passed

.. parsed-literal::

    ... 53%, 7872 KB, 3355 KB/s, 2 seconds passed
... 53%, 7904 KB, 3358 KB/s, 2 seconds passed
... 53%, 7936 KB, 3371 KB/s, 2 seconds passed
... 53%, 7968 KB, 3384 KB/s, 2 seconds passed

.. parsed-literal::

    ... 54%, 8000 KB, 3357 KB/s, 2 seconds passed
... 54%, 8032 KB, 3360 KB/s, 2 seconds passed
... 54%, 8064 KB, 3372 KB/s, 2 seconds passed
... 54%, 8096 KB, 3385 KB/s, 2 seconds passed
... 55%, 8128 KB, 3358 KB/s, 2 seconds passed
... 55%, 8160 KB, 3360 KB/s, 2 seconds passed
... 55%, 8192 KB, 3373 KB/s, 2 seconds passed

.. parsed-literal::

    ... 55%, 8224 KB, 3385 KB/s, 2 seconds passed
... 55%, 8256 KB, 3361 KB/s, 2 seconds passed
... 56%, 8288 KB, 3361 KB/s, 2 seconds passed
... 56%, 8320 KB, 3373 KB/s, 2 seconds passed
... 56%, 8352 KB, 3385 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 8384 KB, 3363 KB/s, 2 seconds passed
... 56%, 8416 KB, 3366 KB/s, 2 seconds passed
... 57%, 8448 KB, 3374 KB/s, 2 seconds passed
... 57%, 8480 KB, 3387 KB/s, 2 seconds passed
... 57%, 8512 KB, 3363 KB/s, 2 seconds passed

.. parsed-literal::

    ... 57%, 8544 KB, 3364 KB/s, 2 seconds passed
... 58%, 8576 KB, 3375 KB/s, 2 seconds passed
... 58%, 8608 KB, 3387 KB/s, 2 seconds passed
... 58%, 8640 KB, 3365 KB/s, 2 seconds passed
... 58%, 8672 KB, 3366 KB/s, 2 seconds passed
... 58%, 8704 KB, 3377 KB/s, 2 seconds passed
... 59%, 8736 KB, 3388 KB/s, 2 seconds passed

.. parsed-literal::

    ... 59%, 8768 KB, 3367 KB/s, 2 seconds passed
... 59%, 8800 KB, 3367 KB/s, 2 seconds passed
... 59%, 8832 KB, 3378 KB/s, 2 seconds passed
... 60%, 8864 KB, 3389 KB/s, 2 seconds passed

.. parsed-literal::

    ... 60%, 8896 KB, 3369 KB/s, 2 seconds passed
... 60%, 8928 KB, 3370 KB/s, 2 seconds passed
... 60%, 8960 KB, 3378 KB/s, 2 seconds passed
... 60%, 8992 KB, 3388 KB/s, 2 seconds passed
... 61%, 9024 KB, 3370 KB/s, 2 seconds passed

.. parsed-literal::

    ... 61%, 9056 KB, 3372 KB/s, 2 seconds passed
... 61%, 9088 KB, 3380 KB/s, 2 seconds passed
... 61%, 9120 KB, 3390 KB/s, 2 seconds passed
... 61%, 9152 KB, 3371 KB/s, 2 seconds passed
... 62%, 9184 KB, 3374 KB/s, 2 seconds passed
... 62%, 9216 KB, 3380 KB/s, 2 seconds passed
... 62%, 9248 KB, 3390 KB/s, 2 seconds passed

.. parsed-literal::

    ... 62%, 9280 KB, 3371 KB/s, 2 seconds passed
... 63%, 9312 KB, 3375 KB/s, 2 seconds passed
... 63%, 9344 KB, 3382 KB/s, 2 seconds passed
... 63%, 9376 KB, 3391 KB/s, 2 seconds passed

.. parsed-literal::

    ... 63%, 9408 KB, 3372 KB/s, 2 seconds passed
... 63%, 9440 KB, 3377 KB/s, 2 seconds passed
... 64%, 9472 KB, 3384 KB/s, 2 seconds passed
... 64%, 9504 KB, 3393 KB/s, 2 seconds passed
... 64%, 9536 KB, 3373 KB/s, 2 seconds passed
... 64%, 9568 KB, 3378 KB/s, 2 seconds passed
... 65%, 9600 KB, 3386 KB/s, 2 seconds passed
... 65%, 9632 KB, 3394 KB/s, 2 seconds passed

.. parsed-literal::

    ... 65%, 9664 KB, 3374 KB/s, 2 seconds passed
... 65%, 9696 KB, 3379 KB/s, 2 seconds passed
... 65%, 9728 KB, 3387 KB/s, 2 seconds passed
... 66%, 9760 KB, 3395 KB/s, 2 seconds passed

.. parsed-literal::

    ... 66%, 9792 KB, 3373 KB/s, 2 seconds passed
... 66%, 9824 KB, 3378 KB/s, 2 seconds passed
... 66%, 9856 KB, 3386 KB/s, 2 seconds passed
... 66%, 9888 KB, 3393 KB/s, 2 seconds passed
... 67%, 9920 KB, 3373 KB/s, 2 seconds passed

.. parsed-literal::

    ... 67%, 9952 KB, 3379 KB/s, 2 seconds passed
... 67%, 9984 KB, 3387 KB/s, 2 seconds passed
... 67%, 10016 KB, 3394 KB/s, 2 seconds passed
... 68%, 10048 KB, 3373 KB/s, 2 seconds passed
... 68%, 10080 KB, 3379 KB/s, 2 seconds passed
... 68%, 10112 KB, 3388 KB/s, 2 seconds passed
... 68%, 10144 KB, 3395 KB/s, 2 seconds passed

.. parsed-literal::

    ... 68%, 10176 KB, 3373 KB/s, 3 seconds passed
... 69%, 10208 KB, 3379 KB/s, 3 seconds passed
... 69%, 10240 KB, 3388 KB/s, 3 seconds passed
... 69%, 10272 KB, 3395 KB/s, 3 seconds passed

.. parsed-literal::

    ... 69%, 10304 KB, 3375 KB/s, 3 seconds passed
... 70%, 10336 KB, 3381 KB/s, 3 seconds passed
... 70%, 10368 KB, 3389 KB/s, 3 seconds passed
... 70%, 10400 KB, 3396 KB/s, 3 seconds passed
... 70%, 10432 KB, 3376 KB/s, 3 seconds passed
... 70%, 10464 KB, 3381 KB/s, 3 seconds passed

.. parsed-literal::

    ... 71%, 10496 KB, 3388 KB/s, 3 seconds passed
... 71%, 10528 KB, 3375 KB/s, 3 seconds passed
... 71%, 10560 KB, 3376 KB/s, 3 seconds passed
... 71%, 10592 KB, 3382 KB/s, 3 seconds passed
... 71%, 10624 KB, 3389 KB/s, 3 seconds passed

.. parsed-literal::

    ... 72%, 10656 KB, 3382 KB/s, 3 seconds passed
... 72%, 10688 KB, 3378 KB/s, 3 seconds passed
... 72%, 10720 KB, 3384 KB/s, 3 seconds passed
... 72%, 10752 KB, 3390 KB/s, 3 seconds passed
... 73%, 10784 KB, 3383 KB/s, 3 seconds passed

.. parsed-literal::

    ... 73%, 10816 KB, 3379 KB/s, 3 seconds passed
... 73%, 10848 KB, 3385 KB/s, 3 seconds passed
... 73%, 10880 KB, 3392 KB/s, 3 seconds passed
... 73%, 10912 KB, 3384 KB/s, 3 seconds passed
... 74%, 10944 KB, 3381 KB/s, 3 seconds passed
... 74%, 10976 KB, 3386 KB/s, 3 seconds passed
... 74%, 11008 KB, 3393 KB/s, 3 seconds passed

.. parsed-literal::

    ... 74%, 11040 KB, 3385 KB/s, 3 seconds passed
... 74%, 11072 KB, 3382 KB/s, 3 seconds passed
... 75%, 11104 KB, 3388 KB/s, 3 seconds passed
... 75%, 11136 KB, 3394 KB/s, 3 seconds passed
... 75%, 11168 KB, 3386 KB/s, 3 seconds passed

.. parsed-literal::

    ... 75%, 11200 KB, 3382 KB/s, 3 seconds passed
... 76%, 11232 KB, 3388 KB/s, 3 seconds passed
... 76%, 11264 KB, 3394 KB/s, 3 seconds passed
... 76%, 11296 KB, 3375 KB/s, 3 seconds passed
... 76%, 11328 KB, 3381 KB/s, 3 seconds passed

.. parsed-literal::

    ... 76%, 11360 KB, 3387 KB/s, 3 seconds passed
... 77%, 11392 KB, 3394 KB/s, 3 seconds passed
... 77%, 11424 KB, 3381 KB/s, 3 seconds passed
... 77%, 11456 KB, 3382 KB/s, 3 seconds passed
... 77%, 11488 KB, 3388 KB/s, 3 seconds passed
... 78%, 11520 KB, 3395 KB/s, 3 seconds passed

.. parsed-literal::

    ... 78%, 11552 KB, 3388 KB/s, 3 seconds passed
... 78%, 11584 KB, 3382 KB/s, 3 seconds passed
... 78%, 11616 KB, 3389 KB/s, 3 seconds passed
... 78%, 11648 KB, 3396 KB/s, 3 seconds passed
... 79%, 11680 KB, 3389 KB/s, 3 seconds passed

.. parsed-literal::

    ... 79%, 11712 KB, 3383 KB/s, 3 seconds passed
... 79%, 11744 KB, 3390 KB/s, 3 seconds passed
... 79%, 11776 KB, 3396 KB/s, 3 seconds passed
... 79%, 11808 KB, 3378 KB/s, 3 seconds passed
... 80%, 11840 KB, 3383 KB/s, 3 seconds passed
... 80%, 11872 KB, 3390 KB/s, 3 seconds passed
... 80%, 11904 KB, 3395 KB/s, 3 seconds passed

.. parsed-literal::

    ... 80%, 11936 KB, 3380 KB/s, 3 seconds passed
... 81%, 11968 KB, 3384 KB/s, 3 seconds passed
... 81%, 12000 KB, 3391 KB/s, 3 seconds passed
... 81%, 12032 KB, 3396 KB/s, 3 seconds passed

.. parsed-literal::

    ... 81%, 12064 KB, 3378 KB/s, 3 seconds passed
... 81%, 12096 KB, 3385 KB/s, 3 seconds passed
... 82%, 12128 KB, 3391 KB/s, 3 seconds passed
... 82%, 12160 KB, 3397 KB/s, 3 seconds passed
... 82%, 12192 KB, 3380 KB/s, 3 seconds passed

.. parsed-literal::

    ... 82%, 12224 KB, 3386 KB/s, 3 seconds passed
... 83%, 12256 KB, 3392 KB/s, 3 seconds passed
... 83%, 12288 KB, 3398 KB/s, 3 seconds passed
... 83%, 12320 KB, 3380 KB/s, 3 seconds passed
... 83%, 12352 KB, 3386 KB/s, 3 seconds passed
... 83%, 12384 KB, 3393 KB/s, 3 seconds passed
... 84%, 12416 KB, 3398 KB/s, 3 seconds passed

.. parsed-literal::

    ... 84%, 12448 KB, 3381 KB/s, 3 seconds passed
... 84%, 12480 KB, 3387 KB/s, 3 seconds passed
... 84%, 12512 KB, 3394 KB/s, 3 seconds passed
... 84%, 12544 KB, 3399 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 12576 KB, 3382 KB/s, 3 seconds passed
... 85%, 12608 KB, 3388 KB/s, 3 seconds passed
... 85%, 12640 KB, 3394 KB/s, 3 seconds passed
... 85%, 12672 KB, 3400 KB/s, 3 seconds passed
... 86%, 12704 KB, 3381 KB/s, 3 seconds passed
... 86%, 12736 KB, 3388 KB/s, 3 seconds passed
... 86%, 12768 KB, 3395 KB/s, 3 seconds passed
... 86%, 12800 KB, 3401 KB/s, 3 seconds passed

.. parsed-literal::

    ... 86%, 12832 KB, 3383 KB/s, 3 seconds passed
... 87%, 12864 KB, 3389 KB/s, 3 seconds passed
... 87%, 12896 KB, 3396 KB/s, 3 seconds passed
... 87%, 12928 KB, 3402 KB/s, 3 seconds passed

.. parsed-literal::

    ... 87%, 12960 KB, 3383 KB/s, 3 seconds passed
... 87%, 12992 KB, 3389 KB/s, 3 seconds passed
... 88%, 13024 KB, 3396 KB/s, 3 seconds passed
... 88%, 13056 KB, 3402 KB/s, 3 seconds passed

.. parsed-literal::

    ... 88%, 13088 KB, 3383 KB/s, 3 seconds passed
... 88%, 13120 KB, 3390 KB/s, 3 seconds passed
... 89%, 13152 KB, 3397 KB/s, 3 seconds passed
... 89%, 13184 KB, 3402 KB/s, 3 seconds passed
... 89%, 13216 KB, 3384 KB/s, 3 seconds passed
... 89%, 13248 KB, 3391 KB/s, 3 seconds passed
... 89%, 13280 KB, 3398 KB/s, 3 seconds passed
... 90%, 13312 KB, 3404 KB/s, 3 seconds passed

.. parsed-literal::

    ... 90%, 13344 KB, 3386 KB/s, 3 seconds passed
... 90%, 13376 KB, 3392 KB/s, 3 seconds passed
... 90%, 13408 KB, 3399 KB/s, 3 seconds passed
... 91%, 13440 KB, 3404 KB/s, 3 seconds passed

.. parsed-literal::

    ... 91%, 13472 KB, 3387 KB/s, 3 seconds passed
... 91%, 13504 KB, 3393 KB/s, 3 seconds passed
... 91%, 13536 KB, 3399 KB/s, 3 seconds passed
... 91%, 13568 KB, 3406 KB/s, 3 seconds passed
... 92%, 13600 KB, 3390 KB/s, 4 seconds passed
... 92%, 13632 KB, 3394 KB/s, 4 seconds passed
... 92%, 13664 KB, 3400 KB/s, 4 seconds passed

.. parsed-literal::

    ... 92%, 13696 KB, 3406 KB/s, 4 seconds passed
... 92%, 13728 KB, 3391 KB/s, 4 seconds passed
... 93%, 13760 KB, 3393 KB/s, 4 seconds passed
... 93%, 13792 KB, 3400 KB/s, 4 seconds passed

.. parsed-literal::

    ... 93%, 13824 KB, 3394 KB/s, 4 seconds passed
... 93%, 13856 KB, 3388 KB/s, 4 seconds passed
... 94%, 13888 KB, 3392 KB/s, 4 seconds passed
... 94%, 13920 KB, 3399 KB/s, 4 seconds passed
... 94%, 13952 KB, 3405 KB/s, 4 seconds passed

.. parsed-literal::

    ... 94%, 13984 KB, 3389 KB/s, 4 seconds passed
... 94%, 14016 KB, 3393 KB/s, 4 seconds passed
... 95%, 14048 KB, 3400 KB/s, 4 seconds passed
... 95%, 14080 KB, 3396 KB/s, 4 seconds passed
... 95%, 14112 KB, 3390 KB/s, 4 seconds passed
... 95%, 14144 KB, 3394 KB/s, 4 seconds passed
... 96%, 14176 KB, 3401 KB/s, 4 seconds passed

.. parsed-literal::

    ... 96%, 14208 KB, 3397 KB/s, 4 seconds passed
... 96%, 14240 KB, 3390 KB/s, 4 seconds passed
... 96%, 14272 KB, 3395 KB/s, 4 seconds passed
... 96%, 14304 KB, 3402 KB/s, 4 seconds passed
... 97%, 14336 KB, 3397 KB/s, 4 seconds passed

.. parsed-literal::

    ... 97%, 14368 KB, 3391 KB/s, 4 seconds passed
... 97%, 14400 KB, 3395 KB/s, 4 seconds passed
... 97%, 14432 KB, 3402 KB/s, 4 seconds passed
... 97%, 14464 KB, 3398 KB/s, 4 seconds passed
... 98%, 14496 KB, 3391 KB/s, 4 seconds passed

.. parsed-literal::

    ... 98%, 14528 KB, 3396 KB/s, 4 seconds passed
... 98%, 14560 KB, 3403 KB/s, 4 seconds passed
... 98%, 14592 KB, 3399 KB/s, 4 seconds passed
... 99%, 14624 KB, 3391 KB/s, 4 seconds passed
... 99%, 14656 KB, 3396 KB/s, 4 seconds passed
... 99%, 14688 KB, 3403 KB/s, 4 seconds passed

.. parsed-literal::

    ... 99%, 14720 KB, 3399 KB/s, 4 seconds passed
... 99%, 14752 KB, 3392 KB/s, 4 seconds passed
... 100%, 14764 KB, 3394 KB/s, 4 seconds passed

    


Load your labels
~~~~~~~~~~~~~~~~



This tutorial uses `Kinetics-400
dataset <https://deepmind.com/research/open-source/kinetics>`__, and
also provides the text file embedded into this notebook.

   **NOTE**: If you want to run
   ``"driver-action-recognition-adas-0002"`` model, replace the
   ``kinetics.txt`` file to ``driver_actions.txt``.

.. code:: ipython3

    # Download the text from the openvino_notebooks storage
    vocab_file_path = utils.download_file(
        "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/text/kinetics.txt",
        directory="data"
    )
    
    with vocab_file_path.open(mode='r') as f:
        labels = [line.strip() for line in f]
    
    print(labels[0:9], np.shape(labels))



.. parsed-literal::

    data/kinetics.txt:   0%|          | 0.00/5.82k [00:00<?, ?B/s]


.. parsed-literal::

    ['abseiling', 'air drumming', 'answering questions', 'applauding', 'applying cream', 'archery', 'arm wrestling', 'arranging flowers', 'assembling computer'] (400,)


Load the models
~~~~~~~~~~~~~~~



Load the two models for this particular architecture, Encoder and
Decoder. Downloaded models are located in a fixed structure, indicating
a vendor, the name of the model, and a precision.

1. Initialize OpenVINO Runtime.
2. Read the network from ``*.bin`` and ``*.xml`` files (weights and
   architecture).
3. Compile the model for specified device.
4. Get input and output names of nodes.

Only a few lines of code are required to run the model.

Select device from dropdown list for running inference using OpenVINO

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



Model Initialization function
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    # Initialize OpenVINO Runtime.
    core = ov.Core()
    
    
    def model_init(model_path: str, device: str) -> Tuple:
        """
        Read the network and weights from a file, load the
        model on CPU and get input and output names of nodes
    
        :param: 
                model: model architecture path *.xml
                device: inference device
        :retuns:
                compiled_model: Compiled model 
                input_key: Input node for model
                output_key: Output node for model
        """
    
        # Read the network and corresponding weights from a file.
        model = core.read_model(model=model_path)
        # Compile the model for specified device.
        compiled_model = core.compile_model(model=model, device_name=device)
        # Get input and output names of nodes.
        input_keys = compiled_model.input(0)
        output_keys = compiled_model.output(0)
        return input_keys, output_keys, compiled_model

Initialization for Encoder and Decoder
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



.. code:: ipython3

    # Encoder initialization
    input_key_en, output_keys_en, compiled_model_en = model_init(model_path_encoder, device.value)
    # Decoder initialization
    input_key_de, output_keys_de, compiled_model_de = model_init(model_path_decoder, device.value)
    
    # Get input size - Encoder.
    height_en, width_en = list(input_key_en.shape)[2:]
    # Get input size - Decoder.
    frames2decode = list(input_key_de.shape)[0:][1]

Helper functions
~~~~~~~~~~~~~~~~



Use the following helper functions for preprocessing and postprocessing
frames:

1. Preprocess the input image before running the Encoder model.
   (``center_crop`` and ``adaptative_resize``)
2. Decode top-3 probabilities into label names. (``decode_output``)
3. Draw the Region of Interest (ROI) over the video.
   (``rec_frame_display``)
4. Prepare the frame for displaying label names over the video.
   (``display_text_fnc``)

.. code:: ipython3

    def center_crop(frame: np.ndarray) -> np.ndarray:
        """
        Center crop squared the original frame to standardize the input image to the encoder model
    
        :param frame: input frame
        :returns: center-crop-squared frame
        """
        img_h, img_w, _ = frame.shape
        min_dim = min(img_h, img_w)
        start_x = int((img_w - min_dim) / 2.0)
        start_y = int((img_h - min_dim) / 2.0)
        roi = [start_y, (start_y + min_dim), start_x, (start_x + min_dim)]
        return frame[start_y : (start_y + min_dim), start_x : (start_x + min_dim), ...], roi
    
    
    def adaptive_resize(frame: np.ndarray, size: int) -> np.ndarray:
        """
         The frame going to be resized to have a height of size or a width of size
    
        :param frame: input frame
        :param size: input size to encoder model
        :returns: resized frame, np.array type
        """
        h, w, _ = frame.shape
        scale = size / min(h, w)
        w_scaled, h_scaled = int(w * scale), int(h * scale)
        if w_scaled == w and h_scaled == h:
            return frame
        return cv2.resize(frame, (w_scaled, h_scaled))
    
    
    def decode_output(probs: np.ndarray, labels: np.ndarray, top_k: int = 3) -> np.ndarray:
        """
        Decodes top probabilities into corresponding label names
    
        :param probs: confidence vector for 400 actions
        :param labels: list of actions
        :param top_k: The k most probable positions in the list of labels
        :returns: decoded_labels: The k most probable actions from the labels list
                  decoded_top_probs: confidence for the k most probable actions
        """
        top_ind = np.argsort(-1 * probs)[:top_k]
        out_label = np.array(labels)[top_ind.astype(int)]
        decoded_labels = [out_label[0][0], out_label[0][1], out_label[0][2]]
        top_probs = np.array(probs)[0][top_ind.astype(int)]
        decoded_top_probs = [top_probs[0][0], top_probs[0][1], top_probs[0][2]]
        return decoded_labels, decoded_top_probs
    
    
    def rec_frame_display(frame: np.ndarray, roi) -> np.ndarray:
        """
        Draw a rec frame over actual frame
    
        :param frame: input frame
        :param roi: Region of interest, image section processed by the Encoder
        :returns: frame with drawed shape
    
        """
    
        cv2.line(frame, (roi[2] + 3, roi[0] + 3), (roi[2] + 3, roi[0] + 100), (0, 200, 0), 2)
        cv2.line(frame, (roi[2] + 3, roi[0] + 3), (roi[2] + 100, roi[0] + 3), (0, 200, 0), 2)
        cv2.line(frame, (roi[3] - 3, roi[1] - 3), (roi[3] - 3, roi[1] - 100), (0, 200, 0), 2)
        cv2.line(frame, (roi[3] - 3, roi[1] - 3), (roi[3] - 100, roi[1] - 3), (0, 200, 0), 2)
        cv2.line(frame, (roi[3] - 3, roi[0] + 3), (roi[3] - 3, roi[0] + 100), (0, 200, 0), 2)
        cv2.line(frame, (roi[3] - 3, roi[0] + 3), (roi[3] - 100, roi[0] + 3), (0, 200, 0), 2)
        cv2.line(frame, (roi[2] + 3, roi[1] - 3), (roi[2] + 3, roi[1] - 100), (0, 200, 0), 2)
        cv2.line(frame, (roi[2] + 3, roi[1] - 3), (roi[2] + 100, roi[1] - 3), (0, 200, 0), 2)
        # Write ROI over actual frame
        FONT_STYLE = cv2.FONT_HERSHEY_SIMPLEX
        org = (roi[2] + 3, roi[1] - 3)
        org2 = (roi[2] + 2, roi[1] - 2)
        FONT_SIZE = 0.5
        FONT_COLOR = (0, 200, 0)
        FONT_COLOR2 = (0, 0, 0)
        cv2.putText(frame, "ROI", org2, FONT_STYLE, FONT_SIZE, FONT_COLOR2)
        cv2.putText(frame, "ROI", org, FONT_STYLE, FONT_SIZE, FONT_COLOR)
        return frame
    
    
    def display_text_fnc(frame: np.ndarray, display_text: str, index: int):
        """
        Include a text on the analyzed frame
    
        :param frame: input frame
        :param display_text: text to add on the frame
        :param index: index line dor adding text
    
        """
        # Configuration for displaying images with text.
        FONT_COLOR = (255, 255, 255)
        FONT_COLOR2 = (0, 0, 0)
        FONT_STYLE = cv2.FONT_HERSHEY_DUPLEX
        FONT_SIZE = 0.7
        TEXT_VERTICAL_INTERVAL = 25
        TEXT_LEFT_MARGIN = 15
        # ROI over actual frame
        (processed, roi) = center_crop(frame)
        # Draw a ROI over actual frame.
        frame = rec_frame_display(frame, roi)
        # Put a text over actual frame.
        text_loc = (TEXT_LEFT_MARGIN, TEXT_VERTICAL_INTERVAL * (index + 1))
        text_loc2 = (TEXT_LEFT_MARGIN + 1, TEXT_VERTICAL_INTERVAL * (index + 1) + 1)
        cv2.putText(frame, display_text, text_loc2, FONT_STYLE, FONT_SIZE, FONT_COLOR2)
        cv2.putText(frame, display_text, text_loc, FONT_STYLE, FONT_SIZE, FONT_COLOR)

AI Functions
~~~~~~~~~~~~



Following the pipeline above, you will use the next functions to:

1. Preprocess a frame before running the Encoder. (``preprocessing``)
2. Encoder Inference per frame. (``encoder``)
3. Decoder inference per set of frames. (``decoder``)
4. Normalize the Decoder output to get confidence values per action
   recognition label. (``softmax``)

.. code:: ipython3

    def preprocessing(frame: np.ndarray, size: int) -> np.ndarray:
        """
        Preparing frame before Encoder.
        The image should be scaled to its shortest dimension at "size"
        and cropped, centered, and squared so that both width and
        height have lengths "size". The frame must be transposed from
        Height-Width-Channels (HWC) to Channels-Height-Width (CHW).
    
        :param frame: input frame
        :param size: input size to encoder model
        :returns: resized and cropped frame
        """
        # Adaptative resize
        preprocessed = adaptive_resize(frame, size)
        # Center_crop
        (preprocessed, roi) = center_crop(preprocessed)
        # Transpose frame HWC -> CHW
        preprocessed = preprocessed.transpose((2, 0, 1))[None,]  # HWC -> CHW
        return preprocessed, roi
    
    
    def encoder(
        preprocessed: np.ndarray,
        compiled_model: CompiledModel
    ) -> List:
        """
        Encoder Inference per frame. This function calls the network previously
        configured for the encoder model (compiled_model), extracts the data
        from the output node, and appends it in an array to be used by the decoder.
    
        :param: preprocessed: preprocessing frame
        :param: compiled_model: Encoder model network
        :returns: encoder_output: embedding layer that is appended with each arriving frame
        """
        output_key_en = compiled_model.output(0)
        
        # Get results on action-recognition-0001-encoder model
        infer_result_encoder = compiled_model([preprocessed])[output_key_en]
        return infer_result_encoder
    
    
    def decoder(encoder_output: List, compiled_model_de: CompiledModel) -> List:
        """
        Decoder inference per set of frames. This function concatenates the embedding layer
        froms the encoder output, transpose the array to match with the decoder input size.
        Calls the network previously configured for the decoder model (compiled_model_de), extracts
        the logits and normalize those to get confidence values along specified axis.
        Decodes top probabilities into corresponding label names
    
        :param: encoder_output: embedding layer for 16 frames
        :param: compiled_model_de: Decoder model network
        :returns: decoded_labels: The k most probable actions from the labels list
                  decoded_top_probs: confidence for the k most probable actions
        """
        # Concatenate sample_duration frames in just one array
        decoder_input = np.concatenate(encoder_output, axis=0)
        # Organize input shape vector to the Decoder (shape: [1x16x512]]
        decoder_input = decoder_input.transpose((2, 0, 1, 3))
        decoder_input = np.squeeze(decoder_input, axis=3)
        output_key_de = compiled_model_de.output(0)
        # Get results on action-recognition-0001-decoder model
        result_de = compiled_model_de([decoder_input])[output_key_de]
        # Normalize logits to get confidence values along specified axis
        probs = softmax(result_de - np.max(result_de))
        # Decodes top probabilities into corresponding label names
        decoded_labels, decoded_top_probs = decode_output(probs, labels, top_k=3)
        return decoded_labels, decoded_top_probs
    
    
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Normalizes logits to get confidence values along specified axis
        x: np.array, axis=None
        """
        exp = np.exp(x)
        return exp / np.sum(exp, axis=None)

Main Processing Function
~~~~~~~~~~~~~~~~~~~~~~~~



Running action recognition function will run in different operations,
either a webcam or a video file. See the list of procedures below:

1. Create a video player to play with target fps
   (``utils.VideoPlayer``).
2. Prepare a set of frames to be encoded-decoded.
3. Run AI functions
4. Visualize the results.

.. code:: ipython3

    def run_action_recognition(
        source: str = "0",
        flip: bool = True,
        use_popup: bool = False,
        compiled_model_en: CompiledModel = compiled_model_en,
        compiled_model_de: CompiledModel = compiled_model_de,
        skip_first_frames: int = 0,
    ):
        """
        Use the "source" webcam or video file to run the complete pipeline for action-recognition problem
        1. Create a video player to play with target fps
        2. Prepare a set of frames to be encoded-decoded
        3. Preprocess frame before Encoder
        4. Encoder Inference per frame
        5. Decoder inference per set of frames
        6. Visualize the results
    
        :param: source: webcam "0" or video path
        :param: flip: to be used by VideoPlayer function for flipping capture image
        :param: use_popup: False for showing encoded frames over this notebook, True for creating a popup window.
        :param: skip_first_frames: Number of frames to skip at the beginning of the video.
        :returns: display video over the notebook or in a popup window
    
        """
        size = height_en  # Endoder input size - From Cell 5_9
        sample_duration = frames2decode  # Decoder input size - From Cell 5_7
        # Select frames per second of your source.
        fps = 30
        player = None
        try:
            # Create a video player.
            player = utils.VideoPlayer(source, flip=flip, fps=fps, skip_first_frames=skip_first_frames)
            # Start capturing.
            player.start()
            if use_popup:
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_GUI_NORMAL | cv2.WINDOW_AUTOSIZE)
    
            processing_times = collections.deque()
            processing_time = 0
            encoder_output = []
            decoded_labels = [0, 0, 0]
            decoded_top_probs = [0, 0, 0]
            counter = 0
            # Create a text template to show inference results over video.
            text_inference_template = "Infer Time:{Time:.1f}ms,{fps:.1f}FPS"
            text_template = "{label},{conf:.2f}%"
    
            while True:
                counter = counter + 1
    
                # Read a frame from the video stream.
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break
    
                scale = 1280 / max(frame.shape)
    
                # Adaptative resize for visualization.
                if scale < 1:
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    
                # Select one frame every two for processing through the encoder.
                # After 16 frames are processed, the decoder will find the action,
                # and the label will be printed over the frames.
    
                if counter % 2 == 0:
                    # Preprocess frame before Encoder.
                    (preprocessed, _) = preprocessing(frame, size)
    
                    # Measure processing time.
                    start_time = time.time()
    
                    # Encoder Inference per frame
                    encoder_output.append(encoder(preprocessed, compiled_model_en))
    
                    # Decoder inference per set of frames
                    # Wait for sample duration to work with decoder model.
                    if len(encoder_output) == sample_duration:
                        decoded_labels, decoded_top_probs = decoder(encoder_output, compiled_model_de)
                        encoder_output = []
    
                    # Inference has finished. Display the results.
                    stop_time = time.time()
    
                    # Calculate processing time.
                    processing_times.append(stop_time - start_time)
    
                    # Use processing times from last 200 frames.
                    if len(processing_times) > 200:
                        processing_times.popleft()
    
                    # Mean processing time [ms]
                    processing_time = np.mean(processing_times) * 1000
                    fps = 1000 / processing_time
    
                # Visualize the results.
                for i in range(0, 3):
                    display_text = text_template.format(
                        label=decoded_labels[i],
                        conf=decoded_top_probs[i] * 100,
                    )
                    display_text_fnc(frame, display_text, i)
    
                display_text = text_inference_template.format(Time=processing_time, fps=fps)
                display_text_fnc(frame, display_text, 3)
    
                # Use this workaround if you experience flickering.
                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27
                    if key == 27:
                        break
                else:
                    # Encode numpy array to jpg.
                    _, encoded_img = cv2.imencode(".jpg", frame, params=[cv2.IMWRITE_JPEG_QUALITY, 90])
                    # Create an IPython image.
                    i = display.Image(data=encoded_img)
                    # Display the image in this notebook.
                    display.clear_output(wait=True)
                    display.display(i)
    
        # ctrl-c
        except KeyboardInterrupt:
            print("Interrupted")
        # Any different error
        except RuntimeError as e:
            print(e)
        finally:
            if player is not None:
                # Stop capturing.
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()

Run Action Recognition
~~~~~~~~~~~~~~~~~~~~~~



Find out how the model works in a video file. `Any format
supported <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
by OpenCV will work. You can press the stop button anytime while the
video file is running, and it will activate the webcam for the next
step.

   **NOTE**: Sometimes, the video can be cut off if there are corrupted
   frames. In that case, you can convert it. If you experience any
   problems with your video, use the
   `HandBrake <https://handbrake.fr/>`__ and select the MPEG format.

if you want to use a web camera as an input source for the demo, please
change the value of ``USE_WEBCAM`` variable to True and specify
``cam_id`` (the default value is 0, which can be different in
multi-camera systems).

.. code:: ipython3

    USE_WEBCAM = False
    
    cam_id = 0
    video_file = "https://archive.org/serve/ISSVideoResourceLifeOnStation720p/ISS%20Video%20Resource_LifeOnStation_720p.mp4"
    
    source = cam_id if USE_WEBCAM else video_file
    additional_options = {"skip_first_frames": 600, "flip": False} if not USE_WEBCAM else {"flip": True}
    run_action_recognition(source=source, use_popup=False, **additional_options)



.. image:: 403-action-recognition-webcam-with-output_files/403-action-recognition-webcam-with-output_22_0.png


.. parsed-literal::

    Source ended

