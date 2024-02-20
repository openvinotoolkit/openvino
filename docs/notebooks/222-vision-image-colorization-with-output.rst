Image Colorization with OpenVINO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This notebook demonstrates how to colorize images with OpenVINO using
the Colorization model
`colorization-v2 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md>`__
or
`colorization-siggraph <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
based on the paper `Colorful Image
Colorization <https://arxiv.org/abs/1603.08511>`__ models from Open
Model Zoo.

.. figure:: https://user-images.githubusercontent.com/18904157/180923280-9caefaf1-742b-4d2f-8943-5d4a6126e2fc.png
   :alt: Let there be color

   Let there be color

Given a grayscale image as input, the model generates colorized version
of the image as the output.

About Colorization-v2
^^^^^^^^^^^^^^^^^^^^^

-  The colorization-v2 model is one of the colorization group of models
   designed to perform image colorization.
-  Model trained on the ImageNet dataset.
-  Model consumes L-channel of LAB-image as input and produces predict
   A- and B-channels of LAB-image as output.

About Colorization-siggraph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  The colorization-siggraph model is one of the colorization group of
   models designed to real-time user-guided image colorization.
-  Model trained on the ImageNet dataset with synthetically generated
   user interaction.
-  Model consumes L-channel of LAB-image as input and produces predict
   A- and B-channels of LAB-image as output.

See the `colorization <https://github.com/richzhang/colorization>`__
repository for more details.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Imports <#imports>`__
-  `Configurations <#configurations>`__

   -  `Select inference device <#select-inference-device>`__

-  `Download the model <#download-the-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Loading the Model <#loading-the-model>`__
-  `Utility Functions <#utility-functions>`__
-  `Load the Image <#load-the-image>`__
-  `Display Colorized Image <#display-colorized-image>`__

.. code:: ipython3

    %pip install "openvino-dev>=2023.1.0"


.. parsed-literal::

    Requirement already satisfied: openvino-dev>=2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2023.3.0)
    Requirement already satisfied: addict>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.4.0)
    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (0.7.1)
    Requirement already satisfied: jstyleson>=0.0.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (0.0.2)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.8.8)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (1.23.5)
    Requirement already satisfied: opencv-python in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (4.9.0.80)
    Requirement already satisfied: openvino-telemetry>=2022.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2023.2.1)
    Requirement already satisfied: pillow>=8.1.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (10.2.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.31.0)
    Requirement already satisfied: scipy>=1.8 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (1.10.1)
    Requirement already satisfied: texttable>=1.6.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (1.7.0)


.. parsed-literal::

    Requirement already satisfied: tqdm>=4.54.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (4.66.1)
    Requirement already satisfied: openvino==2023.3.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2023.3.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (2.2.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (2024.2.2)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import os
    import sys
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov

    sys.path.append("../utils")
    import notebook_utils as utils

Configurations
--------------



-  ``PRECISION`` - {FP16, FP32}, default: FP16.
-  ``MODEL_DIR`` - directory where the model is to be stored, default:
   public.
-  ``MODEL_NAME`` - name of the model used for inference, default:
   colorization-v2.
-  ``DATA_DIR`` - directory where test images are stored, default: data.

.. code:: ipython3

    PRECISION = "FP16"
    MODEL_DIR = "models"
    MODEL_NAME = "colorization-v2"
    # MODEL_NAME="colorization-siggraph"
    MODEL_PATH = f"{MODEL_DIR}/public/{MODEL_NAME}/{PRECISION}/{MODEL_NAME}.xml"
    DATA_DIR = "data"

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

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



Download the model
------------------



``omz_downloader`` downloads model files from online sources and, if
necessary, patches them to make them more usable with Model Converter.

In this case, ``omz_downloader`` downloads the checkpoint and pytorch
model of
`colorization-v2 <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/colorization-v2/README.md>`__
or
`colorization-siggraph <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/colorization-siggraph>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/blob/master/models/public/index.md>`__
and saves it under ``MODEL_DIR``, as specified in the configuration
above.

.. code:: ipython3

    download_command = (
        f"omz_downloader "
        f"--name {MODEL_NAME} "
        f"--output_dir {MODEL_DIR} "
        f"--cache_dir {MODEL_DIR}"
    )
    ! $download_command


.. parsed-literal::

    ################|| Downloading colorization-v2 ||################

    ========== Downloading models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth


.. parsed-literal::

    ... 0%, 32 KB, 1298 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 1111 KB/s, 0 seconds passed
... 0%, 96 KB, 1601 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1382 KB/s, 0 seconds passed
... 0%, 160 KB, 1607 KB/s, 0 seconds passed
... 0%, 192 KB, 1513 KB/s, 0 seconds passed
... 0%, 224 KB, 1759 KB/s, 0 seconds passed
... 0%, 256 KB, 2004 KB/s, 0 seconds passed
... 0%, 288 KB, 2238 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 320 KB, 1988 KB/s, 0 seconds passed
... 0%, 352 KB, 2178 KB/s, 0 seconds passed
... 0%, 384 KB, 2371 KB/s, 0 seconds passed
... 0%, 416 KB, 2552 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 448 KB, 2292 KB/s, 0 seconds passed
... 0%, 480 KB, 2445 KB/s, 0 seconds passed
... 0%, 512 KB, 2603 KB/s, 0 seconds passed
... 0%, 544 KB, 2493 KB/s, 0 seconds passed
... 0%, 576 KB, 2502 KB/s, 0 seconds passed
... 0%, 608 KB, 2637 KB/s, 0 seconds passed
... 0%, 640 KB, 2771 KB/s, 0 seconds passed
... 0%, 672 KB, 2835 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 704 KB, 2664 KB/s, 0 seconds passed
... 0%, 736 KB, 2776 KB/s, 0 seconds passed
... 0%, 768 KB, 2893 KB/s, 0 seconds passed
... 0%, 800 KB, 2947 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 832 KB, 2786 KB/s, 0 seconds passed
... 0%, 864 KB, 2884 KB/s, 0 seconds passed
... 0%, 896 KB, 2984 KB/s, 0 seconds passed
... 0%, 928 KB, 3035 KB/s, 0 seconds passed
... 0%, 960 KB, 2881 KB/s, 0 seconds passed
... 0%, 992 KB, 2969 KB/s, 0 seconds passed
... 0%, 1024 KB, 3060 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 1056 KB, 2969 KB/s, 0 seconds passed
... 0%, 1088 KB, 2957 KB/s, 0 seconds passed
... 0%, 1120 KB, 3038 KB/s, 0 seconds passed
... 0%, 1152 KB, 3120 KB/s, 0 seconds passed
... 0%, 1184 KB, 3036 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 1216 KB, 3023 KB/s, 0 seconds passed
... 0%, 1248 KB, 3095 KB/s, 0 seconds passed
... 1%, 1280 KB, 3170 KB/s, 0 seconds passed
... 1%, 1312 KB, 3092 KB/s, 0 seconds passed
... 1%, 1344 KB, 3078 KB/s, 0 seconds passed
... 1%, 1376 KB, 3145 KB/s, 0 seconds passed
... 1%, 1408 KB, 3215 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1440 KB, 3137 KB/s, 0 seconds passed
... 1%, 1472 KB, 3124 KB/s, 0 seconds passed
... 1%, 1504 KB, 3185 KB/s, 0 seconds passed
... 1%, 1536 KB, 3250 KB/s, 0 seconds passed
... 1%, 1568 KB, 3178 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1600 KB, 3165 KB/s, 0 seconds passed
... 1%, 1632 KB, 3222 KB/s, 0 seconds passed
... 1%, 1664 KB, 3283 KB/s, 0 seconds passed
... 1%, 1696 KB, 3311 KB/s, 0 seconds passed
... 1%, 1728 KB, 3202 KB/s, 0 seconds passed
... 1%, 1760 KB, 3255 KB/s, 0 seconds passed
... 1%, 1792 KB, 3310 KB/s, 0 seconds passed
... 1%, 1824 KB, 3336 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1856 KB, 3233 KB/s, 0 seconds passed
... 1%, 1888 KB, 3282 KB/s, 0 seconds passed
... 1%, 1920 KB, 3334 KB/s, 0 seconds passed
... 1%, 1952 KB, 3273 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1984 KB, 3260 KB/s, 0 seconds passed
... 1%, 2016 KB, 3304 KB/s, 0 seconds passed
... 1%, 2048 KB, 3354 KB/s, 0 seconds passed
... 1%, 2080 KB, 3295 KB/s, 0 seconds passed
... 1%, 2112 KB, 3285 KB/s, 0 seconds passed
... 1%, 2144 KB, 3326 KB/s, 0 seconds passed
... 1%, 2176 KB, 3373 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 2208 KB, 3317 KB/s, 0 seconds passed
... 1%, 2240 KB, 3305 KB/s, 0 seconds passed
... 1%, 2272 KB, 3345 KB/s, 0 seconds passed
... 1%, 2304 KB, 3389 KB/s, 0 seconds passed
... 1%, 2336 KB, 3335 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 2368 KB, 3323 KB/s, 0 seconds passed
... 1%, 2400 KB, 3361 KB/s, 0 seconds passed
... 1%, 2432 KB, 3403 KB/s, 0 seconds passed
... 1%, 2464 KB, 3306 KB/s, 0 seconds passed
... 1%, 2496 KB, 3339 KB/s, 0 seconds passed
... 2%, 2528 KB, 3375 KB/s, 0 seconds passed
... 2%, 2560 KB, 3415 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 2592 KB, 3324 KB/s, 0 seconds passed
... 2%, 2624 KB, 3355 KB/s, 0 seconds passed
... 2%, 2656 KB, 3390 KB/s, 0 seconds passed
... 2%, 2688 KB, 3428 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 2720 KB, 3341 KB/s, 0 seconds passed
... 2%, 2752 KB, 3370 KB/s, 0 seconds passed
... 2%, 2784 KB, 3403 KB/s, 0 seconds passed
... 2%, 2816 KB, 3440 KB/s, 0 seconds passed
... 2%, 2848 KB, 3355 KB/s, 0 seconds passed
... 2%, 2880 KB, 3383 KB/s, 0 seconds passed
... 2%, 2912 KB, 3414 KB/s, 0 seconds passed
... 2%, 2944 KB, 3450 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 2976 KB, 3368 KB/s, 0 seconds passed
... 2%, 3008 KB, 3396 KB/s, 0 seconds passed
... 2%, 3040 KB, 3427 KB/s, 0 seconds passed
... 2%, 3072 KB, 3460 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 3104 KB, 3382 KB/s, 0 seconds passed
... 2%, 3136 KB, 3409 KB/s, 0 seconds passed
... 2%, 3168 KB, 3439 KB/s, 0 seconds passed
... 2%, 3200 KB, 3470 KB/s, 0 seconds passed
... 2%, 3232 KB, 3431 KB/s, 0 seconds passed
... 2%, 3264 KB, 3421 KB/s, 0 seconds passed
... 2%, 3296 KB, 3449 KB/s, 0 seconds passed
... 2%, 3328 KB, 3478 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 3360 KB, 3441 KB/s, 0 seconds passed
... 2%, 3392 KB, 3430 KB/s, 0 seconds passed
... 2%, 3424 KB, 3458 KB/s, 0 seconds passed
... 2%, 3456 KB, 3487 KB/s, 0 seconds passed
... 2%, 3488 KB, 3451 KB/s, 1 seconds passed

.. parsed-literal::

    ... 2%, 3520 KB, 3442 KB/s, 1 seconds passed
... 2%, 3552 KB, 3467 KB/s, 1 seconds passed
... 2%, 3584 KB, 3493 KB/s, 1 seconds passed
... 2%, 3616 KB, 3423 KB/s, 1 seconds passed
... 2%, 3648 KB, 3447 KB/s, 1 seconds passed
... 2%, 3680 KB, 3472 KB/s, 1 seconds passed
... 2%, 3712 KB, 3500 KB/s, 1 seconds passed

.. parsed-literal::

    ... 2%, 3744 KB, 3434 KB/s, 1 seconds passed
... 2%, 3776 KB, 3457 KB/s, 1 seconds passed
... 3%, 3808 KB, 3482 KB/s, 1 seconds passed
... 3%, 3840 KB, 3507 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 3872 KB, 3442 KB/s, 1 seconds passed
... 3%, 3904 KB, 3464 KB/s, 1 seconds passed
... 3%, 3936 KB, 3488 KB/s, 1 seconds passed
... 3%, 3968 KB, 3513 KB/s, 1 seconds passed
... 3%, 4000 KB, 3449 KB/s, 1 seconds passed
... 3%, 4032 KB, 3471 KB/s, 1 seconds passed
... 3%, 4064 KB, 3496 KB/s, 1 seconds passed
... 3%, 4096 KB, 3520 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 4128 KB, 3458 KB/s, 1 seconds passed
... 3%, 4160 KB, 3480 KB/s, 1 seconds passed
... 3%, 4192 KB, 3503 KB/s, 1 seconds passed
... 3%, 4224 KB, 3525 KB/s, 1 seconds passed
... 3%, 4256 KB, 3495 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 4288 KB, 3487 KB/s, 1 seconds passed
... 3%, 4320 KB, 3509 KB/s, 1 seconds passed
... 3%, 4352 KB, 3531 KB/s, 1 seconds passed
... 3%, 4384 KB, 3501 KB/s, 1 seconds passed
... 3%, 4416 KB, 3493 KB/s, 1 seconds passed
... 3%, 4448 KB, 3515 KB/s, 1 seconds passed
... 3%, 4480 KB, 3536 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 4512 KB, 3507 KB/s, 1 seconds passed
... 3%, 4544 KB, 3498 KB/s, 1 seconds passed
... 3%, 4576 KB, 3519 KB/s, 1 seconds passed
... 3%, 4608 KB, 3540 KB/s, 1 seconds passed
... 3%, 4640 KB, 3512 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 4672 KB, 3503 KB/s, 1 seconds passed
... 3%, 4704 KB, 3523 KB/s, 1 seconds passed
... 3%, 4736 KB, 3542 KB/s, 1 seconds passed
... 3%, 4768 KB, 3488 KB/s, 1 seconds passed
... 3%, 4800 KB, 3507 KB/s, 1 seconds passed
... 3%, 4832 KB, 3527 KB/s, 1 seconds passed
... 3%, 4864 KB, 3545 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 4896 KB, 3494 KB/s, 1 seconds passed
... 3%, 4928 KB, 3511 KB/s, 1 seconds passed
... 3%, 4960 KB, 3531 KB/s, 1 seconds passed
... 3%, 4992 KB, 3506 KB/s, 1 seconds passed

.. parsed-literal::

    ... 3%, 5024 KB, 3498 KB/s, 1 seconds passed
... 4%, 5056 KB, 3515 KB/s, 1 seconds passed
... 4%, 5088 KB, 3534 KB/s, 1 seconds passed
... 4%, 5120 KB, 3510 KB/s, 1 seconds passed
... 4%, 5152 KB, 3502 KB/s, 1 seconds passed
... 4%, 5184 KB, 3519 KB/s, 1 seconds passed
... 4%, 5216 KB, 3539 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 5248 KB, 3514 KB/s, 1 seconds passed
... 4%, 5280 KB, 3507 KB/s, 1 seconds passed
... 4%, 5312 KB, 3524 KB/s, 1 seconds passed
... 4%, 5344 KB, 3543 KB/s, 1 seconds passed
... 4%, 5376 KB, 3519 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 5408 KB, 3512 KB/s, 1 seconds passed
... 4%, 5440 KB, 3528 KB/s, 1 seconds passed
... 4%, 5472 KB, 3546 KB/s, 1 seconds passed
... 4%, 5504 KB, 3523 KB/s, 1 seconds passed
... 4%, 5536 KB, 3516 KB/s, 1 seconds passed
... 4%, 5568 KB, 3532 KB/s, 1 seconds passed
... 4%, 5600 KB, 3550 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 5632 KB, 3527 KB/s, 1 seconds passed
... 4%, 5664 KB, 3520 KB/s, 1 seconds passed
... 4%, 5696 KB, 3536 KB/s, 1 seconds passed
... 4%, 5728 KB, 3553 KB/s, 1 seconds passed
... 4%, 5760 KB, 3530 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 5792 KB, 3523 KB/s, 1 seconds passed
... 4%, 5824 KB, 3539 KB/s, 1 seconds passed
... 4%, 5856 KB, 3556 KB/s, 1 seconds passed
... 4%, 5888 KB, 3532 KB/s, 1 seconds passed
... 4%, 5920 KB, 3529 KB/s, 1 seconds passed
... 4%, 5952 KB, 3543 KB/s, 1 seconds passed
... 4%, 5984 KB, 3560 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 6016 KB, 3536 KB/s, 1 seconds passed
... 4%, 6048 KB, 3533 KB/s, 1 seconds passed
... 4%, 6080 KB, 3546 KB/s, 1 seconds passed
... 4%, 6112 KB, 3563 KB/s, 1 seconds passed
... 4%, 6144 KB, 3579 KB/s, 1 seconds passed

.. parsed-literal::

    ... 4%, 6176 KB, 3537 KB/s, 1 seconds passed
... 4%, 6208 KB, 3551 KB/s, 1 seconds passed
... 4%, 6240 KB, 3566 KB/s, 1 seconds passed
... 4%, 6272 KB, 3546 KB/s, 1 seconds passed
... 5%, 6304 KB, 3540 KB/s, 1 seconds passed
... 5%, 6336 KB, 3554 KB/s, 1 seconds passed
... 5%, 6368 KB, 3568 KB/s, 1 seconds passed

.. parsed-literal::

    ... 5%, 6400 KB, 3546 KB/s, 1 seconds passed
... 5%, 6432 KB, 3541 KB/s, 1 seconds passed
... 5%, 6464 KB, 3555 KB/s, 1 seconds passed
... 5%, 6496 KB, 3570 KB/s, 1 seconds passed

.. parsed-literal::

    ... 5%, 6528 KB, 3548 KB/s, 1 seconds passed
... 5%, 6560 KB, 3544 KB/s, 1 seconds passed
... 5%, 6592 KB, 3558 KB/s, 1 seconds passed
... 5%, 6624 KB, 3573 KB/s, 1 seconds passed
... 5%, 6656 KB, 3551 KB/s, 1 seconds passed
... 5%, 6688 KB, 3547 KB/s, 1 seconds passed
... 5%, 6720 KB, 3561 KB/s, 1 seconds passed

.. parsed-literal::

    ... 5%, 6752 KB, 3575 KB/s, 1 seconds passed
... 5%, 6784 KB, 3554 KB/s, 1 seconds passed
... 5%, 6816 KB, 3552 KB/s, 1 seconds passed
... 5%, 6848 KB, 3565 KB/s, 1 seconds passed
... 5%, 6880 KB, 3578 KB/s, 1 seconds passed

.. parsed-literal::

    ... 5%, 6912 KB, 3557 KB/s, 1 seconds passed
... 5%, 6944 KB, 3554 KB/s, 1 seconds passed
... 5%, 6976 KB, 3568 KB/s, 1 seconds passed
... 5%, 7008 KB, 3580 KB/s, 1 seconds passed
... 5%, 7040 KB, 3560 KB/s, 1 seconds passed
... 5%, 7072 KB, 3559 KB/s, 1 seconds passed
... 5%, 7104 KB, 3570 KB/s, 1 seconds passed

.. parsed-literal::

    ... 5%, 7136 KB, 3584 KB/s, 1 seconds passed
... 5%, 7168 KB, 3565 KB/s, 2 seconds passed
... 5%, 7200 KB, 3561 KB/s, 2 seconds passed
... 5%, 7232 KB, 3572 KB/s, 2 seconds passed
... 5%, 7264 KB, 3586 KB/s, 2 seconds passed

.. parsed-literal::

    ... 5%, 7296 KB, 3566 KB/s, 2 seconds passed
... 5%, 7328 KB, 3561 KB/s, 2 seconds passed
... 5%, 7360 KB, 3574 KB/s, 2 seconds passed
... 5%, 7392 KB, 3586 KB/s, 2 seconds passed
... 5%, 7424 KB, 3567 KB/s, 2 seconds passed
... 5%, 7456 KB, 3563 KB/s, 2 seconds passed

.. parsed-literal::

    ... 5%, 7488 KB, 3575 KB/s, 2 seconds passed
... 5%, 7520 KB, 3588 KB/s, 2 seconds passed
... 5%, 7552 KB, 3569 KB/s, 2 seconds passed
... 6%, 7584 KB, 3567 KB/s, 2 seconds passed
... 6%, 7616 KB, 3579 KB/s, 2 seconds passed
... 6%, 7648 KB, 3591 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 7680 KB, 3572 KB/s, 2 seconds passed
... 6%, 7712 KB, 3569 KB/s, 2 seconds passed
... 6%, 7744 KB, 3581 KB/s, 2 seconds passed
... 6%, 7776 KB, 3593 KB/s, 2 seconds passed
... 6%, 7808 KB, 3574 KB/s, 2 seconds passed
... 6%, 7840 KB, 3571 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 7872 KB, 3583 KB/s, 2 seconds passed
... 6%, 7904 KB, 3595 KB/s, 2 seconds passed
... 6%, 7936 KB, 3577 KB/s, 2 seconds passed
... 6%, 7968 KB, 3575 KB/s, 2 seconds passed
... 6%, 8000 KB, 3585 KB/s, 2 seconds passed
... 6%, 8032 KB, 3597 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 8064 KB, 3578 KB/s, 2 seconds passed
... 6%, 8096 KB, 3577 KB/s, 2 seconds passed
... 6%, 8128 KB, 3587 KB/s, 2 seconds passed
... 6%, 8160 KB, 3599 KB/s, 2 seconds passed
... 6%, 8192 KB, 3568 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 8224 KB, 3575 KB/s, 2 seconds passed
... 6%, 8256 KB, 3588 KB/s, 2 seconds passed
... 6%, 8288 KB, 3599 KB/s, 2 seconds passed
... 6%, 8320 KB, 3571 KB/s, 2 seconds passed
... 6%, 8352 KB, 3577 KB/s, 2 seconds passed
... 6%, 8384 KB, 3590 KB/s, 2 seconds passed
... 6%, 8416 KB, 3601 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 8448 KB, 3583 KB/s, 2 seconds passed
... 6%, 8480 KB, 3579 KB/s, 2 seconds passed
... 6%, 8512 KB, 3591 KB/s, 2 seconds passed
... 6%, 8544 KB, 3602 KB/s, 2 seconds passed
... 6%, 8576 KB, 3575 KB/s, 2 seconds passed

.. parsed-literal::

    ... 6%, 8608 KB, 3581 KB/s, 2 seconds passed
... 6%, 8640 KB, 3593 KB/s, 2 seconds passed
... 6%, 8672 KB, 3603 KB/s, 2 seconds passed
... 6%, 8704 KB, 3576 KB/s, 2 seconds passed
... 6%, 8736 KB, 3583 KB/s, 2 seconds passed
... 6%, 8768 KB, 3594 KB/s, 2 seconds passed
... 6%, 8800 KB, 3604 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 8832 KB, 3578 KB/s, 2 seconds passed
... 7%, 8864 KB, 3584 KB/s, 2 seconds passed
... 7%, 8896 KB, 3596 KB/s, 2 seconds passed
... 7%, 8928 KB, 3606 KB/s, 2 seconds passed
... 7%, 8960 KB, 3579 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 8992 KB, 3586 KB/s, 2 seconds passed
... 7%, 9024 KB, 3597 KB/s, 2 seconds passed
... 7%, 9056 KB, 3607 KB/s, 2 seconds passed
... 7%, 9088 KB, 3581 KB/s, 2 seconds passed
... 7%, 9120 KB, 3588 KB/s, 2 seconds passed
... 7%, 9152 KB, 3599 KB/s, 2 seconds passed
... 7%, 9184 KB, 3608 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 9216 KB, 3583 KB/s, 2 seconds passed
... 7%, 9248 KB, 3589 KB/s, 2 seconds passed
... 7%, 9280 KB, 3600 KB/s, 2 seconds passed
... 7%, 9312 KB, 3609 KB/s, 2 seconds passed
... 7%, 9344 KB, 3596 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 9376 KB, 3593 KB/s, 2 seconds passed
... 7%, 9408 KB, 3602 KB/s, 2 seconds passed
... 7%, 9440 KB, 3612 KB/s, 2 seconds passed
... 7%, 9472 KB, 3597 KB/s, 2 seconds passed
... 7%, 9504 KB, 3594 KB/s, 2 seconds passed
... 7%, 9536 KB, 3603 KB/s, 2 seconds passed
... 7%, 9568 KB, 3613 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 9600 KB, 3588 KB/s, 2 seconds passed
... 7%, 9632 KB, 3595 KB/s, 2 seconds passed
... 7%, 9664 KB, 3605 KB/s, 2 seconds passed
... 7%, 9696 KB, 3614 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 9728 KB, 3590 KB/s, 2 seconds passed
... 7%, 9760 KB, 3596 KB/s, 2 seconds passed
... 7%, 9792 KB, 3606 KB/s, 2 seconds passed
... 7%, 9824 KB, 3615 KB/s, 2 seconds passed
... 7%, 9856 KB, 3591 KB/s, 2 seconds passed
... 7%, 9888 KB, 3597 KB/s, 2 seconds passed
... 7%, 9920 KB, 3607 KB/s, 2 seconds passed
... 7%, 9952 KB, 3616 KB/s, 2 seconds passed

.. parsed-literal::

    ... 7%, 9984 KB, 3592 KB/s, 2 seconds passed
... 7%, 10016 KB, 3598 KB/s, 2 seconds passed
... 7%, 10048 KB, 3608 KB/s, 2 seconds passed
... 8%, 10080 KB, 3616 KB/s, 2 seconds passed

.. parsed-literal::

    ... 8%, 10112 KB, 3593 KB/s, 2 seconds passed
... 8%, 10144 KB, 3599 KB/s, 2 seconds passed
... 8%, 10176 KB, 3609 KB/s, 2 seconds passed
... 8%, 10208 KB, 3617 KB/s, 2 seconds passed
... 8%, 10240 KB, 3596 KB/s, 2 seconds passed
... 8%, 10272 KB, 3602 KB/s, 2 seconds passed
... 8%, 10304 KB, 3611 KB/s, 2 seconds passed
... 8%, 10336 KB, 3620 KB/s, 2 seconds passed

.. parsed-literal::

    ... 8%, 10368 KB, 3598 KB/s, 2 seconds passed
... 8%, 10400 KB, 3602 KB/s, 2 seconds passed
... 8%, 10432 KB, 3612 KB/s, 2 seconds passed
... 8%, 10464 KB, 3620 KB/s, 2 seconds passed

.. parsed-literal::

    ... 8%, 10496 KB, 3598 KB/s, 2 seconds passed
... 8%, 10528 KB, 3603 KB/s, 2 seconds passed
... 8%, 10560 KB, 3614 KB/s, 2 seconds passed
... 8%, 10592 KB, 3622 KB/s, 2 seconds passed
... 8%, 10624 KB, 3600 KB/s, 2 seconds passed
... 8%, 10656 KB, 3605 KB/s, 2 seconds passed
... 8%, 10688 KB, 3615 KB/s, 2 seconds passed
... 8%, 10720 KB, 3623 KB/s, 2 seconds passed

.. parsed-literal::

    ... 8%, 10752 KB, 3601 KB/s, 2 seconds passed
... 8%, 10784 KB, 3606 KB/s, 2 seconds passed
... 8%, 10816 KB, 3616 KB/s, 2 seconds passed
... 8%, 10848 KB, 3623 KB/s, 2 seconds passed

.. parsed-literal::

    ... 8%, 10880 KB, 3602 KB/s, 3 seconds passed
... 8%, 10912 KB, 3607 KB/s, 3 seconds passed
... 8%, 10944 KB, 3617 KB/s, 3 seconds passed
... 8%, 10976 KB, 3624 KB/s, 3 seconds passed
... 8%, 11008 KB, 3603 KB/s, 3 seconds passed
... 8%, 11040 KB, 3608 KB/s, 3 seconds passed
... 8%, 11072 KB, 3617 KB/s, 3 seconds passed

.. parsed-literal::

    ... 8%, 11104 KB, 3606 KB/s, 3 seconds passed
... 8%, 11136 KB, 3603 KB/s, 3 seconds passed
... 8%, 11168 KB, 3608 KB/s, 3 seconds passed
... 8%, 11200 KB, 3618 KB/s, 3 seconds passed
... 8%, 11232 KB, 3607 KB/s, 3 seconds passed

.. parsed-literal::

    ... 8%, 11264 KB, 3605 KB/s, 3 seconds passed
... 8%, 11296 KB, 3610 KB/s, 3 seconds passed
... 8%, 11328 KB, 3619 KB/s, 3 seconds passed
... 9%, 11360 KB, 3611 KB/s, 3 seconds passed
... 9%, 11392 KB, 3605 KB/s, 3 seconds passed
... 9%, 11424 KB, 3611 KB/s, 3 seconds passed
... 9%, 11456 KB, 3620 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 11488 KB, 3608 KB/s, 3 seconds passed
... 9%, 11520 KB, 3602 KB/s, 3 seconds passed
... 9%, 11552 KB, 3611 KB/s, 3 seconds passed
... 9%, 11584 KB, 3620 KB/s, 3 seconds passed
... 9%, 11616 KB, 3609 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 11648 KB, 3604 KB/s, 3 seconds passed
... 9%, 11680 KB, 3613 KB/s, 3 seconds passed
... 9%, 11712 KB, 3621 KB/s, 3 seconds passed
... 9%, 11744 KB, 3611 KB/s, 3 seconds passed
... 9%, 11776 KB, 3605 KB/s, 3 seconds passed
... 9%, 11808 KB, 3614 KB/s, 3 seconds passed
... 9%, 11840 KB, 3622 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 11872 KB, 3612 KB/s, 3 seconds passed
... 9%, 11904 KB, 3605 KB/s, 3 seconds passed
... 9%, 11936 KB, 3615 KB/s, 3 seconds passed
... 9%, 11968 KB, 3623 KB/s, 3 seconds passed
... 9%, 12000 KB, 3612 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 12032 KB, 3607 KB/s, 3 seconds passed
... 9%, 12064 KB, 3615 KB/s, 3 seconds passed
... 9%, 12096 KB, 3624 KB/s, 3 seconds passed
... 9%, 12128 KB, 3613 KB/s, 3 seconds passed
... 9%, 12160 KB, 3607 KB/s, 3 seconds passed
... 9%, 12192 KB, 3616 KB/s, 3 seconds passed
... 9%, 12224 KB, 3624 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 12256 KB, 3614 KB/s, 3 seconds passed
... 9%, 12288 KB, 3608 KB/s, 3 seconds passed
... 9%, 12320 KB, 3617 KB/s, 3 seconds passed
... 9%, 12352 KB, 3625 KB/s, 3 seconds passed
... 9%, 12384 KB, 3615 KB/s, 3 seconds passed

.. parsed-literal::

    ... 9%, 12416 KB, 3609 KB/s, 3 seconds passed
... 9%, 12448 KB, 3618 KB/s, 3 seconds passed
... 9%, 12480 KB, 3625 KB/s, 3 seconds passed
... 9%, 12512 KB, 3615 KB/s, 3 seconds passed
... 9%, 12544 KB, 3610 KB/s, 3 seconds passed
... 9%, 12576 KB, 3619 KB/s, 3 seconds passed
... 10%, 12608 KB, 3626 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 12640 KB, 3616 KB/s, 3 seconds passed
... 10%, 12672 KB, 3611 KB/s, 3 seconds passed
... 10%, 12704 KB, 3619 KB/s, 3 seconds passed
... 10%, 12736 KB, 3628 KB/s, 3 seconds passed
... 10%, 12768 KB, 3619 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 12800 KB, 3613 KB/s, 3 seconds passed
... 10%, 12832 KB, 3620 KB/s, 3 seconds passed
... 10%, 12864 KB, 3629 KB/s, 3 seconds passed
... 10%, 12896 KB, 3618 KB/s, 3 seconds passed
... 10%, 12928 KB, 3614 KB/s, 3 seconds passed
... 10%, 12960 KB, 3621 KB/s, 3 seconds passed
... 10%, 12992 KB, 3629 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 13024 KB, 3619 KB/s, 3 seconds passed
... 10%, 13056 KB, 3615 KB/s, 3 seconds passed
... 10%, 13088 KB, 3622 KB/s, 3 seconds passed
... 10%, 13120 KB, 3630 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 13152 KB, 3620 KB/s, 3 seconds passed
... 10%, 13184 KB, 3616 KB/s, 3 seconds passed
... 10%, 13216 KB, 3622 KB/s, 3 seconds passed
... 10%, 13248 KB, 3630 KB/s, 3 seconds passed
... 10%, 13280 KB, 3613 KB/s, 3 seconds passed
... 10%, 13312 KB, 3615 KB/s, 3 seconds passed
... 10%, 13344 KB, 3623 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 13376 KB, 3630 KB/s, 3 seconds passed
... 10%, 13408 KB, 3614 KB/s, 3 seconds passed
... 10%, 13440 KB, 3616 KB/s, 3 seconds passed
... 10%, 13472 KB, 3624 KB/s, 3 seconds passed
... 10%, 13504 KB, 3632 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 13536 KB, 3615 KB/s, 3 seconds passed
... 10%, 13568 KB, 3617 KB/s, 3 seconds passed
... 10%, 13600 KB, 3625 KB/s, 3 seconds passed
... 10%, 13632 KB, 3632 KB/s, 3 seconds passed
... 10%, 13664 KB, 3616 KB/s, 3 seconds passed
... 10%, 13696 KB, 3618 KB/s, 3 seconds passed
... 10%, 13728 KB, 3625 KB/s, 3 seconds passed

.. parsed-literal::

    ... 10%, 13760 KB, 3633 KB/s, 3 seconds passed
... 10%, 13792 KB, 3617 KB/s, 3 seconds passed
... 10%, 13824 KB, 3619 KB/s, 3 seconds passed
... 11%, 13856 KB, 3626 KB/s, 3 seconds passed
... 11%, 13888 KB, 3633 KB/s, 3 seconds passed

.. parsed-literal::

    ... 11%, 13920 KB, 3617 KB/s, 3 seconds passed
... 11%, 13952 KB, 3620 KB/s, 3 seconds passed
... 11%, 13984 KB, 3627 KB/s, 3 seconds passed
... 11%, 14016 KB, 3635 KB/s, 3 seconds passed
... 11%, 14048 KB, 3618 KB/s, 3 seconds passed
... 11%, 14080 KB, 3621 KB/s, 3 seconds passed

.. parsed-literal::

    ... 11%, 14112 KB, 3628 KB/s, 3 seconds passed
... 11%, 14144 KB, 3635 KB/s, 3 seconds passed
... 11%, 14176 KB, 3619 KB/s, 3 seconds passed
... 11%, 14208 KB, 3621 KB/s, 3 seconds passed
... 11%, 14240 KB, 3628 KB/s, 3 seconds passed
... 11%, 14272 KB, 3636 KB/s, 3 seconds passed

.. parsed-literal::

    ... 11%, 14304 KB, 3619 KB/s, 3 seconds passed
... 11%, 14336 KB, 3622 KB/s, 3 seconds passed
... 11%, 14368 KB, 3629 KB/s, 3 seconds passed
... 11%, 14400 KB, 3636 KB/s, 3 seconds passed
... 11%, 14432 KB, 3620 KB/s, 3 seconds passed
... 11%, 14464 KB, 3623 KB/s, 3 seconds passed

.. parsed-literal::

    ... 11%, 14496 KB, 3630 KB/s, 3 seconds passed
... 11%, 14528 KB, 3637 KB/s, 3 seconds passed
... 11%, 14560 KB, 3629 KB/s, 4 seconds passed
... 11%, 14592 KB, 3624 KB/s, 4 seconds passed
... 11%, 14624 KB, 3630 KB/s, 4 seconds passed
... 11%, 14656 KB, 3637 KB/s, 4 seconds passed

.. parsed-literal::

    ... 11%, 14688 KB, 3621 KB/s, 4 seconds passed
... 11%, 14720 KB, 3624 KB/s, 4 seconds passed
... 11%, 14752 KB, 3630 KB/s, 4 seconds passed
... 11%, 14784 KB, 3638 KB/s, 4 seconds passed
... 11%, 14816 KB, 3622 KB/s, 4 seconds passed

.. parsed-literal::

    ... 11%, 14848 KB, 3624 KB/s, 4 seconds passed
... 11%, 14880 KB, 3631 KB/s, 4 seconds passed
... 11%, 14912 KB, 3638 KB/s, 4 seconds passed
... 11%, 14944 KB, 3623 KB/s, 4 seconds passed
... 11%, 14976 KB, 3625 KB/s, 4 seconds passed
... 11%, 15008 KB, 3632 KB/s, 4 seconds passed
... 11%, 15040 KB, 3639 KB/s, 4 seconds passed

.. parsed-literal::

    ... 11%, 15072 KB, 3623 KB/s, 4 seconds passed
... 11%, 15104 KB, 3625 KB/s, 4 seconds passed
... 12%, 15136 KB, 3633 KB/s, 4 seconds passed
... 12%, 15168 KB, 3639 KB/s, 4 seconds passed
... 12%, 15200 KB, 3624 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 15232 KB, 3626 KB/s, 4 seconds passed
... 12%, 15264 KB, 3633 KB/s, 4 seconds passed
... 12%, 15296 KB, 3639 KB/s, 4 seconds passed
... 12%, 15328 KB, 3624 KB/s, 4 seconds passed
... 12%, 15360 KB, 3627 KB/s, 4 seconds passed
... 12%, 15392 KB, 3634 KB/s, 4 seconds passed
... 12%, 15424 KB, 3641 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 15456 KB, 3626 KB/s, 4 seconds passed
... 12%, 15488 KB, 3629 KB/s, 4 seconds passed
... 12%, 15520 KB, 3634 KB/s, 4 seconds passed
... 12%, 15552 KB, 3641 KB/s, 4 seconds passed
... 12%, 15584 KB, 3625 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 15616 KB, 3628 KB/s, 4 seconds passed
... 12%, 15648 KB, 3634 KB/s, 4 seconds passed
... 12%, 15680 KB, 3630 KB/s, 4 seconds passed
... 12%, 15712 KB, 3626 KB/s, 4 seconds passed
... 12%, 15744 KB, 3628 KB/s, 4 seconds passed
... 12%, 15776 KB, 3635 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 15808 KB, 3631 KB/s, 4 seconds passed
... 12%, 15840 KB, 3627 KB/s, 4 seconds passed
... 12%, 15872 KB, 3629 KB/s, 4 seconds passed
... 12%, 15904 KB, 3636 KB/s, 4 seconds passed
... 12%, 15936 KB, 3634 KB/s, 4 seconds passed
... 12%, 15968 KB, 3628 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 16000 KB, 3630 KB/s, 4 seconds passed
... 12%, 16032 KB, 3636 KB/s, 4 seconds passed
... 12%, 16064 KB, 3632 KB/s, 4 seconds passed
... 12%, 16096 KB, 3628 KB/s, 4 seconds passed
... 12%, 16128 KB, 3630 KB/s, 4 seconds passed
... 12%, 16160 KB, 3637 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 16192 KB, 3632 KB/s, 4 seconds passed
... 12%, 16224 KB, 3629 KB/s, 4 seconds passed
... 12%, 16256 KB, 3631 KB/s, 4 seconds passed
... 12%, 16288 KB, 3637 KB/s, 4 seconds passed
... 12%, 16320 KB, 3633 KB/s, 4 seconds passed

.. parsed-literal::

    ... 12%, 16352 KB, 3629 KB/s, 4 seconds passed
... 13%, 16384 KB, 3631 KB/s, 4 seconds passed
... 13%, 16416 KB, 3637 KB/s, 4 seconds passed
... 13%, 16448 KB, 3633 KB/s, 4 seconds passed
... 13%, 16480 KB, 3630 KB/s, 4 seconds passed
... 13%, 16512 KB, 3632 KB/s, 4 seconds passed
... 13%, 16544 KB, 3638 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 16576 KB, 3633 KB/s, 4 seconds passed
... 13%, 16608 KB, 3630 KB/s, 4 seconds passed
... 13%, 16640 KB, 3632 KB/s, 4 seconds passed
... 13%, 16672 KB, 3638 KB/s, 4 seconds passed
... 13%, 16704 KB, 3635 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 16736 KB, 3631 KB/s, 4 seconds passed
... 13%, 16768 KB, 3633 KB/s, 4 seconds passed
... 13%, 16800 KB, 3639 KB/s, 4 seconds passed
... 13%, 16832 KB, 3635 KB/s, 4 seconds passed
... 13%, 16864 KB, 3631 KB/s, 4 seconds passed
... 13%, 16896 KB, 3633 KB/s, 4 seconds passed
... 13%, 16928 KB, 3639 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 16960 KB, 3636 KB/s, 4 seconds passed
... 13%, 16992 KB, 3632 KB/s, 4 seconds passed
... 13%, 17024 KB, 3635 KB/s, 4 seconds passed
... 13%, 17056 KB, 3640 KB/s, 4 seconds passed
... 13%, 17088 KB, 3637 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 17120 KB, 3633 KB/s, 4 seconds passed
... 13%, 17152 KB, 3634 KB/s, 4 seconds passed
... 13%, 17184 KB, 3640 KB/s, 4 seconds passed
... 13%, 17216 KB, 3636 KB/s, 4 seconds passed
... 13%, 17248 KB, 3632 KB/s, 4 seconds passed
... 13%, 17280 KB, 3635 KB/s, 4 seconds passed
... 13%, 17312 KB, 3641 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 17344 KB, 3629 KB/s, 4 seconds passed
... 13%, 17376 KB, 3633 KB/s, 4 seconds passed
... 13%, 17408 KB, 3635 KB/s, 4 seconds passed
... 13%, 17440 KB, 3641 KB/s, 4 seconds passed
... 13%, 17472 KB, 3637 KB/s, 4 seconds passed

.. parsed-literal::

    ... 13%, 17504 KB, 3634 KB/s, 4 seconds passed
... 13%, 17536 KB, 3635 KB/s, 4 seconds passed
... 13%, 17568 KB, 3642 KB/s, 4 seconds passed
... 13%, 17600 KB, 3637 KB/s, 4 seconds passed
... 13%, 17632 KB, 3634 KB/s, 4 seconds passed
... 14%, 17664 KB, 3637 KB/s, 4 seconds passed
... 14%, 17696 KB, 3642 KB/s, 4 seconds passed

.. parsed-literal::

    ... 14%, 17728 KB, 3638 KB/s, 4 seconds passed
... 14%, 17760 KB, 3635 KB/s, 4 seconds passed
... 14%, 17792 KB, 3636 KB/s, 4 seconds passed
... 14%, 17824 KB, 3642 KB/s, 4 seconds passed

.. parsed-literal::

    ... 14%, 17856 KB, 3631 KB/s, 4 seconds passed
... 14%, 17888 KB, 3635 KB/s, 4 seconds passed
... 14%, 17920 KB, 3637 KB/s, 4 seconds passed
... 14%, 17952 KB, 3643 KB/s, 4 seconds passed
... 14%, 17984 KB, 3632 KB/s, 4 seconds passed
... 14%, 18016 KB, 3635 KB/s, 4 seconds passed
... 14%, 18048 KB, 3637 KB/s, 4 seconds passed
... 14%, 18080 KB, 3643 KB/s, 4 seconds passed

.. parsed-literal::

    ... 14%, 18112 KB, 3632 KB/s, 4 seconds passed
... 14%, 18144 KB, 3636 KB/s, 4 seconds passed
... 14%, 18176 KB, 3637 KB/s, 4 seconds passed
... 14%, 18208 KB, 3643 KB/s, 4 seconds passed

.. parsed-literal::

    ... 14%, 18240 KB, 3633 KB/s, 5 seconds passed
... 14%, 18272 KB, 3636 KB/s, 5 seconds passed
... 14%, 18304 KB, 3638 KB/s, 5 seconds passed
... 14%, 18336 KB, 3644 KB/s, 5 seconds passed
... 14%, 18368 KB, 3633 KB/s, 5 seconds passed
... 14%, 18400 KB, 3637 KB/s, 5 seconds passed
... 14%, 18432 KB, 3638 KB/s, 5 seconds passed
... 14%, 18464 KB, 3644 KB/s, 5 seconds passed

.. parsed-literal::

    ... 14%, 18496 KB, 3633 KB/s, 5 seconds passed
... 14%, 18528 KB, 3637 KB/s, 5 seconds passed
... 14%, 18560 KB, 3638 KB/s, 5 seconds passed
... 14%, 18592 KB, 3644 KB/s, 5 seconds passed
... 14%, 18624 KB, 3642 KB/s, 5 seconds passed

.. parsed-literal::

    ... 14%, 18656 KB, 3639 KB/s, 5 seconds passed
... 14%, 18688 KB, 3641 KB/s, 5 seconds passed
... 14%, 18720 KB, 3645 KB/s, 5 seconds passed
... 14%, 18752 KB, 3642 KB/s, 5 seconds passed
... 14%, 18784 KB, 3639 KB/s, 5 seconds passed
... 14%, 18816 KB, 3640 KB/s, 5 seconds passed
... 14%, 18848 KB, 3645 KB/s, 5 seconds passed

.. parsed-literal::

    ... 14%, 18880 KB, 3635 KB/s, 5 seconds passed
... 15%, 18912 KB, 3639 KB/s, 5 seconds passed
... 15%, 18944 KB, 3640 KB/s, 5 seconds passed
... 15%, 18976 KB, 3645 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19008 KB, 3635 KB/s, 5 seconds passed
... 15%, 19040 KB, 3640 KB/s, 5 seconds passed
... 15%, 19072 KB, 3641 KB/s, 5 seconds passed
... 15%, 19104 KB, 3646 KB/s, 5 seconds passed
... 15%, 19136 KB, 3637 KB/s, 5 seconds passed
... 15%, 19168 KB, 3640 KB/s, 5 seconds passed
... 15%, 19200 KB, 3640 KB/s, 5 seconds passed
... 15%, 19232 KB, 3646 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19264 KB, 3636 KB/s, 5 seconds passed
... 15%, 19296 KB, 3640 KB/s, 5 seconds passed
... 15%, 19328 KB, 3641 KB/s, 5 seconds passed
... 15%, 19360 KB, 3646 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19392 KB, 3637 KB/s, 5 seconds passed
... 15%, 19424 KB, 3641 KB/s, 5 seconds passed
... 15%, 19456 KB, 3641 KB/s, 5 seconds passed
... 15%, 19488 KB, 3646 KB/s, 5 seconds passed
... 15%, 19520 KB, 3637 KB/s, 5 seconds passed
... 15%, 19552 KB, 3640 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19584 KB, 3641 KB/s, 5 seconds passed
... 15%, 19616 KB, 3641 KB/s, 5 seconds passed
... 15%, 19648 KB, 3637 KB/s, 5 seconds passed
... 15%, 19680 KB, 3640 KB/s, 5 seconds passed
... 15%, 19712 KB, 3641 KB/s, 5 seconds passed
... 15%, 19744 KB, 3647 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19776 KB, 3638 KB/s, 5 seconds passed
... 15%, 19808 KB, 3641 KB/s, 5 seconds passed
... 15%, 19840 KB, 3642 KB/s, 5 seconds passed
... 15%, 19872 KB, 3647 KB/s, 5 seconds passed
... 15%, 19904 KB, 3639 KB/s, 5 seconds passed
... 15%, 19936 KB, 3641 KB/s, 5 seconds passed

.. parsed-literal::

    ... 15%, 19968 KB, 3642 KB/s, 5 seconds passed
... 15%, 20000 KB, 3648 KB/s, 5 seconds passed
... 15%, 20032 KB, 3639 KB/s, 5 seconds passed
... 15%, 20064 KB, 3642 KB/s, 5 seconds passed
... 15%, 20096 KB, 3643 KB/s, 5 seconds passed
... 15%, 20128 KB, 3648 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 20160 KB, 3640 KB/s, 5 seconds passed
... 16%, 20192 KB, 3642 KB/s, 5 seconds passed
... 16%, 20224 KB, 3643 KB/s, 5 seconds passed
... 16%, 20256 KB, 3649 KB/s, 5 seconds passed
... 16%, 20288 KB, 3640 KB/s, 5 seconds passed
... 16%, 20320 KB, 3643 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 20352 KB, 3644 KB/s, 5 seconds passed
... 16%, 20384 KB, 3648 KB/s, 5 seconds passed
... 16%, 20416 KB, 3640 KB/s, 5 seconds passed
... 16%, 20448 KB, 3643 KB/s, 5 seconds passed
... 16%, 20480 KB, 3643 KB/s, 5 seconds passed
... 16%, 20512 KB, 3644 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 20544 KB, 3641 KB/s, 5 seconds passed
... 16%, 20576 KB, 3643 KB/s, 5 seconds passed
... 16%, 20608 KB, 3643 KB/s, 5 seconds passed
... 16%, 20640 KB, 3645 KB/s, 5 seconds passed
... 16%, 20672 KB, 3641 KB/s, 5 seconds passed
... 16%, 20704 KB, 3643 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 20736 KB, 3644 KB/s, 5 seconds passed
... 16%, 20768 KB, 3645 KB/s, 5 seconds passed
... 16%, 20800 KB, 3641 KB/s, 5 seconds passed
... 16%, 20832 KB, 3643 KB/s, 5 seconds passed
... 16%, 20864 KB, 3644 KB/s, 5 seconds passed
... 16%, 20896 KB, 3645 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 20928 KB, 3641 KB/s, 5 seconds passed
... 16%, 20960 KB, 3644 KB/s, 5 seconds passed
... 16%, 20992 KB, 3644 KB/s, 5 seconds passed
... 16%, 21024 KB, 3645 KB/s, 5 seconds passed
... 16%, 21056 KB, 3642 KB/s, 5 seconds passed
... 16%, 21088 KB, 3644 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 21120 KB, 3644 KB/s, 5 seconds passed
... 16%, 21152 KB, 3645 KB/s, 5 seconds passed
... 16%, 21184 KB, 3642 KB/s, 5 seconds passed
... 16%, 21216 KB, 3645 KB/s, 5 seconds passed
... 16%, 21248 KB, 3645 KB/s, 5 seconds passed
... 16%, 21280 KB, 3646 KB/s, 5 seconds passed

.. parsed-literal::

    ... 16%, 21312 KB, 3642 KB/s, 5 seconds passed
... 16%, 21344 KB, 3645 KB/s, 5 seconds passed
... 16%, 21376 KB, 3645 KB/s, 5 seconds passed
... 16%, 21408 KB, 3646 KB/s, 5 seconds passed
... 17%, 21440 KB, 3643 KB/s, 5 seconds passed
... 17%, 21472 KB, 3646 KB/s, 5 seconds passed

.. parsed-literal::

    ... 17%, 21504 KB, 3645 KB/s, 5 seconds passed
... 17%, 21536 KB, 3647 KB/s, 5 seconds passed
... 17%, 21568 KB, 3643 KB/s, 5 seconds passed
... 17%, 21600 KB, 3646 KB/s, 5 seconds passed
... 17%, 21632 KB, 3647 KB/s, 5 seconds passed
... 17%, 21664 KB, 3649 KB/s, 5 seconds passed

.. parsed-literal::

    ... 17%, 21696 KB, 3644 KB/s, 5 seconds passed
... 17%, 21728 KB, 3647 KB/s, 5 seconds passed
... 17%, 21760 KB, 3647 KB/s, 5 seconds passed
... 17%, 21792 KB, 3649 KB/s, 5 seconds passed
... 17%, 21824 KB, 3645 KB/s, 5 seconds passed
... 17%, 21856 KB, 3648 KB/s, 5 seconds passed

.. parsed-literal::

    ... 17%, 21888 KB, 3648 KB/s, 5 seconds passed
... 17%, 21920 KB, 3641 KB/s, 6 seconds passed
... 17%, 21952 KB, 3644 KB/s, 6 seconds passed
... 17%, 21984 KB, 3647 KB/s, 6 seconds passed
... 17%, 22016 KB, 3647 KB/s, 6 seconds passed

.. parsed-literal::

    ... 17%, 22048 KB, 3642 KB/s, 6 seconds passed
... 17%, 22080 KB, 3644 KB/s, 6 seconds passed
... 17%, 22112 KB, 3647 KB/s, 6 seconds passed
... 17%, 22144 KB, 3647 KB/s, 6 seconds passed
... 17%, 22176 KB, 3642 KB/s, 6 seconds passed
... 17%, 22208 KB, 3645 KB/s, 6 seconds passed

.. parsed-literal::

    ... 17%, 22240 KB, 3647 KB/s, 6 seconds passed
... 17%, 22272 KB, 3647 KB/s, 6 seconds passed
... 17%, 22304 KB, 3643 KB/s, 6 seconds passed
... 17%, 22336 KB, 3645 KB/s, 6 seconds passed
... 17%, 22368 KB, 3647 KB/s, 6 seconds passed
... 17%, 22400 KB, 3648 KB/s, 6 seconds passed

.. parsed-literal::

    ... 17%, 22432 KB, 3649 KB/s, 6 seconds passed
... 17%, 22464 KB, 3646 KB/s, 6 seconds passed
... 17%, 22496 KB, 3648 KB/s, 6 seconds passed
... 17%, 22528 KB, 3648 KB/s, 6 seconds passed
... 17%, 22560 KB, 3643 KB/s, 6 seconds passed
... 17%, 22592 KB, 3646 KB/s, 6 seconds passed

.. parsed-literal::

    ... 17%, 22624 KB, 3649 KB/s, 6 seconds passed
... 17%, 22656 KB, 3648 KB/s, 6 seconds passed
... 18%, 22688 KB, 3650 KB/s, 6 seconds passed
... 18%, 22720 KB, 3646 KB/s, 6 seconds passed
... 18%, 22752 KB, 3649 KB/s, 6 seconds passed
... 18%, 22784 KB, 3649 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 22816 KB, 3650 KB/s, 6 seconds passed
... 18%, 22848 KB, 3644 KB/s, 6 seconds passed
... 18%, 22880 KB, 3648 KB/s, 6 seconds passed
... 18%, 22912 KB, 3649 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 22944 KB, 3636 KB/s, 6 seconds passed
... 18%, 22976 KB, 3641 KB/s, 6 seconds passed
... 18%, 23008 KB, 3645 KB/s, 6 seconds passed
... 18%, 23040 KB, 3649 KB/s, 6 seconds passed
... 18%, 23072 KB, 3636 KB/s, 6 seconds passed
... 18%, 23104 KB, 3641 KB/s, 6 seconds passed
... 18%, 23136 KB, 3646 KB/s, 6 seconds passed
... 18%, 23168 KB, 3650 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 23200 KB, 3637 KB/s, 6 seconds passed
... 18%, 23232 KB, 3641 KB/s, 6 seconds passed
... 18%, 23264 KB, 3646 KB/s, 6 seconds passed
... 18%, 23296 KB, 3650 KB/s, 6 seconds passed
... 18%, 23328 KB, 3652 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 23360 KB, 3642 KB/s, 6 seconds passed
... 18%, 23392 KB, 3646 KB/s, 6 seconds passed
... 18%, 23424 KB, 3651 KB/s, 6 seconds passed
... 18%, 23456 KB, 3652 KB/s, 6 seconds passed
... 18%, 23488 KB, 3642 KB/s, 6 seconds passed
... 18%, 23520 KB, 3647 KB/s, 6 seconds passed
... 18%, 23552 KB, 3651 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 23584 KB, 3653 KB/s, 6 seconds passed
... 18%, 23616 KB, 3642 KB/s, 6 seconds passed
... 18%, 23648 KB, 3647 KB/s, 6 seconds passed
... 18%, 23680 KB, 3651 KB/s, 6 seconds passed

.. parsed-literal::

    ... 18%, 23712 KB, 3638 KB/s, 6 seconds passed
... 18%, 23744 KB, 3643 KB/s, 6 seconds passed
... 18%, 23776 KB, 3647 KB/s, 6 seconds passed
... 18%, 23808 KB, 3651 KB/s, 6 seconds passed
... 18%, 23840 KB, 3638 KB/s, 6 seconds passed
... 18%, 23872 KB, 3643 KB/s, 6 seconds passed
... 18%, 23904 KB, 3647 KB/s, 6 seconds passed
... 19%, 23936 KB, 3651 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 23968 KB, 3639 KB/s, 6 seconds passed
... 19%, 24000 KB, 3643 KB/s, 6 seconds passed
... 19%, 24032 KB, 3648 KB/s, 6 seconds passed
... 19%, 24064 KB, 3652 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 24096 KB, 3639 KB/s, 6 seconds passed
... 19%, 24128 KB, 3644 KB/s, 6 seconds passed
... 19%, 24160 KB, 3648 KB/s, 6 seconds passed
... 19%, 24192 KB, 3652 KB/s, 6 seconds passed
... 19%, 24224 KB, 3639 KB/s, 6 seconds passed
... 19%, 24256 KB, 3644 KB/s, 6 seconds passed
... 19%, 24288 KB, 3648 KB/s, 6 seconds passed
... 19%, 24320 KB, 3652 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 24352 KB, 3639 KB/s, 6 seconds passed
... 19%, 24384 KB, 3644 KB/s, 6 seconds passed
... 19%, 24416 KB, 3648 KB/s, 6 seconds passed
... 19%, 24448 KB, 3652 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 24480 KB, 3639 KB/s, 6 seconds passed
... 19%, 24512 KB, 3644 KB/s, 6 seconds passed
... 19%, 24544 KB, 3648 KB/s, 6 seconds passed
... 19%, 24576 KB, 3652 KB/s, 6 seconds passed
... 19%, 24608 KB, 3640 KB/s, 6 seconds passed
... 19%, 24640 KB, 3644 KB/s, 6 seconds passed
... 19%, 24672 KB, 3649 KB/s, 6 seconds passed
... 19%, 24704 KB, 3653 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 24736 KB, 3642 KB/s, 6 seconds passed
... 19%, 24768 KB, 3644 KB/s, 6 seconds passed
... 19%, 24800 KB, 3649 KB/s, 6 seconds passed
... 19%, 24832 KB, 3653 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 24864 KB, 3642 KB/s, 6 seconds passed
... 19%, 24896 KB, 3645 KB/s, 6 seconds passed
... 19%, 24928 KB, 3649 KB/s, 6 seconds passed
... 19%, 24960 KB, 3654 KB/s, 6 seconds passed
... 19%, 24992 KB, 3642 KB/s, 6 seconds passed
... 19%, 25024 KB, 3645 KB/s, 6 seconds passed

.. parsed-literal::

    ... 19%, 25056 KB, 3649 KB/s, 6 seconds passed
... 19%, 25088 KB, 3654 KB/s, 6 seconds passed
... 19%, 25120 KB, 3641 KB/s, 6 seconds passed
... 19%, 25152 KB, 3645 KB/s, 6 seconds passed
... 19%, 25184 KB, 3649 KB/s, 6 seconds passed
... 20%, 25216 KB, 3654 KB/s, 6 seconds passed

.. parsed-literal::

    ... 20%, 25248 KB, 3641 KB/s, 6 seconds passed
... 20%, 25280 KB, 3645 KB/s, 6 seconds passed
... 20%, 25312 KB, 3650 KB/s, 6 seconds passed
... 20%, 25344 KB, 3654 KB/s, 6 seconds passed
... 20%, 25376 KB, 3642 KB/s, 6 seconds passed

.. parsed-literal::

    ... 20%, 25408 KB, 3646 KB/s, 6 seconds passed
... 20%, 25440 KB, 3650 KB/s, 6 seconds passed
... 20%, 25472 KB, 3654 KB/s, 6 seconds passed
... 20%, 25504 KB, 3642 KB/s, 7 seconds passed
... 20%, 25536 KB, 3646 KB/s, 7 seconds passed
... 20%, 25568 KB, 3650 KB/s, 7 seconds passed
... 20%, 25600 KB, 3654 KB/s, 7 seconds passed

.. parsed-literal::

    ... 20%, 25632 KB, 3642 KB/s, 7 seconds passed
... 20%, 25664 KB, 3646 KB/s, 7 seconds passed
... 20%, 25696 KB, 3650 KB/s, 7 seconds passed
... 20%, 25728 KB, 3655 KB/s, 7 seconds passed

.. parsed-literal::

    ... 20%, 25760 KB, 3642 KB/s, 7 seconds passed
... 20%, 25792 KB, 3646 KB/s, 7 seconds passed
... 20%, 25824 KB, 3651 KB/s, 7 seconds passed
... 20%, 25856 KB, 3655 KB/s, 7 seconds passed
... 20%, 25888 KB, 3644 KB/s, 7 seconds passed
... 20%, 25920 KB, 3647 KB/s, 7 seconds passed
... 20%, 25952 KB, 3651 KB/s, 7 seconds passed
... 20%, 25984 KB, 3655 KB/s, 7 seconds passed

.. parsed-literal::

    ... 20%, 26016 KB, 3643 KB/s, 7 seconds passed
... 20%, 26048 KB, 3647 KB/s, 7 seconds passed
... 20%, 26080 KB, 3651 KB/s, 7 seconds passed
... 20%, 26112 KB, 3655 KB/s, 7 seconds passed

.. parsed-literal::

    ... 20%, 26144 KB, 3643 KB/s, 7 seconds passed
... 20%, 26176 KB, 3647 KB/s, 7 seconds passed
... 20%, 26208 KB, 3651 KB/s, 7 seconds passed
... 20%, 26240 KB, 3655 KB/s, 7 seconds passed
... 20%, 26272 KB, 3644 KB/s, 7 seconds passed
... 20%, 26304 KB, 3647 KB/s, 7 seconds passed
... 20%, 26336 KB, 3652 KB/s, 7 seconds passed
... 20%, 26368 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 20%, 26400 KB, 3644 KB/s, 7 seconds passed
... 20%, 26432 KB, 3648 KB/s, 7 seconds passed
... 21%, 26464 KB, 3652 KB/s, 7 seconds passed
... 21%, 26496 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 26528 KB, 3644 KB/s, 7 seconds passed
... 21%, 26560 KB, 3648 KB/s, 7 seconds passed
... 21%, 26592 KB, 3652 KB/s, 7 seconds passed
... 21%, 26624 KB, 3656 KB/s, 7 seconds passed
... 21%, 26656 KB, 3645 KB/s, 7 seconds passed
... 21%, 26688 KB, 3648 KB/s, 7 seconds passed
... 21%, 26720 KB, 3652 KB/s, 7 seconds passed
... 21%, 26752 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 26784 KB, 3645 KB/s, 7 seconds passed
... 21%, 26816 KB, 3648 KB/s, 7 seconds passed
... 21%, 26848 KB, 3653 KB/s, 7 seconds passed
... 21%, 26880 KB, 3657 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 26912 KB, 3645 KB/s, 7 seconds passed
... 21%, 26944 KB, 3649 KB/s, 7 seconds passed
... 21%, 26976 KB, 3653 KB/s, 7 seconds passed
... 21%, 27008 KB, 3657 KB/s, 7 seconds passed
... 21%, 27040 KB, 3645 KB/s, 7 seconds passed
... 21%, 27072 KB, 3649 KB/s, 7 seconds passed
... 21%, 27104 KB, 3653 KB/s, 7 seconds passed
... 21%, 27136 KB, 3657 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 27168 KB, 3645 KB/s, 7 seconds passed
... 21%, 27200 KB, 3649 KB/s, 7 seconds passed
... 21%, 27232 KB, 3653 KB/s, 7 seconds passed
... 21%, 27264 KB, 3657 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 27296 KB, 3645 KB/s, 7 seconds passed
... 21%, 27328 KB, 3649 KB/s, 7 seconds passed
... 21%, 27360 KB, 3653 KB/s, 7 seconds passed
... 21%, 27392 KB, 3658 KB/s, 7 seconds passed
... 21%, 27424 KB, 3647 KB/s, 7 seconds passed
... 21%, 27456 KB, 3650 KB/s, 7 seconds passed
... 21%, 27488 KB, 3654 KB/s, 7 seconds passed
... 21%, 27520 KB, 3658 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 27552 KB, 3647 KB/s, 7 seconds passed
... 21%, 27584 KB, 3650 KB/s, 7 seconds passed
... 21%, 27616 KB, 3654 KB/s, 7 seconds passed
... 21%, 27648 KB, 3645 KB/s, 7 seconds passed

.. parsed-literal::

    ... 21%, 27680 KB, 3646 KB/s, 7 seconds passed
... 22%, 27712 KB, 3650 KB/s, 7 seconds passed
... 22%, 27744 KB, 3654 KB/s, 7 seconds passed
... 22%, 27776 KB, 3645 KB/s, 7 seconds passed
... 22%, 27808 KB, 3646 KB/s, 7 seconds passed
... 22%, 27840 KB, 3650 KB/s, 7 seconds passed
... 22%, 27872 KB, 3654 KB/s, 7 seconds passed
... 22%, 27904 KB, 3658 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 27936 KB, 3647 KB/s, 7 seconds passed
... 22%, 27968 KB, 3651 KB/s, 7 seconds passed
... 22%, 28000 KB, 3654 KB/s, 7 seconds passed
... 22%, 28032 KB, 3658 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 28064 KB, 3647 KB/s, 7 seconds passed
... 22%, 28096 KB, 3651 KB/s, 7 seconds passed
... 22%, 28128 KB, 3655 KB/s, 7 seconds passed
... 22%, 28160 KB, 3646 KB/s, 7 seconds passed
... 22%, 28192 KB, 3647 KB/s, 7 seconds passed
... 22%, 28224 KB, 3651 KB/s, 7 seconds passed
... 22%, 28256 KB, 3655 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 28288 KB, 3647 KB/s, 7 seconds passed
... 22%, 28320 KB, 3648 KB/s, 7 seconds passed
... 22%, 28352 KB, 3651 KB/s, 7 seconds passed
... 22%, 28384 KB, 3655 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 28416 KB, 3647 KB/s, 7 seconds passed
... 22%, 28448 KB, 3648 KB/s, 7 seconds passed
... 22%, 28480 KB, 3652 KB/s, 7 seconds passed
... 22%, 28512 KB, 3655 KB/s, 7 seconds passed
... 22%, 28544 KB, 3647 KB/s, 7 seconds passed
... 22%, 28576 KB, 3648 KB/s, 7 seconds passed
... 22%, 28608 KB, 3652 KB/s, 7 seconds passed
... 22%, 28640 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 28672 KB, 3648 KB/s, 7 seconds passed
... 22%, 28704 KB, 3648 KB/s, 7 seconds passed
... 22%, 28736 KB, 3652 KB/s, 7 seconds passed
... 22%, 28768 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 22%, 28800 KB, 3648 KB/s, 7 seconds passed
... 22%, 28832 KB, 3649 KB/s, 7 seconds passed
... 22%, 28864 KB, 3652 KB/s, 7 seconds passed
... 22%, 28896 KB, 3656 KB/s, 7 seconds passed
... 22%, 28928 KB, 3648 KB/s, 7 seconds passed
... 22%, 28960 KB, 3649 KB/s, 7 seconds passed
... 23%, 28992 KB, 3652 KB/s, 7 seconds passed
... 23%, 29024 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 23%, 29056 KB, 3648 KB/s, 7 seconds passed
... 23%, 29088 KB, 3649 KB/s, 7 seconds passed
... 23%, 29120 KB, 3653 KB/s, 7 seconds passed
... 23%, 29152 KB, 3656 KB/s, 7 seconds passed

.. parsed-literal::

    ... 23%, 29184 KB, 3647 KB/s, 8 seconds passed
... 23%, 29216 KB, 3649 KB/s, 8 seconds passed
... 23%, 29248 KB, 3653 KB/s, 8 seconds passed
... 23%, 29280 KB, 3657 KB/s, 8 seconds passed
... 23%, 29312 KB, 3648 KB/s, 8 seconds passed
... 23%, 29344 KB, 3649 KB/s, 8 seconds passed
... 23%, 29376 KB, 3653 KB/s, 8 seconds passed
... 23%, 29408 KB, 3657 KB/s, 8 seconds passed

.. parsed-literal::

    ... 23%, 29440 KB, 3648 KB/s, 8 seconds passed
... 23%, 29472 KB, 3649 KB/s, 8 seconds passed
... 23%, 29504 KB, 3653 KB/s, 8 seconds passed
... 23%, 29536 KB, 3657 KB/s, 8 seconds passed

.. parsed-literal::

    ... 23%, 29568 KB, 3648 KB/s, 8 seconds passed
... 23%, 29600 KB, 3650 KB/s, 8 seconds passed
... 23%, 29632 KB, 3653 KB/s, 8 seconds passed
... 23%, 29664 KB, 3657 KB/s, 8 seconds passed
... 23%, 29696 KB, 3648 KB/s, 8 seconds passed
... 23%, 29728 KB, 3650 KB/s, 8 seconds passed
... 23%, 29760 KB, 3654 KB/s, 8 seconds passed
... 23%, 29792 KB, 3657 KB/s, 8 seconds passed

.. parsed-literal::

    ... 23%, 29824 KB, 3648 KB/s, 8 seconds passed
... 23%, 29856 KB, 3650 KB/s, 8 seconds passed
... 23%, 29888 KB, 3654 KB/s, 8 seconds passed
... 23%, 29920 KB, 3657 KB/s, 8 seconds passed

.. parsed-literal::

    ... 23%, 29952 KB, 3649 KB/s, 8 seconds passed
... 23%, 29984 KB, 3650 KB/s, 8 seconds passed
... 23%, 30016 KB, 3654 KB/s, 8 seconds passed
... 23%, 30048 KB, 3658 KB/s, 8 seconds passed
... 23%, 30080 KB, 3649 KB/s, 8 seconds passed
... 23%, 30112 KB, 3651 KB/s, 8 seconds passed
... 23%, 30144 KB, 3654 KB/s, 8 seconds passed
... 23%, 30176 KB, 3658 KB/s, 8 seconds passed

.. parsed-literal::

    ... 23%, 30208 KB, 3648 KB/s, 8 seconds passed
... 24%, 30240 KB, 3651 KB/s, 8 seconds passed
... 24%, 30272 KB, 3654 KB/s, 8 seconds passed
... 24%, 30304 KB, 3658 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 30336 KB, 3649 KB/s, 8 seconds passed
... 24%, 30368 KB, 3651 KB/s, 8 seconds passed
... 24%, 30400 KB, 3654 KB/s, 8 seconds passed
... 24%, 30432 KB, 3658 KB/s, 8 seconds passed
... 24%, 30464 KB, 3649 KB/s, 8 seconds passed
... 24%, 30496 KB, 3651 KB/s, 8 seconds passed
... 24%, 30528 KB, 3655 KB/s, 8 seconds passed
... 24%, 30560 KB, 3658 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 30592 KB, 3649 KB/s, 8 seconds passed
... 24%, 30624 KB, 3651 KB/s, 8 seconds passed
... 24%, 30656 KB, 3655 KB/s, 8 seconds passed
... 24%, 30688 KB, 3658 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 30720 KB, 3650 KB/s, 8 seconds passed
... 24%, 30752 KB, 3652 KB/s, 8 seconds passed
... 24%, 30784 KB, 3655 KB/s, 8 seconds passed
... 24%, 30816 KB, 3658 KB/s, 8 seconds passed
... 24%, 30848 KB, 3651 KB/s, 8 seconds passed
... 24%, 30880 KB, 3653 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 30912 KB, 3655 KB/s, 8 seconds passed
... 24%, 30944 KB, 3659 KB/s, 8 seconds passed
... 24%, 30976 KB, 3651 KB/s, 8 seconds passed
... 24%, 31008 KB, 3653 KB/s, 8 seconds passed
... 24%, 31040 KB, 3655 KB/s, 8 seconds passed
... 24%, 31072 KB, 3659 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 31104 KB, 3651 KB/s, 8 seconds passed
... 24%, 31136 KB, 3653 KB/s, 8 seconds passed
... 24%, 31168 KB, 3655 KB/s, 8 seconds passed
... 24%, 31200 KB, 3659 KB/s, 8 seconds passed
... 24%, 31232 KB, 3650 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 31264 KB, 3652 KB/s, 8 seconds passed
... 24%, 31296 KB, 3655 KB/s, 8 seconds passed
... 24%, 31328 KB, 3659 KB/s, 8 seconds passed
... 24%, 31360 KB, 3650 KB/s, 8 seconds passed
... 24%, 31392 KB, 3653 KB/s, 8 seconds passed
... 24%, 31424 KB, 3656 KB/s, 8 seconds passed
... 24%, 31456 KB, 3659 KB/s, 8 seconds passed

.. parsed-literal::

    ... 24%, 31488 KB, 3651 KB/s, 8 seconds passed
... 25%, 31520 KB, 3653 KB/s, 8 seconds passed
... 25%, 31552 KB, 3656 KB/s, 8 seconds passed
... 25%, 31584 KB, 3659 KB/s, 8 seconds passed
... 25%, 31616 KB, 3651 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 31648 KB, 3653 KB/s, 8 seconds passed
... 25%, 31680 KB, 3656 KB/s, 8 seconds passed
... 25%, 31712 KB, 3659 KB/s, 8 seconds passed
... 25%, 31744 KB, 3651 KB/s, 8 seconds passed
... 25%, 31776 KB, 3654 KB/s, 8 seconds passed
... 25%, 31808 KB, 3656 KB/s, 8 seconds passed
... 25%, 31840 KB, 3660 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 31872 KB, 3651 KB/s, 8 seconds passed
... 25%, 31904 KB, 3653 KB/s, 8 seconds passed
... 25%, 31936 KB, 3656 KB/s, 8 seconds passed
... 25%, 31968 KB, 3660 KB/s, 8 seconds passed
... 25%, 32000 KB, 3652 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 32032 KB, 3654 KB/s, 8 seconds passed
... 25%, 32064 KB, 3657 KB/s, 8 seconds passed
... 25%, 32096 KB, 3660 KB/s, 8 seconds passed
... 25%, 32128 KB, 3652 KB/s, 8 seconds passed
... 25%, 32160 KB, 3654 KB/s, 8 seconds passed
... 25%, 32192 KB, 3657 KB/s, 8 seconds passed
... 25%, 32224 KB, 3660 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 32256 KB, 3652 KB/s, 8 seconds passed
... 25%, 32288 KB, 3655 KB/s, 8 seconds passed
... 25%, 32320 KB, 3657 KB/s, 8 seconds passed
... 25%, 32352 KB, 3660 KB/s, 8 seconds passed
... 25%, 32384 KB, 3652 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 32416 KB, 3655 KB/s, 8 seconds passed
... 25%, 32448 KB, 3657 KB/s, 8 seconds passed
... 25%, 32480 KB, 3652 KB/s, 8 seconds passed
... 25%, 32512 KB, 3652 KB/s, 8 seconds passed
... 25%, 32544 KB, 3655 KB/s, 8 seconds passed
... 25%, 32576 KB, 3657 KB/s, 8 seconds passed

.. parsed-literal::

    ... 25%, 32608 KB, 3653 KB/s, 8 seconds passed
... 25%, 32640 KB, 3652 KB/s, 8 seconds passed
... 25%, 32672 KB, 3655 KB/s, 8 seconds passed
... 25%, 32704 KB, 3658 KB/s, 8 seconds passed
... 25%, 32736 KB, 3653 KB/s, 8 seconds passed

.. parsed-literal::

    ... 26%, 32768 KB, 3652 KB/s, 8 seconds passed
... 26%, 32800 KB, 3655 KB/s, 8 seconds passed
... 26%, 32832 KB, 3658 KB/s, 8 seconds passed
... 26%, 32864 KB, 3653 KB/s, 8 seconds passed
... 26%, 32896 KB, 3653 KB/s, 9 seconds passed
... 26%, 32928 KB, 3655 KB/s, 9 seconds passed
... 26%, 32960 KB, 3658 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 32992 KB, 3653 KB/s, 9 seconds passed
... 26%, 33024 KB, 3653 KB/s, 9 seconds passed
... 26%, 33056 KB, 3655 KB/s, 9 seconds passed
... 26%, 33088 KB, 3658 KB/s, 9 seconds passed
... 26%, 33120 KB, 3653 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 33152 KB, 3653 KB/s, 9 seconds passed
... 26%, 33184 KB, 3655 KB/s, 9 seconds passed
... 26%, 33216 KB, 3658 KB/s, 9 seconds passed
... 26%, 33248 KB, 3653 KB/s, 9 seconds passed
... 26%, 33280 KB, 3653 KB/s, 9 seconds passed
... 26%, 33312 KB, 3655 KB/s, 9 seconds passed
... 26%, 33344 KB, 3658 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 33376 KB, 3653 KB/s, 9 seconds passed
... 26%, 33408 KB, 3653 KB/s, 9 seconds passed
... 26%, 33440 KB, 3656 KB/s, 9 seconds passed
... 26%, 33472 KB, 3659 KB/s, 9 seconds passed
... 26%, 33504 KB, 3654 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 33536 KB, 3654 KB/s, 9 seconds passed
... 26%, 33568 KB, 3656 KB/s, 9 seconds passed
... 26%, 33600 KB, 3659 KB/s, 9 seconds passed
... 26%, 33632 KB, 3654 KB/s, 9 seconds passed
... 26%, 33664 KB, 3654 KB/s, 9 seconds passed
... 26%, 33696 KB, 3656 KB/s, 9 seconds passed
... 26%, 33728 KB, 3659 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 33760 KB, 3655 KB/s, 9 seconds passed
... 26%, 33792 KB, 3654 KB/s, 9 seconds passed
... 26%, 33824 KB, 3657 KB/s, 9 seconds passed
... 26%, 33856 KB, 3659 KB/s, 9 seconds passed
... 26%, 33888 KB, 3654 KB/s, 9 seconds passed

.. parsed-literal::

    ... 26%, 33920 KB, 3655 KB/s, 9 seconds passed
... 26%, 33952 KB, 3657 KB/s, 9 seconds passed
... 26%, 33984 KB, 3660 KB/s, 9 seconds passed
... 27%, 34016 KB, 3663 KB/s, 9 seconds passed
... 27%, 34048 KB, 3655 KB/s, 9 seconds passed
... 27%, 34080 KB, 3658 KB/s, 9 seconds passed
... 27%, 34112 KB, 3660 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 34144 KB, 3656 KB/s, 9 seconds passed
... 27%, 34176 KB, 3655 KB/s, 9 seconds passed
... 27%, 34208 KB, 3657 KB/s, 9 seconds passed
... 27%, 34240 KB, 3660 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 34272 KB, 3654 KB/s, 9 seconds passed
... 27%, 34304 KB, 3655 KB/s, 9 seconds passed
... 27%, 34336 KB, 3657 KB/s, 9 seconds passed
... 27%, 34368 KB, 3660 KB/s, 9 seconds passed
... 27%, 34400 KB, 3654 KB/s, 9 seconds passed
... 27%, 34432 KB, 3655 KB/s, 9 seconds passed
... 27%, 34464 KB, 3657 KB/s, 9 seconds passed
... 27%, 34496 KB, 3660 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 34528 KB, 3655 KB/s, 9 seconds passed
... 27%, 34560 KB, 3655 KB/s, 9 seconds passed
... 27%, 34592 KB, 3658 KB/s, 9 seconds passed
... 27%, 34624 KB, 3660 KB/s, 9 seconds passed
... 27%, 34656 KB, 3655 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 34688 KB, 3656 KB/s, 9 seconds passed
... 27%, 34720 KB, 3658 KB/s, 9 seconds passed
... 27%, 34752 KB, 3661 KB/s, 9 seconds passed
... 27%, 34784 KB, 3655 KB/s, 9 seconds passed
... 27%, 34816 KB, 3656 KB/s, 9 seconds passed
... 27%, 34848 KB, 3658 KB/s, 9 seconds passed
... 27%, 34880 KB, 3661 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 34912 KB, 3656 KB/s, 9 seconds passed
... 27%, 34944 KB, 3656 KB/s, 9 seconds passed
... 27%, 34976 KB, 3658 KB/s, 9 seconds passed
... 27%, 35008 KB, 3661 KB/s, 9 seconds passed
... 27%, 35040 KB, 3656 KB/s, 9 seconds passed

.. parsed-literal::

    ... 27%, 35072 KB, 3656 KB/s, 9 seconds passed
... 27%, 35104 KB, 3658 KB/s, 9 seconds passed
... 27%, 35136 KB, 3661 KB/s, 9 seconds passed
... 27%, 35168 KB, 3656 KB/s, 9 seconds passed
... 27%, 35200 KB, 3656 KB/s, 9 seconds passed
... 27%, 35232 KB, 3658 KB/s, 9 seconds passed
... 27%, 35264 KB, 3661 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 35296 KB, 3656 KB/s, 9 seconds passed
... 28%, 35328 KB, 3657 KB/s, 9 seconds passed
... 28%, 35360 KB, 3659 KB/s, 9 seconds passed
... 28%, 35392 KB, 3662 KB/s, 9 seconds passed
... 28%, 35424 KB, 3656 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 35456 KB, 3657 KB/s, 9 seconds passed
... 28%, 35488 KB, 3659 KB/s, 9 seconds passed
... 28%, 35520 KB, 3662 KB/s, 9 seconds passed
... 28%, 35552 KB, 3656 KB/s, 9 seconds passed
... 28%, 35584 KB, 3657 KB/s, 9 seconds passed
... 28%, 35616 KB, 3659 KB/s, 9 seconds passed
... 28%, 35648 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 35680 KB, 3656 KB/s, 9 seconds passed
... 28%, 35712 KB, 3657 KB/s, 9 seconds passed
... 28%, 35744 KB, 3659 KB/s, 9 seconds passed
... 28%, 35776 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 35808 KB, 3656 KB/s, 9 seconds passed
... 28%, 35840 KB, 3657 KB/s, 9 seconds passed
... 28%, 35872 KB, 3659 KB/s, 9 seconds passed
... 28%, 35904 KB, 3662 KB/s, 9 seconds passed
... 28%, 35936 KB, 3656 KB/s, 9 seconds passed
... 28%, 35968 KB, 3657 KB/s, 9 seconds passed
... 28%, 36000 KB, 3659 KB/s, 9 seconds passed
... 28%, 36032 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 36064 KB, 3656 KB/s, 9 seconds passed
... 28%, 36096 KB, 3657 KB/s, 9 seconds passed
... 28%, 36128 KB, 3659 KB/s, 9 seconds passed
... 28%, 36160 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 36192 KB, 3656 KB/s, 9 seconds passed
... 28%, 36224 KB, 3657 KB/s, 9 seconds passed
... 28%, 36256 KB, 3659 KB/s, 9 seconds passed
... 28%, 36288 KB, 3662 KB/s, 9 seconds passed
... 28%, 36320 KB, 3656 KB/s, 9 seconds passed
... 28%, 36352 KB, 3658 KB/s, 9 seconds passed
... 28%, 36384 KB, 3659 KB/s, 9 seconds passed
... 28%, 36416 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 28%, 36448 KB, 3657 KB/s, 9 seconds passed
... 28%, 36480 KB, 3657 KB/s, 9 seconds passed
... 28%, 36512 KB, 3659 KB/s, 9 seconds passed
... 29%, 36544 KB, 3662 KB/s, 9 seconds passed

.. parsed-literal::

    ... 29%, 36576 KB, 3656 KB/s, 10 seconds passed
... 29%, 36608 KB, 3658 KB/s, 10 seconds passed
... 29%, 36640 KB, 3659 KB/s, 10 seconds passed
... 29%, 36672 KB, 3662 KB/s, 10 seconds passed
... 29%, 36704 KB, 3656 KB/s, 10 seconds passed
... 29%, 36736 KB, 3658 KB/s, 10 seconds passed
... 29%, 36768 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 36800 KB, 3663 KB/s, 10 seconds passed
... 29%, 36832 KB, 3656 KB/s, 10 seconds passed
... 29%, 36864 KB, 3658 KB/s, 10 seconds passed
... 29%, 36896 KB, 3660 KB/s, 10 seconds passed
... 29%, 36928 KB, 3663 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 36960 KB, 3657 KB/s, 10 seconds passed
... 29%, 36992 KB, 3658 KB/s, 10 seconds passed
... 29%, 37024 KB, 3660 KB/s, 10 seconds passed
... 29%, 37056 KB, 3663 KB/s, 10 seconds passed
... 29%, 37088 KB, 3657 KB/s, 10 seconds passed
... 29%, 37120 KB, 3658 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 37152 KB, 3660 KB/s, 10 seconds passed
... 29%, 37184 KB, 3663 KB/s, 10 seconds passed
... 29%, 37216 KB, 3657 KB/s, 10 seconds passed
... 29%, 37248 KB, 3658 KB/s, 10 seconds passed
... 29%, 37280 KB, 3660 KB/s, 10 seconds passed
... 29%, 37312 KB, 3663 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 37344 KB, 3657 KB/s, 10 seconds passed
... 29%, 37376 KB, 3659 KB/s, 10 seconds passed
... 29%, 37408 KB, 3661 KB/s, 10 seconds passed
... 29%, 37440 KB, 3663 KB/s, 10 seconds passed
... 29%, 37472 KB, 3658 KB/s, 10 seconds passed
... 29%, 37504 KB, 3659 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 37536 KB, 3661 KB/s, 10 seconds passed
... 29%, 37568 KB, 3664 KB/s, 10 seconds passed
... 29%, 37600 KB, 3658 KB/s, 10 seconds passed
... 29%, 37632 KB, 3659 KB/s, 10 seconds passed
... 29%, 37664 KB, 3661 KB/s, 10 seconds passed
... 29%, 37696 KB, 3664 KB/s, 10 seconds passed

.. parsed-literal::

    ... 29%, 37728 KB, 3658 KB/s, 10 seconds passed
... 29%, 37760 KB, 3659 KB/s, 10 seconds passed
... 30%, 37792 KB, 3661 KB/s, 10 seconds passed
... 30%, 37824 KB, 3664 KB/s, 10 seconds passed
... 30%, 37856 KB, 3658 KB/s, 10 seconds passed
... 30%, 37888 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 37920 KB, 3661 KB/s, 10 seconds passed
... 30%, 37952 KB, 3664 KB/s, 10 seconds passed
... 30%, 37984 KB, 3658 KB/s, 10 seconds passed
... 30%, 38016 KB, 3660 KB/s, 10 seconds passed
... 30%, 38048 KB, 3661 KB/s, 10 seconds passed
... 30%, 38080 KB, 3664 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 38112 KB, 3658 KB/s, 10 seconds passed
... 30%, 38144 KB, 3660 KB/s, 10 seconds passed
... 30%, 38176 KB, 3661 KB/s, 10 seconds passed
... 30%, 38208 KB, 3658 KB/s, 10 seconds passed
... 30%, 38240 KB, 3658 KB/s, 10 seconds passed
... 30%, 38272 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 38304 KB, 3661 KB/s, 10 seconds passed
... 30%, 38336 KB, 3664 KB/s, 10 seconds passed
... 30%, 38368 KB, 3659 KB/s, 10 seconds passed
... 30%, 38400 KB, 3660 KB/s, 10 seconds passed
... 30%, 38432 KB, 3662 KB/s, 10 seconds passed
... 30%, 38464 KB, 3664 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 38496 KB, 3659 KB/s, 10 seconds passed
... 30%, 38528 KB, 3660 KB/s, 10 seconds passed
... 30%, 38560 KB, 3662 KB/s, 10 seconds passed
... 30%, 38592 KB, 3664 KB/s, 10 seconds passed
... 30%, 38624 KB, 3659 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 38656 KB, 3660 KB/s, 10 seconds passed
... 30%, 38688 KB, 3662 KB/s, 10 seconds passed
... 30%, 38720 KB, 3665 KB/s, 10 seconds passed
... 30%, 38752 KB, 3659 KB/s, 10 seconds passed
... 30%, 38784 KB, 3660 KB/s, 10 seconds passed
... 30%, 38816 KB, 3662 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 38848 KB, 3660 KB/s, 10 seconds passed
... 30%, 38880 KB, 3659 KB/s, 10 seconds passed
... 30%, 38912 KB, 3660 KB/s, 10 seconds passed
... 30%, 38944 KB, 3662 KB/s, 10 seconds passed
... 30%, 38976 KB, 3660 KB/s, 10 seconds passed
... 30%, 39008 KB, 3659 KB/s, 10 seconds passed

.. parsed-literal::

    ... 30%, 39040 KB, 3661 KB/s, 10 seconds passed
... 31%, 39072 KB, 3662 KB/s, 10 seconds passed
... 31%, 39104 KB, 3659 KB/s, 10 seconds passed
... 31%, 39136 KB, 3659 KB/s, 10 seconds passed
... 31%, 39168 KB, 3660 KB/s, 10 seconds passed
... 31%, 39200 KB, 3662 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 39232 KB, 3659 KB/s, 10 seconds passed
... 31%, 39264 KB, 3659 KB/s, 10 seconds passed
... 31%, 39296 KB, 3661 KB/s, 10 seconds passed
... 31%, 39328 KB, 3662 KB/s, 10 seconds passed
... 31%, 39360 KB, 3660 KB/s, 10 seconds passed
... 31%, 39392 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 39424 KB, 3661 KB/s, 10 seconds passed
... 31%, 39456 KB, 3662 KB/s, 10 seconds passed
... 31%, 39488 KB, 3660 KB/s, 10 seconds passed
... 31%, 39520 KB, 3660 KB/s, 10 seconds passed
... 31%, 39552 KB, 3661 KB/s, 10 seconds passed
... 31%, 39584 KB, 3662 KB/s, 10 seconds passed
... 31%, 39616 KB, 3665 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 39648 KB, 3660 KB/s, 10 seconds passed
... 31%, 39680 KB, 3661 KB/s, 10 seconds passed
... 31%, 39712 KB, 3663 KB/s, 10 seconds passed
... 31%, 39744 KB, 3661 KB/s, 10 seconds passed
... 31%, 39776 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 39808 KB, 3661 KB/s, 10 seconds passed
... 31%, 39840 KB, 3663 KB/s, 10 seconds passed
... 31%, 39872 KB, 3660 KB/s, 10 seconds passed
... 31%, 39904 KB, 3660 KB/s, 10 seconds passed
... 31%, 39936 KB, 3661 KB/s, 10 seconds passed
... 31%, 39968 KB, 3663 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 40000 KB, 3660 KB/s, 10 seconds passed
... 31%, 40032 KB, 3660 KB/s, 10 seconds passed
... 31%, 40064 KB, 3661 KB/s, 10 seconds passed
... 31%, 40096 KB, 3663 KB/s, 10 seconds passed
... 31%, 40128 KB, 3660 KB/s, 10 seconds passed

.. parsed-literal::

    ... 31%, 40160 KB, 3660 KB/s, 10 seconds passed
... 31%, 40192 KB, 3661 KB/s, 10 seconds passed
... 31%, 40224 KB, 3663 KB/s, 10 seconds passed
... 31%, 40256 KB, 3660 KB/s, 10 seconds passed
... 31%, 40288 KB, 3660 KB/s, 11 seconds passed
... 32%, 40320 KB, 3662 KB/s, 11 seconds passed
... 32%, 40352 KB, 3663 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 40384 KB, 3660 KB/s, 11 seconds passed
... 32%, 40416 KB, 3660 KB/s, 11 seconds passed
... 32%, 40448 KB, 3662 KB/s, 11 seconds passed
... 32%, 40480 KB, 3663 KB/s, 11 seconds passed
... 32%, 40512 KB, 3660 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 40544 KB, 3660 KB/s, 11 seconds passed
... 32%, 40576 KB, 3662 KB/s, 11 seconds passed
... 32%, 40608 KB, 3663 KB/s, 11 seconds passed
... 32%, 40640 KB, 3660 KB/s, 11 seconds passed
... 32%, 40672 KB, 3660 KB/s, 11 seconds passed
... 32%, 40704 KB, 3662 KB/s, 11 seconds passed
... 32%, 40736 KB, 3663 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 40768 KB, 3660 KB/s, 11 seconds passed
... 32%, 40800 KB, 3660 KB/s, 11 seconds passed
... 32%, 40832 KB, 3662 KB/s, 11 seconds passed
... 32%, 40864 KB, 3664 KB/s, 11 seconds passed
... 32%, 40896 KB, 3661 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 40928 KB, 3661 KB/s, 11 seconds passed
... 32%, 40960 KB, 3662 KB/s, 11 seconds passed
... 32%, 40992 KB, 3664 KB/s, 11 seconds passed
... 32%, 41024 KB, 3661 KB/s, 11 seconds passed
... 32%, 41056 KB, 3661 KB/s, 11 seconds passed
... 32%, 41088 KB, 3662 KB/s, 11 seconds passed
... 32%, 41120 KB, 3664 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 41152 KB, 3661 KB/s, 11 seconds passed
... 32%, 41184 KB, 3661 KB/s, 11 seconds passed
... 32%, 41216 KB, 3662 KB/s, 11 seconds passed
... 32%, 41248 KB, 3664 KB/s, 11 seconds passed
... 32%, 41280 KB, 3661 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 41312 KB, 3661 KB/s, 11 seconds passed
... 32%, 41344 KB, 3662 KB/s, 11 seconds passed
... 32%, 41376 KB, 3664 KB/s, 11 seconds passed
... 32%, 41408 KB, 3661 KB/s, 11 seconds passed
... 32%, 41440 KB, 3661 KB/s, 11 seconds passed
... 32%, 41472 KB, 3662 KB/s, 11 seconds passed
... 32%, 41504 KB, 3664 KB/s, 11 seconds passed

.. parsed-literal::

    ... 32%, 41536 KB, 3661 KB/s, 11 seconds passed
... 33%, 41568 KB, 3661 KB/s, 11 seconds passed
... 33%, 41600 KB, 3663 KB/s, 11 seconds passed
... 33%, 41632 KB, 3664 KB/s, 11 seconds passed
... 33%, 41664 KB, 3661 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 41696 KB, 3661 KB/s, 11 seconds passed
... 33%, 41728 KB, 3663 KB/s, 11 seconds passed
... 33%, 41760 KB, 3665 KB/s, 11 seconds passed
... 33%, 41792 KB, 3661 KB/s, 11 seconds passed
... 33%, 41824 KB, 3661 KB/s, 11 seconds passed
... 33%, 41856 KB, 3663 KB/s, 11 seconds passed
... 33%, 41888 KB, 3665 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 41920 KB, 3661 KB/s, 11 seconds passed
... 33%, 41952 KB, 3661 KB/s, 11 seconds passed
... 33%, 41984 KB, 3663 KB/s, 11 seconds passed
... 33%, 42016 KB, 3665 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 42048 KB, 3661 KB/s, 11 seconds passed
... 33%, 42080 KB, 3661 KB/s, 11 seconds passed
... 33%, 42112 KB, 3663 KB/s, 11 seconds passed
... 33%, 42144 KB, 3665 KB/s, 11 seconds passed
... 33%, 42176 KB, 3661 KB/s, 11 seconds passed
... 33%, 42208 KB, 3661 KB/s, 11 seconds passed
... 33%, 42240 KB, 3663 KB/s, 11 seconds passed
... 33%, 42272 KB, 3665 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 42304 KB, 3661 KB/s, 11 seconds passed
... 33%, 42336 KB, 3661 KB/s, 11 seconds passed
... 33%, 42368 KB, 3663 KB/s, 11 seconds passed
... 33%, 42400 KB, 3665 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 42432 KB, 3662 KB/s, 11 seconds passed
... 33%, 42464 KB, 3663 KB/s, 11 seconds passed
... 33%, 42496 KB, 3664 KB/s, 11 seconds passed
... 33%, 42528 KB, 3665 KB/s, 11 seconds passed
... 33%, 42560 KB, 3662 KB/s, 11 seconds passed
... 33%, 42592 KB, 3663 KB/s, 11 seconds passed
... 33%, 42624 KB, 3664 KB/s, 11 seconds passed
... 33%, 42656 KB, 3665 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 42688 KB, 3662 KB/s, 11 seconds passed
... 33%, 42720 KB, 3663 KB/s, 11 seconds passed
... 33%, 42752 KB, 3663 KB/s, 11 seconds passed
... 33%, 42784 KB, 3662 KB/s, 11 seconds passed

.. parsed-literal::

    ... 33%, 42816 KB, 3662 KB/s, 11 seconds passed
... 34%, 42848 KB, 3663 KB/s, 11 seconds passed
... 34%, 42880 KB, 3664 KB/s, 11 seconds passed
... 34%, 42912 KB, 3666 KB/s, 11 seconds passed
... 34%, 42944 KB, 3662 KB/s, 11 seconds passed
... 34%, 42976 KB, 3663 KB/s, 11 seconds passed
... 34%, 43008 KB, 3664 KB/s, 11 seconds passed
... 34%, 43040 KB, 3666 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43072 KB, 3662 KB/s, 11 seconds passed
... 34%, 43104 KB, 3662 KB/s, 11 seconds passed
... 34%, 43136 KB, 3664 KB/s, 11 seconds passed
... 34%, 43168 KB, 3662 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43200 KB, 3662 KB/s, 11 seconds passed
... 34%, 43232 KB, 3662 KB/s, 11 seconds passed
... 34%, 43264 KB, 3664 KB/s, 11 seconds passed
... 34%, 43296 KB, 3666 KB/s, 11 seconds passed
... 34%, 43328 KB, 3662 KB/s, 11 seconds passed
... 34%, 43360 KB, 3663 KB/s, 11 seconds passed
... 34%, 43392 KB, 3664 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43424 KB, 3666 KB/s, 11 seconds passed
... 34%, 43456 KB, 3663 KB/s, 11 seconds passed
... 34%, 43488 KB, 3663 KB/s, 11 seconds passed
... 34%, 43520 KB, 3665 KB/s, 11 seconds passed
... 34%, 43552 KB, 3666 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43584 KB, 3663 KB/s, 11 seconds passed
... 34%, 43616 KB, 3663 KB/s, 11 seconds passed
... 34%, 43648 KB, 3664 KB/s, 11 seconds passed
... 34%, 43680 KB, 3662 KB/s, 11 seconds passed
... 34%, 43712 KB, 3662 KB/s, 11 seconds passed
... 34%, 43744 KB, 3663 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43776 KB, 3664 KB/s, 11 seconds passed
... 34%, 43808 KB, 3662 KB/s, 11 seconds passed
... 34%, 43840 KB, 3663 KB/s, 11 seconds passed
... 34%, 43872 KB, 3663 KB/s, 11 seconds passed
... 34%, 43904 KB, 3664 KB/s, 11 seconds passed

.. parsed-literal::

    ... 34%, 43936 KB, 3662 KB/s, 11 seconds passed
... 34%, 43968 KB, 3663 KB/s, 12 seconds passed
... 34%, 44000 KB, 3663 KB/s, 12 seconds passed
... 34%, 44032 KB, 3665 KB/s, 12 seconds passed
... 34%, 44064 KB, 3663 KB/s, 12 seconds passed
... 35%, 44096 KB, 3663 KB/s, 12 seconds passed
... 35%, 44128 KB, 3663 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 44160 KB, 3665 KB/s, 12 seconds passed
... 35%, 44192 KB, 3663 KB/s, 12 seconds passed
... 35%, 44224 KB, 3663 KB/s, 12 seconds passed
... 35%, 44256 KB, 3663 KB/s, 12 seconds passed
... 35%, 44288 KB, 3665 KB/s, 12 seconds passed
... 35%, 44320 KB, 3663 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 44352 KB, 3663 KB/s, 12 seconds passed
... 35%, 44384 KB, 3664 KB/s, 12 seconds passed
... 35%, 44416 KB, 3665 KB/s, 12 seconds passed
... 35%, 44448 KB, 3663 KB/s, 12 seconds passed
... 35%, 44480 KB, 3663 KB/s, 12 seconds passed
... 35%, 44512 KB, 3664 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 44544 KB, 3665 KB/s, 12 seconds passed
... 35%, 44576 KB, 3663 KB/s, 12 seconds passed
... 35%, 44608 KB, 3663 KB/s, 12 seconds passed
... 35%, 44640 KB, 3664 KB/s, 12 seconds passed
... 35%, 44672 KB, 3665 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 44704 KB, 3663 KB/s, 12 seconds passed
... 35%, 44736 KB, 3664 KB/s, 12 seconds passed
... 35%, 44768 KB, 3664 KB/s, 12 seconds passed
... 35%, 44800 KB, 3666 KB/s, 12 seconds passed
... 35%, 44832 KB, 3663 KB/s, 12 seconds passed
... 35%, 44864 KB, 3664 KB/s, 12 seconds passed
... 35%, 44896 KB, 3664 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 44928 KB, 3665 KB/s, 12 seconds passed
... 35%, 44960 KB, 3663 KB/s, 12 seconds passed
... 35%, 44992 KB, 3664 KB/s, 12 seconds passed
... 35%, 45024 KB, 3664 KB/s, 12 seconds passed
... 35%, 45056 KB, 3666 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 45088 KB, 3663 KB/s, 12 seconds passed
... 35%, 45120 KB, 3664 KB/s, 12 seconds passed
... 35%, 45152 KB, 3664 KB/s, 12 seconds passed
... 35%, 45184 KB, 3666 KB/s, 12 seconds passed
... 35%, 45216 KB, 3663 KB/s, 12 seconds passed
... 35%, 45248 KB, 3664 KB/s, 12 seconds passed

.. parsed-literal::

    ... 35%, 45280 KB, 3664 KB/s, 12 seconds passed
... 35%, 45312 KB, 3666 KB/s, 12 seconds passed
... 36%, 45344 KB, 3663 KB/s, 12 seconds passed
... 36%, 45376 KB, 3664 KB/s, 12 seconds passed
... 36%, 45408 KB, 3664 KB/s, 12 seconds passed
... 36%, 45440 KB, 3666 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 45472 KB, 3663 KB/s, 12 seconds passed
... 36%, 45504 KB, 3664 KB/s, 12 seconds passed
... 36%, 45536 KB, 3664 KB/s, 12 seconds passed
... 36%, 45568 KB, 3666 KB/s, 12 seconds passed
... 36%, 45600 KB, 3663 KB/s, 12 seconds passed
... 36%, 45632 KB, 3664 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 45664 KB, 3664 KB/s, 12 seconds passed
... 36%, 45696 KB, 3666 KB/s, 12 seconds passed
... 36%, 45728 KB, 3663 KB/s, 12 seconds passed
... 36%, 45760 KB, 3664 KB/s, 12 seconds passed
... 36%, 45792 KB, 3665 KB/s, 12 seconds passed
... 36%, 45824 KB, 3666 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 45856 KB, 3664 KB/s, 12 seconds passed
... 36%, 45888 KB, 3665 KB/s, 12 seconds passed
... 36%, 45920 KB, 3665 KB/s, 12 seconds passed
... 36%, 45952 KB, 3666 KB/s, 12 seconds passed
... 36%, 45984 KB, 3664 KB/s, 12 seconds passed
... 36%, 46016 KB, 3665 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 46048 KB, 3665 KB/s, 12 seconds passed
... 36%, 46080 KB, 3666 KB/s, 12 seconds passed
... 36%, 46112 KB, 3663 KB/s, 12 seconds passed
... 36%, 46144 KB, 3665 KB/s, 12 seconds passed
... 36%, 46176 KB, 3665 KB/s, 12 seconds passed
... 36%, 46208 KB, 3666 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 46240 KB, 3663 KB/s, 12 seconds passed
... 36%, 46272 KB, 3665 KB/s, 12 seconds passed
... 36%, 46304 KB, 3665 KB/s, 12 seconds passed
... 36%, 46336 KB, 3667 KB/s, 12 seconds passed
... 36%, 46368 KB, 3663 KB/s, 12 seconds passed
... 36%, 46400 KB, 3664 KB/s, 12 seconds passed

.. parsed-literal::

    ... 36%, 46432 KB, 3665 KB/s, 12 seconds passed
... 36%, 46464 KB, 3667 KB/s, 12 seconds passed
... 36%, 46496 KB, 3665 KB/s, 12 seconds passed
... 36%, 46528 KB, 3665 KB/s, 12 seconds passed
... 36%, 46560 KB, 3666 KB/s, 12 seconds passed
... 36%, 46592 KB, 3667 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 46624 KB, 3665 KB/s, 12 seconds passed
... 37%, 46656 KB, 3665 KB/s, 12 seconds passed
... 37%, 46688 KB, 3666 KB/s, 12 seconds passed
... 37%, 46720 KB, 3667 KB/s, 12 seconds passed
... 37%, 46752 KB, 3663 KB/s, 12 seconds passed
... 37%, 46784 KB, 3665 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 46816 KB, 3665 KB/s, 12 seconds passed
... 37%, 46848 KB, 3663 KB/s, 12 seconds passed
... 37%, 46880 KB, 3664 KB/s, 12 seconds passed
... 37%, 46912 KB, 3665 KB/s, 12 seconds passed
... 37%, 46944 KB, 3666 KB/s, 12 seconds passed
... 37%, 46976 KB, 3667 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 47008 KB, 3664 KB/s, 12 seconds passed
... 37%, 47040 KB, 3665 KB/s, 12 seconds passed
... 37%, 47072 KB, 3666 KB/s, 12 seconds passed
... 37%, 47104 KB, 3664 KB/s, 12 seconds passed
... 37%, 47136 KB, 3664 KB/s, 12 seconds passed
... 37%, 47168 KB, 3665 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 47200 KB, 3666 KB/s, 12 seconds passed
... 37%, 47232 KB, 3667 KB/s, 12 seconds passed
... 37%, 47264 KB, 3664 KB/s, 12 seconds passed
... 37%, 47296 KB, 3665 KB/s, 12 seconds passed
... 37%, 47328 KB, 3666 KB/s, 12 seconds passed
... 37%, 47360 KB, 3668 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 47392 KB, 3664 KB/s, 12 seconds passed
... 37%, 47424 KB, 3665 KB/s, 12 seconds passed
... 37%, 47456 KB, 3666 KB/s, 12 seconds passed
... 37%, 47488 KB, 3668 KB/s, 12 seconds passed
... 37%, 47520 KB, 3664 KB/s, 12 seconds passed
... 37%, 47552 KB, 3666 KB/s, 12 seconds passed

.. parsed-literal::

    ... 37%, 47584 KB, 3666 KB/s, 12 seconds passed
... 37%, 47616 KB, 3668 KB/s, 12 seconds passed
... 37%, 47648 KB, 3664 KB/s, 13 seconds passed
... 37%, 47680 KB, 3666 KB/s, 13 seconds passed
... 37%, 47712 KB, 3666 KB/s, 13 seconds passed
... 37%, 47744 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 37%, 47776 KB, 3664 KB/s, 13 seconds passed
... 37%, 47808 KB, 3666 KB/s, 13 seconds passed
... 37%, 47840 KB, 3666 KB/s, 13 seconds passed
... 38%, 47872 KB, 3668 KB/s, 13 seconds passed
... 38%, 47904 KB, 3664 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 47936 KB, 3666 KB/s, 13 seconds passed
... 38%, 47968 KB, 3666 KB/s, 13 seconds passed
... 38%, 48000 KB, 3668 KB/s, 13 seconds passed
... 38%, 48032 KB, 3664 KB/s, 13 seconds passed
... 38%, 48064 KB, 3666 KB/s, 13 seconds passed
... 38%, 48096 KB, 3666 KB/s, 13 seconds passed
... 38%, 48128 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 48160 KB, 3664 KB/s, 13 seconds passed
... 38%, 48192 KB, 3666 KB/s, 13 seconds passed
... 38%, 48224 KB, 3667 KB/s, 13 seconds passed
... 38%, 48256 KB, 3668 KB/s, 13 seconds passed
... 38%, 48288 KB, 3665 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 48320 KB, 3666 KB/s, 13 seconds passed
... 38%, 48352 KB, 3667 KB/s, 13 seconds passed
... 38%, 48384 KB, 3669 KB/s, 13 seconds passed
... 38%, 48416 KB, 3665 KB/s, 13 seconds passed
... 38%, 48448 KB, 3666 KB/s, 13 seconds passed
... 38%, 48480 KB, 3667 KB/s, 13 seconds passed
... 38%, 48512 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 48544 KB, 3665 KB/s, 13 seconds passed
... 38%, 48576 KB, 3666 KB/s, 13 seconds passed
... 38%, 48608 KB, 3667 KB/s, 13 seconds passed
... 38%, 48640 KB, 3668 KB/s, 13 seconds passed
... 38%, 48672 KB, 3665 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 48704 KB, 3667 KB/s, 13 seconds passed
... 38%, 48736 KB, 3667 KB/s, 13 seconds passed
... 38%, 48768 KB, 3665 KB/s, 13 seconds passed
... 38%, 48800 KB, 3665 KB/s, 13 seconds passed
... 38%, 48832 KB, 3667 KB/s, 13 seconds passed
... 38%, 48864 KB, 3667 KB/s, 13 seconds passed
... 38%, 48896 KB, 3669 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 48928 KB, 3665 KB/s, 13 seconds passed
... 38%, 48960 KB, 3666 KB/s, 13 seconds passed
... 38%, 48992 KB, 3667 KB/s, 13 seconds passed
... 38%, 49024 KB, 3664 KB/s, 13 seconds passed

.. parsed-literal::

    ... 38%, 49056 KB, 3665 KB/s, 13 seconds passed
... 38%, 49088 KB, 3666 KB/s, 13 seconds passed
... 38%, 49120 KB, 3667 KB/s, 13 seconds passed
... 39%, 49152 KB, 3664 KB/s, 13 seconds passed
... 39%, 49184 KB, 3665 KB/s, 13 seconds passed
... 39%, 49216 KB, 3666 KB/s, 13 seconds passed
... 39%, 49248 KB, 3667 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 49280 KB, 3664 KB/s, 13 seconds passed
... 39%, 49312 KB, 3665 KB/s, 13 seconds passed
... 39%, 49344 KB, 3666 KB/s, 13 seconds passed
... 39%, 49376 KB, 3668 KB/s, 13 seconds passed
... 39%, 49408 KB, 3665 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 49440 KB, 3665 KB/s, 13 seconds passed
... 39%, 49472 KB, 3666 KB/s, 13 seconds passed
... 39%, 49504 KB, 3668 KB/s, 13 seconds passed
... 39%, 49536 KB, 3665 KB/s, 13 seconds passed
... 39%, 49568 KB, 3665 KB/s, 13 seconds passed
... 39%, 49600 KB, 3666 KB/s, 13 seconds passed
... 39%, 49632 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 49664 KB, 3665 KB/s, 13 seconds passed
... 39%, 49696 KB, 3666 KB/s, 13 seconds passed
... 39%, 49728 KB, 3666 KB/s, 13 seconds passed
... 39%, 49760 KB, 3668 KB/s, 13 seconds passed
... 39%, 49792 KB, 3665 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 49824 KB, 3666 KB/s, 13 seconds passed
... 39%, 49856 KB, 3666 KB/s, 13 seconds passed
... 39%, 49888 KB, 3668 KB/s, 13 seconds passed
... 39%, 49920 KB, 3665 KB/s, 13 seconds passed
... 39%, 49952 KB, 3666 KB/s, 13 seconds passed
... 39%, 49984 KB, 3666 KB/s, 13 seconds passed
... 39%, 50016 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 50048 KB, 3665 KB/s, 13 seconds passed
... 39%, 50080 KB, 3666 KB/s, 13 seconds passed
... 39%, 50112 KB, 3666 KB/s, 13 seconds passed
... 39%, 50144 KB, 3668 KB/s, 13 seconds passed
... 39%, 50176 KB, 3665 KB/s, 13 seconds passed

.. parsed-literal::

    ... 39%, 50208 KB, 3666 KB/s, 13 seconds passed
... 39%, 50240 KB, 3666 KB/s, 13 seconds passed
... 39%, 50272 KB, 3668 KB/s, 13 seconds passed
... 39%, 50304 KB, 3665 KB/s, 13 seconds passed
... 39%, 50336 KB, 3666 KB/s, 13 seconds passed
... 39%, 50368 KB, 3666 KB/s, 13 seconds passed
... 40%, 50400 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 50432 KB, 3665 KB/s, 13 seconds passed
... 40%, 50464 KB, 3666 KB/s, 13 seconds passed
... 40%, 50496 KB, 3666 KB/s, 13 seconds passed
... 40%, 50528 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 50560 KB, 3665 KB/s, 13 seconds passed
... 40%, 50592 KB, 3666 KB/s, 13 seconds passed
... 40%, 50624 KB, 3666 KB/s, 13 seconds passed
... 40%, 50656 KB, 3668 KB/s, 13 seconds passed
... 40%, 50688 KB, 3665 KB/s, 13 seconds passed
... 40%, 50720 KB, 3666 KB/s, 13 seconds passed
... 40%, 50752 KB, 3667 KB/s, 13 seconds passed
... 40%, 50784 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 50816 KB, 3666 KB/s, 13 seconds passed
... 40%, 50848 KB, 3666 KB/s, 13 seconds passed
... 40%, 50880 KB, 3667 KB/s, 13 seconds passed
... 40%, 50912 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 50944 KB, 3666 KB/s, 13 seconds passed
... 40%, 50976 KB, 3666 KB/s, 13 seconds passed
... 40%, 51008 KB, 3667 KB/s, 13 seconds passed
... 40%, 51040 KB, 3668 KB/s, 13 seconds passed
... 40%, 51072 KB, 3666 KB/s, 13 seconds passed
... 40%, 51104 KB, 3666 KB/s, 13 seconds passed
... 40%, 51136 KB, 3667 KB/s, 13 seconds passed
... 40%, 51168 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 51200 KB, 3665 KB/s, 13 seconds passed
... 40%, 51232 KB, 3666 KB/s, 13 seconds passed
... 40%, 51264 KB, 3667 KB/s, 13 seconds passed
... 40%, 51296 KB, 3668 KB/s, 13 seconds passed

.. parsed-literal::

    ... 40%, 51328 KB, 3665 KB/s, 14 seconds passed
... 40%, 51360 KB, 3666 KB/s, 14 seconds passed
... 40%, 51392 KB, 3667 KB/s, 14 seconds passed
... 40%, 51424 KB, 3669 KB/s, 14 seconds passed
... 40%, 51456 KB, 3665 KB/s, 14 seconds passed
... 40%, 51488 KB, 3667 KB/s, 14 seconds passed
... 40%, 51520 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 40%, 51552 KB, 3669 KB/s, 14 seconds passed
... 40%, 51584 KB, 3666 KB/s, 14 seconds passed
... 40%, 51616 KB, 3667 KB/s, 14 seconds passed
... 41%, 51648 KB, 3667 KB/s, 14 seconds passed
... 41%, 51680 KB, 3669 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 51712 KB, 3666 KB/s, 14 seconds passed
... 41%, 51744 KB, 3666 KB/s, 14 seconds passed
... 41%, 51776 KB, 3667 KB/s, 14 seconds passed
... 41%, 51808 KB, 3665 KB/s, 14 seconds passed
... 41%, 51840 KB, 3665 KB/s, 14 seconds passed
... 41%, 51872 KB, 3666 KB/s, 14 seconds passed
... 41%, 51904 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 51936 KB, 3665 KB/s, 14 seconds passed
... 41%, 51968 KB, 3666 KB/s, 14 seconds passed
... 41%, 52000 KB, 3666 KB/s, 14 seconds passed
... 41%, 52032 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 52064 KB, 3665 KB/s, 14 seconds passed
... 41%, 52096 KB, 3666 KB/s, 14 seconds passed
... 41%, 52128 KB, 3666 KB/s, 14 seconds passed
... 41%, 52160 KB, 3667 KB/s, 14 seconds passed
... 41%, 52192 KB, 3665 KB/s, 14 seconds passed
... 41%, 52224 KB, 3666 KB/s, 14 seconds passed
... 41%, 52256 KB, 3667 KB/s, 14 seconds passed
... 41%, 52288 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 52320 KB, 3665 KB/s, 14 seconds passed
... 41%, 52352 KB, 3666 KB/s, 14 seconds passed
... 41%, 52384 KB, 3667 KB/s, 14 seconds passed
... 41%, 52416 KB, 3668 KB/s, 14 seconds passed
... 41%, 52448 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 52480 KB, 3666 KB/s, 14 seconds passed
... 41%, 52512 KB, 3667 KB/s, 14 seconds passed
... 41%, 52544 KB, 3668 KB/s, 14 seconds passed
... 41%, 52576 KB, 3665 KB/s, 14 seconds passed
... 41%, 52608 KB, 3666 KB/s, 14 seconds passed
... 41%, 52640 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 52672 KB, 3668 KB/s, 14 seconds passed
... 41%, 52704 KB, 3665 KB/s, 14 seconds passed
... 41%, 52736 KB, 3666 KB/s, 14 seconds passed
... 41%, 52768 KB, 3667 KB/s, 14 seconds passed
... 41%, 52800 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 41%, 52832 KB, 3665 KB/s, 14 seconds passed
... 41%, 52864 KB, 3666 KB/s, 14 seconds passed
... 41%, 52896 KB, 3667 KB/s, 14 seconds passed
... 42%, 52928 KB, 3668 KB/s, 14 seconds passed
... 42%, 52960 KB, 3665 KB/s, 14 seconds passed
... 42%, 52992 KB, 3667 KB/s, 14 seconds passed
... 42%, 53024 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53056 KB, 3668 KB/s, 14 seconds passed
... 42%, 53088 KB, 3665 KB/s, 14 seconds passed
... 42%, 53120 KB, 3667 KB/s, 14 seconds passed
... 42%, 53152 KB, 3668 KB/s, 14 seconds passed
... 42%, 53184 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53216 KB, 3666 KB/s, 14 seconds passed
... 42%, 53248 KB, 3667 KB/s, 14 seconds passed
... 42%, 53280 KB, 3667 KB/s, 14 seconds passed
... 42%, 53312 KB, 3668 KB/s, 14 seconds passed
... 42%, 53344 KB, 3666 KB/s, 14 seconds passed
... 42%, 53376 KB, 3667 KB/s, 14 seconds passed
... 42%, 53408 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53440 KB, 3668 KB/s, 14 seconds passed
... 42%, 53472 KB, 3665 KB/s, 14 seconds passed
... 42%, 53504 KB, 3667 KB/s, 14 seconds passed
... 42%, 53536 KB, 3667 KB/s, 14 seconds passed
... 42%, 53568 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53600 KB, 3665 KB/s, 14 seconds passed
... 42%, 53632 KB, 3667 KB/s, 14 seconds passed
... 42%, 53664 KB, 3667 KB/s, 14 seconds passed
... 42%, 53696 KB, 3668 KB/s, 14 seconds passed
... 42%, 53728 KB, 3666 KB/s, 14 seconds passed
... 42%, 53760 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53792 KB, 3667 KB/s, 14 seconds passed
... 42%, 53824 KB, 3669 KB/s, 14 seconds passed
... 42%, 53856 KB, 3665 KB/s, 14 seconds passed
... 42%, 53888 KB, 3667 KB/s, 14 seconds passed
... 42%, 53920 KB, 3667 KB/s, 14 seconds passed
... 42%, 53952 KB, 3669 KB/s, 14 seconds passed

.. parsed-literal::

    ... 42%, 53984 KB, 3666 KB/s, 14 seconds passed
... 42%, 54016 KB, 3667 KB/s, 14 seconds passed
... 42%, 54048 KB, 3668 KB/s, 14 seconds passed
... 42%, 54080 KB, 3668 KB/s, 14 seconds passed
... 42%, 54112 KB, 3666 KB/s, 14 seconds passed
... 42%, 54144 KB, 3667 KB/s, 14 seconds passed

.. parsed-literal::

    ... 43%, 54176 KB, 3668 KB/s, 14 seconds passed
... 43%, 54208 KB, 3668 KB/s, 14 seconds passed
... 43%, 54240 KB, 3666 KB/s, 14 seconds passed
... 43%, 54272 KB, 3667 KB/s, 14 seconds passed
... 43%, 54304 KB, 3668 KB/s, 14 seconds passed
... 43%, 54336 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 43%, 54368 KB, 3666 KB/s, 14 seconds passed
... 43%, 54400 KB, 3668 KB/s, 14 seconds passed
... 43%, 54432 KB, 3668 KB/s, 14 seconds passed
... 43%, 54464 KB, 3669 KB/s, 14 seconds passed
... 43%, 54496 KB, 3666 KB/s, 14 seconds passed
... 43%, 54528 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 43%, 54560 KB, 3668 KB/s, 14 seconds passed
... 43%, 54592 KB, 3668 KB/s, 14 seconds passed
... 43%, 54624 KB, 3666 KB/s, 14 seconds passed
... 43%, 54656 KB, 3668 KB/s, 14 seconds passed
... 43%, 54688 KB, 3668 KB/s, 14 seconds passed
... 43%, 54720 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 43%, 54752 KB, 3666 KB/s, 14 seconds passed
... 43%, 54784 KB, 3668 KB/s, 14 seconds passed
... 43%, 54816 KB, 3668 KB/s, 14 seconds passed
... 43%, 54848 KB, 3669 KB/s, 14 seconds passed
... 43%, 54880 KB, 3666 KB/s, 14 seconds passed
... 43%, 54912 KB, 3668 KB/s, 14 seconds passed

.. parsed-literal::

    ... 43%, 54944 KB, 3668 KB/s, 14 seconds passed
... 43%, 54976 KB, 3669 KB/s, 14 seconds passed
... 43%, 55008 KB, 3667 KB/s, 15 seconds passed
... 43%, 55040 KB, 3668 KB/s, 15 seconds passed
... 43%, 55072 KB, 3668 KB/s, 15 seconds passed
... 43%, 55104 KB, 3669 KB/s, 15 seconds passed

.. parsed-literal::

    ... 43%, 55136 KB, 3666 KB/s, 15 seconds passed
... 43%, 55168 KB, 3668 KB/s, 15 seconds passed
... 43%, 55200 KB, 3668 KB/s, 15 seconds passed
... 43%, 55232 KB, 3669 KB/s, 15 seconds passed
... 43%, 55264 KB, 3666 KB/s, 15 seconds passed
... 43%, 55296 KB, 3668 KB/s, 15 seconds passed

.. parsed-literal::

    ... 43%, 55328 KB, 3668 KB/s, 15 seconds passed
... 43%, 55360 KB, 3669 KB/s, 15 seconds passed
... 43%, 55392 KB, 3666 KB/s, 15 seconds passed
... 44%, 55424 KB, 3668 KB/s, 15 seconds passed
... 44%, 55456 KB, 3668 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 55488 KB, 3667 KB/s, 15 seconds passed
... 44%, 55520 KB, 3667 KB/s, 15 seconds passed
... 44%, 55552 KB, 3668 KB/s, 15 seconds passed
... 44%, 55584 KB, 3669 KB/s, 15 seconds passed
... 44%, 55616 KB, 3669 KB/s, 15 seconds passed
... 44%, 55648 KB, 3667 KB/s, 15 seconds passed
... 44%, 55680 KB, 3668 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 55712 KB, 3669 KB/s, 15 seconds passed
... 44%, 55744 KB, 3669 KB/s, 15 seconds passed
... 44%, 55776 KB, 3667 KB/s, 15 seconds passed
... 44%, 55808 KB, 3669 KB/s, 15 seconds passed
... 44%, 55840 KB, 3669 KB/s, 15 seconds passed
... 44%, 55872 KB, 3669 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 55904 KB, 3667 KB/s, 15 seconds passed
... 44%, 55936 KB, 3669 KB/s, 15 seconds passed
... 44%, 55968 KB, 3669 KB/s, 15 seconds passed
... 44%, 56000 KB, 3667 KB/s, 15 seconds passed
... 44%, 56032 KB, 3667 KB/s, 15 seconds passed
... 44%, 56064 KB, 3669 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 56096 KB, 3669 KB/s, 15 seconds passed
... 44%, 56128 KB, 3668 KB/s, 15 seconds passed
... 44%, 56160 KB, 3667 KB/s, 15 seconds passed
... 44%, 56192 KB, 3669 KB/s, 15 seconds passed
... 44%, 56224 KB, 3669 KB/s, 15 seconds passed
... 44%, 56256 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 56288 KB, 3667 KB/s, 15 seconds passed
... 44%, 56320 KB, 3669 KB/s, 15 seconds passed
... 44%, 56352 KB, 3669 KB/s, 15 seconds passed
... 44%, 56384 KB, 3670 KB/s, 15 seconds passed
... 44%, 56416 KB, 3667 KB/s, 15 seconds passed
... 44%, 56448 KB, 3669 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 56480 KB, 3669 KB/s, 15 seconds passed
... 44%, 56512 KB, 3670 KB/s, 15 seconds passed
... 44%, 56544 KB, 3667 KB/s, 15 seconds passed
... 44%, 56576 KB, 3669 KB/s, 15 seconds passed
... 44%, 56608 KB, 3669 KB/s, 15 seconds passed
... 44%, 56640 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 44%, 56672 KB, 3667 KB/s, 15 seconds passed
... 45%, 56704 KB, 3669 KB/s, 15 seconds passed
... 45%, 56736 KB, 3669 KB/s, 15 seconds passed
... 45%, 56768 KB, 3670 KB/s, 15 seconds passed
... 45%, 56800 KB, 3667 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 56832 KB, 3669 KB/s, 15 seconds passed
... 45%, 56864 KB, 3669 KB/s, 15 seconds passed
... 45%, 56896 KB, 3668 KB/s, 15 seconds passed
... 45%, 56928 KB, 3667 KB/s, 15 seconds passed
... 45%, 56960 KB, 3668 KB/s, 15 seconds passed
... 45%, 56992 KB, 3669 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 57024 KB, 3667 KB/s, 15 seconds passed
... 45%, 57056 KB, 3667 KB/s, 15 seconds passed
... 45%, 57088 KB, 3668 KB/s, 15 seconds passed
... 45%, 57120 KB, 3669 KB/s, 15 seconds passed
... 45%, 57152 KB, 3667 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 57184 KB, 3667 KB/s, 15 seconds passed
... 45%, 57216 KB, 3668 KB/s, 15 seconds passed
... 45%, 57248 KB, 3669 KB/s, 15 seconds passed
... 45%, 57280 KB, 3667 KB/s, 15 seconds passed
... 45%, 57312 KB, 3668 KB/s, 15 seconds passed
... 45%, 57344 KB, 3668 KB/s, 15 seconds passed
... 45%, 57376 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 57408 KB, 3667 KB/s, 15 seconds passed
... 45%, 57440 KB, 3668 KB/s, 15 seconds passed
... 45%, 57472 KB, 3668 KB/s, 15 seconds passed
... 45%, 57504 KB, 3670 KB/s, 15 seconds passed
... 45%, 57536 KB, 3667 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 57568 KB, 3668 KB/s, 15 seconds passed
... 45%, 57600 KB, 3668 KB/s, 15 seconds passed
... 45%, 57632 KB, 3670 KB/s, 15 seconds passed
... 45%, 57664 KB, 3667 KB/s, 15 seconds passed
... 45%, 57696 KB, 3668 KB/s, 15 seconds passed
... 45%, 57728 KB, 3669 KB/s, 15 seconds passed
... 45%, 57760 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 45%, 57792 KB, 3667 KB/s, 15 seconds passed
... 45%, 57824 KB, 3668 KB/s, 15 seconds passed
... 45%, 57856 KB, 3669 KB/s, 15 seconds passed
... 45%, 57888 KB, 3670 KB/s, 15 seconds passed
... 45%, 57920 KB, 3667 KB/s, 15 seconds passed

.. parsed-literal::

    ... 46%, 57952 KB, 3668 KB/s, 15 seconds passed
... 46%, 57984 KB, 3669 KB/s, 15 seconds passed
... 46%, 58016 KB, 3670 KB/s, 15 seconds passed
... 46%, 58048 KB, 3667 KB/s, 15 seconds passed
... 46%, 58080 KB, 3668 KB/s, 15 seconds passed
... 46%, 58112 KB, 3669 KB/s, 15 seconds passed
... 46%, 58144 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 46%, 58176 KB, 3668 KB/s, 15 seconds passed
... 46%, 58208 KB, 3668 KB/s, 15 seconds passed
... 46%, 58240 KB, 3669 KB/s, 15 seconds passed
... 46%, 58272 KB, 3670 KB/s, 15 seconds passed
... 46%, 58304 KB, 3667 KB/s, 15 seconds passed

.. parsed-literal::

    ... 46%, 58336 KB, 3668 KB/s, 15 seconds passed
... 46%, 58368 KB, 3669 KB/s, 15 seconds passed
... 46%, 58400 KB, 3670 KB/s, 15 seconds passed
... 46%, 58432 KB, 3667 KB/s, 15 seconds passed
... 46%, 58464 KB, 3668 KB/s, 15 seconds passed
... 46%, 58496 KB, 3669 KB/s, 15 seconds passed
... 46%, 58528 KB, 3670 KB/s, 15 seconds passed

.. parsed-literal::

    ... 46%, 58560 KB, 3667 KB/s, 15 seconds passed
... 46%, 58592 KB, 3668 KB/s, 15 seconds passed
... 46%, 58624 KB, 3669 KB/s, 15 seconds passed
... 46%, 58656 KB, 3671 KB/s, 15 seconds passed
... 46%, 58688 KB, 3668 KB/s, 15 seconds passed

.. parsed-literal::

    ... 46%, 58720 KB, 3669 KB/s, 16 seconds passed
... 46%, 58752 KB, 3670 KB/s, 16 seconds passed
... 46%, 58784 KB, 3671 KB/s, 16 seconds passed
... 46%, 58816 KB, 3668 KB/s, 16 seconds passed
... 46%, 58848 KB, 3669 KB/s, 16 seconds passed
... 46%, 58880 KB, 3670 KB/s, 16 seconds passed
... 46%, 58912 KB, 3671 KB/s, 16 seconds passed

.. parsed-literal::

    ... 46%, 58944 KB, 3668 KB/s, 16 seconds passed
... 46%, 58976 KB, 3669 KB/s, 16 seconds passed
... 46%, 59008 KB, 3669 KB/s, 16 seconds passed
... 46%, 59040 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 46%, 59072 KB, 3668 KB/s, 16 seconds passed
... 46%, 59104 KB, 3669 KB/s, 16 seconds passed
... 46%, 59136 KB, 3669 KB/s, 16 seconds passed
... 46%, 59168 KB, 3670 KB/s, 16 seconds passed
... 47%, 59200 KB, 3668 KB/s, 16 seconds passed
... 47%, 59232 KB, 3669 KB/s, 16 seconds passed
... 47%, 59264 KB, 3669 KB/s, 16 seconds passed
... 47%, 59296 KB, 3671 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 59328 KB, 3668 KB/s, 16 seconds passed
... 47%, 59360 KB, 3669 KB/s, 16 seconds passed
... 47%, 59392 KB, 3669 KB/s, 16 seconds passed
... 47%, 59424 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 59456 KB, 3668 KB/s, 16 seconds passed
... 47%, 59488 KB, 3669 KB/s, 16 seconds passed
... 47%, 59520 KB, 3670 KB/s, 16 seconds passed
... 47%, 59552 KB, 3670 KB/s, 16 seconds passed
... 47%, 59584 KB, 3668 KB/s, 16 seconds passed
... 47%, 59616 KB, 3669 KB/s, 16 seconds passed
... 47%, 59648 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 59680 KB, 3670 KB/s, 16 seconds passed
... 47%, 59712 KB, 3668 KB/s, 16 seconds passed
... 47%, 59744 KB, 3669 KB/s, 16 seconds passed
... 47%, 59776 KB, 3670 KB/s, 16 seconds passed
... 47%, 59808 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 59840 KB, 3668 KB/s, 16 seconds passed
... 47%, 59872 KB, 3669 KB/s, 16 seconds passed
... 47%, 59904 KB, 3670 KB/s, 16 seconds passed
... 47%, 59936 KB, 3670 KB/s, 16 seconds passed
... 47%, 59968 KB, 3668 KB/s, 16 seconds passed
... 47%, 60000 KB, 3669 KB/s, 16 seconds passed
... 47%, 60032 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 60064 KB, 3670 KB/s, 16 seconds passed
... 47%, 60096 KB, 3668 KB/s, 16 seconds passed
... 47%, 60128 KB, 3669 KB/s, 16 seconds passed
... 47%, 60160 KB, 3670 KB/s, 16 seconds passed
... 47%, 60192 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 60224 KB, 3668 KB/s, 16 seconds passed
... 47%, 60256 KB, 3669 KB/s, 16 seconds passed
... 47%, 60288 KB, 3670 KB/s, 16 seconds passed
... 47%, 60320 KB, 3671 KB/s, 16 seconds passed
... 47%, 60352 KB, 3668 KB/s, 16 seconds passed
... 47%, 60384 KB, 3669 KB/s, 16 seconds passed

.. parsed-literal::

    ... 47%, 60416 KB, 3669 KB/s, 16 seconds passed
... 47%, 60448 KB, 3671 KB/s, 16 seconds passed
... 48%, 60480 KB, 3668 KB/s, 16 seconds passed
... 48%, 60512 KB, 3668 KB/s, 16 seconds passed
... 48%, 60544 KB, 3669 KB/s, 16 seconds passed
... 48%, 60576 KB, 3669 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 60608 KB, 3666 KB/s, 16 seconds passed
... 48%, 60640 KB, 3666 KB/s, 16 seconds passed
... 48%, 60672 KB, 3668 KB/s, 16 seconds passed
... 48%, 60704 KB, 3669 KB/s, 16 seconds passed
... 48%, 60736 KB, 3666 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 60768 KB, 3666 KB/s, 16 seconds passed
... 48%, 60800 KB, 3668 KB/s, 16 seconds passed
... 48%, 60832 KB, 3669 KB/s, 16 seconds passed
... 48%, 60864 KB, 3666 KB/s, 16 seconds passed
... 48%, 60896 KB, 3666 KB/s, 16 seconds passed
... 48%, 60928 KB, 3668 KB/s, 16 seconds passed
... 48%, 60960 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 60992 KB, 3666 KB/s, 16 seconds passed
... 48%, 61024 KB, 3666 KB/s, 16 seconds passed
... 48%, 61056 KB, 3668 KB/s, 16 seconds passed
... 48%, 61088 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 61120 KB, 3666 KB/s, 16 seconds passed
... 48%, 61152 KB, 3666 KB/s, 16 seconds passed
... 48%, 61184 KB, 3668 KB/s, 16 seconds passed
... 48%, 61216 KB, 3670 KB/s, 16 seconds passed
... 48%, 61248 KB, 3666 KB/s, 16 seconds passed
... 48%, 61280 KB, 3668 KB/s, 16 seconds passed
... 48%, 61312 KB, 3668 KB/s, 16 seconds passed
... 48%, 61344 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 61376 KB, 3667 KB/s, 16 seconds passed
... 48%, 61408 KB, 3666 KB/s, 16 seconds passed
... 48%, 61440 KB, 3668 KB/s, 16 seconds passed
... 48%, 61472 KB, 3668 KB/s, 16 seconds passed

.. parsed-literal::

    ... 48%, 61504 KB, 3666 KB/s, 16 seconds passed
... 48%, 61536 KB, 3668 KB/s, 16 seconds passed
... 48%, 61568 KB, 3668 KB/s, 16 seconds passed
... 48%, 61600 KB, 3670 KB/s, 16 seconds passed
... 48%, 61632 KB, 3667 KB/s, 16 seconds passed
... 48%, 61664 KB, 3668 KB/s, 16 seconds passed
... 48%, 61696 KB, 3668 KB/s, 16 seconds passed
... 49%, 61728 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 49%, 61760 KB, 3667 KB/s, 16 seconds passed
... 49%, 61792 KB, 3667 KB/s, 16 seconds passed
... 49%, 61824 KB, 3668 KB/s, 16 seconds passed
... 49%, 61856 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 49%, 61888 KB, 3667 KB/s, 16 seconds passed
... 49%, 61920 KB, 3667 KB/s, 16 seconds passed
... 49%, 61952 KB, 3668 KB/s, 16 seconds passed
... 49%, 61984 KB, 3670 KB/s, 16 seconds passed
... 49%, 62016 KB, 3667 KB/s, 16 seconds passed
... 49%, 62048 KB, 3667 KB/s, 16 seconds passed
... 49%, 62080 KB, 3668 KB/s, 16 seconds passed
... 49%, 62112 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 49%, 62144 KB, 3667 KB/s, 16 seconds passed
... 49%, 62176 KB, 3667 KB/s, 16 seconds passed
... 49%, 62208 KB, 3668 KB/s, 16 seconds passed
... 49%, 62240 KB, 3670 KB/s, 16 seconds passed

.. parsed-literal::

    ... 49%, 62272 KB, 3667 KB/s, 16 seconds passed
... 49%, 62304 KB, 3667 KB/s, 16 seconds passed
... 49%, 62336 KB, 3669 KB/s, 16 seconds passed
... 49%, 62368 KB, 3670 KB/s, 16 seconds passed
... 49%, 62400 KB, 3667 KB/s, 17 seconds passed
... 49%, 62432 KB, 3667 KB/s, 17 seconds passed
... 49%, 62464 KB, 3669 KB/s, 17 seconds passed
... 49%, 62496 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 49%, 62528 KB, 3667 KB/s, 17 seconds passed
... 49%, 62560 KB, 3667 KB/s, 17 seconds passed
... 49%, 62592 KB, 3669 KB/s, 17 seconds passed
... 49%, 62624 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 49%, 62656 KB, 3667 KB/s, 17 seconds passed
... 49%, 62688 KB, 3667 KB/s, 17 seconds passed
... 49%, 62720 KB, 3669 KB/s, 17 seconds passed
... 49%, 62752 KB, 3670 KB/s, 17 seconds passed
... 49%, 62784 KB, 3667 KB/s, 17 seconds passed
... 49%, 62816 KB, 3667 KB/s, 17 seconds passed
... 49%, 62848 KB, 3669 KB/s, 17 seconds passed

.. parsed-literal::

    ... 49%, 62880 KB, 3670 KB/s, 17 seconds passed
... 49%, 62912 KB, 3667 KB/s, 17 seconds passed
... 49%, 62944 KB, 3667 KB/s, 17 seconds passed
... 49%, 62976 KB, 3669 KB/s, 17 seconds passed
... 50%, 63008 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63040 KB, 3667 KB/s, 17 seconds passed
... 50%, 63072 KB, 3667 KB/s, 17 seconds passed
... 50%, 63104 KB, 3669 KB/s, 17 seconds passed
... 50%, 63136 KB, 3670 KB/s, 17 seconds passed
... 50%, 63168 KB, 3668 KB/s, 17 seconds passed
... 50%, 63200 KB, 3667 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63232 KB, 3669 KB/s, 17 seconds passed
... 50%, 63264 KB, 3670 KB/s, 17 seconds passed
... 50%, 63296 KB, 3668 KB/s, 17 seconds passed
... 50%, 63328 KB, 3667 KB/s, 17 seconds passed
... 50%, 63360 KB, 3669 KB/s, 17 seconds passed
... 50%, 63392 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63424 KB, 3668 KB/s, 17 seconds passed
... 50%, 63456 KB, 3667 KB/s, 17 seconds passed
... 50%, 63488 KB, 3669 KB/s, 17 seconds passed
... 50%, 63520 KB, 3670 KB/s, 17 seconds passed
... 50%, 63552 KB, 3668 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63584 KB, 3667 KB/s, 17 seconds passed
... 50%, 63616 KB, 3669 KB/s, 17 seconds passed
... 50%, 63648 KB, 3670 KB/s, 17 seconds passed
... 50%, 63680 KB, 3668 KB/s, 17 seconds passed
... 50%, 63712 KB, 3668 KB/s, 17 seconds passed
... 50%, 63744 KB, 3669 KB/s, 17 seconds passed
... 50%, 63776 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63808 KB, 3668 KB/s, 17 seconds passed
... 50%, 63840 KB, 3667 KB/s, 17 seconds passed
... 50%, 63872 KB, 3669 KB/s, 17 seconds passed
... 50%, 63904 KB, 3669 KB/s, 17 seconds passed
... 50%, 63936 KB, 3668 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 63968 KB, 3667 KB/s, 17 seconds passed
... 50%, 64000 KB, 3669 KB/s, 17 seconds passed
... 50%, 64032 KB, 3669 KB/s, 17 seconds passed
... 50%, 64064 KB, 3668 KB/s, 17 seconds passed
... 50%, 64096 KB, 3668 KB/s, 17 seconds passed
... 50%, 64128 KB, 3669 KB/s, 17 seconds passed
... 50%, 64160 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 50%, 64192 KB, 3668 KB/s, 17 seconds passed
... 50%, 64224 KB, 3668 KB/s, 17 seconds passed
... 51%, 64256 KB, 3669 KB/s, 17 seconds passed
... 51%, 64288 KB, 3670 KB/s, 17 seconds passed
... 51%, 64320 KB, 3668 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 64352 KB, 3668 KB/s, 17 seconds passed
... 51%, 64384 KB, 3669 KB/s, 17 seconds passed
... 51%, 64416 KB, 3670 KB/s, 17 seconds passed
... 51%, 64448 KB, 3668 KB/s, 17 seconds passed
... 51%, 64480 KB, 3668 KB/s, 17 seconds passed
... 51%, 64512 KB, 3669 KB/s, 17 seconds passed
... 51%, 64544 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 64576 KB, 3668 KB/s, 17 seconds passed
... 51%, 64608 KB, 3668 KB/s, 17 seconds passed
... 51%, 64640 KB, 3670 KB/s, 17 seconds passed
... 51%, 64672 KB, 3667 KB/s, 17 seconds passed
... 51%, 64704 KB, 3668 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 64736 KB, 3668 KB/s, 17 seconds passed
... 51%, 64768 KB, 3669 KB/s, 17 seconds passed
... 51%, 64800 KB, 3667 KB/s, 17 seconds passed
... 51%, 64832 KB, 3668 KB/s, 17 seconds passed
... 51%, 64864 KB, 3668 KB/s, 17 seconds passed
... 51%, 64896 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 64928 KB, 3667 KB/s, 17 seconds passed
... 51%, 64960 KB, 3668 KB/s, 17 seconds passed
... 51%, 64992 KB, 3668 KB/s, 17 seconds passed
... 51%, 65024 KB, 3670 KB/s, 17 seconds passed
... 51%, 65056 KB, 3667 KB/s, 17 seconds passed
... 51%, 65088 KB, 3669 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 65120 KB, 3668 KB/s, 17 seconds passed
... 51%, 65152 KB, 3670 KB/s, 17 seconds passed
... 51%, 65184 KB, 3670 KB/s, 17 seconds passed
... 51%, 65216 KB, 3669 KB/s, 17 seconds passed
... 51%, 65248 KB, 3668 KB/s, 17 seconds passed
... 51%, 65280 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 51%, 65312 KB, 3667 KB/s, 17 seconds passed
... 51%, 65344 KB, 3669 KB/s, 17 seconds passed
... 51%, 65376 KB, 3668 KB/s, 17 seconds passed
... 51%, 65408 KB, 3670 KB/s, 17 seconds passed
... 51%, 65440 KB, 3667 KB/s, 17 seconds passed
... 51%, 65472 KB, 3669 KB/s, 17 seconds passed

.. parsed-literal::

    ... 52%, 65504 KB, 3668 KB/s, 17 seconds passed
... 52%, 65536 KB, 3670 KB/s, 17 seconds passed
... 52%, 65568 KB, 3668 KB/s, 17 seconds passed
... 52%, 65600 KB, 3669 KB/s, 17 seconds passed
... 52%, 65632 KB, 3668 KB/s, 17 seconds passed
... 52%, 65664 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 52%, 65696 KB, 3668 KB/s, 17 seconds passed
... 52%, 65728 KB, 3669 KB/s, 17 seconds passed
... 52%, 65760 KB, 3668 KB/s, 17 seconds passed
... 52%, 65792 KB, 3670 KB/s, 17 seconds passed
... 52%, 65824 KB, 3668 KB/s, 17 seconds passed
... 52%, 65856 KB, 3669 KB/s, 17 seconds passed

.. parsed-literal::

    ... 52%, 65888 KB, 3668 KB/s, 17 seconds passed
... 52%, 65920 KB, 3670 KB/s, 17 seconds passed
... 52%, 65952 KB, 3668 KB/s, 17 seconds passed
... 52%, 65984 KB, 3669 KB/s, 17 seconds passed
... 52%, 66016 KB, 3669 KB/s, 17 seconds passed
... 52%, 66048 KB, 3670 KB/s, 17 seconds passed

.. parsed-literal::

    ... 52%, 66080 KB, 3668 KB/s, 18 seconds passed
... 52%, 66112 KB, 3667 KB/s, 18 seconds passed
... 52%, 66144 KB, 3668 KB/s, 18 seconds passed
... 52%, 66176 KB, 3670 KB/s, 18 seconds passed
... 52%, 66208 KB, 3668 KB/s, 18 seconds passed

.. parsed-literal::

    ... 52%, 66240 KB, 3667 KB/s, 18 seconds passed
... 52%, 66272 KB, 3669 KB/s, 18 seconds passed
... 52%, 66304 KB, 3670 KB/s, 18 seconds passed
... 52%, 66336 KB, 3668 KB/s, 18 seconds passed
... 52%, 66368 KB, 3668 KB/s, 18 seconds passed
... 52%, 66400 KB, 3669 KB/s, 18 seconds passed
... 52%, 66432 KB, 3670 KB/s, 18 seconds passed

.. parsed-literal::

    ... 52%, 66464 KB, 3668 KB/s, 18 seconds passed
... 52%, 66496 KB, 3668 KB/s, 18 seconds passed
... 52%, 66528 KB, 3669 KB/s, 18 seconds passed
... 52%, 66560 KB, 3670 KB/s, 18 seconds passed
... 52%, 66592 KB, 3668 KB/s, 18 seconds passed

.. parsed-literal::

    ... 52%, 66624 KB, 3668 KB/s, 18 seconds passed
... 52%, 66656 KB, 3669 KB/s, 18 seconds passed
... 52%, 66688 KB, 3670 KB/s, 18 seconds passed
... 52%, 66720 KB, 3668 KB/s, 18 seconds passed
... 52%, 66752 KB, 3668 KB/s, 18 seconds passed
... 53%, 66784 KB, 3669 KB/s, 18 seconds passed
... 53%, 66816 KB, 3670 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 66848 KB, 3668 KB/s, 18 seconds passed
... 53%, 66880 KB, 3668 KB/s, 18 seconds passed
... 53%, 66912 KB, 3669 KB/s, 18 seconds passed
... 53%, 66944 KB, 3671 KB/s, 18 seconds passed
... 53%, 66976 KB, 3668 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67008 KB, 3668 KB/s, 18 seconds passed
... 53%, 67040 KB, 3669 KB/s, 18 seconds passed
... 53%, 67072 KB, 3671 KB/s, 18 seconds passed
... 53%, 67104 KB, 3668 KB/s, 18 seconds passed
... 53%, 67136 KB, 3668 KB/s, 18 seconds passed
... 53%, 67168 KB, 3669 KB/s, 18 seconds passed
... 53%, 67200 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67232 KB, 3668 KB/s, 18 seconds passed
... 53%, 67264 KB, 3668 KB/s, 18 seconds passed
... 53%, 67296 KB, 3669 KB/s, 18 seconds passed
... 53%, 67328 KB, 3671 KB/s, 18 seconds passed
... 53%, 67360 KB, 3668 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67392 KB, 3668 KB/s, 18 seconds passed
... 53%, 67424 KB, 3669 KB/s, 18 seconds passed
... 53%, 67456 KB, 3671 KB/s, 18 seconds passed
... 53%, 67488 KB, 3668 KB/s, 18 seconds passed
... 53%, 67520 KB, 3668 KB/s, 18 seconds passed
... 53%, 67552 KB, 3669 KB/s, 18 seconds passed
... 53%, 67584 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67616 KB, 3669 KB/s, 18 seconds passed
... 53%, 67648 KB, 3669 KB/s, 18 seconds passed
... 53%, 67680 KB, 3670 KB/s, 18 seconds passed
... 53%, 67712 KB, 3671 KB/s, 18 seconds passed
... 53%, 67744 KB, 3669 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67776 KB, 3670 KB/s, 18 seconds passed
... 53%, 67808 KB, 3670 KB/s, 18 seconds passed
... 53%, 67840 KB, 3671 KB/s, 18 seconds passed
... 53%, 67872 KB, 3669 KB/s, 18 seconds passed
... 53%, 67904 KB, 3669 KB/s, 18 seconds passed
... 53%, 67936 KB, 3669 KB/s, 18 seconds passed

.. parsed-literal::

    ... 53%, 67968 KB, 3670 KB/s, 18 seconds passed
... 53%, 68000 KB, 3668 KB/s, 18 seconds passed
... 54%, 68032 KB, 3669 KB/s, 18 seconds passed
... 54%, 68064 KB, 3670 KB/s, 18 seconds passed
... 54%, 68096 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 68128 KB, 3669 KB/s, 18 seconds passed
... 54%, 68160 KB, 3669 KB/s, 18 seconds passed
... 54%, 68192 KB, 3670 KB/s, 18 seconds passed
... 54%, 68224 KB, 3671 KB/s, 18 seconds passed
... 54%, 68256 KB, 3669 KB/s, 18 seconds passed
... 54%, 68288 KB, 3669 KB/s, 18 seconds passed
... 54%, 68320 KB, 3670 KB/s, 18 seconds passed
... 54%, 68352 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 68384 KB, 3669 KB/s, 18 seconds passed
... 54%, 68416 KB, 3669 KB/s, 18 seconds passed
... 54%, 68448 KB, 3670 KB/s, 18 seconds passed
... 54%, 68480 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 68512 KB, 3669 KB/s, 18 seconds passed
... 54%, 68544 KB, 3669 KB/s, 18 seconds passed
... 54%, 68576 KB, 3670 KB/s, 18 seconds passed
... 54%, 68608 KB, 3671 KB/s, 18 seconds passed
... 54%, 68640 KB, 3669 KB/s, 18 seconds passed
... 54%, 68672 KB, 3669 KB/s, 18 seconds passed
... 54%, 68704 KB, 3670 KB/s, 18 seconds passed
... 54%, 68736 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 68768 KB, 3669 KB/s, 18 seconds passed
... 54%, 68800 KB, 3669 KB/s, 18 seconds passed
... 54%, 68832 KB, 3670 KB/s, 18 seconds passed
... 54%, 68864 KB, 3671 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 68896 KB, 3669 KB/s, 18 seconds passed
... 54%, 68928 KB, 3669 KB/s, 18 seconds passed
... 54%, 68960 KB, 3670 KB/s, 18 seconds passed
... 54%, 68992 KB, 3671 KB/s, 18 seconds passed
... 54%, 69024 KB, 3669 KB/s, 18 seconds passed
... 54%, 69056 KB, 3669 KB/s, 18 seconds passed
... 54%, 69088 KB, 3670 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 69120 KB, 3668 KB/s, 18 seconds passed
... 54%, 69152 KB, 3669 KB/s, 18 seconds passed
... 54%, 69184 KB, 3669 KB/s, 18 seconds passed
... 54%, 69216 KB, 3670 KB/s, 18 seconds passed

.. parsed-literal::

    ... 54%, 69248 KB, 3668 KB/s, 18 seconds passed
... 55%, 69280 KB, 3669 KB/s, 18 seconds passed
... 55%, 69312 KB, 3669 KB/s, 18 seconds passed
... 55%, 69344 KB, 3670 KB/s, 18 seconds passed
... 55%, 69376 KB, 3668 KB/s, 18 seconds passed
... 55%, 69408 KB, 3669 KB/s, 18 seconds passed
... 55%, 69440 KB, 3669 KB/s, 18 seconds passed

.. parsed-literal::

    ... 55%, 69472 KB, 3670 KB/s, 18 seconds passed
... 55%, 69504 KB, 3668 KB/s, 18 seconds passed
... 55%, 69536 KB, 3669 KB/s, 18 seconds passed
... 55%, 69568 KB, 3669 KB/s, 18 seconds passed
... 55%, 69600 KB, 3670 KB/s, 18 seconds passed

.. parsed-literal::

    ... 55%, 69632 KB, 3668 KB/s, 18 seconds passed
... 55%, 69664 KB, 3669 KB/s, 18 seconds passed
... 55%, 69696 KB, 3669 KB/s, 18 seconds passed
... 55%, 69728 KB, 3670 KB/s, 18 seconds passed
... 55%, 69760 KB, 3668 KB/s, 19 seconds passed
... 55%, 69792 KB, 3669 KB/s, 19 seconds passed
... 55%, 69824 KB, 3669 KB/s, 19 seconds passed

.. parsed-literal::

    ... 55%, 69856 KB, 3671 KB/s, 19 seconds passed
... 55%, 69888 KB, 3668 KB/s, 19 seconds passed
... 55%, 69920 KB, 3669 KB/s, 19 seconds passed
... 55%, 69952 KB, 3669 KB/s, 19 seconds passed
... 55%, 69984 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 55%, 70016 KB, 3668 KB/s, 19 seconds passed
... 55%, 70048 KB, 3669 KB/s, 19 seconds passed
... 55%, 70080 KB, 3670 KB/s, 19 seconds passed
... 55%, 70112 KB, 3671 KB/s, 19 seconds passed
... 55%, 70144 KB, 3668 KB/s, 19 seconds passed
... 55%, 70176 KB, 3669 KB/s, 19 seconds passed
... 55%, 70208 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 55%, 70240 KB, 3671 KB/s, 19 seconds passed
... 55%, 70272 KB, 3668 KB/s, 19 seconds passed
... 55%, 70304 KB, 3669 KB/s, 19 seconds passed
... 55%, 70336 KB, 3670 KB/s, 19 seconds passed
... 55%, 70368 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 55%, 70400 KB, 3669 KB/s, 19 seconds passed
... 55%, 70432 KB, 3669 KB/s, 19 seconds passed
... 55%, 70464 KB, 3670 KB/s, 19 seconds passed
... 55%, 70496 KB, 3671 KB/s, 19 seconds passed
... 55%, 70528 KB, 3669 KB/s, 19 seconds passed
... 56%, 70560 KB, 3669 KB/s, 19 seconds passed
... 56%, 70592 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 70624 KB, 3671 KB/s, 19 seconds passed
... 56%, 70656 KB, 3669 KB/s, 19 seconds passed
... 56%, 70688 KB, 3670 KB/s, 19 seconds passed
... 56%, 70720 KB, 3670 KB/s, 19 seconds passed
... 56%, 70752 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 70784 KB, 3669 KB/s, 19 seconds passed
... 56%, 70816 KB, 3669 KB/s, 19 seconds passed
... 56%, 70848 KB, 3670 KB/s, 19 seconds passed
... 56%, 70880 KB, 3671 KB/s, 19 seconds passed
... 56%, 70912 KB, 3669 KB/s, 19 seconds passed
... 56%, 70944 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 70976 KB, 3670 KB/s, 19 seconds passed
... 56%, 71008 KB, 3671 KB/s, 19 seconds passed
... 56%, 71040 KB, 3669 KB/s, 19 seconds passed
... 56%, 71072 KB, 3670 KB/s, 19 seconds passed
... 56%, 71104 KB, 3670 KB/s, 19 seconds passed
... 56%, 71136 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 71168 KB, 3669 KB/s, 19 seconds passed
... 56%, 71200 KB, 3670 KB/s, 19 seconds passed
... 56%, 71232 KB, 3670 KB/s, 19 seconds passed
... 56%, 71264 KB, 3671 KB/s, 19 seconds passed
... 56%, 71296 KB, 3669 KB/s, 19 seconds passed
... 56%, 71328 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 71360 KB, 3670 KB/s, 19 seconds passed
... 56%, 71392 KB, 3671 KB/s, 19 seconds passed
... 56%, 71424 KB, 3669 KB/s, 19 seconds passed
... 56%, 71456 KB, 3670 KB/s, 19 seconds passed
... 56%, 71488 KB, 3670 KB/s, 19 seconds passed
... 56%, 71520 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 71552 KB, 3669 KB/s, 19 seconds passed
... 56%, 71584 KB, 3670 KB/s, 19 seconds passed
... 56%, 71616 KB, 3670 KB/s, 19 seconds passed
... 56%, 71648 KB, 3671 KB/s, 19 seconds passed
... 56%, 71680 KB, 3669 KB/s, 19 seconds passed
... 56%, 71712 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 56%, 71744 KB, 3670 KB/s, 19 seconds passed
... 56%, 71776 KB, 3671 KB/s, 19 seconds passed
... 57%, 71808 KB, 3669 KB/s, 19 seconds passed
... 57%, 71840 KB, 3670 KB/s, 19 seconds passed
... 57%, 71872 KB, 3670 KB/s, 19 seconds passed
... 57%, 71904 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 71936 KB, 3669 KB/s, 19 seconds passed
... 57%, 71968 KB, 3669 KB/s, 19 seconds passed
... 57%, 72000 KB, 3670 KB/s, 19 seconds passed
... 57%, 72032 KB, 3668 KB/s, 19 seconds passed
... 57%, 72064 KB, 3669 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 72096 KB, 3669 KB/s, 19 seconds passed
... 57%, 72128 KB, 3670 KB/s, 19 seconds passed
... 57%, 72160 KB, 3668 KB/s, 19 seconds passed
... 57%, 72192 KB, 3669 KB/s, 19 seconds passed
... 57%, 72224 KB, 3669 KB/s, 19 seconds passed
... 57%, 72256 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 72288 KB, 3668 KB/s, 19 seconds passed
... 57%, 72320 KB, 3669 KB/s, 19 seconds passed
... 57%, 72352 KB, 3669 KB/s, 19 seconds passed
... 57%, 72384 KB, 3670 KB/s, 19 seconds passed
... 57%, 72416 KB, 3669 KB/s, 19 seconds passed
... 57%, 72448 KB, 3669 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 72480 KB, 3670 KB/s, 19 seconds passed
... 57%, 72512 KB, 3670 KB/s, 19 seconds passed
... 57%, 72544 KB, 3672 KB/s, 19 seconds passed
... 57%, 72576 KB, 3669 KB/s, 19 seconds passed
... 57%, 72608 KB, 3670 KB/s, 19 seconds passed
... 57%, 72640 KB, 3670 KB/s, 19 seconds passed
... 57%, 72672 KB, 3672 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 72704 KB, 3669 KB/s, 19 seconds passed
... 57%, 72736 KB, 3669 KB/s, 19 seconds passed
... 57%, 72768 KB, 3670 KB/s, 19 seconds passed
... 57%, 72800 KB, 3669 KB/s, 19 seconds passed
... 57%, 72832 KB, 3669 KB/s, 19 seconds passed

.. parsed-literal::

    ... 57%, 72864 KB, 3669 KB/s, 19 seconds passed
... 57%, 72896 KB, 3671 KB/s, 19 seconds passed
... 57%, 72928 KB, 3669 KB/s, 19 seconds passed
... 57%, 72960 KB, 3669 KB/s, 19 seconds passed
... 57%, 72992 KB, 3670 KB/s, 19 seconds passed
... 57%, 73024 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 58%, 73056 KB, 3669 KB/s, 19 seconds passed
... 58%, 73088 KB, 3669 KB/s, 19 seconds passed
... 58%, 73120 KB, 3670 KB/s, 19 seconds passed
... 58%, 73152 KB, 3671 KB/s, 19 seconds passed
... 58%, 73184 KB, 3669 KB/s, 19 seconds passed
... 58%, 73216 KB, 3670 KB/s, 19 seconds passed

.. parsed-literal::

    ... 58%, 73248 KB, 3670 KB/s, 19 seconds passed
... 58%, 73280 KB, 3671 KB/s, 19 seconds passed
... 58%, 73312 KB, 3669 KB/s, 19 seconds passed
... 58%, 73344 KB, 3670 KB/s, 19 seconds passed
... 58%, 73376 KB, 3670 KB/s, 19 seconds passed
... 58%, 73408 KB, 3671 KB/s, 19 seconds passed

.. parsed-literal::

    ... 58%, 73440 KB, 3669 KB/s, 20 seconds passed
... 58%, 73472 KB, 3669 KB/s, 20 seconds passed
... 58%, 73504 KB, 3670 KB/s, 20 seconds passed
... 58%, 73536 KB, 3671 KB/s, 20 seconds passed
... 58%, 73568 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 58%, 73600 KB, 3670 KB/s, 20 seconds passed
... 58%, 73632 KB, 3670 KB/s, 20 seconds passed
... 58%, 73664 KB, 3671 KB/s, 20 seconds passed
... 58%, 73696 KB, 3669 KB/s, 20 seconds passed
... 58%, 73728 KB, 3670 KB/s, 20 seconds passed
... 58%, 73760 KB, 3670 KB/s, 20 seconds passed
... 58%, 73792 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 58%, 73824 KB, 3669 KB/s, 20 seconds passed
... 58%, 73856 KB, 3670 KB/s, 20 seconds passed
... 58%, 73888 KB, 3670 KB/s, 20 seconds passed
... 58%, 73920 KB, 3671 KB/s, 20 seconds passed
... 58%, 73952 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 58%, 73984 KB, 3670 KB/s, 20 seconds passed
... 58%, 74016 KB, 3670 KB/s, 20 seconds passed
... 58%, 74048 KB, 3671 KB/s, 20 seconds passed
... 58%, 74080 KB, 3669 KB/s, 20 seconds passed
... 58%, 74112 KB, 3670 KB/s, 20 seconds passed
... 58%, 74144 KB, 3670 KB/s, 20 seconds passed
... 58%, 74176 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 58%, 74208 KB, 3669 KB/s, 20 seconds passed
... 58%, 74240 KB, 3670 KB/s, 20 seconds passed
... 58%, 74272 KB, 3670 KB/s, 20 seconds passed
... 58%, 74304 KB, 3671 KB/s, 20 seconds passed
... 59%, 74336 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 74368 KB, 3670 KB/s, 20 seconds passed
... 59%, 74400 KB, 3670 KB/s, 20 seconds passed
... 59%, 74432 KB, 3671 KB/s, 20 seconds passed
... 59%, 74464 KB, 3669 KB/s, 20 seconds passed
... 59%, 74496 KB, 3670 KB/s, 20 seconds passed
... 59%, 74528 KB, 3670 KB/s, 20 seconds passed
... 59%, 74560 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 74592 KB, 3669 KB/s, 20 seconds passed
... 59%, 74624 KB, 3670 KB/s, 20 seconds passed
... 59%, 74656 KB, 3670 KB/s, 20 seconds passed
... 59%, 74688 KB, 3671 KB/s, 20 seconds passed
... 59%, 74720 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 74752 KB, 3670 KB/s, 20 seconds passed
... 59%, 74784 KB, 3670 KB/s, 20 seconds passed
... 59%, 74816 KB, 3671 KB/s, 20 seconds passed
... 59%, 74848 KB, 3669 KB/s, 20 seconds passed
... 59%, 74880 KB, 3670 KB/s, 20 seconds passed
... 59%, 74912 KB, 3670 KB/s, 20 seconds passed
... 59%, 74944 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 74976 KB, 3669 KB/s, 20 seconds passed
... 59%, 75008 KB, 3670 KB/s, 20 seconds passed
... 59%, 75040 KB, 3670 KB/s, 20 seconds passed
... 59%, 75072 KB, 3672 KB/s, 20 seconds passed
... 59%, 75104 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 75136 KB, 3670 KB/s, 20 seconds passed
... 59%, 75168 KB, 3670 KB/s, 20 seconds passed
... 59%, 75200 KB, 3672 KB/s, 20 seconds passed
... 59%, 75232 KB, 3669 KB/s, 20 seconds passed
... 59%, 75264 KB, 3670 KB/s, 20 seconds passed
... 59%, 75296 KB, 3670 KB/s, 20 seconds passed
... 59%, 75328 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 75360 KB, 3670 KB/s, 20 seconds passed
... 59%, 75392 KB, 3670 KB/s, 20 seconds passed
... 59%, 75424 KB, 3671 KB/s, 20 seconds passed
... 59%, 75456 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 59%, 75488 KB, 3670 KB/s, 20 seconds passed
... 59%, 75520 KB, 3670 KB/s, 20 seconds passed
... 59%, 75552 KB, 3671 KB/s, 20 seconds passed
... 60%, 75584 KB, 3672 KB/s, 20 seconds passed
... 60%, 75616 KB, 3670 KB/s, 20 seconds passed
... 60%, 75648 KB, 3671 KB/s, 20 seconds passed
... 60%, 75680 KB, 3671 KB/s, 20 seconds passed
... 60%, 75712 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 75744 KB, 3670 KB/s, 20 seconds passed
... 60%, 75776 KB, 3671 KB/s, 20 seconds passed
... 60%, 75808 KB, 3671 KB/s, 20 seconds passed
... 60%, 75840 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 75872 KB, 3670 KB/s, 20 seconds passed
... 60%, 75904 KB, 3671 KB/s, 20 seconds passed
... 60%, 75936 KB, 3671 KB/s, 20 seconds passed
... 60%, 75968 KB, 3669 KB/s, 20 seconds passed
... 60%, 76000 KB, 3670 KB/s, 20 seconds passed
... 60%, 76032 KB, 3670 KB/s, 20 seconds passed
... 60%, 76064 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 76096 KB, 3669 KB/s, 20 seconds passed
... 60%, 76128 KB, 3670 KB/s, 20 seconds passed
... 60%, 76160 KB, 3670 KB/s, 20 seconds passed
... 60%, 76192 KB, 3671 KB/s, 20 seconds passed
... 60%, 76224 KB, 3669 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 76256 KB, 3670 KB/s, 20 seconds passed
... 60%, 76288 KB, 3670 KB/s, 20 seconds passed
... 60%, 76320 KB, 3671 KB/s, 20 seconds passed
... 60%, 76352 KB, 3669 KB/s, 20 seconds passed
... 60%, 76384 KB, 3670 KB/s, 20 seconds passed
... 60%, 76416 KB, 3670 KB/s, 20 seconds passed
... 60%, 76448 KB, 3671 KB/s, 20 seconds passed
... 60%, 76480 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 76512 KB, 3670 KB/s, 20 seconds passed
... 60%, 76544 KB, 3671 KB/s, 20 seconds passed
... 60%, 76576 KB, 3671 KB/s, 20 seconds passed
... 60%, 76608 KB, 3672 KB/s, 20 seconds passed

.. parsed-literal::

    ... 60%, 76640 KB, 3670 KB/s, 20 seconds passed
... 60%, 76672 KB, 3671 KB/s, 20 seconds passed
... 60%, 76704 KB, 3671 KB/s, 20 seconds passed
... 60%, 76736 KB, 3670 KB/s, 20 seconds passed
... 60%, 76768 KB, 3670 KB/s, 20 seconds passed
... 60%, 76800 KB, 3671 KB/s, 20 seconds passed
... 61%, 76832 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 61%, 76864 KB, 3670 KB/s, 20 seconds passed
... 61%, 76896 KB, 3670 KB/s, 20 seconds passed
... 61%, 76928 KB, 3670 KB/s, 20 seconds passed
... 61%, 76960 KB, 3671 KB/s, 20 seconds passed

.. parsed-literal::

    ... 61%, 76992 KB, 3670 KB/s, 20 seconds passed
... 61%, 77024 KB, 3670 KB/s, 20 seconds passed
... 61%, 77056 KB, 3670 KB/s, 20 seconds passed
... 61%, 77088 KB, 3671 KB/s, 20 seconds passed
... 61%, 77120 KB, 3670 KB/s, 21 seconds passed
... 61%, 77152 KB, 3670 KB/s, 21 seconds passed
... 61%, 77184 KB, 3670 KB/s, 21 seconds passed

.. parsed-literal::

    ... 61%, 77216 KB, 3671 KB/s, 21 seconds passed
... 61%, 77248 KB, 3670 KB/s, 21 seconds passed
... 61%, 77280 KB, 3670 KB/s, 21 seconds passed
... 61%, 77312 KB, 3670 KB/s, 21 seconds passed
... 61%, 77344 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 61%, 77376 KB, 3670 KB/s, 21 seconds passed
... 61%, 77408 KB, 3670 KB/s, 21 seconds passed
... 61%, 77440 KB, 3670 KB/s, 21 seconds passed
... 61%, 77472 KB, 3671 KB/s, 21 seconds passed
... 61%, 77504 KB, 3670 KB/s, 21 seconds passed
... 61%, 77536 KB, 3670 KB/s, 21 seconds passed
... 61%, 77568 KB, 3670 KB/s, 21 seconds passed

.. parsed-literal::

    ... 61%, 77600 KB, 3671 KB/s, 21 seconds passed
... 61%, 77632 KB, 3670 KB/s, 21 seconds passed
... 61%, 77664 KB, 3670 KB/s, 21 seconds passed
... 61%, 77696 KB, 3670 KB/s, 21 seconds passed
... 61%, 77728 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 61%, 77760 KB, 3670 KB/s, 21 seconds passed
... 61%, 77792 KB, 3670 KB/s, 21 seconds passed
... 61%, 77824 KB, 3671 KB/s, 21 seconds passed
... 61%, 77856 KB, 3671 KB/s, 21 seconds passed
... 61%, 77888 KB, 3670 KB/s, 21 seconds passed
... 61%, 77920 KB, 3671 KB/s, 21 seconds passed
... 61%, 77952 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 61%, 77984 KB, 3672 KB/s, 21 seconds passed
... 61%, 78016 KB, 3670 KB/s, 21 seconds passed
... 61%, 78048 KB, 3671 KB/s, 21 seconds passed
... 61%, 78080 KB, 3671 KB/s, 21 seconds passed
... 62%, 78112 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 78144 KB, 3670 KB/s, 21 seconds passed
... 62%, 78176 KB, 3671 KB/s, 21 seconds passed
... 62%, 78208 KB, 3671 KB/s, 21 seconds passed
... 62%, 78240 KB, 3671 KB/s, 21 seconds passed
... 62%, 78272 KB, 3670 KB/s, 21 seconds passed
... 62%, 78304 KB, 3671 KB/s, 21 seconds passed
... 62%, 78336 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 78368 KB, 3671 KB/s, 21 seconds passed
... 62%, 78400 KB, 3670 KB/s, 21 seconds passed
... 62%, 78432 KB, 3670 KB/s, 21 seconds passed
... 62%, 78464 KB, 3671 KB/s, 21 seconds passed
... 62%, 78496 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 78528 KB, 3670 KB/s, 21 seconds passed
... 62%, 78560 KB, 3671 KB/s, 21 seconds passed
... 62%, 78592 KB, 3671 KB/s, 21 seconds passed
... 62%, 78624 KB, 3672 KB/s, 21 seconds passed
... 62%, 78656 KB, 3670 KB/s, 21 seconds passed
... 62%, 78688 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 78720 KB, 3671 KB/s, 21 seconds passed
... 62%, 78752 KB, 3672 KB/s, 21 seconds passed
... 62%, 78784 KB, 3670 KB/s, 21 seconds passed
... 62%, 78816 KB, 3671 KB/s, 21 seconds passed
... 62%, 78848 KB, 3671 KB/s, 21 seconds passed
... 62%, 78880 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 78912 KB, 3670 KB/s, 21 seconds passed
... 62%, 78944 KB, 3671 KB/s, 21 seconds passed
... 62%, 78976 KB, 3671 KB/s, 21 seconds passed
... 62%, 79008 KB, 3672 KB/s, 21 seconds passed
... 62%, 79040 KB, 3670 KB/s, 21 seconds passed
... 62%, 79072 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 79104 KB, 3671 KB/s, 21 seconds passed
... 62%, 79136 KB, 3672 KB/s, 21 seconds passed
... 62%, 79168 KB, 3670 KB/s, 21 seconds passed
... 62%, 79200 KB, 3671 KB/s, 21 seconds passed
... 62%, 79232 KB, 3671 KB/s, 21 seconds passed
... 62%, 79264 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 62%, 79296 KB, 3670 KB/s, 21 seconds passed
... 62%, 79328 KB, 3671 KB/s, 21 seconds passed
... 63%, 79360 KB, 3671 KB/s, 21 seconds passed
... 63%, 79392 KB, 3672 KB/s, 21 seconds passed
... 63%, 79424 KB, 3670 KB/s, 21 seconds passed
... 63%, 79456 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 79488 KB, 3671 KB/s, 21 seconds passed
... 63%, 79520 KB, 3672 KB/s, 21 seconds passed
... 63%, 79552 KB, 3670 KB/s, 21 seconds passed
... 63%, 79584 KB, 3671 KB/s, 21 seconds passed
... 63%, 79616 KB, 3671 KB/s, 21 seconds passed
... 63%, 79648 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 79680 KB, 3670 KB/s, 21 seconds passed
... 63%, 79712 KB, 3671 KB/s, 21 seconds passed
... 63%, 79744 KB, 3671 KB/s, 21 seconds passed
... 63%, 79776 KB, 3672 KB/s, 21 seconds passed
... 63%, 79808 KB, 3670 KB/s, 21 seconds passed
... 63%, 79840 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 79872 KB, 3671 KB/s, 21 seconds passed
... 63%, 79904 KB, 3672 KB/s, 21 seconds passed
... 63%, 79936 KB, 3670 KB/s, 21 seconds passed
... 63%, 79968 KB, 3671 KB/s, 21 seconds passed
... 63%, 80000 KB, 3672 KB/s, 21 seconds passed
... 63%, 80032 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 80064 KB, 3670 KB/s, 21 seconds passed
... 63%, 80096 KB, 3671 KB/s, 21 seconds passed
... 63%, 80128 KB, 3671 KB/s, 21 seconds passed
... 63%, 80160 KB, 3672 KB/s, 21 seconds passed
... 63%, 80192 KB, 3670 KB/s, 21 seconds passed
... 63%, 80224 KB, 3671 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 80256 KB, 3671 KB/s, 21 seconds passed
... 63%, 80288 KB, 3672 KB/s, 21 seconds passed
... 63%, 80320 KB, 3670 KB/s, 21 seconds passed
... 63%, 80352 KB, 3671 KB/s, 21 seconds passed
... 63%, 80384 KB, 3671 KB/s, 21 seconds passed
... 63%, 80416 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 63%, 80448 KB, 3670 KB/s, 21 seconds passed
... 63%, 80480 KB, 3671 KB/s, 21 seconds passed
... 63%, 80512 KB, 3671 KB/s, 21 seconds passed
... 63%, 80544 KB, 3670 KB/s, 21 seconds passed
... 63%, 80576 KB, 3670 KB/s, 21 seconds passed
... 63%, 80608 KB, 3672 KB/s, 21 seconds passed

.. parsed-literal::

    ... 64%, 80640 KB, 3672 KB/s, 21 seconds passed
... 64%, 80672 KB, 3672 KB/s, 21 seconds passed
... 64%, 80704 KB, 3671 KB/s, 21 seconds passed
... 64%, 80736 KB, 3672 KB/s, 21 seconds passed
... 64%, 80768 KB, 3672 KB/s, 21 seconds passed
... 64%, 80800 KB, 3673 KB/s, 21 seconds passed

.. parsed-literal::

    ... 64%, 80832 KB, 3670 KB/s, 22 seconds passed
... 64%, 80864 KB, 3671 KB/s, 22 seconds passed
... 64%, 80896 KB, 3672 KB/s, 22 seconds passed
... 64%, 80928 KB, 3670 KB/s, 22 seconds passed
... 64%, 80960 KB, 3670 KB/s, 22 seconds passed

.. parsed-literal::

    ... 64%, 80992 KB, 3672 KB/s, 22 seconds passed
... 64%, 81024 KB, 3672 KB/s, 22 seconds passed
... 64%, 81056 KB, 3670 KB/s, 22 seconds passed
... 64%, 81088 KB, 3670 KB/s, 22 seconds passed
... 64%, 81120 KB, 3672 KB/s, 22 seconds passed
... 64%, 81152 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 64%, 81184 KB, 3670 KB/s, 22 seconds passed
... 64%, 81216 KB, 3671 KB/s, 22 seconds passed
... 64%, 81248 KB, 3672 KB/s, 22 seconds passed
... 64%, 81280 KB, 3672 KB/s, 22 seconds passed
... 64%, 81312 KB, 3673 KB/s, 22 seconds passed
... 64%, 81344 KB, 3671 KB/s, 22 seconds passed

.. parsed-literal::

    ... 64%, 81376 KB, 3672 KB/s, 22 seconds passed
... 64%, 81408 KB, 3672 KB/s, 22 seconds passed
... 64%, 81440 KB, 3671 KB/s, 22 seconds passed
... 64%, 81472 KB, 3671 KB/s, 22 seconds passed
... 64%, 81504 KB, 3672 KB/s, 22 seconds passed
... 64%, 81536 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 64%, 81568 KB, 3671 KB/s, 22 seconds passed
... 64%, 81600 KB, 3671 KB/s, 22 seconds passed
... 64%, 81632 KB, 3671 KB/s, 22 seconds passed
... 64%, 81664 KB, 3672 KB/s, 22 seconds passed
... 64%, 81696 KB, 3670 KB/s, 22 seconds passed

.. parsed-literal::

    ... 64%, 81728 KB, 3671 KB/s, 22 seconds passed
... 64%, 81760 KB, 3672 KB/s, 22 seconds passed
... 64%, 81792 KB, 3672 KB/s, 22 seconds passed
... 64%, 81824 KB, 3671 KB/s, 22 seconds passed
... 64%, 81856 KB, 3671 KB/s, 22 seconds passed
... 65%, 81888 KB, 3672 KB/s, 22 seconds passed
... 65%, 81920 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 81952 KB, 3670 KB/s, 22 seconds passed
... 65%, 81984 KB, 3671 KB/s, 22 seconds passed
... 65%, 82016 KB, 3672 KB/s, 22 seconds passed
... 65%, 82048 KB, 3672 KB/s, 22 seconds passed
... 65%, 82080 KB, 3671 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 82112 KB, 3671 KB/s, 22 seconds passed
... 65%, 82144 KB, 3672 KB/s, 22 seconds passed
... 65%, 82176 KB, 3672 KB/s, 22 seconds passed
... 65%, 82208 KB, 3671 KB/s, 22 seconds passed
... 65%, 82240 KB, 3671 KB/s, 22 seconds passed
... 65%, 82272 KB, 3672 KB/s, 22 seconds passed
... 65%, 82304 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 82336 KB, 3671 KB/s, 22 seconds passed
... 65%, 82368 KB, 3671 KB/s, 22 seconds passed
... 65%, 82400 KB, 3672 KB/s, 22 seconds passed
... 65%, 82432 KB, 3672 KB/s, 22 seconds passed
... 65%, 82464 KB, 3671 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 82496 KB, 3671 KB/s, 22 seconds passed
... 65%, 82528 KB, 3672 KB/s, 22 seconds passed
... 65%, 82560 KB, 3672 KB/s, 22 seconds passed
... 65%, 82592 KB, 3671 KB/s, 22 seconds passed
... 65%, 82624 KB, 3671 KB/s, 22 seconds passed
... 65%, 82656 KB, 3672 KB/s, 22 seconds passed
... 65%, 82688 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 82720 KB, 3671 KB/s, 22 seconds passed
... 65%, 82752 KB, 3671 KB/s, 22 seconds passed
... 65%, 82784 KB, 3672 KB/s, 22 seconds passed
... 65%, 82816 KB, 3672 KB/s, 22 seconds passed
... 65%, 82848 KB, 3671 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 82880 KB, 3671 KB/s, 22 seconds passed
... 65%, 82912 KB, 3672 KB/s, 22 seconds passed
... 65%, 82944 KB, 3672 KB/s, 22 seconds passed
... 65%, 82976 KB, 3671 KB/s, 22 seconds passed
... 65%, 83008 KB, 3671 KB/s, 22 seconds passed
... 65%, 83040 KB, 3672 KB/s, 22 seconds passed
... 65%, 83072 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 65%, 83104 KB, 3671 KB/s, 22 seconds passed
... 66%, 83136 KB, 3671 KB/s, 22 seconds passed
... 66%, 83168 KB, 3672 KB/s, 22 seconds passed
... 66%, 83200 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 83232 KB, 3671 KB/s, 22 seconds passed
... 66%, 83264 KB, 3671 KB/s, 22 seconds passed
... 66%, 83296 KB, 3672 KB/s, 22 seconds passed
... 66%, 83328 KB, 3672 KB/s, 22 seconds passed
... 66%, 83360 KB, 3671 KB/s, 22 seconds passed
... 66%, 83392 KB, 3671 KB/s, 22 seconds passed
... 66%, 83424 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 83456 KB, 3672 KB/s, 22 seconds passed
... 66%, 83488 KB, 3671 KB/s, 22 seconds passed
... 66%, 83520 KB, 3671 KB/s, 22 seconds passed
... 66%, 83552 KB, 3672 KB/s, 22 seconds passed
... 66%, 83584 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 83616 KB, 3671 KB/s, 22 seconds passed
... 66%, 83648 KB, 3671 KB/s, 22 seconds passed
... 66%, 83680 KB, 3672 KB/s, 22 seconds passed
... 66%, 83712 KB, 3672 KB/s, 22 seconds passed
... 66%, 83744 KB, 3671 KB/s, 22 seconds passed
... 66%, 83776 KB, 3671 KB/s, 22 seconds passed
... 66%, 83808 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 83840 KB, 3672 KB/s, 22 seconds passed
... 66%, 83872 KB, 3671 KB/s, 22 seconds passed
... 66%, 83904 KB, 3672 KB/s, 22 seconds passed
... 66%, 83936 KB, 3673 KB/s, 22 seconds passed
... 66%, 83968 KB, 3673 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 84000 KB, 3671 KB/s, 22 seconds passed
... 66%, 84032 KB, 3671 KB/s, 22 seconds passed
... 66%, 84064 KB, 3672 KB/s, 22 seconds passed
... 66%, 84096 KB, 3672 KB/s, 22 seconds passed
... 66%, 84128 KB, 3671 KB/s, 22 seconds passed
... 66%, 84160 KB, 3671 KB/s, 22 seconds passed
... 66%, 84192 KB, 3672 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 84224 KB, 3672 KB/s, 22 seconds passed
... 66%, 84256 KB, 3671 KB/s, 22 seconds passed
... 66%, 84288 KB, 3671 KB/s, 22 seconds passed
... 66%, 84320 KB, 3672 KB/s, 22 seconds passed
... 66%, 84352 KB, 3673 KB/s, 22 seconds passed

.. parsed-literal::

    ... 66%, 84384 KB, 3671 KB/s, 22 seconds passed
... 67%, 84416 KB, 3672 KB/s, 22 seconds passed
... 67%, 84448 KB, 3673 KB/s, 22 seconds passed
... 67%, 84480 KB, 3673 KB/s, 22 seconds passed
... 67%, 84512 KB, 3671 KB/s, 23 seconds passed
... 67%, 84544 KB, 3672 KB/s, 23 seconds passed
... 67%, 84576 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 84608 KB, 3673 KB/s, 23 seconds passed
... 67%, 84640 KB, 3672 KB/s, 23 seconds passed
... 67%, 84672 KB, 3672 KB/s, 23 seconds passed
... 67%, 84704 KB, 3673 KB/s, 23 seconds passed
... 67%, 84736 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 84768 KB, 3672 KB/s, 23 seconds passed
... 67%, 84800 KB, 3672 KB/s, 23 seconds passed
... 67%, 84832 KB, 3672 KB/s, 23 seconds passed
... 67%, 84864 KB, 3671 KB/s, 23 seconds passed
... 67%, 84896 KB, 3671 KB/s, 23 seconds passed
... 67%, 84928 KB, 3672 KB/s, 23 seconds passed
... 67%, 84960 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 84992 KB, 3671 KB/s, 23 seconds passed
... 67%, 85024 KB, 3671 KB/s, 23 seconds passed
... 67%, 85056 KB, 3672 KB/s, 23 seconds passed
... 67%, 85088 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 85120 KB, 3671 KB/s, 23 seconds passed
... 67%, 85152 KB, 3671 KB/s, 23 seconds passed
... 67%, 85184 KB, 3672 KB/s, 23 seconds passed
... 67%, 85216 KB, 3672 KB/s, 23 seconds passed
... 67%, 85248 KB, 3673 KB/s, 23 seconds passed
... 67%, 85280 KB, 3671 KB/s, 23 seconds passed
... 67%, 85312 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 85344 KB, 3673 KB/s, 23 seconds passed
... 67%, 85376 KB, 3671 KB/s, 23 seconds passed
... 67%, 85408 KB, 3671 KB/s, 23 seconds passed
... 67%, 85440 KB, 3672 KB/s, 23 seconds passed
... 67%, 85472 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 67%, 85504 KB, 3671 KB/s, 23 seconds passed
... 67%, 85536 KB, 3671 KB/s, 23 seconds passed
... 67%, 85568 KB, 3672 KB/s, 23 seconds passed
... 67%, 85600 KB, 3673 KB/s, 23 seconds passed
... 67%, 85632 KB, 3671 KB/s, 23 seconds passed
... 68%, 85664 KB, 3671 KB/s, 23 seconds passed
... 68%, 85696 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 85728 KB, 3673 KB/s, 23 seconds passed
... 68%, 85760 KB, 3671 KB/s, 23 seconds passed
... 68%, 85792 KB, 3671 KB/s, 23 seconds passed
... 68%, 85824 KB, 3672 KB/s, 23 seconds passed
... 68%, 85856 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 85888 KB, 3671 KB/s, 23 seconds passed
... 68%, 85920 KB, 3671 KB/s, 23 seconds passed
... 68%, 85952 KB, 3672 KB/s, 23 seconds passed
... 68%, 85984 KB, 3673 KB/s, 23 seconds passed
... 68%, 86016 KB, 3671 KB/s, 23 seconds passed
... 68%, 86048 KB, 3671 KB/s, 23 seconds passed
... 68%, 86080 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 86112 KB, 3673 KB/s, 23 seconds passed
... 68%, 86144 KB, 3671 KB/s, 23 seconds passed
... 68%, 86176 KB, 3672 KB/s, 23 seconds passed
... 68%, 86208 KB, 3672 KB/s, 23 seconds passed
... 68%, 86240 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 86272 KB, 3671 KB/s, 23 seconds passed
... 68%, 86304 KB, 3671 KB/s, 23 seconds passed
... 68%, 86336 KB, 3672 KB/s, 23 seconds passed
... 68%, 86368 KB, 3673 KB/s, 23 seconds passed
... 68%, 86400 KB, 3671 KB/s, 23 seconds passed
... 68%, 86432 KB, 3672 KB/s, 23 seconds passed
... 68%, 86464 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 86496 KB, 3673 KB/s, 23 seconds passed
... 68%, 86528 KB, 3672 KB/s, 23 seconds passed
... 68%, 86560 KB, 3672 KB/s, 23 seconds passed
... 68%, 86592 KB, 3673 KB/s, 23 seconds passed
... 68%, 86624 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 86656 KB, 3672 KB/s, 23 seconds passed
... 68%, 86688 KB, 3672 KB/s, 23 seconds passed
... 68%, 86720 KB, 3673 KB/s, 23 seconds passed
... 68%, 86752 KB, 3673 KB/s, 23 seconds passed
... 68%, 86784 KB, 3672 KB/s, 23 seconds passed
... 68%, 86816 KB, 3672 KB/s, 23 seconds passed
... 68%, 86848 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 68%, 86880 KB, 3673 KB/s, 23 seconds passed
... 69%, 86912 KB, 3672 KB/s, 23 seconds passed
... 69%, 86944 KB, 3672 KB/s, 23 seconds passed
... 69%, 86976 KB, 3673 KB/s, 23 seconds passed
... 69%, 87008 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 87040 KB, 3672 KB/s, 23 seconds passed
... 69%, 87072 KB, 3672 KB/s, 23 seconds passed
... 69%, 87104 KB, 3673 KB/s, 23 seconds passed
... 69%, 87136 KB, 3673 KB/s, 23 seconds passed
... 69%, 87168 KB, 3672 KB/s, 23 seconds passed
... 69%, 87200 KB, 3672 KB/s, 23 seconds passed
... 69%, 87232 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 87264 KB, 3673 KB/s, 23 seconds passed
... 69%, 87296 KB, 3672 KB/s, 23 seconds passed
... 69%, 87328 KB, 3672 KB/s, 23 seconds passed
... 69%, 87360 KB, 3673 KB/s, 23 seconds passed
... 69%, 87392 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 87424 KB, 3672 KB/s, 23 seconds passed
... 69%, 87456 KB, 3672 KB/s, 23 seconds passed
... 69%, 87488 KB, 3673 KB/s, 23 seconds passed
... 69%, 87520 KB, 3673 KB/s, 23 seconds passed
... 69%, 87552 KB, 3672 KB/s, 23 seconds passed
... 69%, 87584 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 87616 KB, 3673 KB/s, 23 seconds passed
... 69%, 87648 KB, 3673 KB/s, 23 seconds passed
... 69%, 87680 KB, 3672 KB/s, 23 seconds passed
... 69%, 87712 KB, 3672 KB/s, 23 seconds passed
... 69%, 87744 KB, 3673 KB/s, 23 seconds passed
... 69%, 87776 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 87808 KB, 3672 KB/s, 23 seconds passed
... 69%, 87840 KB, 3672 KB/s, 23 seconds passed
... 69%, 87872 KB, 3673 KB/s, 23 seconds passed
... 69%, 87904 KB, 3673 KB/s, 23 seconds passed
... 69%, 87936 KB, 3672 KB/s, 23 seconds passed
... 69%, 87968 KB, 3672 KB/s, 23 seconds passed

.. parsed-literal::

    ... 69%, 88000 KB, 3673 KB/s, 23 seconds passed
... 69%, 88032 KB, 3673 KB/s, 23 seconds passed
... 69%, 88064 KB, 3672 KB/s, 23 seconds passed
... 69%, 88096 KB, 3672 KB/s, 23 seconds passed
... 69%, 88128 KB, 3673 KB/s, 23 seconds passed
... 69%, 88160 KB, 3673 KB/s, 23 seconds passed

.. parsed-literal::

    ... 70%, 88192 KB, 3672 KB/s, 24 seconds passed
... 70%, 88224 KB, 3672 KB/s, 24 seconds passed
... 70%, 88256 KB, 3673 KB/s, 24 seconds passed
... 70%, 88288 KB, 3673 KB/s, 24 seconds passed
... 70%, 88320 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 88352 KB, 3672 KB/s, 24 seconds passed
... 70%, 88384 KB, 3673 KB/s, 24 seconds passed
... 70%, 88416 KB, 3673 KB/s, 24 seconds passed
... 70%, 88448 KB, 3672 KB/s, 24 seconds passed
... 70%, 88480 KB, 3672 KB/s, 24 seconds passed
... 70%, 88512 KB, 3673 KB/s, 24 seconds passed
... 70%, 88544 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 88576 KB, 3672 KB/s, 24 seconds passed
... 70%, 88608 KB, 3672 KB/s, 24 seconds passed
... 70%, 88640 KB, 3673 KB/s, 24 seconds passed
... 70%, 88672 KB, 3673 KB/s, 24 seconds passed
... 70%, 88704 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 88736 KB, 3672 KB/s, 24 seconds passed
... 70%, 88768 KB, 3673 KB/s, 24 seconds passed
... 70%, 88800 KB, 3673 KB/s, 24 seconds passed
... 70%, 88832 KB, 3672 KB/s, 24 seconds passed
... 70%, 88864 KB, 3672 KB/s, 24 seconds passed
... 70%, 88896 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 88928 KB, 3671 KB/s, 24 seconds passed
... 70%, 88960 KB, 3672 KB/s, 24 seconds passed
... 70%, 88992 KB, 3672 KB/s, 24 seconds passed
... 70%, 89024 KB, 3673 KB/s, 24 seconds passed
... 70%, 89056 KB, 3671 KB/s, 24 seconds passed
... 70%, 89088 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 89120 KB, 3672 KB/s, 24 seconds passed
... 70%, 89152 KB, 3673 KB/s, 24 seconds passed
... 70%, 89184 KB, 3672 KB/s, 24 seconds passed
... 70%, 89216 KB, 3672 KB/s, 24 seconds passed
... 70%, 89248 KB, 3672 KB/s, 24 seconds passed
... 70%, 89280 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 70%, 89312 KB, 3672 KB/s, 24 seconds passed
... 70%, 89344 KB, 3672 KB/s, 24 seconds passed
... 70%, 89376 KB, 3672 KB/s, 24 seconds passed
... 70%, 89408 KB, 3673 KB/s, 24 seconds passed
... 71%, 89440 KB, 3672 KB/s, 24 seconds passed
... 71%, 89472 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 89504 KB, 3672 KB/s, 24 seconds passed
... 71%, 89536 KB, 3673 KB/s, 24 seconds passed
... 71%, 89568 KB, 3672 KB/s, 24 seconds passed
... 71%, 89600 KB, 3672 KB/s, 24 seconds passed
... 71%, 89632 KB, 3672 KB/s, 24 seconds passed
... 71%, 89664 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 89696 KB, 3671 KB/s, 24 seconds passed
... 71%, 89728 KB, 3672 KB/s, 24 seconds passed
... 71%, 89760 KB, 3672 KB/s, 24 seconds passed
... 71%, 89792 KB, 3673 KB/s, 24 seconds passed
... 71%, 89824 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 89856 KB, 3672 KB/s, 24 seconds passed
... 71%, 89888 KB, 3673 KB/s, 24 seconds passed
... 71%, 89920 KB, 3673 KB/s, 24 seconds passed
... 71%, 89952 KB, 3672 KB/s, 24 seconds passed
... 71%, 89984 KB, 3672 KB/s, 24 seconds passed
... 71%, 90016 KB, 3673 KB/s, 24 seconds passed
... 71%, 90048 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 90080 KB, 3672 KB/s, 24 seconds passed
... 71%, 90112 KB, 3672 KB/s, 24 seconds passed
... 71%, 90144 KB, 3673 KB/s, 24 seconds passed
... 71%, 90176 KB, 3673 KB/s, 24 seconds passed
... 71%, 90208 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 90240 KB, 3672 KB/s, 24 seconds passed
... 71%, 90272 KB, 3673 KB/s, 24 seconds passed
... 71%, 90304 KB, 3673 KB/s, 24 seconds passed
... 71%, 90336 KB, 3672 KB/s, 24 seconds passed
... 71%, 90368 KB, 3672 KB/s, 24 seconds passed
... 71%, 90400 KB, 3673 KB/s, 24 seconds passed
... 71%, 90432 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 90464 KB, 3672 KB/s, 24 seconds passed
... 71%, 90496 KB, 3672 KB/s, 24 seconds passed
... 71%, 90528 KB, 3673 KB/s, 24 seconds passed
... 71%, 90560 KB, 3674 KB/s, 24 seconds passed
... 71%, 90592 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 71%, 90624 KB, 3672 KB/s, 24 seconds passed
... 71%, 90656 KB, 3673 KB/s, 24 seconds passed
... 72%, 90688 KB, 3673 KB/s, 24 seconds passed
... 72%, 90720 KB, 3672 KB/s, 24 seconds passed
... 72%, 90752 KB, 3672 KB/s, 24 seconds passed
... 72%, 90784 KB, 3673 KB/s, 24 seconds passed
... 72%, 90816 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 90848 KB, 3672 KB/s, 24 seconds passed
... 72%, 90880 KB, 3672 KB/s, 24 seconds passed
... 72%, 90912 KB, 3673 KB/s, 24 seconds passed
... 72%, 90944 KB, 3674 KB/s, 24 seconds passed
... 72%, 90976 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 91008 KB, 3672 KB/s, 24 seconds passed
... 72%, 91040 KB, 3673 KB/s, 24 seconds passed
... 72%, 91072 KB, 3674 KB/s, 24 seconds passed
... 72%, 91104 KB, 3672 KB/s, 24 seconds passed
... 72%, 91136 KB, 3672 KB/s, 24 seconds passed
... 72%, 91168 KB, 3673 KB/s, 24 seconds passed
... 72%, 91200 KB, 3674 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 91232 KB, 3672 KB/s, 24 seconds passed
... 72%, 91264 KB, 3672 KB/s, 24 seconds passed
... 72%, 91296 KB, 3673 KB/s, 24 seconds passed
... 72%, 91328 KB, 3673 KB/s, 24 seconds passed
... 72%, 91360 KB, 3672 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 91392 KB, 3672 KB/s, 24 seconds passed
... 72%, 91424 KB, 3673 KB/s, 24 seconds passed
... 72%, 91456 KB, 3673 KB/s, 24 seconds passed
... 72%, 91488 KB, 3672 KB/s, 24 seconds passed
... 72%, 91520 KB, 3672 KB/s, 24 seconds passed
... 72%, 91552 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 91584 KB, 3673 KB/s, 24 seconds passed
... 72%, 91616 KB, 3672 KB/s, 24 seconds passed
... 72%, 91648 KB, 3672 KB/s, 24 seconds passed
... 72%, 91680 KB, 3673 KB/s, 24 seconds passed
... 72%, 91712 KB, 3673 KB/s, 24 seconds passed

.. parsed-literal::

    ... 72%, 91744 KB, 3672 KB/s, 24 seconds passed
... 72%, 91776 KB, 3672 KB/s, 24 seconds passed
... 72%, 91808 KB, 3673 KB/s, 24 seconds passed
... 72%, 91840 KB, 3673 KB/s, 25 seconds passed
... 72%, 91872 KB, 3672 KB/s, 25 seconds passed
... 72%, 91904 KB, 3672 KB/s, 25 seconds passed
... 72%, 91936 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 91968 KB, 3673 KB/s, 25 seconds passed
... 73%, 92000 KB, 3672 KB/s, 25 seconds passed
... 73%, 92032 KB, 3672 KB/s, 25 seconds passed
... 73%, 92064 KB, 3673 KB/s, 25 seconds passed
... 73%, 92096 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 92128 KB, 3672 KB/s, 25 seconds passed
... 73%, 92160 KB, 3672 KB/s, 25 seconds passed
... 73%, 92192 KB, 3673 KB/s, 25 seconds passed
... 73%, 92224 KB, 3673 KB/s, 25 seconds passed
... 73%, 92256 KB, 3672 KB/s, 25 seconds passed
... 73%, 92288 KB, 3672 KB/s, 25 seconds passed
... 73%, 92320 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 92352 KB, 3673 KB/s, 25 seconds passed
... 73%, 92384 KB, 3672 KB/s, 25 seconds passed
... 73%, 92416 KB, 3672 KB/s, 25 seconds passed
... 73%, 92448 KB, 3673 KB/s, 25 seconds passed
... 73%, 92480 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 92512 KB, 3672 KB/s, 25 seconds passed
... 73%, 92544 KB, 3672 KB/s, 25 seconds passed
... 73%, 92576 KB, 3673 KB/s, 25 seconds passed
... 73%, 92608 KB, 3673 KB/s, 25 seconds passed
... 73%, 92640 KB, 3672 KB/s, 25 seconds passed
... 73%, 92672 KB, 3672 KB/s, 25 seconds passed
... 73%, 92704 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 92736 KB, 3673 KB/s, 25 seconds passed
... 73%, 92768 KB, 3672 KB/s, 25 seconds passed
... 73%, 92800 KB, 3672 KB/s, 25 seconds passed
... 73%, 92832 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 92864 KB, 3671 KB/s, 25 seconds passed
... 73%, 92896 KB, 3671 KB/s, 25 seconds passed
... 73%, 92928 KB, 3672 KB/s, 25 seconds passed
... 73%, 92960 KB, 3673 KB/s, 25 seconds passed
... 73%, 92992 KB, 3671 KB/s, 25 seconds passed
... 73%, 93024 KB, 3672 KB/s, 25 seconds passed
... 73%, 93056 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 73%, 93088 KB, 3673 KB/s, 25 seconds passed
... 73%, 93120 KB, 3672 KB/s, 25 seconds passed
... 73%, 93152 KB, 3672 KB/s, 25 seconds passed
... 73%, 93184 KB, 3672 KB/s, 25 seconds passed
... 74%, 93216 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 93248 KB, 3672 KB/s, 25 seconds passed
... 74%, 93280 KB, 3672 KB/s, 25 seconds passed
... 74%, 93312 KB, 3672 KB/s, 25 seconds passed
... 74%, 93344 KB, 3673 KB/s, 25 seconds passed
... 74%, 93376 KB, 3673 KB/s, 25 seconds passed
... 74%, 93408 KB, 3672 KB/s, 25 seconds passed
... 74%, 93440 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 93472 KB, 3673 KB/s, 25 seconds passed
... 74%, 93504 KB, 3672 KB/s, 25 seconds passed
... 74%, 93536 KB, 3672 KB/s, 25 seconds passed
... 74%, 93568 KB, 3672 KB/s, 25 seconds passed
... 74%, 93600 KB, 3673 KB/s, 25 seconds passed
... 74%, 93632 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 93664 KB, 3672 KB/s, 25 seconds passed
... 74%, 93696 KB, 3673 KB/s, 25 seconds passed
... 74%, 93728 KB, 3673 KB/s, 25 seconds passed
... 74%, 93760 KB, 3674 KB/s, 25 seconds passed
... 74%, 93792 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 93824 KB, 3672 KB/s, 25 seconds passed
... 74%, 93856 KB, 3673 KB/s, 25 seconds passed
... 74%, 93888 KB, 3672 KB/s, 25 seconds passed
... 74%, 93920 KB, 3672 KB/s, 25 seconds passed
... 74%, 93952 KB, 3673 KB/s, 25 seconds passed
... 74%, 93984 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 94016 KB, 3672 KB/s, 25 seconds passed
... 74%, 94048 KB, 3672 KB/s, 25 seconds passed
... 74%, 94080 KB, 3673 KB/s, 25 seconds passed
... 74%, 94112 KB, 3673 KB/s, 25 seconds passed
... 74%, 94144 KB, 3672 KB/s, 25 seconds passed
... 74%, 94176 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 94208 KB, 3673 KB/s, 25 seconds passed
... 74%, 94240 KB, 3673 KB/s, 25 seconds passed
... 74%, 94272 KB, 3672 KB/s, 25 seconds passed
... 74%, 94304 KB, 3672 KB/s, 25 seconds passed
... 74%, 94336 KB, 3673 KB/s, 25 seconds passed
... 74%, 94368 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 74%, 94400 KB, 3672 KB/s, 25 seconds passed
... 74%, 94432 KB, 3673 KB/s, 25 seconds passed
... 74%, 94464 KB, 3673 KB/s, 25 seconds passed
... 75%, 94496 KB, 3674 KB/s, 25 seconds passed
... 75%, 94528 KB, 3672 KB/s, 25 seconds passed
... 75%, 94560 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 75%, 94592 KB, 3673 KB/s, 25 seconds passed
... 75%, 94624 KB, 3673 KB/s, 25 seconds passed
... 75%, 94656 KB, 3672 KB/s, 25 seconds passed
... 75%, 94688 KB, 3672 KB/s, 25 seconds passed
... 75%, 94720 KB, 3673 KB/s, 25 seconds passed
... 75%, 94752 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 75%, 94784 KB, 3672 KB/s, 25 seconds passed
... 75%, 94816 KB, 3672 KB/s, 25 seconds passed
... 75%, 94848 KB, 3673 KB/s, 25 seconds passed
... 75%, 94880 KB, 3673 KB/s, 25 seconds passed
... 75%, 94912 KB, 3672 KB/s, 25 seconds passed
... 75%, 94944 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 75%, 94976 KB, 3673 KB/s, 25 seconds passed
... 75%, 95008 KB, 3673 KB/s, 25 seconds passed
... 75%, 95040 KB, 3672 KB/s, 25 seconds passed
... 75%, 95072 KB, 3672 KB/s, 25 seconds passed
... 75%, 95104 KB, 3673 KB/s, 25 seconds passed
... 75%, 95136 KB, 3673 KB/s, 25 seconds passed

.. parsed-literal::

    ... 75%, 95168 KB, 3672 KB/s, 25 seconds passed
... 75%, 95200 KB, 3672 KB/s, 25 seconds passed
... 75%, 95232 KB, 3673 KB/s, 25 seconds passed
... 75%, 95264 KB, 3673 KB/s, 25 seconds passed
... 75%, 95296 KB, 3672 KB/s, 25 seconds passed

.. parsed-literal::

    ... 75%, 95328 KB, 3672 KB/s, 25 seconds passed
... 75%, 95360 KB, 3673 KB/s, 25 seconds passed
... 75%, 95392 KB, 3673 KB/s, 25 seconds passed
... 75%, 95424 KB, 3672 KB/s, 25 seconds passed
... 75%, 95456 KB, 3672 KB/s, 25 seconds passed
... 75%, 95488 KB, 3673 KB/s, 25 seconds passed
... 75%, 95520 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 75%, 95552 KB, 3672 KB/s, 26 seconds passed
... 75%, 95584 KB, 3672 KB/s, 26 seconds passed
... 75%, 95616 KB, 3673 KB/s, 26 seconds passed
... 75%, 95648 KB, 3673 KB/s, 26 seconds passed
... 75%, 95680 KB, 3672 KB/s, 26 seconds passed

.. parsed-literal::

    ... 75%, 95712 KB, 3672 KB/s, 26 seconds passed
... 76%, 95744 KB, 3673 KB/s, 26 seconds passed
... 76%, 95776 KB, 3673 KB/s, 26 seconds passed
... 76%, 95808 KB, 3672 KB/s, 26 seconds passed
... 76%, 95840 KB, 3672 KB/s, 26 seconds passed
... 76%, 95872 KB, 3673 KB/s, 26 seconds passed
... 76%, 95904 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 95936 KB, 3672 KB/s, 26 seconds passed
... 76%, 95968 KB, 3672 KB/s, 26 seconds passed
... 76%, 96000 KB, 3673 KB/s, 26 seconds passed
... 76%, 96032 KB, 3673 KB/s, 26 seconds passed
... 76%, 96064 KB, 3672 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 96096 KB, 3672 KB/s, 26 seconds passed
... 76%, 96128 KB, 3673 KB/s, 26 seconds passed
... 76%, 96160 KB, 3673 KB/s, 26 seconds passed
... 76%, 96192 KB, 3672 KB/s, 26 seconds passed
... 76%, 96224 KB, 3672 KB/s, 26 seconds passed
... 76%, 96256 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 96288 KB, 3673 KB/s, 26 seconds passed
... 76%, 96320 KB, 3672 KB/s, 26 seconds passed
... 76%, 96352 KB, 3672 KB/s, 26 seconds passed
... 76%, 96384 KB, 3673 KB/s, 26 seconds passed
... 76%, 96416 KB, 3673 KB/s, 26 seconds passed
... 76%, 96448 KB, 3672 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 96480 KB, 3672 KB/s, 26 seconds passed
... 76%, 96512 KB, 3673 KB/s, 26 seconds passed
... 76%, 96544 KB, 3673 KB/s, 26 seconds passed
... 76%, 96576 KB, 3672 KB/s, 26 seconds passed
... 76%, 96608 KB, 3672 KB/s, 26 seconds passed
... 76%, 96640 KB, 3673 KB/s, 26 seconds passed
... 76%, 96672 KB, 3674 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 96704 KB, 3672 KB/s, 26 seconds passed
... 76%, 96736 KB, 3673 KB/s, 26 seconds passed
... 76%, 96768 KB, 3673 KB/s, 26 seconds passed
... 76%, 96800 KB, 3674 KB/s, 26 seconds passed
... 76%, 96832 KB, 3672 KB/s, 26 seconds passed

.. parsed-literal::

    ... 76%, 96864 KB, 3673 KB/s, 26 seconds passed
... 76%, 96896 KB, 3673 KB/s, 26 seconds passed
... 76%, 96928 KB, 3673 KB/s, 26 seconds passed
... 76%, 96960 KB, 3672 KB/s, 26 seconds passed
... 77%, 96992 KB, 3672 KB/s, 26 seconds passed
... 77%, 97024 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97056 KB, 3673 KB/s, 26 seconds passed
... 77%, 97088 KB, 3672 KB/s, 26 seconds passed
... 77%, 97120 KB, 3672 KB/s, 26 seconds passed
... 77%, 97152 KB, 3673 KB/s, 26 seconds passed
... 77%, 97184 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97216 KB, 3672 KB/s, 26 seconds passed
... 77%, 97248 KB, 3673 KB/s, 26 seconds passed
... 77%, 97280 KB, 3673 KB/s, 26 seconds passed
... 77%, 97312 KB, 3673 KB/s, 26 seconds passed
... 77%, 97344 KB, 3672 KB/s, 26 seconds passed
... 77%, 97376 KB, 3673 KB/s, 26 seconds passed
... 77%, 97408 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97440 KB, 3673 KB/s, 26 seconds passed
... 77%, 97472 KB, 3672 KB/s, 26 seconds passed
... 77%, 97504 KB, 3673 KB/s, 26 seconds passed
... 77%, 97536 KB, 3674 KB/s, 26 seconds passed
... 77%, 97568 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97600 KB, 3673 KB/s, 26 seconds passed
... 77%, 97632 KB, 3673 KB/s, 26 seconds passed
... 77%, 97664 KB, 3673 KB/s, 26 seconds passed
... 77%, 97696 KB, 3672 KB/s, 26 seconds passed
... 77%, 97728 KB, 3672 KB/s, 26 seconds passed
... 77%, 97760 KB, 3673 KB/s, 26 seconds passed
... 77%, 97792 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97824 KB, 3672 KB/s, 26 seconds passed
... 77%, 97856 KB, 3672 KB/s, 26 seconds passed
... 77%, 97888 KB, 3673 KB/s, 26 seconds passed
... 77%, 97920 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 97952 KB, 3672 KB/s, 26 seconds passed
... 77%, 97984 KB, 3672 KB/s, 26 seconds passed
... 77%, 98016 KB, 3673 KB/s, 26 seconds passed
... 77%, 98048 KB, 3673 KB/s, 26 seconds passed
... 77%, 98080 KB, 3672 KB/s, 26 seconds passed
... 77%, 98112 KB, 3672 KB/s, 26 seconds passed
... 77%, 98144 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 77%, 98176 KB, 3673 KB/s, 26 seconds passed
... 77%, 98208 KB, 3671 KB/s, 26 seconds passed
... 77%, 98240 KB, 3672 KB/s, 26 seconds passed
... 78%, 98272 KB, 3673 KB/s, 26 seconds passed
... 78%, 98304 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 78%, 98336 KB, 3671 KB/s, 26 seconds passed
... 78%, 98368 KB, 3672 KB/s, 26 seconds passed
... 78%, 98400 KB, 3672 KB/s, 26 seconds passed
... 78%, 98432 KB, 3673 KB/s, 26 seconds passed
... 78%, 98464 KB, 3671 KB/s, 26 seconds passed
... 78%, 98496 KB, 3672 KB/s, 26 seconds passed
... 78%, 98528 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 78%, 98560 KB, 3673 KB/s, 26 seconds passed
... 78%, 98592 KB, 3671 KB/s, 26 seconds passed
... 78%, 98624 KB, 3672 KB/s, 26 seconds passed
... 78%, 98656 KB, 3673 KB/s, 26 seconds passed
... 78%, 98688 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 78%, 98720 KB, 3671 KB/s, 26 seconds passed
... 78%, 98752 KB, 3672 KB/s, 26 seconds passed
... 78%, 98784 KB, 3673 KB/s, 26 seconds passed
... 78%, 98816 KB, 3673 KB/s, 26 seconds passed
... 78%, 98848 KB, 3671 KB/s, 26 seconds passed
... 78%, 98880 KB, 3672 KB/s, 26 seconds passed
... 78%, 98912 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 78%, 98944 KB, 3673 KB/s, 26 seconds passed
... 78%, 98976 KB, 3671 KB/s, 26 seconds passed
... 78%, 99008 KB, 3672 KB/s, 26 seconds passed
... 78%, 99040 KB, 3673 KB/s, 26 seconds passed
... 78%, 99072 KB, 3673 KB/s, 26 seconds passed

.. parsed-literal::

    ... 78%, 99104 KB, 3672 KB/s, 26 seconds passed
... 78%, 99136 KB, 3672 KB/s, 26 seconds passed
... 78%, 99168 KB, 3673 KB/s, 26 seconds passed
... 78%, 99200 KB, 3673 KB/s, 27 seconds passed
... 78%, 99232 KB, 3672 KB/s, 27 seconds passed
... 78%, 99264 KB, 3672 KB/s, 27 seconds passed
... 78%, 99296 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 78%, 99328 KB, 3673 KB/s, 27 seconds passed
... 78%, 99360 KB, 3671 KB/s, 27 seconds passed
... 78%, 99392 KB, 3672 KB/s, 27 seconds passed
... 78%, 99424 KB, 3673 KB/s, 27 seconds passed
... 78%, 99456 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 78%, 99488 KB, 3671 KB/s, 27 seconds passed
... 79%, 99520 KB, 3672 KB/s, 27 seconds passed
... 79%, 99552 KB, 3673 KB/s, 27 seconds passed
... 79%, 99584 KB, 3673 KB/s, 27 seconds passed
... 79%, 99616 KB, 3671 KB/s, 27 seconds passed
... 79%, 99648 KB, 3672 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 99680 KB, 3673 KB/s, 27 seconds passed
... 79%, 99712 KB, 3674 KB/s, 27 seconds passed
... 79%, 99744 KB, 3672 KB/s, 27 seconds passed
... 79%, 99776 KB, 3672 KB/s, 27 seconds passed
... 79%, 99808 KB, 3673 KB/s, 27 seconds passed
... 79%, 99840 KB, 3674 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 99872 KB, 3672 KB/s, 27 seconds passed
... 79%, 99904 KB, 3673 KB/s, 27 seconds passed
... 79%, 99936 KB, 3673 KB/s, 27 seconds passed
... 79%, 99968 KB, 3673 KB/s, 27 seconds passed
... 79%, 100000 KB, 3672 KB/s, 27 seconds passed
... 79%, 100032 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 100064 KB, 3673 KB/s, 27 seconds passed
... 79%, 100096 KB, 3673 KB/s, 27 seconds passed
... 79%, 100128 KB, 3672 KB/s, 27 seconds passed
... 79%, 100160 KB, 3673 KB/s, 27 seconds passed
... 79%, 100192 KB, 3673 KB/s, 27 seconds passed
... 79%, 100224 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 100256 KB, 3672 KB/s, 27 seconds passed
... 79%, 100288 KB, 3673 KB/s, 27 seconds passed
... 79%, 100320 KB, 3673 KB/s, 27 seconds passed
... 79%, 100352 KB, 3673 KB/s, 27 seconds passed
... 79%, 100384 KB, 3672 KB/s, 27 seconds passed
... 79%, 100416 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 100448 KB, 3673 KB/s, 27 seconds passed
... 79%, 100480 KB, 3673 KB/s, 27 seconds passed
... 79%, 100512 KB, 3672 KB/s, 27 seconds passed
... 79%, 100544 KB, 3673 KB/s, 27 seconds passed
... 79%, 100576 KB, 3673 KB/s, 27 seconds passed
... 79%, 100608 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 79%, 100640 KB, 3672 KB/s, 27 seconds passed
... 79%, 100672 KB, 3673 KB/s, 27 seconds passed
... 79%, 100704 KB, 3673 KB/s, 27 seconds passed
... 79%, 100736 KB, 3673 KB/s, 27 seconds passed
... 80%, 100768 KB, 3672 KB/s, 27 seconds passed
... 80%, 100800 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 100832 KB, 3673 KB/s, 27 seconds passed
... 80%, 100864 KB, 3673 KB/s, 27 seconds passed
... 80%, 100896 KB, 3672 KB/s, 27 seconds passed
... 80%, 100928 KB, 3673 KB/s, 27 seconds passed
... 80%, 100960 KB, 3673 KB/s, 27 seconds passed
... 80%, 100992 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101024 KB, 3672 KB/s, 27 seconds passed
... 80%, 101056 KB, 3673 KB/s, 27 seconds passed
... 80%, 101088 KB, 3673 KB/s, 27 seconds passed
... 80%, 101120 KB, 3673 KB/s, 27 seconds passed
... 80%, 101152 KB, 3672 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101184 KB, 3673 KB/s, 27 seconds passed
... 80%, 101216 KB, 3673 KB/s, 27 seconds passed
... 80%, 101248 KB, 3673 KB/s, 27 seconds passed
... 80%, 101280 KB, 3672 KB/s, 27 seconds passed
... 80%, 101312 KB, 3673 KB/s, 27 seconds passed
... 80%, 101344 KB, 3673 KB/s, 27 seconds passed
... 80%, 101376 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101408 KB, 3672 KB/s, 27 seconds passed
... 80%, 101440 KB, 3673 KB/s, 27 seconds passed
... 80%, 101472 KB, 3673 KB/s, 27 seconds passed
... 80%, 101504 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101536 KB, 3672 KB/s, 27 seconds passed
... 80%, 101568 KB, 3673 KB/s, 27 seconds passed
... 80%, 101600 KB, 3673 KB/s, 27 seconds passed
... 80%, 101632 KB, 3672 KB/s, 27 seconds passed
... 80%, 101664 KB, 3672 KB/s, 27 seconds passed
... 80%, 101696 KB, 3673 KB/s, 27 seconds passed
... 80%, 101728 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101760 KB, 3672 KB/s, 27 seconds passed
... 80%, 101792 KB, 3672 KB/s, 27 seconds passed
... 80%, 101824 KB, 3673 KB/s, 27 seconds passed
... 80%, 101856 KB, 3673 KB/s, 27 seconds passed
... 80%, 101888 KB, 3672 KB/s, 27 seconds passed

.. parsed-literal::

    ... 80%, 101920 KB, 3672 KB/s, 27 seconds passed
... 80%, 101952 KB, 3673 KB/s, 27 seconds passed
... 80%, 101984 KB, 3673 KB/s, 27 seconds passed
... 80%, 102016 KB, 3672 KB/s, 27 seconds passed
... 81%, 102048 KB, 3672 KB/s, 27 seconds passed
... 81%, 102080 KB, 3673 KB/s, 27 seconds passed
... 81%, 102112 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 81%, 102144 KB, 3672 KB/s, 27 seconds passed
... 81%, 102176 KB, 3672 KB/s, 27 seconds passed
... 81%, 102208 KB, 3673 KB/s, 27 seconds passed
... 81%, 102240 KB, 3673 KB/s, 27 seconds passed
... 81%, 102272 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 81%, 102304 KB, 3672 KB/s, 27 seconds passed
... 81%, 102336 KB, 3673 KB/s, 27 seconds passed
... 81%, 102368 KB, 3673 KB/s, 27 seconds passed
... 81%, 102400 KB, 3672 KB/s, 27 seconds passed
... 81%, 102432 KB, 3672 KB/s, 27 seconds passed
... 81%, 102464 KB, 3673 KB/s, 27 seconds passed
... 81%, 102496 KB, 3673 KB/s, 27 seconds passed

.. parsed-literal::

    ... 81%, 102528 KB, 3673 KB/s, 27 seconds passed
... 81%, 102560 KB, 3672 KB/s, 27 seconds passed
... 81%, 102592 KB, 3673 KB/s, 27 seconds passed
... 81%, 102624 KB, 3673 KB/s, 27 seconds passed
... 81%, 102656 KB, 3672 KB/s, 27 seconds passed

.. parsed-literal::

    ... 81%, 102688 KB, 3672 KB/s, 27 seconds passed
... 81%, 102720 KB, 3673 KB/s, 27 seconds passed
... 81%, 102752 KB, 3673 KB/s, 27 seconds passed
... 81%, 102784 KB, 3674 KB/s, 27 seconds passed
... 81%, 102816 KB, 3672 KB/s, 27 seconds passed
... 81%, 102848 KB, 3673 KB/s, 27 seconds passed
... 81%, 102880 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 81%, 102912 KB, 3673 KB/s, 28 seconds passed
... 81%, 102944 KB, 3672 KB/s, 28 seconds passed
... 81%, 102976 KB, 3673 KB/s, 28 seconds passed
... 81%, 103008 KB, 3673 KB/s, 28 seconds passed
... 81%, 103040 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 81%, 103072 KB, 3672 KB/s, 28 seconds passed
... 81%, 103104 KB, 3673 KB/s, 28 seconds passed
... 81%, 103136 KB, 3674 KB/s, 28 seconds passed
... 81%, 103168 KB, 3673 KB/s, 28 seconds passed
... 81%, 103200 KB, 3672 KB/s, 28 seconds passed
... 81%, 103232 KB, 3673 KB/s, 28 seconds passed
... 81%, 103264 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 103296 KB, 3673 KB/s, 28 seconds passed
... 82%, 103328 KB, 3672 KB/s, 28 seconds passed
... 82%, 103360 KB, 3673 KB/s, 28 seconds passed
... 82%, 103392 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 103424 KB, 3672 KB/s, 28 seconds passed
... 82%, 103456 KB, 3672 KB/s, 28 seconds passed
... 82%, 103488 KB, 3673 KB/s, 28 seconds passed
... 82%, 103520 KB, 3673 KB/s, 28 seconds passed
... 82%, 103552 KB, 3673 KB/s, 28 seconds passed
... 82%, 103584 KB, 3672 KB/s, 28 seconds passed
... 82%, 103616 KB, 3673 KB/s, 28 seconds passed
... 82%, 103648 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 103680 KB, 3672 KB/s, 28 seconds passed
... 82%, 103712 KB, 3672 KB/s, 28 seconds passed
... 82%, 103744 KB, 3673 KB/s, 28 seconds passed
... 82%, 103776 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 103808 KB, 3672 KB/s, 28 seconds passed
... 82%, 103840 KB, 3672 KB/s, 28 seconds passed
... 82%, 103872 KB, 3673 KB/s, 28 seconds passed
... 82%, 103904 KB, 3673 KB/s, 28 seconds passed
... 82%, 103936 KB, 3672 KB/s, 28 seconds passed
... 82%, 103968 KB, 3672 KB/s, 28 seconds passed
... 82%, 104000 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 104032 KB, 3673 KB/s, 28 seconds passed
... 82%, 104064 KB, 3672 KB/s, 28 seconds passed
... 82%, 104096 KB, 3672 KB/s, 28 seconds passed
... 82%, 104128 KB, 3673 KB/s, 28 seconds passed
... 82%, 104160 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 104192 KB, 3672 KB/s, 28 seconds passed
... 82%, 104224 KB, 3672 KB/s, 28 seconds passed
... 82%, 104256 KB, 3673 KB/s, 28 seconds passed
... 82%, 104288 KB, 3673 KB/s, 28 seconds passed
... 82%, 104320 KB, 3672 KB/s, 28 seconds passed
... 82%, 104352 KB, 3673 KB/s, 28 seconds passed
... 82%, 104384 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 82%, 104416 KB, 3673 KB/s, 28 seconds passed
... 82%, 104448 KB, 3672 KB/s, 28 seconds passed
... 82%, 104480 KB, 3673 KB/s, 28 seconds passed
... 82%, 104512 KB, 3673 KB/s, 28 seconds passed
... 83%, 104544 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 104576 KB, 3672 KB/s, 28 seconds passed
... 83%, 104608 KB, 3673 KB/s, 28 seconds passed
... 83%, 104640 KB, 3674 KB/s, 28 seconds passed
... 83%, 104672 KB, 3674 KB/s, 28 seconds passed
... 83%, 104704 KB, 3672 KB/s, 28 seconds passed
... 83%, 104736 KB, 3673 KB/s, 28 seconds passed
... 83%, 104768 KB, 3674 KB/s, 28 seconds passed
... 83%, 104800 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 104832 KB, 3672 KB/s, 28 seconds passed
... 83%, 104864 KB, 3673 KB/s, 28 seconds passed
... 83%, 104896 KB, 3674 KB/s, 28 seconds passed
... 83%, 104928 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 104960 KB, 3672 KB/s, 28 seconds passed
... 83%, 104992 KB, 3673 KB/s, 28 seconds passed
... 83%, 105024 KB, 3674 KB/s, 28 seconds passed
... 83%, 105056 KB, 3674 KB/s, 28 seconds passed
... 83%, 105088 KB, 3672 KB/s, 28 seconds passed
... 83%, 105120 KB, 3673 KB/s, 28 seconds passed
... 83%, 105152 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 105184 KB, 3674 KB/s, 28 seconds passed
... 83%, 105216 KB, 3672 KB/s, 28 seconds passed
... 83%, 105248 KB, 3673 KB/s, 28 seconds passed
... 83%, 105280 KB, 3674 KB/s, 28 seconds passed
... 83%, 105312 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 105344 KB, 3672 KB/s, 28 seconds passed
... 83%, 105376 KB, 3673 KB/s, 28 seconds passed
... 83%, 105408 KB, 3674 KB/s, 28 seconds passed
... 83%, 105440 KB, 3672 KB/s, 28 seconds passed
... 83%, 105472 KB, 3672 KB/s, 28 seconds passed
... 83%, 105504 KB, 3673 KB/s, 28 seconds passed
... 83%, 105536 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 105568 KB, 3673 KB/s, 28 seconds passed
... 83%, 105600 KB, 3672 KB/s, 28 seconds passed
... 83%, 105632 KB, 3673 KB/s, 28 seconds passed
... 83%, 105664 KB, 3674 KB/s, 28 seconds passed
... 83%, 105696 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 83%, 105728 KB, 3672 KB/s, 28 seconds passed
... 83%, 105760 KB, 3673 KB/s, 28 seconds passed
... 83%, 105792 KB, 3674 KB/s, 28 seconds passed
... 84%, 105824 KB, 3673 KB/s, 28 seconds passed
... 84%, 105856 KB, 3672 KB/s, 28 seconds passed
... 84%, 105888 KB, 3673 KB/s, 28 seconds passed
... 84%, 105920 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 84%, 105952 KB, 3673 KB/s, 28 seconds passed
... 84%, 105984 KB, 3672 KB/s, 28 seconds passed
... 84%, 106016 KB, 3673 KB/s, 28 seconds passed
... 84%, 106048 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 84%, 106080 KB, 3673 KB/s, 28 seconds passed
... 84%, 106112 KB, 3672 KB/s, 28 seconds passed
... 84%, 106144 KB, 3673 KB/s, 28 seconds passed
... 84%, 106176 KB, 3674 KB/s, 28 seconds passed
... 84%, 106208 KB, 3673 KB/s, 28 seconds passed
... 84%, 106240 KB, 3672 KB/s, 28 seconds passed
... 84%, 106272 KB, 3673 KB/s, 28 seconds passed

.. parsed-literal::

    ... 84%, 106304 KB, 3674 KB/s, 28 seconds passed
... 84%, 106336 KB, 3674 KB/s, 28 seconds passed
... 84%, 106368 KB, 3672 KB/s, 28 seconds passed
... 84%, 106400 KB, 3673 KB/s, 28 seconds passed
... 84%, 106432 KB, 3674 KB/s, 28 seconds passed
... 84%, 106464 KB, 3674 KB/s, 28 seconds passed

.. parsed-literal::

    ... 84%, 106496 KB, 3672 KB/s, 28 seconds passed
... 84%, 106528 KB, 3673 KB/s, 28 seconds passed
... 84%, 106560 KB, 3674 KB/s, 29 seconds passed
... 84%, 106592 KB, 3674 KB/s, 29 seconds passed
... 84%, 106624 KB, 3672 KB/s, 29 seconds passed

.. parsed-literal::

    ... 84%, 106656 KB, 3673 KB/s, 29 seconds passed
... 84%, 106688 KB, 3674 KB/s, 29 seconds passed
... 84%, 106720 KB, 3674 KB/s, 29 seconds passed
... 84%, 106752 KB, 3672 KB/s, 29 seconds passed
... 84%, 106784 KB, 3673 KB/s, 29 seconds passed
... 84%, 106816 KB, 3674 KB/s, 29 seconds passed
... 84%, 106848 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 84%, 106880 KB, 3672 KB/s, 29 seconds passed
... 84%, 106912 KB, 3673 KB/s, 29 seconds passed
... 84%, 106944 KB, 3674 KB/s, 29 seconds passed
... 84%, 106976 KB, 3674 KB/s, 29 seconds passed
... 84%, 107008 KB, 3672 KB/s, 29 seconds passed

.. parsed-literal::

    ... 84%, 107040 KB, 3673 KB/s, 29 seconds passed
... 85%, 107072 KB, 3674 KB/s, 29 seconds passed
... 85%, 107104 KB, 3674 KB/s, 29 seconds passed
... 85%, 107136 KB, 3672 KB/s, 29 seconds passed
... 85%, 107168 KB, 3673 KB/s, 29 seconds passed
... 85%, 107200 KB, 3674 KB/s, 29 seconds passed
... 85%, 107232 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 107264 KB, 3672 KB/s, 29 seconds passed
... 85%, 107296 KB, 3673 KB/s, 29 seconds passed
... 85%, 107328 KB, 3674 KB/s, 29 seconds passed
... 85%, 107360 KB, 3672 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 107392 KB, 3672 KB/s, 29 seconds passed
... 85%, 107424 KB, 3673 KB/s, 29 seconds passed
... 85%, 107456 KB, 3674 KB/s, 29 seconds passed
... 85%, 107488 KB, 3672 KB/s, 29 seconds passed
... 85%, 107520 KB, 3672 KB/s, 29 seconds passed
... 85%, 107552 KB, 3673 KB/s, 29 seconds passed
... 85%, 107584 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 107616 KB, 3672 KB/s, 29 seconds passed
... 85%, 107648 KB, 3672 KB/s, 29 seconds passed
... 85%, 107680 KB, 3673 KB/s, 29 seconds passed
... 85%, 107712 KB, 3674 KB/s, 29 seconds passed
... 85%, 107744 KB, 3672 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 107776 KB, 3672 KB/s, 29 seconds passed
... 85%, 107808 KB, 3673 KB/s, 29 seconds passed
... 85%, 107840 KB, 3674 KB/s, 29 seconds passed
... 85%, 107872 KB, 3672 KB/s, 29 seconds passed
... 85%, 107904 KB, 3672 KB/s, 29 seconds passed
... 85%, 107936 KB, 3673 KB/s, 29 seconds passed
... 85%, 107968 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 108000 KB, 3672 KB/s, 29 seconds passed
... 85%, 108032 KB, 3672 KB/s, 29 seconds passed
... 85%, 108064 KB, 3673 KB/s, 29 seconds passed
... 85%, 108096 KB, 3674 KB/s, 29 seconds passed
... 85%, 108128 KB, 3672 KB/s, 29 seconds passed

.. parsed-literal::

    ... 85%, 108160 KB, 3672 KB/s, 29 seconds passed
... 85%, 108192 KB, 3673 KB/s, 29 seconds passed
... 85%, 108224 KB, 3674 KB/s, 29 seconds passed
... 85%, 108256 KB, 3672 KB/s, 29 seconds passed
... 85%, 108288 KB, 3672 KB/s, 29 seconds passed
... 86%, 108320 KB, 3673 KB/s, 29 seconds passed
... 86%, 108352 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 108384 KB, 3672 KB/s, 29 seconds passed
... 86%, 108416 KB, 3672 KB/s, 29 seconds passed
... 86%, 108448 KB, 3673 KB/s, 29 seconds passed
... 86%, 108480 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 108512 KB, 3672 KB/s, 29 seconds passed
... 86%, 108544 KB, 3672 KB/s, 29 seconds passed
... 86%, 108576 KB, 3673 KB/s, 29 seconds passed
... 86%, 108608 KB, 3674 KB/s, 29 seconds passed
... 86%, 108640 KB, 3672 KB/s, 29 seconds passed
... 86%, 108672 KB, 3672 KB/s, 29 seconds passed
... 86%, 108704 KB, 3673 KB/s, 29 seconds passed
... 86%, 108736 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 108768 KB, 3672 KB/s, 29 seconds passed
... 86%, 108800 KB, 3672 KB/s, 29 seconds passed
... 86%, 108832 KB, 3673 KB/s, 29 seconds passed
... 86%, 108864 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 108896 KB, 3672 KB/s, 29 seconds passed
... 86%, 108928 KB, 3672 KB/s, 29 seconds passed
... 86%, 108960 KB, 3673 KB/s, 29 seconds passed
... 86%, 108992 KB, 3674 KB/s, 29 seconds passed
... 86%, 109024 KB, 3672 KB/s, 29 seconds passed
... 86%, 109056 KB, 3673 KB/s, 29 seconds passed
... 86%, 109088 KB, 3673 KB/s, 29 seconds passed
... 86%, 109120 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 109152 KB, 3672 KB/s, 29 seconds passed
... 86%, 109184 KB, 3672 KB/s, 29 seconds passed
... 86%, 109216 KB, 3673 KB/s, 29 seconds passed
... 86%, 109248 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 109280 KB, 3672 KB/s, 29 seconds passed
... 86%, 109312 KB, 3672 KB/s, 29 seconds passed
... 86%, 109344 KB, 3673 KB/s, 29 seconds passed
... 86%, 109376 KB, 3674 KB/s, 29 seconds passed
... 86%, 109408 KB, 3672 KB/s, 29 seconds passed
... 86%, 109440 KB, 3672 KB/s, 29 seconds passed
... 86%, 109472 KB, 3673 KB/s, 29 seconds passed
... 86%, 109504 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 86%, 109536 KB, 3672 KB/s, 29 seconds passed
... 86%, 109568 KB, 3672 KB/s, 29 seconds passed
... 87%, 109600 KB, 3673 KB/s, 29 seconds passed
... 87%, 109632 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 87%, 109664 KB, 3672 KB/s, 29 seconds passed
... 87%, 109696 KB, 3672 KB/s, 29 seconds passed
... 87%, 109728 KB, 3673 KB/s, 29 seconds passed
... 87%, 109760 KB, 3674 KB/s, 29 seconds passed
... 87%, 109792 KB, 3672 KB/s, 29 seconds passed
... 87%, 109824 KB, 3672 KB/s, 29 seconds passed
... 87%, 109856 KB, 3673 KB/s, 29 seconds passed

.. parsed-literal::

    ... 87%, 109888 KB, 3674 KB/s, 29 seconds passed
... 87%, 109920 KB, 3672 KB/s, 29 seconds passed
... 87%, 109952 KB, 3672 KB/s, 29 seconds passed
... 87%, 109984 KB, 3673 KB/s, 29 seconds passed
... 87%, 110016 KB, 3674 KB/s, 29 seconds passed

.. parsed-literal::

    ... 87%, 110048 KB, 3672 KB/s, 29 seconds passed
... 87%, 110080 KB, 3672 KB/s, 29 seconds passed
... 87%, 110112 KB, 3673 KB/s, 29 seconds passed
... 87%, 110144 KB, 3674 KB/s, 29 seconds passed
... 87%, 110176 KB, 3672 KB/s, 30 seconds passed
... 87%, 110208 KB, 3673 KB/s, 30 seconds passed
... 87%, 110240 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 87%, 110272 KB, 3674 KB/s, 30 seconds passed
... 87%, 110304 KB, 3672 KB/s, 30 seconds passed
... 87%, 110336 KB, 3673 KB/s, 30 seconds passed
... 87%, 110368 KB, 3673 KB/s, 30 seconds passed
... 87%, 110400 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 87%, 110432 KB, 3672 KB/s, 30 seconds passed
... 87%, 110464 KB, 3673 KB/s, 30 seconds passed
... 87%, 110496 KB, 3673 KB/s, 30 seconds passed
... 87%, 110528 KB, 3673 KB/s, 30 seconds passed
... 87%, 110560 KB, 3672 KB/s, 30 seconds passed
... 87%, 110592 KB, 3673 KB/s, 30 seconds passed
... 87%, 110624 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 87%, 110656 KB, 3673 KB/s, 30 seconds passed
... 87%, 110688 KB, 3672 KB/s, 30 seconds passed
... 87%, 110720 KB, 3673 KB/s, 30 seconds passed
... 87%, 110752 KB, 3674 KB/s, 30 seconds passed
... 87%, 110784 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 87%, 110816 KB, 3672 KB/s, 30 seconds passed
... 88%, 110848 KB, 3673 KB/s, 30 seconds passed
... 88%, 110880 KB, 3674 KB/s, 30 seconds passed
... 88%, 110912 KB, 3674 KB/s, 30 seconds passed
... 88%, 110944 KB, 3672 KB/s, 30 seconds passed
... 88%, 110976 KB, 3673 KB/s, 30 seconds passed
... 88%, 111008 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111040 KB, 3674 KB/s, 30 seconds passed
... 88%, 111072 KB, 3672 KB/s, 30 seconds passed
... 88%, 111104 KB, 3673 KB/s, 30 seconds passed
... 88%, 111136 KB, 3674 KB/s, 30 seconds passed
... 88%, 111168 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111200 KB, 3672 KB/s, 30 seconds passed
... 88%, 111232 KB, 3673 KB/s, 30 seconds passed
... 88%, 111264 KB, 3674 KB/s, 30 seconds passed
... 88%, 111296 KB, 3673 KB/s, 30 seconds passed
... 88%, 111328 KB, 3672 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111360 KB, 3673 KB/s, 30 seconds passed
... 88%, 111392 KB, 3674 KB/s, 30 seconds passed
... 88%, 111424 KB, 3673 KB/s, 30 seconds passed
... 88%, 111456 KB, 3672 KB/s, 30 seconds passed
... 88%, 111488 KB, 3673 KB/s, 30 seconds passed
... 88%, 111520 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111552 KB, 3673 KB/s, 30 seconds passed
... 88%, 111584 KB, 3672 KB/s, 30 seconds passed
... 88%, 111616 KB, 3672 KB/s, 30 seconds passed
... 88%, 111648 KB, 3673 KB/s, 30 seconds passed
... 88%, 111680 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111712 KB, 3672 KB/s, 30 seconds passed
... 88%, 111744 KB, 3673 KB/s, 30 seconds passed
... 88%, 111776 KB, 3673 KB/s, 30 seconds passed
... 88%, 111808 KB, 3673 KB/s, 30 seconds passed
... 88%, 111840 KB, 3672 KB/s, 30 seconds passed
... 88%, 111872 KB, 3673 KB/s, 30 seconds passed
... 88%, 111904 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 111936 KB, 3673 KB/s, 30 seconds passed
... 88%, 111968 KB, 3672 KB/s, 30 seconds passed
... 88%, 112000 KB, 3673 KB/s, 30 seconds passed
... 88%, 112032 KB, 3673 KB/s, 30 seconds passed
... 88%, 112064 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 88%, 112096 KB, 3672 KB/s, 30 seconds passed
... 89%, 112128 KB, 3673 KB/s, 30 seconds passed
... 89%, 112160 KB, 3674 KB/s, 30 seconds passed
... 89%, 112192 KB, 3674 KB/s, 30 seconds passed
... 89%, 112224 KB, 3672 KB/s, 30 seconds passed
... 89%, 112256 KB, 3673 KB/s, 30 seconds passed
... 89%, 112288 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 112320 KB, 3673 KB/s, 30 seconds passed
... 89%, 112352 KB, 3672 KB/s, 30 seconds passed
... 89%, 112384 KB, 3673 KB/s, 30 seconds passed
... 89%, 112416 KB, 3674 KB/s, 30 seconds passed
... 89%, 112448 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 112480 KB, 3672 KB/s, 30 seconds passed
... 89%, 112512 KB, 3673 KB/s, 30 seconds passed
... 89%, 112544 KB, 3674 KB/s, 30 seconds passed
... 89%, 112576 KB, 3673 KB/s, 30 seconds passed
... 89%, 112608 KB, 3672 KB/s, 30 seconds passed
... 89%, 112640 KB, 3673 KB/s, 30 seconds passed
... 89%, 112672 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 112704 KB, 3673 KB/s, 30 seconds passed
... 89%, 112736 KB, 3672 KB/s, 30 seconds passed
... 89%, 112768 KB, 3673 KB/s, 30 seconds passed
... 89%, 112800 KB, 3674 KB/s, 30 seconds passed
... 89%, 112832 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 112864 KB, 3672 KB/s, 30 seconds passed
... 89%, 112896 KB, 3673 KB/s, 30 seconds passed
... 89%, 112928 KB, 3674 KB/s, 30 seconds passed
... 89%, 112960 KB, 3673 KB/s, 30 seconds passed
... 89%, 112992 KB, 3672 KB/s, 30 seconds passed
... 89%, 113024 KB, 3673 KB/s, 30 seconds passed
... 89%, 113056 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 113088 KB, 3673 KB/s, 30 seconds passed
... 89%, 113120 KB, 3672 KB/s, 30 seconds passed
... 89%, 113152 KB, 3673 KB/s, 30 seconds passed
... 89%, 113184 KB, 3674 KB/s, 30 seconds passed
... 89%, 113216 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 89%, 113248 KB, 3672 KB/s, 30 seconds passed
... 89%, 113280 KB, 3673 KB/s, 30 seconds passed
... 89%, 113312 KB, 3674 KB/s, 30 seconds passed
... 89%, 113344 KB, 3673 KB/s, 30 seconds passed
... 90%, 113376 KB, 3672 KB/s, 30 seconds passed
... 90%, 113408 KB, 3673 KB/s, 30 seconds passed
... 90%, 113440 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 90%, 113472 KB, 3672 KB/s, 30 seconds passed
... 90%, 113504 KB, 3672 KB/s, 30 seconds passed
... 90%, 113536 KB, 3673 KB/s, 30 seconds passed
... 90%, 113568 KB, 3674 KB/s, 30 seconds passed
... 90%, 113600 KB, 3673 KB/s, 30 seconds passed

.. parsed-literal::

    ... 90%, 113632 KB, 3672 KB/s, 30 seconds passed
... 90%, 113664 KB, 3673 KB/s, 30 seconds passed
... 90%, 113696 KB, 3674 KB/s, 30 seconds passed
... 90%, 113728 KB, 3673 KB/s, 30 seconds passed
... 90%, 113760 KB, 3672 KB/s, 30 seconds passed
... 90%, 113792 KB, 3673 KB/s, 30 seconds passed
... 90%, 113824 KB, 3674 KB/s, 30 seconds passed

.. parsed-literal::

    ... 90%, 113856 KB, 3673 KB/s, 30 seconds passed
... 90%, 113888 KB, 3672 KB/s, 31 seconds passed
... 90%, 113920 KB, 3673 KB/s, 31 seconds passed
... 90%, 113952 KB, 3674 KB/s, 31 seconds passed
... 90%, 113984 KB, 3673 KB/s, 31 seconds passed

.. parsed-literal::

    ... 90%, 114016 KB, 3672 KB/s, 31 seconds passed
... 90%, 114048 KB, 3673 KB/s, 31 seconds passed
... 90%, 114080 KB, 3674 KB/s, 31 seconds passed
... 90%, 114112 KB, 3672 KB/s, 31 seconds passed
... 90%, 114144 KB, 3672 KB/s, 31 seconds passed
... 90%, 114176 KB, 3673 KB/s, 31 seconds passed
... 90%, 114208 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 90%, 114240 KB, 3672 KB/s, 31 seconds passed
... 90%, 114272 KB, 3672 KB/s, 31 seconds passed
... 90%, 114304 KB, 3673 KB/s, 31 seconds passed
... 90%, 114336 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 90%, 114368 KB, 3672 KB/s, 31 seconds passed
... 90%, 114400 KB, 3672 KB/s, 31 seconds passed
... 90%, 114432 KB, 3673 KB/s, 31 seconds passed
... 90%, 114464 KB, 3674 KB/s, 31 seconds passed
... 90%, 114496 KB, 3672 KB/s, 31 seconds passed
... 90%, 114528 KB, 3672 KB/s, 31 seconds passed
... 90%, 114560 KB, 3673 KB/s, 31 seconds passed
... 90%, 114592 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 114624 KB, 3673 KB/s, 31 seconds passed
... 91%, 114656 KB, 3672 KB/s, 31 seconds passed
... 91%, 114688 KB, 3673 KB/s, 31 seconds passed
... 91%, 114720 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 114752 KB, 3672 KB/s, 31 seconds passed
... 91%, 114784 KB, 3672 KB/s, 31 seconds passed
... 91%, 114816 KB, 3673 KB/s, 31 seconds passed
... 91%, 114848 KB, 3674 KB/s, 31 seconds passed
... 91%, 114880 KB, 3672 KB/s, 31 seconds passed
... 91%, 114912 KB, 3672 KB/s, 31 seconds passed
... 91%, 114944 KB, 3673 KB/s, 31 seconds passed
... 91%, 114976 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 115008 KB, 3672 KB/s, 31 seconds passed
... 91%, 115040 KB, 3672 KB/s, 31 seconds passed
... 91%, 115072 KB, 3673 KB/s, 31 seconds passed
... 91%, 115104 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 115136 KB, 3672 KB/s, 31 seconds passed
... 91%, 115168 KB, 3672 KB/s, 31 seconds passed
... 91%, 115200 KB, 3673 KB/s, 31 seconds passed
... 91%, 115232 KB, 3674 KB/s, 31 seconds passed
... 91%, 115264 KB, 3672 KB/s, 31 seconds passed
... 91%, 115296 KB, 3672 KB/s, 31 seconds passed
... 91%, 115328 KB, 3673 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 115360 KB, 3674 KB/s, 31 seconds passed
... 91%, 115392 KB, 3672 KB/s, 31 seconds passed
... 91%, 115424 KB, 3672 KB/s, 31 seconds passed
... 91%, 115456 KB, 3673 KB/s, 31 seconds passed
... 91%, 115488 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 115520 KB, 3672 KB/s, 31 seconds passed
... 91%, 115552 KB, 3672 KB/s, 31 seconds passed
... 91%, 115584 KB, 3673 KB/s, 31 seconds passed
... 91%, 115616 KB, 3674 KB/s, 31 seconds passed
... 91%, 115648 KB, 3672 KB/s, 31 seconds passed
... 91%, 115680 KB, 3672 KB/s, 31 seconds passed
... 91%, 115712 KB, 3673 KB/s, 31 seconds passed

.. parsed-literal::

    ... 91%, 115744 KB, 3674 KB/s, 31 seconds passed
... 91%, 115776 KB, 3672 KB/s, 31 seconds passed
... 91%, 115808 KB, 3672 KB/s, 31 seconds passed
... 91%, 115840 KB, 3673 KB/s, 31 seconds passed
... 91%, 115872 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 115904 KB, 3672 KB/s, 31 seconds passed
... 92%, 115936 KB, 3673 KB/s, 31 seconds passed
... 92%, 115968 KB, 3673 KB/s, 31 seconds passed
... 92%, 116000 KB, 3674 KB/s, 31 seconds passed
... 92%, 116032 KB, 3672 KB/s, 31 seconds passed
... 92%, 116064 KB, 3673 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 116096 KB, 3673 KB/s, 31 seconds passed
... 92%, 116128 KB, 3674 KB/s, 31 seconds passed
... 92%, 116160 KB, 3672 KB/s, 31 seconds passed
... 92%, 116192 KB, 3673 KB/s, 31 seconds passed
... 92%, 116224 KB, 3673 KB/s, 31 seconds passed
... 92%, 116256 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 116288 KB, 3672 KB/s, 31 seconds passed
... 92%, 116320 KB, 3673 KB/s, 31 seconds passed
... 92%, 116352 KB, 3673 KB/s, 31 seconds passed
... 92%, 116384 KB, 3674 KB/s, 31 seconds passed
... 92%, 116416 KB, 3672 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 116448 KB, 3673 KB/s, 31 seconds passed
... 92%, 116480 KB, 3673 KB/s, 31 seconds passed
... 92%, 116512 KB, 3674 KB/s, 31 seconds passed
... 92%, 116544 KB, 3672 KB/s, 31 seconds passed
... 92%, 116576 KB, 3673 KB/s, 31 seconds passed
... 92%, 116608 KB, 3673 KB/s, 31 seconds passed
... 92%, 116640 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 116672 KB, 3672 KB/s, 31 seconds passed
... 92%, 116704 KB, 3673 KB/s, 31 seconds passed
... 92%, 116736 KB, 3673 KB/s, 31 seconds passed
... 92%, 116768 KB, 3674 KB/s, 31 seconds passed
... 92%, 116800 KB, 3672 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 116832 KB, 3673 KB/s, 31 seconds passed
... 92%, 116864 KB, 3674 KB/s, 31 seconds passed
... 92%, 116896 KB, 3674 KB/s, 31 seconds passed
... 92%, 116928 KB, 3673 KB/s, 31 seconds passed
... 92%, 116960 KB, 3673 KB/s, 31 seconds passed
... 92%, 116992 KB, 3674 KB/s, 31 seconds passed
... 92%, 117024 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 92%, 117056 KB, 3672 KB/s, 31 seconds passed
... 92%, 117088 KB, 3673 KB/s, 31 seconds passed
... 92%, 117120 KB, 3673 KB/s, 31 seconds passed
... 93%, 117152 KB, 3674 KB/s, 31 seconds passed
... 93%, 117184 KB, 3672 KB/s, 31 seconds passed

.. parsed-literal::

    ... 93%, 117216 KB, 3673 KB/s, 31 seconds passed
... 93%, 117248 KB, 3673 KB/s, 31 seconds passed
... 93%, 117280 KB, 3674 KB/s, 31 seconds passed
... 93%, 117312 KB, 3672 KB/s, 31 seconds passed
... 93%, 117344 KB, 3673 KB/s, 31 seconds passed
... 93%, 117376 KB, 3674 KB/s, 31 seconds passed
... 93%, 117408 KB, 3674 KB/s, 31 seconds passed

.. parsed-literal::

    ... 93%, 117440 KB, 3672 KB/s, 31 seconds passed
... 93%, 117472 KB, 3673 KB/s, 31 seconds passed
... 93%, 117504 KB, 3674 KB/s, 31 seconds passed
... 93%, 117536 KB, 3674 KB/s, 31 seconds passed
... 93%, 117568 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 93%, 117600 KB, 3673 KB/s, 32 seconds passed
... 93%, 117632 KB, 3674 KB/s, 32 seconds passed
... 93%, 117664 KB, 3674 KB/s, 32 seconds passed
... 93%, 117696 KB, 3673 KB/s, 32 seconds passed
... 93%, 117728 KB, 3673 KB/s, 32 seconds passed
... 93%, 117760 KB, 3674 KB/s, 32 seconds passed
... 93%, 117792 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 93%, 117824 KB, 3673 KB/s, 32 seconds passed
... 93%, 117856 KB, 3673 KB/s, 32 seconds passed
... 93%, 117888 KB, 3674 KB/s, 32 seconds passed
... 93%, 117920 KB, 3674 KB/s, 32 seconds passed
... 93%, 117952 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 93%, 117984 KB, 3673 KB/s, 32 seconds passed
... 93%, 118016 KB, 3674 KB/s, 32 seconds passed
... 93%, 118048 KB, 3674 KB/s, 32 seconds passed
... 93%, 118080 KB, 3673 KB/s, 32 seconds passed
... 93%, 118112 KB, 3673 KB/s, 32 seconds passed
... 93%, 118144 KB, 3674 KB/s, 32 seconds passed
... 93%, 118176 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 93%, 118208 KB, 3672 KB/s, 32 seconds passed
... 93%, 118240 KB, 3673 KB/s, 32 seconds passed
... 93%, 118272 KB, 3674 KB/s, 32 seconds passed
... 93%, 118304 KB, 3672 KB/s, 32 seconds passed

.. parsed-literal::

    ... 93%, 118336 KB, 3672 KB/s, 32 seconds passed
... 93%, 118368 KB, 3673 KB/s, 32 seconds passed
... 94%, 118400 KB, 3674 KB/s, 32 seconds passed
... 94%, 118432 KB, 3674 KB/s, 32 seconds passed
... 94%, 118464 KB, 3672 KB/s, 32 seconds passed
... 94%, 118496 KB, 3673 KB/s, 32 seconds passed
... 94%, 118528 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 118560 KB, 3674 KB/s, 32 seconds passed
... 94%, 118592 KB, 3672 KB/s, 32 seconds passed
... 94%, 118624 KB, 3673 KB/s, 32 seconds passed
... 94%, 118656 KB, 3674 KB/s, 32 seconds passed
... 94%, 118688 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 118720 KB, 3672 KB/s, 32 seconds passed
... 94%, 118752 KB, 3673 KB/s, 32 seconds passed
... 94%, 118784 KB, 3674 KB/s, 32 seconds passed
... 94%, 118816 KB, 3673 KB/s, 32 seconds passed
... 94%, 118848 KB, 3672 KB/s, 32 seconds passed
... 94%, 118880 KB, 3673 KB/s, 32 seconds passed
... 94%, 118912 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 118944 KB, 3673 KB/s, 32 seconds passed
... 94%, 118976 KB, 3672 KB/s, 32 seconds passed
... 94%, 119008 KB, 3673 KB/s, 32 seconds passed
... 94%, 119040 KB, 3674 KB/s, 32 seconds passed
... 94%, 119072 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 119104 KB, 3672 KB/s, 32 seconds passed
... 94%, 119136 KB, 3673 KB/s, 32 seconds passed
... 94%, 119168 KB, 3674 KB/s, 32 seconds passed
... 94%, 119200 KB, 3673 KB/s, 32 seconds passed
... 94%, 119232 KB, 3672 KB/s, 32 seconds passed
... 94%, 119264 KB, 3673 KB/s, 32 seconds passed
... 94%, 119296 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 119328 KB, 3672 KB/s, 32 seconds passed
... 94%, 119360 KB, 3672 KB/s, 32 seconds passed
... 94%, 119392 KB, 3673 KB/s, 32 seconds passed
... 94%, 119424 KB, 3674 KB/s, 32 seconds passed
... 94%, 119456 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 94%, 119488 KB, 3672 KB/s, 32 seconds passed
... 94%, 119520 KB, 3673 KB/s, 32 seconds passed
... 94%, 119552 KB, 3674 KB/s, 32 seconds passed
... 94%, 119584 KB, 3673 KB/s, 32 seconds passed
... 94%, 119616 KB, 3672 KB/s, 32 seconds passed
... 94%, 119648 KB, 3673 KB/s, 32 seconds passed
... 95%, 119680 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 119712 KB, 3672 KB/s, 32 seconds passed
... 95%, 119744 KB, 3672 KB/s, 32 seconds passed
... 95%, 119776 KB, 3673 KB/s, 32 seconds passed
... 95%, 119808 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 119840 KB, 3672 KB/s, 32 seconds passed
... 95%, 119872 KB, 3672 KB/s, 32 seconds passed
... 95%, 119904 KB, 3673 KB/s, 32 seconds passed
... 95%, 119936 KB, 3674 KB/s, 32 seconds passed
... 95%, 119968 KB, 3673 KB/s, 32 seconds passed
... 95%, 120000 KB, 3672 KB/s, 32 seconds passed
... 95%, 120032 KB, 3673 KB/s, 32 seconds passed
... 95%, 120064 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 120096 KB, 3673 KB/s, 32 seconds passed
... 95%, 120128 KB, 3672 KB/s, 32 seconds passed
... 95%, 120160 KB, 3673 KB/s, 32 seconds passed
... 95%, 120192 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 120224 KB, 3673 KB/s, 32 seconds passed
... 95%, 120256 KB, 3672 KB/s, 32 seconds passed
... 95%, 120288 KB, 3673 KB/s, 32 seconds passed
... 95%, 120320 KB, 3673 KB/s, 32 seconds passed
... 95%, 120352 KB, 3672 KB/s, 32 seconds passed
... 95%, 120384 KB, 3672 KB/s, 32 seconds passed
... 95%, 120416 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 120448 KB, 3673 KB/s, 32 seconds passed
... 95%, 120480 KB, 3672 KB/s, 32 seconds passed
... 95%, 120512 KB, 3672 KB/s, 32 seconds passed
... 95%, 120544 KB, 3673 KB/s, 32 seconds passed
... 95%, 120576 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 120608 KB, 3672 KB/s, 32 seconds passed
... 95%, 120640 KB, 3673 KB/s, 32 seconds passed
... 95%, 120672 KB, 3673 KB/s, 32 seconds passed
... 95%, 120704 KB, 3674 KB/s, 32 seconds passed
... 95%, 120736 KB, 3672 KB/s, 32 seconds passed
... 95%, 120768 KB, 3673 KB/s, 32 seconds passed
... 95%, 120800 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 95%, 120832 KB, 3674 KB/s, 32 seconds passed
... 95%, 120864 KB, 3672 KB/s, 32 seconds passed
... 95%, 120896 KB, 3673 KB/s, 32 seconds passed
... 96%, 120928 KB, 3673 KB/s, 32 seconds passed
... 96%, 120960 KB, 3674 KB/s, 32 seconds passed

.. parsed-literal::

    ... 96%, 120992 KB, 3673 KB/s, 32 seconds passed
... 96%, 121024 KB, 3673 KB/s, 32 seconds passed
... 96%, 121056 KB, 3673 KB/s, 32 seconds passed
... 96%, 121088 KB, 3673 KB/s, 32 seconds passed
... 96%, 121120 KB, 3672 KB/s, 32 seconds passed
... 96%, 121152 KB, 3673 KB/s, 32 seconds passed
... 96%, 121184 KB, 3673 KB/s, 32 seconds passed

.. parsed-literal::

    ... 96%, 121216 KB, 3673 KB/s, 32 seconds passed
... 96%, 121248 KB, 3672 KB/s, 33 seconds passed
... 96%, 121280 KB, 3673 KB/s, 33 seconds passed
... 96%, 121312 KB, 3673 KB/s, 33 seconds passed
... 96%, 121344 KB, 3674 KB/s, 33 seconds passed

.. parsed-literal::

    ... 96%, 121376 KB, 3672 KB/s, 33 seconds passed
... 96%, 121408 KB, 3673 KB/s, 33 seconds passed
... 96%, 121440 KB, 3674 KB/s, 33 seconds passed
... 96%, 121472 KB, 3673 KB/s, 33 seconds passed
... 96%, 121504 KB, 3672 KB/s, 33 seconds passed
... 96%, 121536 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 96%, 121568 KB, 3674 KB/s, 33 seconds passed
... 96%, 121600 KB, 3673 KB/s, 33 seconds passed
... 96%, 121632 KB, 3672 KB/s, 33 seconds passed
... 96%, 121664 KB, 3673 KB/s, 33 seconds passed
... 96%, 121696 KB, 3674 KB/s, 33 seconds passed
... 96%, 121728 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 96%, 121760 KB, 3672 KB/s, 33 seconds passed
... 96%, 121792 KB, 3673 KB/s, 33 seconds passed
... 96%, 121824 KB, 3674 KB/s, 33 seconds passed
... 96%, 121856 KB, 3674 KB/s, 33 seconds passed
... 96%, 121888 KB, 3672 KB/s, 33 seconds passed

.. parsed-literal::

    ... 96%, 121920 KB, 3673 KB/s, 33 seconds passed
... 96%, 121952 KB, 3674 KB/s, 33 seconds passed
... 96%, 121984 KB, 3673 KB/s, 33 seconds passed
... 96%, 122016 KB, 3672 KB/s, 33 seconds passed
... 96%, 122048 KB, 3673 KB/s, 33 seconds passed
... 96%, 122080 KB, 3674 KB/s, 33 seconds passed
... 96%, 122112 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 96%, 122144 KB, 3672 KB/s, 33 seconds passed
... 97%, 122176 KB, 3673 KB/s, 33 seconds passed
... 97%, 122208 KB, 3674 KB/s, 33 seconds passed
... 97%, 122240 KB, 3673 KB/s, 33 seconds passed
... 97%, 122272 KB, 3672 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 122304 KB, 3673 KB/s, 33 seconds passed
... 97%, 122336 KB, 3673 KB/s, 33 seconds passed
... 97%, 122368 KB, 3673 KB/s, 33 seconds passed
... 97%, 122400 KB, 3672 KB/s, 33 seconds passed
... 97%, 122432 KB, 3673 KB/s, 33 seconds passed
... 97%, 122464 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 122496 KB, 3673 KB/s, 33 seconds passed
... 97%, 122528 KB, 3672 KB/s, 33 seconds passed
... 97%, 122560 KB, 3673 KB/s, 33 seconds passed
... 97%, 122592 KB, 3673 KB/s, 33 seconds passed
... 97%, 122624 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 122656 KB, 3672 KB/s, 33 seconds passed
... 97%, 122688 KB, 3673 KB/s, 33 seconds passed
... 97%, 122720 KB, 3673 KB/s, 33 seconds passed
... 97%, 122752 KB, 3673 KB/s, 33 seconds passed
... 97%, 122784 KB, 3672 KB/s, 33 seconds passed
... 97%, 122816 KB, 3672 KB/s, 33 seconds passed
... 97%, 122848 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 122880 KB, 3673 KB/s, 33 seconds passed
... 97%, 122912 KB, 3672 KB/s, 33 seconds passed
... 97%, 122944 KB, 3672 KB/s, 33 seconds passed
... 97%, 122976 KB, 3673 KB/s, 33 seconds passed
... 97%, 123008 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 123040 KB, 3672 KB/s, 33 seconds passed
... 97%, 123072 KB, 3672 KB/s, 33 seconds passed
... 97%, 123104 KB, 3673 KB/s, 33 seconds passed
... 97%, 123136 KB, 3673 KB/s, 33 seconds passed
... 97%, 123168 KB, 3672 KB/s, 33 seconds passed
... 97%, 123200 KB, 3672 KB/s, 33 seconds passed
... 97%, 123232 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 123264 KB, 3673 KB/s, 33 seconds passed
... 97%, 123296 KB, 3672 KB/s, 33 seconds passed
... 97%, 123328 KB, 3672 KB/s, 33 seconds passed
... 97%, 123360 KB, 3673 KB/s, 33 seconds passed
... 97%, 123392 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 97%, 123424 KB, 3672 KB/s, 33 seconds passed
... 98%, 123456 KB, 3672 KB/s, 33 seconds passed
... 98%, 123488 KB, 3673 KB/s, 33 seconds passed
... 98%, 123520 KB, 3672 KB/s, 33 seconds passed
... 98%, 123552 KB, 3672 KB/s, 33 seconds passed
... 98%, 123584 KB, 3672 KB/s, 33 seconds passed
... 98%, 123616 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 123648 KB, 3672 KB/s, 33 seconds passed
... 98%, 123680 KB, 3672 KB/s, 33 seconds passed
... 98%, 123712 KB, 3672 KB/s, 33 seconds passed
... 98%, 123744 KB, 3673 KB/s, 33 seconds passed
... 98%, 123776 KB, 3672 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 123808 KB, 3672 KB/s, 33 seconds passed
... 98%, 123840 KB, 3672 KB/s, 33 seconds passed
... 98%, 123872 KB, 3673 KB/s, 33 seconds passed
... 98%, 123904 KB, 3672 KB/s, 33 seconds passed
... 98%, 123936 KB, 3672 KB/s, 33 seconds passed
... 98%, 123968 KB, 3672 KB/s, 33 seconds passed
... 98%, 124000 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 124032 KB, 3672 KB/s, 33 seconds passed
... 98%, 124064 KB, 3672 KB/s, 33 seconds passed
... 98%, 124096 KB, 3673 KB/s, 33 seconds passed
... 98%, 124128 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 124160 KB, 3672 KB/s, 33 seconds passed
... 98%, 124192 KB, 3672 KB/s, 33 seconds passed
... 98%, 124224 KB, 3672 KB/s, 33 seconds passed
... 98%, 124256 KB, 3673 KB/s, 33 seconds passed
... 98%, 124288 KB, 3672 KB/s, 33 seconds passed
... 98%, 124320 KB, 3672 KB/s, 33 seconds passed
... 98%, 124352 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 124384 KB, 3673 KB/s, 33 seconds passed
... 98%, 124416 KB, 3672 KB/s, 33 seconds passed
... 98%, 124448 KB, 3672 KB/s, 33 seconds passed
... 98%, 124480 KB, 3673 KB/s, 33 seconds passed
... 98%, 124512 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 98%, 124544 KB, 3672 KB/s, 33 seconds passed
... 98%, 124576 KB, 3672 KB/s, 33 seconds passed
... 98%, 124608 KB, 3673 KB/s, 33 seconds passed
... 98%, 124640 KB, 3673 KB/s, 33 seconds passed
... 98%, 124672 KB, 3672 KB/s, 33 seconds passed
... 99%, 124704 KB, 3672 KB/s, 33 seconds passed
... 99%, 124736 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 99%, 124768 KB, 3673 KB/s, 33 seconds passed
... 99%, 124800 KB, 3672 KB/s, 33 seconds passed
... 99%, 124832 KB, 3672 KB/s, 33 seconds passed
... 99%, 124864 KB, 3673 KB/s, 33 seconds passed
... 99%, 124896 KB, 3673 KB/s, 33 seconds passed

.. parsed-literal::

    ... 99%, 124928 KB, 3673 KB/s, 34 seconds passed
... 99%, 124960 KB, 3672 KB/s, 34 seconds passed
... 99%, 124992 KB, 3673 KB/s, 34 seconds passed
... 99%, 125024 KB, 3673 KB/s, 34 seconds passed
... 99%, 125056 KB, 3673 KB/s, 34 seconds passed
... 99%, 125088 KB, 3672 KB/s, 34 seconds passed
... 99%, 125120 KB, 3673 KB/s, 34 seconds passed

.. parsed-literal::

    ... 99%, 125152 KB, 3673 KB/s, 34 seconds passed
... 99%, 125184 KB, 3673 KB/s, 34 seconds passed
... 99%, 125216 KB, 3672 KB/s, 34 seconds passed
... 99%, 125248 KB, 3673 KB/s, 34 seconds passed
... 99%, 125280 KB, 3674 KB/s, 34 seconds passed

.. parsed-literal::

    ... 99%, 125312 KB, 3673 KB/s, 34 seconds passed
... 99%, 125344 KB, 3672 KB/s, 34 seconds passed
... 99%, 125376 KB, 3673 KB/s, 34 seconds passed
... 99%, 125408 KB, 3674 KB/s, 34 seconds passed
... 99%, 125440 KB, 3673 KB/s, 34 seconds passed
... 99%, 125472 KB, 3672 KB/s, 34 seconds passed

.. parsed-literal::

    ... 99%, 125504 KB, 3673 KB/s, 34 seconds passed
... 99%, 125536 KB, 3673 KB/s, 34 seconds passed
... 99%, 125568 KB, 3673 KB/s, 34 seconds passed
... 99%, 125600 KB, 3672 KB/s, 34 seconds passed
... 99%, 125632 KB, 3673 KB/s, 34 seconds passed
... 99%, 125664 KB, 3673 KB/s, 34 seconds passed

.. parsed-literal::

    ... 99%, 125696 KB, 3673 KB/s, 34 seconds passed
... 99%, 125728 KB, 3672 KB/s, 34 seconds passed
... 99%, 125760 KB, 3673 KB/s, 34 seconds passed
... 99%, 125792 KB, 3674 KB/s, 34 seconds passed
... 99%, 125824 KB, 3673 KB/s, 34 seconds passed
... 99%, 125856 KB, 3673 KB/s, 34 seconds passed

.. parsed-literal::

    ... 99%, 125888 KB, 3673 KB/s, 34 seconds passed
... 99%, 125920 KB, 3674 KB/s, 34 seconds passed
... 99%, 125952 KB, 3673 KB/s, 34 seconds passed
... 100%, 125953 KB, 3673 KB/s, 34 seconds passed



.. parsed-literal::


    ========== Downloading models/public/colorization-v2/model/__init__.py


.. parsed-literal::

    ... 100%, 0 KB, 283 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/base_color.py


.. parsed-literal::

    ... 100%, 0 KB, 1753 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/eccv16.py


.. parsed-literal::

    ... 100%, 4 KB, 17370 KB/s, 0 seconds passed


    ========== Replacing text in models/public/colorization-v2/model/__init__.py
    ========== Replacing text in models/public/colorization-v2/model/__init__.py
    ========== Replacing text in models/public/colorization-v2/model/eccv16.py



Convert the model to OpenVINO IR
--------------------------------



``omz_converter`` converts the models that are not in the OpenVINO™ IR
format into that format using model conversion API.

The downloaded pytorch model is not in OpenVINO IR format which is
required for inference with OpenVINO runtime. ``omz_converter`` is used
to convert the downloaded pytorch model into ONNX and OpenVINO IR format
respectively

.. code:: ipython3

    if not os.path.exists(MODEL_PATH):
        convert_command = (
            f"omz_converter "
            f"--name {MODEL_NAME} "
            f"--download_dir {MODEL_DIR} "
            f"--precisions {PRECISION}"
        )
        ! $convert_command


.. parsed-literal::

    ========== Converting colorization-v2 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py --model-path=models/public/colorization-v2 --model-name=ECCVGenerator --weights=models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth --import-module=model --input-shape=1,1,256,256 --output-file=models/public/colorization-v2/colorization-v2-eccv16.onnx --input-names=data_l --output-names=color_ab



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting colorization-v2 to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=models/public/colorization-v2/FP16 --model_name=colorization-v2 --input=data_l --output=color_ab --input_model=models/public/colorization-v2/colorization-v2-eccv16.onnx '--layout=data_l(NCHW)' '--input_shape=[1, 1, 256, 256]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.bin




Loading the Model
-----------------



Load the model in OpenVINO Runtime with ``ie.read_model`` and compile it
for the specified device with ``ie.compile_model``.

.. code:: ipython3

    core = ov.Core()
    model = core.read_model(model=MODEL_PATH)
    compiled_model = core.compile_model(model=model, device_name=device.value)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    N, C, H, W = list(input_layer.shape)

Utility Functions
-----------------



.. code:: ipython3

    def read_image(impath: str) -> np.ndarray:
        """
        Returns an image as ndarra, given path to an image reads the
        (BGR) image using opencv's imread() API.

            Parameter:
                impath (string): Path of the image to be read and returned.

            Returns:
                image (ndarray): Numpy array representing the read image.
        """

        raw_image = cv2.imread(impath)
        if raw_image.shape[2] > 1:
            image = cv2.cvtColor(
                cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB
            )
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return image


    def plot_image(image: np.ndarray, title: str = "") -> None:
        """
        Given a image as ndarray and title as string, display it using
        matplotlib.

            Parameters:
                image (ndarray): Numpy array representing the image to be
                                 displayed.
                title (string): String representing the title of the plot.

            Returns:
                None

        """

        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()


    def plot_output(gray_img: np.ndarray, color_img: np.ndarray) -> None:
        """
        Plots the original (bw or grayscale) image and colorized image
        on different column axes for comparing side by side.

            Parameters:
                gray_image (ndarray): Numpy array representing the original image.
                color_image (ndarray): Numpy array representing the model output.

            Returns:
                None
        """

        fig = plt.figure(figsize=(12, 12))

        ax1 = fig.add_subplot(1, 2, 1)
        plt.title("Input", fontsize=20)
        ax1.axis("off")

        ax2 = fig.add_subplot(1, 2, 2)
        plt.title("Colorized", fontsize=20)
        ax2.axis("off")

        ax1.imshow(gray_img)
        ax2.imshow(color_img)

        plt.show()

Load the Image
--------------



.. code:: ipython3

    img_url_0 = "https://user-images.githubusercontent.com/18904157/180923287-20339d01-b1bf-493f-9a0d-55eff997aff1.jpg"
    img_url_1 = "https://user-images.githubusercontent.com/18904157/180923289-0bb71e09-25e1-46a6-aaf1-e8f666b62d26.jpg"

    image_file_0 = utils.download_file(
        img_url_0, filename="test_0.jpg", directory="data", show_progress=False, silent=True, timeout=30
    )
    assert Path(image_file_0).exists()

    image_file_1 = utils.download_file(
        img_url_1, filename="test_1.jpg", directory="data", show_progress=False, silent=True, timeout=30
    )
    assert Path(image_file_1).exists()

    test_img_0 = read_image("data/test_0.jpg")
    test_img_1 = read_image("data/test_1.jpg")

.. code:: ipython3

    def colorize(gray_img: np.ndarray) -> np.ndarray:

        """
        Given an image as ndarray for inference convert the image into LAB image,
        the model consumes as input L-Channel of LAB image and provides output
        A & B - Channels of LAB image. i.e returns a colorized image

            Parameters:
                gray_img (ndarray): Numpy array representing the original
                                    image.

            Returns:
                colorize_image (ndarray): Numpy arrray depicting the
                                          colorized version of the original
                                          image.
        """

        # Preprocess
        h_in, w_in, _ = gray_img.shape
        img_rgb = gray_img.astype(np.float32) / 255
        img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)
        img_l_rs = cv2.resize(img_lab.copy(), (W, H))[:, :, 0]

        # Inference
        inputs = np.expand_dims(img_l_rs, axis=[0, 1])
        res = compiled_model([inputs])[output_layer]
        update_res = np.squeeze(res)

        # Post-process
        out = update_res.transpose((1, 2, 0))
        out = cv2.resize(out, (w_in, h_in))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis],
                                      out), axis=2)
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)
        colorized_image = (cv2.resize(img_bgr_out, (w_in, h_in))
                           * 255).astype(np.uint8)
        return colorized_image

.. code:: ipython3

    color_img_0 = colorize(test_img_0)
    color_img_1 = colorize(test_img_1)

Display Colorized Image
-----------------------



.. code:: ipython3

    plot_output(test_img_0, color_img_0)



.. image:: 222-vision-image-colorization-with-output_files/222-vision-image-colorization-with-output_21_0.png


.. code:: ipython3

    plot_output(test_img_1, color_img_1)



.. image:: 222-vision-image-colorization-with-output_files/222-vision-image-colorization-with-output_22_0.png

