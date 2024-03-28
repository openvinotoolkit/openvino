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

    %pip install "openvino-dev>=2024.0.0"


.. parsed-literal::

    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.8.8)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


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

    ... 0%, 32 KB, 917 KB/s, 0 seconds passed
    ... 0%, 64 KB, 918 KB/s, 0 seconds passed
    ... 0%, 96 KB, 1327 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1237 KB/s, 0 seconds passed
    ... 0%, 160 KB, 1539 KB/s, 0 seconds passed
    ... 0%, 192 KB, 1837 KB/s, 0 seconds passed
    ... 0%, 224 KB, 2097 KB/s, 0 seconds passed
    ... 0%, 256 KB, 2388 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 288 KB, 2085 KB/s, 0 seconds passed
    ... 0%, 320 KB, 2306 KB/s, 0 seconds passed
    ... 0%, 352 KB, 2526 KB/s, 0 seconds passed
    ... 0%, 384 KB, 2730 KB/s, 0 seconds passed
    ... 0%, 416 KB, 2943 KB/s, 0 seconds passed
    ... 0%, 448 KB, 3162 KB/s, 0 seconds passed
    ... 0%, 480 KB, 3364 KB/s, 0 seconds passed
    ... 0%, 512 KB, 3569 KB/s, 0 seconds passed
    ... 0%, 544 KB, 3756 KB/s, 0 seconds passed
    ... 0%, 576 KB, 3334 KB/s, 0 seconds passed
    ... 0%, 608 KB, 3508 KB/s, 0 seconds passed
    ... 0%, 640 KB, 3683 KB/s, 0 seconds passed
    ... 0%, 672 KB, 3851 KB/s, 0 seconds passed
    ... 0%, 704 KB, 4024 KB/s, 0 seconds passed
    ... 0%, 736 KB, 4197 KB/s, 0 seconds passed
    ... 0%, 768 KB, 4358 KB/s, 0 seconds passed
    ... 0%, 800 KB, 4530 KB/s, 0 seconds passed
    ... 0%, 832 KB, 4701 KB/s, 0 seconds passed
    ... 0%, 864 KB, 4848 KB/s, 0 seconds passed
    ... 0%, 896 KB, 5015 KB/s, 0 seconds passed
    ... 0%, 928 KB, 5180 KB/s, 0 seconds passed
    ... 0%, 960 KB, 5348 KB/s, 0 seconds passed
    ... 0%, 992 KB, 5486 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 1024 KB, 5648 KB/s, 0 seconds passed
    ... 0%, 1056 KB, 5811 KB/s, 0 seconds passed
    ... 0%, 1088 KB, 5976 KB/s, 0 seconds passed
    ... 0%, 1120 KB, 6137 KB/s, 0 seconds passed
    ... 0%, 1152 KB, 6297 KB/s, 0 seconds passed
    ... 0%, 1184 KB, 5694 KB/s, 0 seconds passed
    ... 0%, 1216 KB, 5833 KB/s, 0 seconds passed
    ... 0%, 1248 KB, 5973 KB/s, 0 seconds passed
    ... 1%, 1280 KB, 6115 KB/s, 0 seconds passed
    ... 1%, 1312 KB, 6258 KB/s, 0 seconds passed
    ... 1%, 1344 KB, 6402 KB/s, 0 seconds passed
    ... 1%, 1376 KB, 6531 KB/s, 0 seconds passed
    ... 1%, 1408 KB, 6671 KB/s, 0 seconds passed
    ... 1%, 1440 KB, 6810 KB/s, 0 seconds passed
    ... 1%, 1472 KB, 6948 KB/s, 0 seconds passed
    ... 1%, 1504 KB, 7087 KB/s, 0 seconds passed
    ... 1%, 1536 KB, 7166 KB/s, 0 seconds passed
    ... 1%, 1568 KB, 7301 KB/s, 0 seconds passed
    ... 1%, 1600 KB, 7436 KB/s, 0 seconds passed
    ... 1%, 1632 KB, 7571 KB/s, 0 seconds passed
    ... 1%, 1664 KB, 7706 KB/s, 0 seconds passed
    ... 1%, 1696 KB, 7842 KB/s, 0 seconds passed
    ... 1%, 1728 KB, 7975 KB/s, 0 seconds passed
    ... 1%, 1760 KB, 8108 KB/s, 0 seconds passed
    ... 1%, 1792 KB, 8241 KB/s, 0 seconds passed
    ... 1%, 1824 KB, 8374 KB/s, 0 seconds passed
    ... 1%, 1856 KB, 8506 KB/s, 0 seconds passed
    ... 1%, 1888 KB, 8638 KB/s, 0 seconds passed
    ... 1%, 1920 KB, 8769 KB/s, 0 seconds passed
    ... 1%, 1952 KB, 8899 KB/s, 0 seconds passed
    ... 1%, 1984 KB, 9029 KB/s, 0 seconds passed
    ... 1%, 2016 KB, 9158 KB/s, 0 seconds passed
    ... 1%, 2048 KB, 9288 KB/s, 0 seconds passed
    ... 1%, 2080 KB, 9418 KB/s, 0 seconds passed
    ... 1%, 2112 KB, 9548 KB/s, 0 seconds passed
    ... 1%, 2144 KB, 9678 KB/s, 0 seconds passed
    ... 1%, 2176 KB, 9808 KB/s, 0 seconds passed
    ... 1%, 2208 KB, 9939 KB/s, 0 seconds passed
    ... 1%, 2240 KB, 10071 KB/s, 0 seconds passed
    ... 1%, 2272 KB, 10203 KB/s, 0 seconds passed
    ... 1%, 2304 KB, 10336 KB/s, 0 seconds passed
    ... 1%, 2336 KB, 10467 KB/s, 0 seconds passed
    ... 1%, 2368 KB, 10599 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 2400 KB, 9808 KB/s, 0 seconds passed
    ... 1%, 2432 KB, 9918 KB/s, 0 seconds passed
    ... 1%, 2464 KB, 10032 KB/s, 0 seconds passed
    ... 1%, 2496 KB, 10146 KB/s, 0 seconds passed
    ... 2%, 2528 KB, 10262 KB/s, 0 seconds passed
    ... 2%, 2560 KB, 10375 KB/s, 0 seconds passed
    ... 2%, 2592 KB, 10489 KB/s, 0 seconds passed
    ... 2%, 2624 KB, 10603 KB/s, 0 seconds passed
    ... 2%, 2656 KB, 10716 KB/s, 0 seconds passed
    ... 2%, 2688 KB, 10829 KB/s, 0 seconds passed
    ... 2%, 2720 KB, 10942 KB/s, 0 seconds passed
    ... 2%, 2752 KB, 11056 KB/s, 0 seconds passed
    ... 2%, 2784 KB, 11166 KB/s, 0 seconds passed
    ... 2%, 2816 KB, 11279 KB/s, 0 seconds passed
    ... 2%, 2848 KB, 11391 KB/s, 0 seconds passed
    ... 2%, 2880 KB, 11502 KB/s, 0 seconds passed
    ... 2%, 2912 KB, 11613 KB/s, 0 seconds passed
    ... 2%, 2944 KB, 11723 KB/s, 0 seconds passed
    ... 2%, 2976 KB, 11833 KB/s, 0 seconds passed
    ... 2%, 3008 KB, 11943 KB/s, 0 seconds passed
    ... 2%, 3040 KB, 12052 KB/s, 0 seconds passed
    ... 2%, 3072 KB, 12160 KB/s, 0 seconds passed
    ... 2%, 3104 KB, 12268 KB/s, 0 seconds passed
    ... 2%, 3136 KB, 12376 KB/s, 0 seconds passed
    ... 2%, 3168 KB, 12485 KB/s, 0 seconds passed
    ... 2%, 3200 KB, 12592 KB/s, 0 seconds passed
    ... 2%, 3232 KB, 12700 KB/s, 0 seconds passed
    ... 2%, 3264 KB, 12807 KB/s, 0 seconds passed
    ... 2%, 3296 KB, 12914 KB/s, 0 seconds passed
    ... 2%, 3328 KB, 13021 KB/s, 0 seconds passed
    ... 2%, 3360 KB, 13128 KB/s, 0 seconds passed
    ... 2%, 3392 KB, 13234 KB/s, 0 seconds passed
    ... 2%, 3424 KB, 13340 KB/s, 0 seconds passed
    ... 2%, 3456 KB, 13444 KB/s, 0 seconds passed
    ... 2%, 3488 KB, 13548 KB/s, 0 seconds passed
    ... 2%, 3520 KB, 13653 KB/s, 0 seconds passed
    ... 2%, 3552 KB, 13758 KB/s, 0 seconds passed
    ... 2%, 3584 KB, 13861 KB/s, 0 seconds passed
    ... 2%, 3616 KB, 13965 KB/s, 0 seconds passed
    ... 2%, 3648 KB, 14069 KB/s, 0 seconds passed
    ... 2%, 3680 KB, 14172 KB/s, 0 seconds passed
    ... 2%, 3712 KB, 14276 KB/s, 0 seconds passed
    ... 2%, 3744 KB, 14379 KB/s, 0 seconds passed
    ... 2%, 3776 KB, 14487 KB/s, 0 seconds passed
    ... 3%, 3808 KB, 14597 KB/s, 0 seconds passed
    ... 3%, 3840 KB, 14705 KB/s, 0 seconds passed
    ... 3%, 3872 KB, 14814 KB/s, 0 seconds passed
    ... 3%, 3904 KB, 14923 KB/s, 0 seconds passed
    ... 3%, 3936 KB, 15032 KB/s, 0 seconds passed
    ... 3%, 3968 KB, 15141 KB/s, 0 seconds passed
    ... 3%, 4000 KB, 15248 KB/s, 0 seconds passed
    ... 3%, 4032 KB, 15342 KB/s, 0 seconds passed
    ... 3%, 4064 KB, 15445 KB/s, 0 seconds passed
    ... 3%, 4096 KB, 15550 KB/s, 0 seconds passed
    ... 3%, 4128 KB, 15637 KB/s, 0 seconds passed
    ... 3%, 4160 KB, 15744 KB/s, 0 seconds passed
    ... 3%, 4192 KB, 15848 KB/s, 0 seconds passed
    ... 3%, 4224 KB, 15953 KB/s, 0 seconds passed
    ... 3%, 4256 KB, 16058 KB/s, 0 seconds passed
    ... 3%, 4288 KB, 16159 KB/s, 0 seconds passed
    ... 3%, 4320 KB, 16263 KB/s, 0 seconds passed
    ... 3%, 4352 KB, 16367 KB/s, 0 seconds passed
    ... 3%, 4384 KB, 16471 KB/s, 0 seconds passed
    ... 3%, 4416 KB, 16572 KB/s, 0 seconds passed
    ... 3%, 4448 KB, 16675 KB/s, 0 seconds passed
    ... 3%, 4480 KB, 16778 KB/s, 0 seconds passed
    ... 3%, 4512 KB, 16881 KB/s, 0 seconds passed
    ... 3%, 4544 KB, 16198 KB/s, 0 seconds passed
    ... 3%, 4576 KB, 16282 KB/s, 0 seconds passed
    ... 3%, 4608 KB, 16373 KB/s, 0 seconds passed
    ... 3%, 4640 KB, 16463 KB/s, 0 seconds passed
    ... 3%, 4672 KB, 16553 KB/s, 0 seconds passed
    ... 3%, 4704 KB, 16645 KB/s, 0 seconds passed
    ... 3%, 4736 KB, 16738 KB/s, 0 seconds passed
    ... 3%, 4768 KB, 16833 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 4800 KB, 16791 KB/s, 0 seconds passed
    ... 3%, 4832 KB, 16875 KB/s, 0 seconds passed
    ... 3%, 4864 KB, 16963 KB/s, 0 seconds passed
    ... 3%, 4896 KB, 17051 KB/s, 0 seconds passed
    ... 3%, 4928 KB, 17140 KB/s, 0 seconds passed
    ... 3%, 4960 KB, 17228 KB/s, 0 seconds passed
    ... 3%, 4992 KB, 17316 KB/s, 0 seconds passed
    ... 3%, 5024 KB, 17405 KB/s, 0 seconds passed
    ... 4%, 5056 KB, 17492 KB/s, 0 seconds passed
    ... 4%, 5088 KB, 17580 KB/s, 0 seconds passed
    ... 4%, 5120 KB, 17668 KB/s, 0 seconds passed
    ... 4%, 5152 KB, 17755 KB/s, 0 seconds passed
    ... 4%, 5184 KB, 17843 KB/s, 0 seconds passed
    ... 4%, 5216 KB, 17930 KB/s, 0 seconds passed
    ... 4%, 5248 KB, 18016 KB/s, 0 seconds passed
    ... 4%, 5280 KB, 18103 KB/s, 0 seconds passed
    ... 4%, 5312 KB, 18188 KB/s, 0 seconds passed
    ... 4%, 5344 KB, 18274 KB/s, 0 seconds passed
    ... 4%, 5376 KB, 18360 KB/s, 0 seconds passed
    ... 4%, 5408 KB, 18444 KB/s, 0 seconds passed
    ... 4%, 5440 KB, 18530 KB/s, 0 seconds passed
    ... 4%, 5472 KB, 18615 KB/s, 0 seconds passed
    ... 4%, 5504 KB, 18700 KB/s, 0 seconds passed
    ... 4%, 5536 KB, 18786 KB/s, 0 seconds passed
    ... 4%, 5568 KB, 18870 KB/s, 0 seconds passed
    ... 4%, 5600 KB, 18955 KB/s, 0 seconds passed
    ... 4%, 5632 KB, 19039 KB/s, 0 seconds passed
    ... 4%, 5664 KB, 19122 KB/s, 0 seconds passed
    ... 4%, 5696 KB, 19204 KB/s, 0 seconds passed
    ... 4%, 5728 KB, 19285 KB/s, 0 seconds passed
    ... 4%, 5760 KB, 19367 KB/s, 0 seconds passed
    ... 4%, 5792 KB, 19450 KB/s, 0 seconds passed
    ... 4%, 5824 KB, 19533 KB/s, 0 seconds passed
    ... 4%, 5856 KB, 19615 KB/s, 0 seconds passed
    ... 4%, 5888 KB, 19697 KB/s, 0 seconds passed
    ... 4%, 5920 KB, 19778 KB/s, 0 seconds passed
    ... 4%, 5952 KB, 19859 KB/s, 0 seconds passed
    ... 4%, 5984 KB, 19941 KB/s, 0 seconds passed
    ... 4%, 6016 KB, 20023 KB/s, 0 seconds passed
    ... 4%, 6048 KB, 20113 KB/s, 0 seconds passed
    ... 4%, 6080 KB, 20203 KB/s, 0 seconds passed
    ... 4%, 6112 KB, 20292 KB/s, 0 seconds passed
    ... 4%, 6144 KB, 20382 KB/s, 0 seconds passed
    ... 4%, 6176 KB, 20472 KB/s, 0 seconds passed
    ... 4%, 6208 KB, 20562 KB/s, 0 seconds passed
    ... 4%, 6240 KB, 20650 KB/s, 0 seconds passed
    ... 4%, 6272 KB, 20739 KB/s, 0 seconds passed
    ... 5%, 6304 KB, 20828 KB/s, 0 seconds passed
    ... 5%, 6336 KB, 20918 KB/s, 0 seconds passed
    ... 5%, 6368 KB, 21007 KB/s, 0 seconds passed
    ... 5%, 6400 KB, 21095 KB/s, 0 seconds passed
    ... 5%, 6432 KB, 21184 KB/s, 0 seconds passed
    ... 5%, 6464 KB, 21272 KB/s, 0 seconds passed
    ... 5%, 6496 KB, 21361 KB/s, 0 seconds passed
    ... 5%, 6528 KB, 21449 KB/s, 0 seconds passed
    ... 5%, 6560 KB, 21538 KB/s, 0 seconds passed
    ... 5%, 6592 KB, 21626 KB/s, 0 seconds passed
    ... 5%, 6624 KB, 21713 KB/s, 0 seconds passed
    ... 5%, 6656 KB, 21801 KB/s, 0 seconds passed
    ... 5%, 6688 KB, 21889 KB/s, 0 seconds passed
    ... 5%, 6720 KB, 21976 KB/s, 0 seconds passed
    ... 5%, 6752 KB, 22063 KB/s, 0 seconds passed
    ... 5%, 6784 KB, 22149 KB/s, 0 seconds passed
    ... 5%, 6816 KB, 22237 KB/s, 0 seconds passed
    ... 5%, 6848 KB, 22324 KB/s, 0 seconds passed
    ... 5%, 6880 KB, 22411 KB/s, 0 seconds passed
    ... 5%, 6912 KB, 22498 KB/s, 0 seconds passed
    ... 5%, 6944 KB, 22585 KB/s, 0 seconds passed
    ... 5%, 6976 KB, 22671 KB/s, 0 seconds passed
    ... 5%, 7008 KB, 22755 KB/s, 0 seconds passed
    ... 5%, 7040 KB, 22832 KB/s, 0 seconds passed
    ... 5%, 7072 KB, 22916 KB/s, 0 seconds passed
    ... 5%, 7104 KB, 23001 KB/s, 0 seconds passed
    ... 5%, 7136 KB, 23083 KB/s, 0 seconds passed
    ... 5%, 7168 KB, 23168 KB/s, 0 seconds passed
    ... 5%, 7200 KB, 23253 KB/s, 0 seconds passed
    ... 5%, 7232 KB, 23338 KB/s, 0 seconds passed
    ... 5%, 7264 KB, 23423 KB/s, 0 seconds passed
    ... 5%, 7296 KB, 23508 KB/s, 0 seconds passed
    ... 5%, 7328 KB, 23599 KB/s, 0 seconds passed
    ... 5%, 7360 KB, 23689 KB/s, 0 seconds passed
    ... 5%, 7392 KB, 23780 KB/s, 0 seconds passed
    ... 5%, 7424 KB, 23870 KB/s, 0 seconds passed
    ... 5%, 7456 KB, 23960 KB/s, 0 seconds passed
    ... 5%, 7488 KB, 24049 KB/s, 0 seconds passed
    ... 5%, 7520 KB, 24139 KB/s, 0 seconds passed
    ... 5%, 7552 KB, 24229 KB/s, 0 seconds passed
    ... 6%, 7584 KB, 24319 KB/s, 0 seconds passed
    ... 6%, 7616 KB, 24409 KB/s, 0 seconds passed
    ... 6%, 7648 KB, 24499 KB/s, 0 seconds passed
    ... 6%, 7680 KB, 24589 KB/s, 0 seconds passed
    ... 6%, 7712 KB, 24679 KB/s, 0 seconds passed
    ... 6%, 7744 KB, 24768 KB/s, 0 seconds passed
    ... 6%, 7776 KB, 24858 KB/s, 0 seconds passed
    ... 6%, 7808 KB, 24946 KB/s, 0 seconds passed
    ... 6%, 7840 KB, 25035 KB/s, 0 seconds passed
    ... 6%, 7872 KB, 25124 KB/s, 0 seconds passed
    ... 6%, 7904 KB, 25214 KB/s, 0 seconds passed
    ... 6%, 7936 KB, 25303 KB/s, 0 seconds passed
    ... 6%, 7968 KB, 25392 KB/s, 0 seconds passed
    ... 6%, 8000 KB, 25481 KB/s, 0 seconds passed
    ... 6%, 8032 KB, 25569 KB/s, 0 seconds passed
    ... 6%, 8064 KB, 25658 KB/s, 0 seconds passed
    ... 6%, 8096 KB, 25744 KB/s, 0 seconds passed
    ... 6%, 8128 KB, 25820 KB/s, 0 seconds passed
    ... 6%, 8160 KB, 25887 KB/s, 0 seconds passed
    ... 6%, 8192 KB, 25962 KB/s, 0 seconds passed
    ... 6%, 8224 KB, 26050 KB/s, 0 seconds passed
    ... 6%, 8256 KB, 26136 KB/s, 0 seconds passed
    ... 6%, 8288 KB, 26217 KB/s, 0 seconds passed
    ... 6%, 8320 KB, 26296 KB/s, 0 seconds passed
    ... 6%, 8352 KB, 26370 KB/s, 0 seconds passed
    ... 6%, 8384 KB, 26443 KB/s, 0 seconds passed
    ... 6%, 8416 KB, 26511 KB/s, 0 seconds passed
    ... 6%, 8448 KB, 26591 KB/s, 0 seconds passed
    ... 6%, 8480 KB, 26674 KB/s, 0 seconds passed
    ... 6%, 8512 KB, 26754 KB/s, 0 seconds passed
    ... 6%, 8544 KB, 26832 KB/s, 0 seconds passed
    ... 6%, 8576 KB, 26911 KB/s, 0 seconds passed
    ... 6%, 8608 KB, 26990 KB/s, 0 seconds passed
    ... 6%, 8640 KB, 27068 KB/s, 0 seconds passed
    ... 6%, 8672 KB, 27147 KB/s, 0 seconds passed
    ... 6%, 8704 KB, 27202 KB/s, 0 seconds passed
    ... 6%, 8736 KB, 27271 KB/s, 0 seconds passed
    ... 6%, 8768 KB, 27345 KB/s, 0 seconds passed
    ... 6%, 8800 KB, 27423 KB/s, 0 seconds passed
    ... 7%, 8832 KB, 27496 KB/s, 0 seconds passed
    ... 7%, 8864 KB, 27573 KB/s, 0 seconds passed
    ... 7%, 8896 KB, 27650 KB/s, 0 seconds passed
    ... 7%, 8928 KB, 27728 KB/s, 0 seconds passed
    ... 7%, 8960 KB, 27805 KB/s, 0 seconds passed
    ... 7%, 8992 KB, 27881 KB/s, 0 seconds passed
    ... 7%, 9024 KB, 27958 KB/s, 0 seconds passed
    ... 7%, 9056 KB, 28034 KB/s, 0 seconds passed
    ... 7%, 9088 KB, 28111 KB/s, 0 seconds passed
    ... 7%, 9120 KB, 28188 KB/s, 0 seconds passed
    ... 7%, 9152 KB, 28264 KB/s, 0 seconds passed
    ... 7%, 9184 KB, 28340 KB/s, 0 seconds passed
    ... 7%, 9216 KB, 28416 KB/s, 0 seconds passed
    ... 7%, 9248 KB, 28492 KB/s, 0 seconds passed
    ... 7%, 9280 KB, 28569 KB/s, 0 seconds passed
    ... 7%, 9312 KB, 28626 KB/s, 0 seconds passed
    ... 7%, 9344 KB, 28700 KB/s, 0 seconds passed
    ... 7%, 9376 KB, 28775 KB/s, 0 seconds passed
    ... 7%, 9408 KB, 28851 KB/s, 0 seconds passed
    ... 7%, 9440 KB, 28927 KB/s, 0 seconds passed
    ... 7%, 9472 KB, 29003 KB/s, 0 seconds passed
    ... 7%, 9504 KB, 29078 KB/s, 0 seconds passed
    ... 7%, 9536 KB, 29154 KB/s, 0 seconds passed
    ... 7%, 9568 KB, 29227 KB/s, 0 seconds passed
    ... 7%, 9600 KB, 29301 KB/s, 0 seconds passed
    ... 7%, 9632 KB, 29355 KB/s, 0 seconds passed
    ... 7%, 9664 KB, 29421 KB/s, 0 seconds passed
    ... 7%, 9696 KB, 29483 KB/s, 0 seconds passed
    ... 7%, 9728 KB, 29556 KB/s, 0 seconds passed
    ... 7%, 9760 KB, 29630 KB/s, 0 seconds passed
    ... 7%, 9792 KB, 29704 KB/s, 0 seconds passed
    ... 7%, 9824 KB, 29779 KB/s, 0 seconds passed
    ... 7%, 9856 KB, 29853 KB/s, 0 seconds passed
    ... 7%, 9888 KB, 29926 KB/s, 0 seconds passed
    ... 7%, 9920 KB, 30000 KB/s, 0 seconds passed
    ... 7%, 9952 KB, 30072 KB/s, 0 seconds passed
    ... 7%, 9984 KB, 30147 KB/s, 0 seconds passed
    ... 7%, 10016 KB, 30220 KB/s, 0 seconds passed
    ... 7%, 10048 KB, 30297 KB/s, 0 seconds passed
    ... 8%, 10080 KB, 30370 KB/s, 0 seconds passed
    ... 8%, 10112 KB, 30444 KB/s, 0 seconds passed
    ... 8%, 10144 KB, 30516 KB/s, 0 seconds passed
    ... 8%, 10176 KB, 30588 KB/s, 0 seconds passed
    ... 8%, 10208 KB, 30661 KB/s, 0 seconds passed
    ... 8%, 10240 KB, 30733 KB/s, 0 seconds passed
    ... 8%, 10272 KB, 30780 KB/s, 0 seconds passed
    ... 8%, 10304 KB, 30843 KB/s, 0 seconds passed
    ... 8%, 10336 KB, 30907 KB/s, 0 seconds passed
    ... 8%, 10368 KB, 30987 KB/s, 0 seconds passed
    ... 8%, 10400 KB, 31066 KB/s, 0 seconds passed
    ... 8%, 10432 KB, 31145 KB/s, 0 seconds passed
    ... 8%, 10464 KB, 31216 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 10496 KB, 31288 KB/s, 0 seconds passed
    ... 8%, 10528 KB, 31359 KB/s, 0 seconds passed
    ... 8%, 10560 KB, 31431 KB/s, 0 seconds passed
    ... 8%, 10592 KB, 31503 KB/s, 0 seconds passed
    ... 8%, 10624 KB, 31552 KB/s, 0 seconds passed
    ... 8%, 10656 KB, 31622 KB/s, 0 seconds passed
    ... 8%, 10688 KB, 31694 KB/s, 0 seconds passed
    ... 8%, 10720 KB, 31765 KB/s, 0 seconds passed
    ... 8%, 10752 KB, 31834 KB/s, 0 seconds passed
    ... 8%, 10784 KB, 31905 KB/s, 0 seconds passed
    ... 8%, 10816 KB, 31976 KB/s, 0 seconds passed
    ... 8%, 10848 KB, 32046 KB/s, 0 seconds passed
    ... 8%, 10880 KB, 32117 KB/s, 0 seconds passed
    ... 8%, 10912 KB, 32165 KB/s, 0 seconds passed
    ... 8%, 10944 KB, 32234 KB/s, 0 seconds passed
    ... 8%, 10976 KB, 32304 KB/s, 0 seconds passed
    ... 8%, 11008 KB, 32374 KB/s, 0 seconds passed
    ... 8%, 11040 KB, 32445 KB/s, 0 seconds passed
    ... 8%, 11072 KB, 32513 KB/s, 0 seconds passed
    ... 8%, 11104 KB, 32583 KB/s, 0 seconds passed
    ... 8%, 11136 KB, 32652 KB/s, 0 seconds passed
    ... 8%, 11168 KB, 32721 KB/s, 0 seconds passed
    ... 8%, 11200 KB, 32764 KB/s, 0 seconds passed
    ... 8%, 11232 KB, 32823 KB/s, 0 seconds passed
    ... 8%, 11264 KB, 32889 KB/s, 0 seconds passed
    ... 8%, 11296 KB, 32967 KB/s, 0 seconds passed
    ... 8%, 11328 KB, 33044 KB/s, 0 seconds passed
    ... 9%, 11360 KB, 33107 KB/s, 0 seconds passed
    ... 9%, 11392 KB, 33167 KB/s, 0 seconds passed
    ... 9%, 11424 KB, 33238 KB/s, 0 seconds passed
    ... 9%, 11456 KB, 33316 KB/s, 0 seconds passed
    ... 9%, 11488 KB, 33386 KB/s, 0 seconds passed
    ... 9%, 11520 KB, 33454 KB/s, 0 seconds passed
    ... 9%, 11552 KB, 33495 KB/s, 0 seconds passed
    ... 9%, 11584 KB, 33550 KB/s, 0 seconds passed
    ... 9%, 11616 KB, 33618 KB/s, 0 seconds passed
    ... 9%, 11648 KB, 33695 KB/s, 0 seconds passed
    ... 9%, 11680 KB, 33770 KB/s, 0 seconds passed
    ... 9%, 11712 KB, 33838 KB/s, 0 seconds passed
    ... 9%, 11744 KB, 33905 KB/s, 0 seconds passed
    ... 9%, 11776 KB, 33972 KB/s, 0 seconds passed
    ... 9%, 11808 KB, 34040 KB/s, 0 seconds passed
    ... 9%, 11840 KB, 34107 KB/s, 0 seconds passed
    ... 9%, 11872 KB, 34174 KB/s, 0 seconds passed
    ... 9%, 11904 KB, 34218 KB/s, 0 seconds passed
    ... 9%, 11936 KB, 34285 KB/s, 0 seconds passed
    ... 9%, 11968 KB, 34351 KB/s, 0 seconds passed
    ... 9%, 12000 KB, 34418 KB/s, 0 seconds passed
    ... 9%, 12032 KB, 34485 KB/s, 0 seconds passed
    ... 9%, 12064 KB, 34550 KB/s, 0 seconds passed
    ... 9%, 12096 KB, 34617 KB/s, 0 seconds passed
    ... 9%, 12128 KB, 34683 KB/s, 0 seconds passed
    ... 9%, 12160 KB, 34750 KB/s, 0 seconds passed
    ... 9%, 12192 KB, 32822 KB/s, 0 seconds passed
    ... 9%, 12224 KB, 32869 KB/s, 0 seconds passed
    ... 9%, 12256 KB, 32921 KB/s, 0 seconds passed
    ... 9%, 12288 KB, 32974 KB/s, 0 seconds passed
    ... 9%, 12320 KB, 33025 KB/s, 0 seconds passed
    ... 9%, 12352 KB, 33076 KB/s, 0 seconds passed
    ... 9%, 12384 KB, 33127 KB/s, 0 seconds passed
    ... 9%, 12416 KB, 33180 KB/s, 0 seconds passed
    ... 9%, 12448 KB, 33234 KB/s, 0 seconds passed
    ... 9%, 12480 KB, 33287 KB/s, 0 seconds passed
    ... 9%, 12512 KB, 33339 KB/s, 0 seconds passed
    ... 9%, 12544 KB, 33392 KB/s, 0 seconds passed
    ... 9%, 12576 KB, 33444 KB/s, 0 seconds passed
    ... 10%, 12608 KB, 33497 KB/s, 0 seconds passed
    ... 10%, 12640 KB, 33550 KB/s, 0 seconds passed
    ... 10%, 12672 KB, 33601 KB/s, 0 seconds passed
    ... 10%, 12704 KB, 33653 KB/s, 0 seconds passed
    ... 10%, 12736 KB, 33706 KB/s, 0 seconds passed
    ... 10%, 12768 KB, 33757 KB/s, 0 seconds passed
    ... 10%, 12800 KB, 33809 KB/s, 0 seconds passed
    ... 10%, 12832 KB, 33860 KB/s, 0 seconds passed
    ... 10%, 12864 KB, 33912 KB/s, 0 seconds passed
    ... 10%, 12896 KB, 33964 KB/s, 0 seconds passed
    ... 10%, 12928 KB, 34016 KB/s, 0 seconds passed
    ... 10%, 12960 KB, 34066 KB/s, 0 seconds passed
    ... 10%, 12992 KB, 34117 KB/s, 0 seconds passed
    ... 10%, 13024 KB, 34166 KB/s, 0 seconds passed
    ... 10%, 13056 KB, 34216 KB/s, 0 seconds passed
    ... 10%, 13088 KB, 34278 KB/s, 0 seconds passed
    ... 10%, 13120 KB, 34339 KB/s, 0 seconds passed
    ... 10%, 13152 KB, 34401 KB/s, 0 seconds passed
    ... 10%, 13184 KB, 34462 KB/s, 0 seconds passed
    ... 10%, 13216 KB, 34523 KB/s, 0 seconds passed
    ... 10%, 13248 KB, 34584 KB/s, 0 seconds passed
    ... 10%, 13280 KB, 34645 KB/s, 0 seconds passed
    ... 10%, 13312 KB, 34705 KB/s, 0 seconds passed
    ... 10%, 13344 KB, 34766 KB/s, 0 seconds passed
    ... 10%, 13376 KB, 34826 KB/s, 0 seconds passed
    ... 10%, 13408 KB, 34887 KB/s, 0 seconds passed
    ... 10%, 13440 KB, 34948 KB/s, 0 seconds passed
    ... 10%, 13472 KB, 35009 KB/s, 0 seconds passed
    ... 10%, 13504 KB, 35068 KB/s, 0 seconds passed
    ... 10%, 13536 KB, 35127 KB/s, 0 seconds passed
    ... 10%, 13568 KB, 35188 KB/s, 0 seconds passed
    ... 10%, 13600 KB, 35249 KB/s, 0 seconds passed
    ... 10%, 13632 KB, 35309 KB/s, 0 seconds passed
    ... 10%, 13664 KB, 35369 KB/s, 0 seconds passed
    ... 10%, 13696 KB, 35429 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 13728 KB, 35489 KB/s, 0 seconds passed
    ... 10%, 13760 KB, 35549 KB/s, 0 seconds passed
    ... 10%, 13792 KB, 35609 KB/s, 0 seconds passed
    ... 10%, 13824 KB, 35667 KB/s, 0 seconds passed
    ... 11%, 13856 KB, 35726 KB/s, 0 seconds passed
    ... 11%, 13888 KB, 35785 KB/s, 0 seconds passed
    ... 11%, 13920 KB, 35845 KB/s, 0 seconds passed
    ... 11%, 13952 KB, 35903 KB/s, 0 seconds passed
    ... 11%, 13984 KB, 35962 KB/s, 0 seconds passed
    ... 11%, 14016 KB, 36019 KB/s, 0 seconds passed
    ... 11%, 14048 KB, 36078 KB/s, 0 seconds passed
    ... 11%, 14080 KB, 36136 KB/s, 0 seconds passed
    ... 11%, 14112 KB, 36195 KB/s, 0 seconds passed
    ... 11%, 14144 KB, 36254 KB/s, 0 seconds passed
    ... 11%, 14176 KB, 36313 KB/s, 0 seconds passed
    ... 11%, 14208 KB, 36372 KB/s, 0 seconds passed
    ... 11%, 14240 KB, 36431 KB/s, 0 seconds passed
    ... 11%, 14272 KB, 36489 KB/s, 0 seconds passed
    ... 11%, 14304 KB, 36548 KB/s, 0 seconds passed
    ... 11%, 14336 KB, 36607 KB/s, 0 seconds passed
    ... 11%, 14368 KB, 36673 KB/s, 0 seconds passed
    ... 11%, 14400 KB, 36739 KB/s, 0 seconds passed
    ... 11%, 14432 KB, 36804 KB/s, 0 seconds passed
    ... 11%, 14464 KB, 36870 KB/s, 0 seconds passed
    ... 11%, 14496 KB, 36935 KB/s, 0 seconds passed
    ... 11%, 14528 KB, 37001 KB/s, 0 seconds passed
    ... 11%, 14560 KB, 37066 KB/s, 0 seconds passed
    ... 11%, 14592 KB, 37131 KB/s, 0 seconds passed
    ... 11%, 14624 KB, 37196 KB/s, 0 seconds passed
    ... 11%, 14656 KB, 37261 KB/s, 0 seconds passed
    ... 11%, 14688 KB, 37326 KB/s, 0 seconds passed
    ... 11%, 14720 KB, 37391 KB/s, 0 seconds passed
    ... 11%, 14752 KB, 37457 KB/s, 0 seconds passed
    ... 11%, 14784 KB, 37521 KB/s, 0 seconds passed
    ... 11%, 14816 KB, 37578 KB/s, 0 seconds passed
    ... 11%, 14848 KB, 37634 KB/s, 0 seconds passed
    ... 11%, 14880 KB, 37691 KB/s, 0 seconds passed
    ... 11%, 14912 KB, 37742 KB/s, 0 seconds passed
    ... 11%, 14944 KB, 37797 KB/s, 0 seconds passed
    ... 11%, 14976 KB, 37853 KB/s, 0 seconds passed
    ... 11%, 15008 KB, 37909 KB/s, 0 seconds passed
    ... 11%, 15040 KB, 37960 KB/s, 0 seconds passed
    ... 11%, 15072 KB, 38015 KB/s, 0 seconds passed
    ... 11%, 15104 KB, 38070 KB/s, 0 seconds passed
    ... 12%, 15136 KB, 38125 KB/s, 0 seconds passed
    ... 12%, 15168 KB, 38175 KB/s, 0 seconds passed
    ... 12%, 15200 KB, 38228 KB/s, 0 seconds passed
    ... 12%, 15232 KB, 38282 KB/s, 0 seconds passed
    ... 12%, 15264 KB, 38337 KB/s, 0 seconds passed
    ... 12%, 15296 KB, 38387 KB/s, 0 seconds passed
    ... 12%, 15328 KB, 38442 KB/s, 0 seconds passed
    ... 12%, 15360 KB, 38497 KB/s, 0 seconds passed
    ... 12%, 15392 KB, 38547 KB/s, 0 seconds passed
    ... 12%, 15424 KB, 38603 KB/s, 0 seconds passed
    ... 12%, 15456 KB, 38657 KB/s, 0 seconds passed
    ... 12%, 15488 KB, 38712 KB/s, 0 seconds passed
    ... 12%, 15520 KB, 38761 KB/s, 0 seconds passed
    ... 12%, 15552 KB, 38816 KB/s, 0 seconds passed
    ... 12%, 15584 KB, 38871 KB/s, 0 seconds passed
    ... 12%, 15616 KB, 38920 KB/s, 0 seconds passed
    ... 12%, 15648 KB, 38975 KB/s, 0 seconds passed
    ... 12%, 15680 KB, 39029 KB/s, 0 seconds passed
    ... 12%, 15712 KB, 39078 KB/s, 0 seconds passed
    ... 12%, 15744 KB, 39133 KB/s, 0 seconds passed
    ... 12%, 15776 KB, 39187 KB/s, 0 seconds passed
    ... 12%, 15808 KB, 39241 KB/s, 0 seconds passed
    ... 12%, 15840 KB, 39289 KB/s, 0 seconds passed
    ... 12%, 15872 KB, 39345 KB/s, 0 seconds passed
    ... 12%, 15904 KB, 39398 KB/s, 0 seconds passed
    ... 12%, 15936 KB, 39446 KB/s, 0 seconds passed
    ... 12%, 15968 KB, 39501 KB/s, 0 seconds passed
    ... 12%, 16000 KB, 39555 KB/s, 0 seconds passed
    ... 12%, 16032 KB, 39609 KB/s, 0 seconds passed
    ... 12%, 16064 KB, 39656 KB/s, 0 seconds passed
    ... 12%, 16096 KB, 39711 KB/s, 0 seconds passed
    ... 12%, 16128 KB, 39764 KB/s, 0 seconds passed
    ... 12%, 16160 KB, 39812 KB/s, 0 seconds passed
    ... 12%, 16192 KB, 39866 KB/s, 0 seconds passed
    ... 12%, 16224 KB, 39913 KB/s, 0 seconds passed
    ... 12%, 16256 KB, 39967 KB/s, 0 seconds passed
    ... 12%, 16288 KB, 40020 KB/s, 0 seconds passed
    ... 12%, 16320 KB, 40067 KB/s, 0 seconds passed
    ... 12%, 16352 KB, 40121 KB/s, 0 seconds passed
    ... 13%, 16384 KB, 40173 KB/s, 0 seconds passed
    ... 13%, 16416 KB, 40220 KB/s, 0 seconds passed
    ... 13%, 16448 KB, 40274 KB/s, 0 seconds passed
    ... 13%, 16480 KB, 40320 KB/s, 0 seconds passed
    ... 13%, 16512 KB, 40369 KB/s, 0 seconds passed
    ... 13%, 16544 KB, 40426 KB/s, 0 seconds passed
    ... 13%, 16576 KB, 40473 KB/s, 0 seconds passed
    ... 13%, 16608 KB, 40526 KB/s, 0 seconds passed
    ... 13%, 16640 KB, 40578 KB/s, 0 seconds passed
    ... 13%, 16672 KB, 40630 KB/s, 0 seconds passed
    ... 13%, 16704 KB, 40677 KB/s, 0 seconds passed
    ... 13%, 16736 KB, 40723 KB/s, 0 seconds passed
    ... 13%, 16768 KB, 40775 KB/s, 0 seconds passed
    ... 13%, 16800 KB, 40817 KB/s, 0 seconds passed
    ... 13%, 16832 KB, 40854 KB/s, 0 seconds passed
    ... 13%, 16864 KB, 40893 KB/s, 0 seconds passed
    ... 13%, 16896 KB, 40956 KB/s, 0 seconds passed
    ... 13%, 16928 KB, 41019 KB/s, 0 seconds passed
    ... 13%, 16960 KB, 41072 KB/s, 0 seconds passed
    ... 13%, 16992 KB, 41118 KB/s, 0 seconds passed
    ... 13%, 17024 KB, 41169 KB/s, 0 seconds passed
    ... 13%, 17056 KB, 41221 KB/s, 0 seconds passed
    ... 13%, 17088 KB, 41272 KB/s, 0 seconds passed
    ... 13%, 17120 KB, 41317 KB/s, 0 seconds passed
    ... 13%, 17152 KB, 41369 KB/s, 0 seconds passed
    ... 13%, 17184 KB, 41420 KB/s, 0 seconds passed
    ... 13%, 17216 KB, 41465 KB/s, 0 seconds passed
    ... 13%, 17248 KB, 41516 KB/s, 0 seconds passed
    ... 13%, 17280 KB, 41567 KB/s, 0 seconds passed
    ... 13%, 17312 KB, 41612 KB/s, 0 seconds passed
    ... 13%, 17344 KB, 41659 KB/s, 0 seconds passed
    ... 13%, 17376 KB, 41696 KB/s, 0 seconds passed
    ... 13%, 17408 KB, 41756 KB/s, 0 seconds passed
    ... 13%, 17440 KB, 41808 KB/s, 0 seconds passed
    ... 13%, 17472 KB, 41854 KB/s, 0 seconds passed
    ... 13%, 17504 KB, 41891 KB/s, 0 seconds passed
    ... 13%, 17536 KB, 41926 KB/s, 0 seconds passed
    ... 13%, 17568 KB, 41975 KB/s, 0 seconds passed
    ... 13%, 17600 KB, 42037 KB/s, 0 seconds passed
    ... 13%, 17632 KB, 42100 KB/s, 0 seconds passed
    ... 14%, 17664 KB, 42152 KB/s, 0 seconds passed
    ... 14%, 17696 KB, 42190 KB/s, 0 seconds passed
    ... 14%, 17728 KB, 42227 KB/s, 0 seconds passed
    ... 14%, 17760 KB, 42262 KB/s, 0 seconds passed
    ... 14%, 17792 KB, 42313 KB/s, 0 seconds passed
    ... 14%, 17824 KB, 42375 KB/s, 0 seconds passed
    ... 14%, 17856 KB, 42436 KB/s, 0 seconds passed
    ... 14%, 17888 KB, 42485 KB/s, 0 seconds passed
    ... 14%, 17920 KB, 42535 KB/s, 0 seconds passed
    ... 14%, 17952 KB, 42578 KB/s, 0 seconds passed
    ... 14%, 17984 KB, 42630 KB/s, 0 seconds passed
    ... 14%, 18016 KB, 42679 KB/s, 0 seconds passed
    ... 14%, 18048 KB, 42723 KB/s, 0 seconds passed
    ... 14%, 18080 KB, 42772 KB/s, 0 seconds passed
    ... 14%, 18112 KB, 42821 KB/s, 0 seconds passed
    ... 14%, 18144 KB, 42865 KB/s, 0 seconds passed
    ... 14%, 18176 KB, 42908 KB/s, 0 seconds passed
    ... 14%, 18208 KB, 42942 KB/s, 0 seconds passed
    ... 14%, 18240 KB, 42980 KB/s, 0 seconds passed
    ... 14%, 18272 KB, 43043 KB/s, 0 seconds passed
    ... 14%, 18304 KB, 43103 KB/s, 0 seconds passed
    ... 14%, 18336 KB, 43143 KB/s, 0 seconds passed
    ... 14%, 18368 KB, 43177 KB/s, 0 seconds passed
    ... 14%, 18400 KB, 43236 KB/s, 0 seconds passed
    ... 14%, 18432 KB, 43290 KB/s, 0 seconds passed
    ... 14%, 18464 KB, 43338 KB/s, 0 seconds passed
    ... 14%, 18496 KB, 43381 KB/s, 0 seconds passed
    ... 14%, 18528 KB, 43430 KB/s, 0 seconds passed
    ... 14%, 18560 KB, 43478 KB/s, 0 seconds passed
    ... 14%, 18592 KB, 43521 KB/s, 0 seconds passed
    ... 14%, 18624 KB, 43569 KB/s, 0 seconds passed
    ... 14%, 18656 KB, 43618 KB/s, 0 seconds passed
    ... 14%, 18688 KB, 43660 KB/s, 0 seconds passed
    ... 14%, 18720 KB, 43708 KB/s, 0 seconds passed
    ... 14%, 18752 KB, 43757 KB/s, 0 seconds passed
    ... 14%, 18784 KB, 43799 KB/s, 0 seconds passed
    ... 14%, 18816 KB, 43847 KB/s, 0 seconds passed
    ... 14%, 18848 KB, 43894 KB/s, 0 seconds passed
    ... 14%, 18880 KB, 43937 KB/s, 0 seconds passed
    ... 15%, 18912 KB, 43985 KB/s, 0 seconds passed
    ... 15%, 18944 KB, 44027 KB/s, 0 seconds passed
    ... 15%, 18976 KB, 44068 KB/s, 0 seconds passed
    ... 15%, 19008 KB, 44100 KB/s, 0 seconds passed
    ... 15%, 19040 KB, 44145 KB/s, 0 seconds passed
    ... 15%, 19072 KB, 44207 KB/s, 0 seconds passed
    ... 15%, 19104 KB, 44257 KB/s, 0 seconds passed
    ... 15%, 19136 KB, 44298 KB/s, 0 seconds passed
    ... 15%, 19168 KB, 44335 KB/s, 0 seconds passed
    ... 15%, 19200 KB, 44372 KB/s, 0 seconds passed
    ... 15%, 19232 KB, 44408 KB/s, 0 seconds passed
    ... 15%, 19264 KB, 44444 KB/s, 0 seconds passed
    ... 15%, 19296 KB, 44481 KB/s, 0 seconds passed
    ... 15%, 19328 KB, 44518 KB/s, 0 seconds passed
    ... 15%, 19360 KB, 44555 KB/s, 0 seconds passed
    ... 15%, 19392 KB, 44592 KB/s, 0 seconds passed
    ... 15%, 19424 KB, 44629 KB/s, 0 seconds passed
    ... 15%, 19456 KB, 44665 KB/s, 0 seconds passed
    ... 15%, 19488 KB, 44703 KB/s, 0 seconds passed
    ... 15%, 19520 KB, 44740 KB/s, 0 seconds passed
    ... 15%, 19552 KB, 44777 KB/s, 0 seconds passed
    ... 15%, 19584 KB, 44810 KB/s, 0 seconds passed
    ... 15%, 19616 KB, 44846 KB/s, 0 seconds passed
    ... 15%, 19648 KB, 44883 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 19680 KB, 44918 KB/s, 0 seconds passed
    ... 15%, 19712 KB, 44955 KB/s, 0 seconds passed
    ... 15%, 19744 KB, 44991 KB/s, 0 seconds passed
    ... 15%, 19776 KB, 45025 KB/s, 0 seconds passed
    ... 15%, 19808 KB, 45061 KB/s, 0 seconds passed
    ... 15%, 19840 KB, 45098 KB/s, 0 seconds passed
    ... 15%, 19872 KB, 45134 KB/s, 0 seconds passed
    ... 15%, 19904 KB, 45170 KB/s, 0 seconds passed
    ... 15%, 19936 KB, 45205 KB/s, 0 seconds passed
    ... 15%, 19968 KB, 45238 KB/s, 0 seconds passed
    ... 15%, 20000 KB, 45282 KB/s, 0 seconds passed
    ... 15%, 20032 KB, 45330 KB/s, 0 seconds passed
    ... 15%, 20064 KB, 45378 KB/s, 0 seconds passed
    ... 15%, 20096 KB, 45426 KB/s, 0 seconds passed
    ... 15%, 20128 KB, 45473 KB/s, 0 seconds passed
    ... 16%, 20160 KB, 45520 KB/s, 0 seconds passed
    ... 16%, 20192 KB, 45568 KB/s, 0 seconds passed
    ... 16%, 20224 KB, 45615 KB/s, 0 seconds passed
    ... 16%, 20256 KB, 45663 KB/s, 0 seconds passed
    ... 16%, 20288 KB, 45709 KB/s, 0 seconds passed
    ... 16%, 20320 KB, 45757 KB/s, 0 seconds passed
    ... 16%, 20352 KB, 45802 KB/s, 0 seconds passed
    ... 16%, 20384 KB, 45847 KB/s, 0 seconds passed
    ... 16%, 20416 KB, 45894 KB/s, 0 seconds passed
    ... 16%, 20448 KB, 45938 KB/s, 0 seconds passed
    ... 16%, 20480 KB, 45984 KB/s, 0 seconds passed
    ... 16%, 20512 KB, 46031 KB/s, 0 seconds passed
    ... 16%, 20544 KB, 46077 KB/s, 0 seconds passed
    ... 16%, 20576 KB, 46124 KB/s, 0 seconds passed
    ... 16%, 20608 KB, 46170 KB/s, 0 seconds passed
    ... 16%, 20640 KB, 46215 KB/s, 0 seconds passed
    ... 16%, 20672 KB, 46261 KB/s, 0 seconds passed
    ... 16%, 20704 KB, 46308 KB/s, 0 seconds passed
    ... 16%, 20736 KB, 46355 KB/s, 0 seconds passed
    ... 16%, 20768 KB, 46401 KB/s, 0 seconds passed
    ... 16%, 20800 KB, 46447 KB/s, 0 seconds passed
    ... 16%, 20832 KB, 46493 KB/s, 0 seconds passed
    ... 16%, 20864 KB, 46540 KB/s, 0 seconds passed
    ... 16%, 20896 KB, 46586 KB/s, 0 seconds passed
    ... 16%, 20928 KB, 46631 KB/s, 0 seconds passed
    ... 16%, 20960 KB, 46675 KB/s, 0 seconds passed
    ... 16%, 20992 KB, 46721 KB/s, 0 seconds passed
    ... 16%, 21024 KB, 46766 KB/s, 0 seconds passed
    ... 16%, 21056 KB, 46811 KB/s, 0 seconds passed
    ... 16%, 21088 KB, 46857 KB/s, 0 seconds passed
    ... 16%, 21120 KB, 46903 KB/s, 0 seconds passed
    ... 16%, 21152 KB, 46949 KB/s, 0 seconds passed
    ... 16%, 21184 KB, 46994 KB/s, 0 seconds passed
    ... 16%, 21216 KB, 47040 KB/s, 0 seconds passed
    ... 16%, 21248 KB, 47084 KB/s, 0 seconds passed
    ... 16%, 21280 KB, 47130 KB/s, 0 seconds passed
    ... 16%, 21312 KB, 47180 KB/s, 0 seconds passed
    ... 16%, 21344 KB, 47235 KB/s, 0 seconds passed
    ... 16%, 21376 KB, 47288 KB/s, 0 seconds passed
    ... 16%, 21408 KB, 47341 KB/s, 0 seconds passed
    ... 17%, 21440 KB, 47395 KB/s, 0 seconds passed
    ... 17%, 21472 KB, 47449 KB/s, 0 seconds passed
    ... 17%, 21504 KB, 47502 KB/s, 0 seconds passed
    ... 17%, 21536 KB, 47557 KB/s, 0 seconds passed
    ... 17%, 21568 KB, 47596 KB/s, 0 seconds passed
    ... 17%, 21600 KB, 47639 KB/s, 0 seconds passed
    ... 17%, 21632 KB, 47683 KB/s, 0 seconds passed
    ... 17%, 21664 KB, 47720 KB/s, 0 seconds passed
    ... 17%, 21696 KB, 47762 KB/s, 0 seconds passed
    ... 17%, 21728 KB, 47806 KB/s, 0 seconds passed
    ... 17%, 21760 KB, 47848 KB/s, 0 seconds passed
    ... 17%, 21792 KB, 47885 KB/s, 0 seconds passed
    ... 17%, 21824 KB, 47928 KB/s, 0 seconds passed
    ... 17%, 21856 KB, 47965 KB/s, 0 seconds passed
    ... 17%, 21888 KB, 48007 KB/s, 0 seconds passed
    ... 17%, 21920 KB, 48050 KB/s, 0 seconds passed
    ... 17%, 21952 KB, 48093 KB/s, 0 seconds passed
    ... 17%, 21984 KB, 48129 KB/s, 0 seconds passed
    ... 17%, 22016 KB, 48171 KB/s, 0 seconds passed
    ... 17%, 22048 KB, 48214 KB/s, 0 seconds passed
    ... 17%, 22080 KB, 48251 KB/s, 0 seconds passed
    ... 17%, 22112 KB, 48293 KB/s, 0 seconds passed
    ... 17%, 22144 KB, 48330 KB/s, 0 seconds passed
    ... 17%, 22176 KB, 48372 KB/s, 0 seconds passed
    ... 17%, 22208 KB, 48414 KB/s, 0 seconds passed
    ... 17%, 22240 KB, 48456 KB/s, 0 seconds passed
    ... 17%, 22272 KB, 48493 KB/s, 0 seconds passed
    ... 17%, 22304 KB, 48534 KB/s, 0 seconds passed
    ... 17%, 22336 KB, 48576 KB/s, 0 seconds passed
    ... 17%, 22368 KB, 48618 KB/s, 0 seconds passed
    ... 17%, 22400 KB, 48654 KB/s, 0 seconds passed
    ... 17%, 22432 KB, 48702 KB/s, 0 seconds passed
    ... 17%, 22464 KB, 48732 KB/s, 0 seconds passed
    ... 17%, 22496 KB, 48774 KB/s, 0 seconds passed
    ... 17%, 22528 KB, 48816 KB/s, 0 seconds passed
    ... 17%, 22560 KB, 48857 KB/s, 0 seconds passed
    ... 17%, 22592 KB, 48894 KB/s, 0 seconds passed
    ... 17%, 22624 KB, 48935 KB/s, 0 seconds passed
    ... 17%, 22656 KB, 48977 KB/s, 0 seconds passed
    ... 18%, 22688 KB, 49013 KB/s, 0 seconds passed
    ... 18%, 22720 KB, 49054 KB/s, 0 seconds passed
    ... 18%, 22752 KB, 49086 KB/s, 0 seconds passed
    ... 18%, 22784 KB, 49113 KB/s, 0 seconds passed
    ... 18%, 22816 KB, 49140 KB/s, 0 seconds passed
    ... 18%, 22848 KB, 49167 KB/s, 0 seconds passed
    ... 18%, 22880 KB, 49222 KB/s, 0 seconds passed
    ... 18%, 22912 KB, 49276 KB/s, 0 seconds passed
    ... 18%, 22944 KB, 49329 KB/s, 0 seconds passed
    ... 18%, 22976 KB, 49370 KB/s, 0 seconds passed
    ... 18%, 23008 KB, 49411 KB/s, 0 seconds passed
    ... 18%, 23040 KB, 49447 KB/s, 0 seconds passed
    ... 18%, 23072 KB, 49487 KB/s, 0 seconds passed
    ... 18%, 23104 KB, 49523 KB/s, 0 seconds passed
    ... 18%, 23136 KB, 49564 KB/s, 0 seconds passed
    ... 18%, 23168 KB, 49604 KB/s, 0 seconds passed
    ... 18%, 23200 KB, 49639 KB/s, 0 seconds passed
    ... 18%, 23232 KB, 49680 KB/s, 0 seconds passed
    ... 18%, 23264 KB, 49715 KB/s, 0 seconds passed
    ... 18%, 23296 KB, 49756 KB/s, 0 seconds passed
    ... 18%, 23328 KB, 49796 KB/s, 0 seconds passed
    ... 18%, 23360 KB, 49837 KB/s, 0 seconds passed
    ... 18%, 23392 KB, 49870 KB/s, 0 seconds passed
    ... 18%, 23424 KB, 49912 KB/s, 0 seconds passed
    ... 18%, 23456 KB, 49952 KB/s, 0 seconds passed
    ... 18%, 23488 KB, 49986 KB/s, 0 seconds passed
    ... 18%, 23520 KB, 50028 KB/s, 0 seconds passed
    ... 18%, 23552 KB, 50068 KB/s, 0 seconds passed
    ... 18%, 23584 KB, 50103 KB/s, 0 seconds passed
    ... 18%, 23616 KB, 50137 KB/s, 0 seconds passed
    ... 18%, 23648 KB, 50178 KB/s, 0 seconds passed
    ... 18%, 23680 KB, 50218 KB/s, 0 seconds passed
    ... 18%, 23712 KB, 50258 KB/s, 0 seconds passed
    ... 18%, 23744 KB, 50297 KB/s, 0 seconds passed
    ... 18%, 23776 KB, 50331 KB/s, 0 seconds passed
    ... 18%, 23808 KB, 50371 KB/s, 0 seconds passed
    ... 18%, 23840 KB, 50411 KB/s, 0 seconds passed
    ... 18%, 23872 KB, 50445 KB/s, 0 seconds passed
    ... 18%, 23904 KB, 50485 KB/s, 0 seconds passed
    ... 19%, 23936 KB, 50520 KB/s, 0 seconds passed
    ... 19%, 23968 KB, 50559 KB/s, 0 seconds passed
    ... 19%, 24000 KB, 50598 KB/s, 0 seconds passed
    ... 19%, 24032 KB, 50632 KB/s, 0 seconds passed
    ... 19%, 24064 KB, 50672 KB/s, 0 seconds passed
    ... 19%, 24096 KB, 50711 KB/s, 0 seconds passed
    ... 19%, 24128 KB, 50751 KB/s, 0 seconds passed
    ... 19%, 24160 KB, 50785 KB/s, 0 seconds passed
    ... 19%, 24192 KB, 50823 KB/s, 0 seconds passed
    ... 19%, 24224 KB, 50857 KB/s, 0 seconds passed
    ... 19%, 24256 KB, 50897 KB/s, 0 seconds passed
    ... 19%, 24288 KB, 50936 KB/s, 0 seconds passed
    ... 19%, 24320 KB, 50969 KB/s, 0 seconds passed
    ... 19%, 24352 KB, 51008 KB/s, 0 seconds passed
    ... 19%, 24384 KB, 51047 KB/s, 0 seconds passed
    ... 19%, 24416 KB, 51086 KB/s, 0 seconds passed
    ... 19%, 24448 KB, 51120 KB/s, 0 seconds passed
    ... 19%, 24480 KB, 51159 KB/s, 0 seconds passed
    ... 19%, 24512 KB, 51192 KB/s, 0 seconds passed
    ... 19%, 24544 KB, 51231 KB/s, 0 seconds passed
    ... 19%, 24576 KB, 51269 KB/s, 0 seconds passed
    ... 19%, 24608 KB, 51309 KB/s, 0 seconds passed
    ... 19%, 24640 KB, 51342 KB/s, 0 seconds passed
    ... 19%, 24672 KB, 51380 KB/s, 0 seconds passed
    ... 19%, 24704 KB, 51414 KB/s, 0 seconds passed
    ... 19%, 24736 KB, 51452 KB/s, 0 seconds passed
    ... 19%, 24768 KB, 51490 KB/s, 0 seconds passed
    ... 19%, 24800 KB, 51523 KB/s, 0 seconds passed
    ... 19%, 24832 KB, 51556 KB/s, 0 seconds passed
    ... 19%, 24864 KB, 51580 KB/s, 0 seconds passed
    ... 19%, 24896 KB, 51604 KB/s, 0 seconds passed
    ... 19%, 24928 KB, 51650 KB/s, 0 seconds passed
    ... 19%, 24960 KB, 51698 KB/s, 0 seconds passed
    ... 19%, 24992 KB, 51745 KB/s, 0 seconds passed
    ... 19%, 25024 KB, 51783 KB/s, 0 seconds passed
    ... 19%, 25056 KB, 51816 KB/s, 0 seconds passed
    ... 19%, 25088 KB, 51854 KB/s, 0 seconds passed
    ... 19%, 25120 KB, 51892 KB/s, 0 seconds passed
    ... 19%, 25152 KB, 51924 KB/s, 0 seconds passed
    ... 19%, 25184 KB, 51963 KB/s, 0 seconds passed
    ... 20%, 25216 KB, 52000 KB/s, 0 seconds passed
    ... 20%, 25248 KB, 52032 KB/s, 0 seconds passed
    ... 20%, 25280 KB, 52071 KB/s, 0 seconds passed
    ... 20%, 25312 KB, 52108 KB/s, 0 seconds passed
    ... 20%, 25344 KB, 52141 KB/s, 0 seconds passed
    ... 20%, 25376 KB, 52179 KB/s, 0 seconds passed
    ... 20%, 25408 KB, 52210 KB/s, 0 seconds passed
    ... 20%, 25440 KB, 52248 KB/s, 0 seconds passed
    ... 20%, 25472 KB, 52286 KB/s, 0 seconds passed
    ... 20%, 25504 KB, 52318 KB/s, 0 seconds passed
    ... 20%, 25536 KB, 52356 KB/s, 0 seconds passed
    ... 20%, 25568 KB, 52393 KB/s, 0 seconds passed
    ... 20%, 25600 KB, 52442 KB/s, 0 seconds passed
    ... 20%, 25632 KB, 52468 KB/s, 0 seconds passed
    ... 20%, 25664 KB, 52500 KB/s, 0 seconds passed
    ... 20%, 25696 KB, 52537 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 25728 KB, 52580 KB/s, 0 seconds passed
    ... 20%, 25760 KB, 52612 KB/s, 0 seconds passed
    ... 20%, 25792 KB, 52643 KB/s, 0 seconds passed
    ... 20%, 25824 KB, 52681 KB/s, 0 seconds passed
    ... 20%, 25856 KB, 52718 KB/s, 0 seconds passed
    ... 20%, 25888 KB, 52755 KB/s, 0 seconds passed
    ... 20%, 25920 KB, 52792 KB/s, 0 seconds passed
    ... 20%, 25952 KB, 52823 KB/s, 0 seconds passed
    ... 20%, 25984 KB, 52866 KB/s, 0 seconds passed
    ... 20%, 26016 KB, 52898 KB/s, 0 seconds passed
    ... 20%, 26048 KB, 52929 KB/s, 0 seconds passed
    ... 20%, 26080 KB, 52960 KB/s, 0 seconds passed
    ... 20%, 26112 KB, 53003 KB/s, 0 seconds passed
    ... 20%, 26144 KB, 53033 KB/s, 0 seconds passed
    ... 20%, 26176 KB, 53065 KB/s, 0 seconds passed
    ... 20%, 26208 KB, 53096 KB/s, 0 seconds passed
    ... 20%, 26240 KB, 53138 KB/s, 0 seconds passed
    ... 20%, 26272 KB, 53170 KB/s, 0 seconds passed
    ... 20%, 26304 KB, 53207 KB/s, 0 seconds passed
    ... 20%, 26336 KB, 53243 KB/s, 0 seconds passed
    ... 20%, 26368 KB, 53269 KB/s, 0 seconds passed
    ... 20%, 26400 KB, 53305 KB/s, 0 seconds passed
    ... 20%, 26432 KB, 53341 KB/s, 0 seconds passed
    ... 21%, 26464 KB, 53372 KB/s, 0 seconds passed
    ... 21%, 26496 KB, 53414 KB/s, 0 seconds passed
    ... 21%, 26528 KB, 53451 KB/s, 0 seconds passed
    ... 21%, 26560 KB, 53482 KB/s, 0 seconds passed
    ... 21%, 26592 KB, 53512 KB/s, 0 seconds passed
    ... 21%, 26624 KB, 53543 KB/s, 0 seconds passed
    ... 21%, 26656 KB, 53579 KB/s, 0 seconds passed
    ... 21%, 26688 KB, 53627 KB/s, 0 seconds passed
    ... 21%, 26720 KB, 53657 KB/s, 0 seconds passed
    ... 21%, 26752 KB, 53693 KB/s, 0 seconds passed
    ... 21%, 26784 KB, 53724 KB/s, 0 seconds passed
    ... 21%, 26816 KB, 53748 KB/s, 0 seconds passed
    ... 21%, 26848 KB, 53784 KB/s, 0 seconds passed
    ... 21%, 26880 KB, 53820 KB/s, 0 seconds passed
    ... 21%, 26912 KB, 53856 KB/s, 0 seconds passed
    ... 21%, 26944 KB, 53892 KB/s, 0 seconds passed
    ... 21%, 26976 KB, 53928 KB/s, 0 seconds passed
    ... 21%, 27008 KB, 53964 KB/s, 0 seconds passed
    ... 21%, 27040 KB, 53994 KB/s, 0 seconds passed
    ... 21%, 27072 KB, 54024 KB/s, 0 seconds passed
    ... 21%, 27104 KB, 54054 KB/s, 0 seconds passed
    ... 21%, 27136 KB, 54100 KB/s, 0 seconds passed
    ... 21%, 27168 KB, 54125 KB/s, 0 seconds passed
    ... 21%, 27200 KB, 54161 KB/s, 0 seconds passed
    ... 21%, 27232 KB, 54202 KB/s, 0 seconds passed
    ... 21%, 27264 KB, 54232 KB/s, 0 seconds passed
    ... 21%, 27296 KB, 54262 KB/s, 0 seconds passed
    ... 21%, 27328 KB, 54303 KB/s, 0 seconds passed
    ... 21%, 27360 KB, 54333 KB/s, 0 seconds passed
    ... 21%, 27392 KB, 54362 KB/s, 0 seconds passed
    ... 21%, 27424 KB, 54392 KB/s, 0 seconds passed
    ... 21%, 27456 KB, 54438 KB/s, 0 seconds passed
    ... 21%, 27488 KB, 54468 KB/s, 0 seconds passed
    ... 21%, 27520 KB, 54503 KB/s, 0 seconds passed
    ... 21%, 27552 KB, 54526 KB/s, 0 seconds passed
    ... 21%, 27584 KB, 54562 KB/s, 0 seconds passed
    ... 21%, 27616 KB, 54597 KB/s, 0 seconds passed
    ... 21%, 27648 KB, 54621 KB/s, 0 seconds passed
    ... 21%, 27680 KB, 54667 KB/s, 0 seconds passed
    ... 22%, 27712 KB, 54702 KB/s, 0 seconds passed
    ... 22%, 27744 KB, 54737 KB/s, 0 seconds passed
    ... 22%, 27776 KB, 54761 KB/s, 0 seconds passed
    ... 22%, 27808 KB, 54790 KB/s, 0 seconds passed
    ... 22%, 27840 KB, 54836 KB/s, 0 seconds passed
    ... 22%, 27872 KB, 54865 KB/s, 0 seconds passed
    ... 22%, 27904 KB, 54900 KB/s, 0 seconds passed
    ... 22%, 27936 KB, 54935 KB/s, 0 seconds passed
    ... 22%, 27968 KB, 54952 KB/s, 0 seconds passed
    ... 22%, 28000 KB, 54987 KB/s, 0 seconds passed
    ... 22%, 28032 KB, 55033 KB/s, 0 seconds passed
    ... 22%, 28064 KB, 55062 KB/s, 0 seconds passed
    ... 22%, 28096 KB, 55085 KB/s, 0 seconds passed
    ... 22%, 28128 KB, 55120 KB/s, 0 seconds passed
    ... 22%, 28160 KB, 55149 KB/s, 0 seconds passed
    ... 22%, 28192 KB, 55177 KB/s, 0 seconds passed
    ... 22%, 28224 KB, 55217 KB/s, 0 seconds passed
    ... 22%, 28256 KB, 55247 KB/s, 0 seconds passed
    ... 22%, 28288 KB, 55281 KB/s, 0 seconds passed
    ... 22%, 28320 KB, 55315 KB/s, 0 seconds passed
    ... 22%, 28352 KB, 55349 KB/s, 0 seconds passed
    ... 22%, 28384 KB, 55379 KB/s, 0 seconds passed
    ... 22%, 28416 KB, 55399 KB/s, 0 seconds passed
    ... 22%, 28448 KB, 55414 KB/s, 0 seconds passed
    ... 22%, 28480 KB, 55461 KB/s, 0 seconds passed
    ... 22%, 28512 KB, 55507 KB/s, 0 seconds passed
    ... 22%, 28544 KB, 55551 KB/s, 0 seconds passed
    ... 22%, 28576 KB, 55569 KB/s, 0 seconds passed
    ... 22%, 28608 KB, 55598 KB/s, 0 seconds passed
    ... 22%, 28640 KB, 55632 KB/s, 0 seconds passed
    ... 22%, 28672 KB, 55671 KB/s, 0 seconds passed
    ... 22%, 28704 KB, 55705 KB/s, 0 seconds passed
    ... 22%, 28736 KB, 55739 KB/s, 0 seconds passed
    ... 22%, 28768 KB, 55767 KB/s, 0 seconds passed
    ... 22%, 28800 KB, 55801 KB/s, 0 seconds passed
    ... 22%, 28832 KB, 55835 KB/s, 0 seconds passed
    ... 22%, 28864 KB, 55869 KB/s, 0 seconds passed
    ... 22%, 28896 KB, 55896 KB/s, 0 seconds passed
    ... 22%, 28928 KB, 55919 KB/s, 0 seconds passed
    ... 22%, 28960 KB, 55953 KB/s, 0 seconds passed
    ... 23%, 28992 KB, 55981 KB/s, 0 seconds passed
    ... 23%, 29024 KB, 56014 KB/s, 0 seconds passed
    ... 23%, 29056 KB, 56048 KB/s, 0 seconds passed
    ... 23%, 29088 KB, 56076 KB/s, 0 seconds passed
    ... 23%, 29120 KB, 56109 KB/s, 0 seconds passed
    ... 23%, 29152 KB, 56143 KB/s, 0 seconds passed
    ... 23%, 29184 KB, 56171 KB/s, 0 seconds passed
    ... 23%, 29216 KB, 56204 KB/s, 0 seconds passed
    ... 23%, 29248 KB, 56232 KB/s, 0 seconds passed
    ... 23%, 29280 KB, 56265 KB/s, 0 seconds passed
    ... 23%, 29312 KB, 56298 KB/s, 0 seconds passed
    ... 23%, 29344 KB, 56320 KB/s, 0 seconds passed
    ... 23%, 29376 KB, 56338 KB/s, 0 seconds passed
    ... 23%, 29408 KB, 56357 KB/s, 0 seconds passed
    ... 23%, 29440 KB, 56379 KB/s, 0 seconds passed
    ... 23%, 29472 KB, 56402 KB/s, 0 seconds passed
    ... 23%, 29504 KB, 56427 KB/s, 0 seconds passed
    ... 23%, 29536 KB, 56457 KB/s, 0 seconds passed
    ... 23%, 29568 KB, 56358 KB/s, 0 seconds passed
    ... 23%, 29600 KB, 56376 KB/s, 0 seconds passed
    ... 23%, 29632 KB, 56398 KB/s, 0 seconds passed
    ... 23%, 29664 KB, 56421 KB/s, 0 seconds passed
    ... 23%, 29696 KB, 56442 KB/s, 0 seconds passed
    ... 23%, 29728 KB, 56464 KB/s, 0 seconds passed
    ... 23%, 29760 KB, 56485 KB/s, 0 seconds passed
    ... 23%, 29792 KB, 56506 KB/s, 0 seconds passed
    ... 23%, 29824 KB, 56528 KB/s, 0 seconds passed
    ... 23%, 29856 KB, 56550 KB/s, 0 seconds passed
    ... 23%, 29888 KB, 56573 KB/s, 0 seconds passed
    ... 23%, 29920 KB, 56592 KB/s, 0 seconds passed
    ... 23%, 29952 KB, 56611 KB/s, 0 seconds passed
    ... 23%, 29984 KB, 56632 KB/s, 0 seconds passed
    ... 23%, 30016 KB, 56654 KB/s, 0 seconds passed
    ... 23%, 30048 KB, 56676 KB/s, 0 seconds passed
    ... 23%, 30080 KB, 56698 KB/s, 0 seconds passed
    ... 23%, 30112 KB, 56718 KB/s, 0 seconds passed
    ... 23%, 30144 KB, 56741 KB/s, 0 seconds passed
    ... 23%, 30176 KB, 56762 KB/s, 0 seconds passed
    ... 23%, 30208 KB, 56786 KB/s, 0 seconds passed
    ... 24%, 30240 KB, 56818 KB/s, 0 seconds passed
    ... 24%, 30272 KB, 56852 KB/s, 0 seconds passed
    ... 24%, 30304 KB, 56885 KB/s, 0 seconds passed
    ... 24%, 30336 KB, 56915 KB/s, 0 seconds passed
    ... 24%, 30368 KB, 56947 KB/s, 0 seconds passed
    ... 24%, 30400 KB, 56980 KB/s, 0 seconds passed
    ... 24%, 30432 KB, 57011 KB/s, 0 seconds passed
    ... 24%, 30464 KB, 57043 KB/s, 0 seconds passed
    ... 24%, 30496 KB, 57076 KB/s, 0 seconds passed
    ... 24%, 30528 KB, 57110 KB/s, 0 seconds passed
    ... 24%, 30560 KB, 57147 KB/s, 0 seconds passed
    ... 24%, 30592 KB, 57185 KB/s, 0 seconds passed
    ... 24%, 30624 KB, 57223 KB/s, 0 seconds passed
    ... 24%, 30656 KB, 57260 KB/s, 0 seconds passed
    ... 24%, 30688 KB, 57298 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 30720 KB, 52867 KB/s, 0 seconds passed
    ... 24%, 30752 KB, 52786 KB/s, 0 seconds passed
    ... 24%, 30784 KB, 52797 KB/s, 0 seconds passed
    ... 24%, 30816 KB, 52817 KB/s, 0 seconds passed
    ... 24%, 30848 KB, 52832 KB/s, 0 seconds passed
    ... 24%, 30880 KB, 52849 KB/s, 0 seconds passed
    ... 24%, 30912 KB, 52869 KB/s, 0 seconds passed
    ... 24%, 30944 KB, 52893 KB/s, 0 seconds passed
    ... 24%, 30976 KB, 52921 KB/s, 0 seconds passed
    ... 24%, 31008 KB, 52890 KB/s, 0 seconds passed
    ... 24%, 31040 KB, 52909 KB/s, 0 seconds passed
    ... 24%, 31072 KB, 52929 KB/s, 0 seconds passed
    ... 24%, 31104 KB, 52949 KB/s, 0 seconds passed
    ... 24%, 31136 KB, 52970 KB/s, 0 seconds passed
    ... 24%, 31168 KB, 52988 KB/s, 0 seconds passed
    ... 24%, 31200 KB, 53006 KB/s, 0 seconds passed
    ... 24%, 31232 KB, 53025 KB/s, 0 seconds passed
    ... 24%, 31264 KB, 53044 KB/s, 0 seconds passed
    ... 24%, 31296 KB, 53064 KB/s, 0 seconds passed
    ... 24%, 31328 KB, 53085 KB/s, 0 seconds passed
    ... 24%, 31360 KB, 53102 KB/s, 0 seconds passed
    ... 24%, 31392 KB, 53123 KB/s, 0 seconds passed
    ... 24%, 31424 KB, 53140 KB/s, 0 seconds passed
    ... 24%, 31456 KB, 53160 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 31488 KB, 53179 KB/s, 0 seconds passed
    ... 25%, 31520 KB, 53200 KB/s, 0 seconds passed
    ... 25%, 31552 KB, 53220 KB/s, 0 seconds passed
    ... 25%, 31584 KB, 53245 KB/s, 0 seconds passed
    ... 25%, 31616 KB, 53272 KB/s, 0 seconds passed
    ... 25%, 31648 KB, 53300 KB/s, 0 seconds passed
    ... 25%, 31680 KB, 53328 KB/s, 0 seconds passed
    ... 25%, 31712 KB, 53356 KB/s, 0 seconds passed
    ... 25%, 31744 KB, 53384 KB/s, 0 seconds passed
    ... 25%, 31776 KB, 53413 KB/s, 0 seconds passed
    ... 25%, 31808 KB, 53442 KB/s, 0 seconds passed
    ... 25%, 31840 KB, 53469 KB/s, 0 seconds passed
    ... 25%, 31872 KB, 53497 KB/s, 0 seconds passed
    ... 25%, 31904 KB, 53523 KB/s, 0 seconds passed
    ... 25%, 31936 KB, 53553 KB/s, 0 seconds passed
    ... 25%, 31968 KB, 53580 KB/s, 0 seconds passed
    ... 25%, 32000 KB, 53609 KB/s, 0 seconds passed
    ... 25%, 32032 KB, 53636 KB/s, 0 seconds passed
    ... 25%, 32064 KB, 53661 KB/s, 0 seconds passed
    ... 25%, 32096 KB, 53689 KB/s, 0 seconds passed
    ... 25%, 32128 KB, 53717 KB/s, 0 seconds passed
    ... 25%, 32160 KB, 53745 KB/s, 0 seconds passed
    ... 25%, 32192 KB, 53773 KB/s, 0 seconds passed
    ... 25%, 32224 KB, 53802 KB/s, 0 seconds passed
    ... 25%, 32256 KB, 53827 KB/s, 0 seconds passed
    ... 25%, 32288 KB, 53856 KB/s, 0 seconds passed
    ... 25%, 32320 KB, 53883 KB/s, 0 seconds passed
    ... 25%, 32352 KB, 53911 KB/s, 0 seconds passed
    ... 25%, 32384 KB, 53938 KB/s, 0 seconds passed
    ... 25%, 32416 KB, 53965 KB/s, 0 seconds passed
    ... 25%, 32448 KB, 53992 KB/s, 0 seconds passed
    ... 25%, 32480 KB, 54017 KB/s, 0 seconds passed
    ... 25%, 32512 KB, 54045 KB/s, 0 seconds passed
    ... 25%, 32544 KB, 54071 KB/s, 0 seconds passed
    ... 25%, 32576 KB, 54099 KB/s, 0 seconds passed
    ... 25%, 32608 KB, 54125 KB/s, 0 seconds passed
    ... 25%, 32640 KB, 54152 KB/s, 0 seconds passed
    ... 25%, 32672 KB, 54181 KB/s, 0 seconds passed
    ... 25%, 32704 KB, 54216 KB/s, 0 seconds passed
    ... 25%, 32736 KB, 54251 KB/s, 0 seconds passed
    ... 26%, 32768 KB, 54286 KB/s, 0 seconds passed
    ... 26%, 32800 KB, 54322 KB/s, 0 seconds passed
    ... 26%, 32832 KB, 54357 KB/s, 0 seconds passed
    ... 26%, 32864 KB, 54393 KB/s, 0 seconds passed
    ... 26%, 32896 KB, 54430 KB/s, 0 seconds passed
    ... 26%, 32928 KB, 54466 KB/s, 0 seconds passed
    ... 26%, 32960 KB, 54501 KB/s, 0 seconds passed
    ... 26%, 32992 KB, 54535 KB/s, 0 seconds passed
    ... 26%, 33024 KB, 54571 KB/s, 0 seconds passed
    ... 26%, 33056 KB, 54605 KB/s, 0 seconds passed
    ... 26%, 33088 KB, 54641 KB/s, 0 seconds passed
    ... 26%, 33120 KB, 54676 KB/s, 0 seconds passed
    ... 26%, 33152 KB, 54710 KB/s, 0 seconds passed
    ... 26%, 33184 KB, 54745 KB/s, 0 seconds passed
    ... 26%, 33216 KB, 54782 KB/s, 0 seconds passed
    ... 26%, 33248 KB, 54817 KB/s, 0 seconds passed
    ... 26%, 33280 KB, 54853 KB/s, 0 seconds passed
    ... 26%, 33312 KB, 54888 KB/s, 0 seconds passed
    ... 26%, 33344 KB, 54924 KB/s, 0 seconds passed
    ... 26%, 33376 KB, 54960 KB/s, 0 seconds passed
    ... 26%, 33408 KB, 54996 KB/s, 0 seconds passed
    ... 26%, 33440 KB, 55032 KB/s, 0 seconds passed
    ... 26%, 33472 KB, 55060 KB/s, 0 seconds passed
    ... 26%, 33504 KB, 55084 KB/s, 0 seconds passed
    ... 26%, 33536 KB, 55113 KB/s, 0 seconds passed
    ... 26%, 33568 KB, 55146 KB/s, 0 seconds passed
    ... 26%, 33600 KB, 55168 KB/s, 0 seconds passed
    ... 26%, 33632 KB, 55183 KB/s, 0 seconds passed
    ... 26%, 33664 KB, 55202 KB/s, 0 seconds passed
    ... 26%, 33696 KB, 55238 KB/s, 0 seconds passed
    ... 26%, 33728 KB, 55274 KB/s, 0 seconds passed
    ... 26%, 33760 KB, 55310 KB/s, 0 seconds passed
    ... 26%, 33792 KB, 55335 KB/s, 0 seconds passed
    ... 26%, 33824 KB, 55364 KB/s, 0 seconds passed
    ... 26%, 33856 KB, 55387 KB/s, 0 seconds passed
    ... 26%, 33888 KB, 55420 KB/s, 0 seconds passed
    ... 26%, 33920 KB, 55449 KB/s, 0 seconds passed
    ... 26%, 33952 KB, 55472 KB/s, 0 seconds passed
    ... 26%, 33984 KB, 55496 KB/s, 0 seconds passed
    ... 27%, 34016 KB, 55519 KB/s, 0 seconds passed
    ... 27%, 34048 KB, 55555 KB/s, 0 seconds passed
    ... 27%, 34080 KB, 55577 KB/s, 0 seconds passed
    ... 27%, 34112 KB, 55601 KB/s, 0 seconds passed
    ... 27%, 34144 KB, 55629 KB/s, 0 seconds passed
    ... 27%, 34176 KB, 55657 KB/s, 0 seconds passed
    ... 27%, 34208 KB, 55681 KB/s, 0 seconds passed
    ... 27%, 34240 KB, 55709 KB/s, 0 seconds passed
    ... 27%, 34272 KB, 55742 KB/s, 0 seconds passed
    ... 27%, 34304 KB, 54747 KB/s, 0 seconds passed
    ... 27%, 34336 KB, 54755 KB/s, 0 seconds passed
    ... 27%, 34368 KB, 54767 KB/s, 0 seconds passed
    ... 27%, 34400 KB, 54786 KB/s, 0 seconds passed
    ... 27%, 34432 KB, 54805 KB/s, 0 seconds passed
    ... 27%, 34464 KB, 54824 KB/s, 0 seconds passed
    ... 27%, 34496 KB, 54842 KB/s, 0 seconds passed
    ... 27%, 34528 KB, 54860 KB/s, 0 seconds passed
    ... 27%, 34560 KB, 54878 KB/s, 0 seconds passed
    ... 27%, 34592 KB, 54898 KB/s, 0 seconds passed
    ... 27%, 34624 KB, 54918 KB/s, 0 seconds passed
    ... 27%, 34656 KB, 54937 KB/s, 0 seconds passed
    ... 27%, 34688 KB, 54957 KB/s, 0 seconds passed
    ... 27%, 34720 KB, 54974 KB/s, 0 seconds passed
    ... 27%, 34752 KB, 54993 KB/s, 0 seconds passed
    ... 27%, 34784 KB, 55012 KB/s, 0 seconds passed
    ... 27%, 34816 KB, 55029 KB/s, 0 seconds passed
    ... 27%, 34848 KB, 55047 KB/s, 0 seconds passed
    ... 27%, 34880 KB, 55065 KB/s, 0 seconds passed
    ... 27%, 34912 KB, 55084 KB/s, 0 seconds passed
    ... 27%, 34944 KB, 55104 KB/s, 0 seconds passed
    ... 27%, 34976 KB, 55123 KB/s, 0 seconds passed
    ... 27%, 35008 KB, 55140 KB/s, 0 seconds passed
    ... 27%, 35040 KB, 55159 KB/s, 0 seconds passed
    ... 27%, 35072 KB, 55175 KB/s, 0 seconds passed
    ... 27%, 35104 KB, 55194 KB/s, 0 seconds passed
    ... 27%, 35136 KB, 55213 KB/s, 0 seconds passed
    ... 27%, 35168 KB, 55232 KB/s, 0 seconds passed
    ... 27%, 35200 KB, 55250 KB/s, 0 seconds passed
    ... 27%, 35232 KB, 55267 KB/s, 0 seconds passed
    ... 27%, 35264 KB, 55286 KB/s, 0 seconds passed
    ... 28%, 35296 KB, 55304 KB/s, 0 seconds passed
    ... 28%, 35328 KB, 55322 KB/s, 0 seconds passed
    ... 28%, 35360 KB, 55340 KB/s, 0 seconds passed
    ... 28%, 35392 KB, 55358 KB/s, 0 seconds passed
    ... 28%, 35424 KB, 55375 KB/s, 0 seconds passed
    ... 28%, 35456 KB, 55393 KB/s, 0 seconds passed
    ... 28%, 35488 KB, 55410 KB/s, 0 seconds passed
    ... 28%, 35520 KB, 55429 KB/s, 0 seconds passed
    ... 28%, 35552 KB, 55444 KB/s, 0 seconds passed
    ... 28%, 35584 KB, 55463 KB/s, 0 seconds passed
    ... 28%, 35616 KB, 55479 KB/s, 0 seconds passed
    ... 28%, 35648 KB, 55497 KB/s, 0 seconds passed
    ... 28%, 35680 KB, 55515 KB/s, 0 seconds passed
    ... 28%, 35712 KB, 55533 KB/s, 0 seconds passed

.. parsed-literal::

    ... 28%, 35744 KB, 55562 KB/s, 0 seconds passed
    ... 28%, 35776 KB, 55591 KB/s, 0 seconds passed
    ... 28%, 35808 KB, 55621 KB/s, 0 seconds passed
    ... 28%, 35840 KB, 55648 KB/s, 0 seconds passed
    ... 28%, 35872 KB, 55678 KB/s, 0 seconds passed
    ... 28%, 35904 KB, 55707 KB/s, 0 seconds passed
    ... 28%, 35936 KB, 55735 KB/s, 0 seconds passed
    ... 28%, 35968 KB, 55762 KB/s, 0 seconds passed
    ... 28%, 36000 KB, 55791 KB/s, 0 seconds passed
    ... 28%, 36032 KB, 55823 KB/s, 0 seconds passed
    ... 28%, 36064 KB, 55855 KB/s, 0 seconds passed
    ... 28%, 36096 KB, 55888 KB/s, 0 seconds passed
    ... 28%, 36128 KB, 55922 KB/s, 0 seconds passed
    ... 28%, 36160 KB, 55955 KB/s, 0 seconds passed
    ... 28%, 36192 KB, 55987 KB/s, 0 seconds passed
    ... 28%, 36224 KB, 56020 KB/s, 0 seconds passed
    ... 28%, 36256 KB, 56053 KB/s, 0 seconds passed
    ... 28%, 36288 KB, 56085 KB/s, 0 seconds passed
    ... 28%, 36320 KB, 56118 KB/s, 0 seconds passed
    ... 28%, 36352 KB, 56152 KB/s, 0 seconds passed
    ... 28%, 36384 KB, 56184 KB/s, 0 seconds passed
    ... 28%, 36416 KB, 56217 KB/s, 0 seconds passed
    ... 28%, 36448 KB, 56250 KB/s, 0 seconds passed
    ... 28%, 36480 KB, 56122 KB/s, 0 seconds passed
    ... 28%, 36512 KB, 56129 KB/s, 0 seconds passed
    ... 29%, 36544 KB, 56150 KB/s, 0 seconds passed
    ... 29%, 36576 KB, 56176 KB/s, 0 seconds passed
    ... 29%, 36608 KB, 56205 KB/s, 0 seconds passed
    ... 29%, 36640 KB, 56232 KB/s, 0 seconds passed
    ... 29%, 36672 KB, 56261 KB/s, 0 seconds passed
    ... 29%, 36704 KB, 56288 KB/s, 0 seconds passed
    ... 29%, 36736 KB, 56315 KB/s, 0 seconds passed
    ... 29%, 36768 KB, 56343 KB/s, 0 seconds passed
    ... 29%, 36800 KB, 56372 KB/s, 0 seconds passed
    ... 29%, 36832 KB, 56399 KB/s, 0 seconds passed
    ... 29%, 36864 KB, 56427 KB/s, 0 seconds passed
    ... 29%, 36896 KB, 56453 KB/s, 0 seconds passed
    ... 29%, 36928 KB, 56474 KB/s, 0 seconds passed
    ... 29%, 36960 KB, 56505 KB/s, 0 seconds passed
    ... 29%, 36992 KB, 56536 KB/s, 0 seconds passed
    ... 29%, 37024 KB, 56554 KB/s, 0 seconds passed
    ... 29%, 37056 KB, 56566 KB/s, 0 seconds passed
    ... 29%, 37088 KB, 56586 KB/s, 0 seconds passed
    ... 29%, 37120 KB, 56619 KB/s, 0 seconds passed
    ... 29%, 37152 KB, 56650 KB/s, 0 seconds passed
    ... 29%, 37184 KB, 56677 KB/s, 0 seconds passed
    ... 29%, 37216 KB, 56703 KB/s, 0 seconds passed
    ... 29%, 37248 KB, 56720 KB/s, 0 seconds passed
    ... 29%, 37280 KB, 56745 KB/s, 0 seconds passed
    ... 29%, 37312 KB, 56777 KB/s, 0 seconds passed
    ... 29%, 37344 KB, 56802 KB/s, 0 seconds passed
    ... 29%, 37376 KB, 56819 KB/s, 0 seconds passed
    ... 29%, 37408 KB, 56844 KB/s, 0 seconds passed
    ... 29%, 37440 KB, 56866 KB/s, 0 seconds passed
    ... 29%, 37472 KB, 56898 KB/s, 0 seconds passed
    ... 29%, 37504 KB, 56927 KB/s, 0 seconds passed
    ... 29%, 37536 KB, 56948 KB/s, 0 seconds passed
    ... 29%, 37568 KB, 56973 KB/s, 0 seconds passed
    ... 29%, 37600 KB, 56999 KB/s, 0 seconds passed
    ... 29%, 37632 KB, 57024 KB/s, 0 seconds passed
    ... 29%, 37664 KB, 57046 KB/s, 0 seconds passed
    ... 29%, 37696 KB, 57067 KB/s, 0 seconds passed
    ... 29%, 37728 KB, 57092 KB/s, 0 seconds passed
    ... 29%, 37760 KB, 57113 KB/s, 0 seconds passed
    ... 30%, 37792 KB, 57139 KB/s, 0 seconds passed
    ... 30%, 37824 KB, 57164 KB/s, 0 seconds passed
    ... 30%, 37856 KB, 57190 KB/s, 0 seconds passed
    ... 30%, 37888 KB, 57211 KB/s, 0 seconds passed
    ... 30%, 37920 KB, 57236 KB/s, 0 seconds passed
    ... 30%, 37952 KB, 57266 KB/s, 0 seconds passed
    ... 30%, 37984 KB, 57288 KB/s, 0 seconds passed
    ... 30%, 38016 KB, 57318 KB/s, 0 seconds passed
    ... 30%, 38048 KB, 57343 KB/s, 0 seconds passed
    ... 30%, 38080 KB, 57349 KB/s, 0 seconds passed
    ... 30%, 38112 KB, 57362 KB/s, 0 seconds passed
    ... 30%, 38144 KB, 57385 KB/s, 0 seconds passed
    ... 30%, 38176 KB, 57421 KB/s, 0 seconds passed
    ... 30%, 38208 KB, 57456 KB/s, 0 seconds passed
    ... 30%, 38240 KB, 57479 KB/s, 0 seconds passed
    ... 30%, 38272 KB, 57505 KB/s, 0 seconds passed
    ... 30%, 38304 KB, 57526 KB/s, 0 seconds passed
    ... 30%, 38336 KB, 57551 KB/s, 0 seconds passed
    ... 30%, 38368 KB, 57572 KB/s, 0 seconds passed
    ... 30%, 38400 KB, 57597 KB/s, 0 seconds passed
    ... 30%, 38432 KB, 57618 KB/s, 0 seconds passed
    ... 30%, 38464 KB, 57455 KB/s, 0 seconds passed
    ... 30%, 38496 KB, 57480 KB/s, 0 seconds passed
    ... 30%, 38528 KB, 57505 KB/s, 0 seconds passed
    ... 30%, 38560 KB, 57526 KB/s, 0 seconds passed
    ... 30%, 38592 KB, 57542 KB/s, 0 seconds passed
    ... 30%, 38624 KB, 57567 KB/s, 0 seconds passed
    ... 30%, 38656 KB, 57588 KB/s, 0 seconds passed
    ... 30%, 38688 KB, 57613 KB/s, 0 seconds passed
    ... 30%, 38720 KB, 57638 KB/s, 0 seconds passed
    ... 30%, 38752 KB, 57663 KB/s, 0 seconds passed
    ... 30%, 38784 KB, 57688 KB/s, 0 seconds passed
    ... 30%, 38816 KB, 57709 KB/s, 0 seconds passed
    ... 30%, 38848 KB, 57734 KB/s, 0 seconds passed
    ... 30%, 38880 KB, 57757 KB/s, 0 seconds passed
    ... 30%, 38912 KB, 57745 KB/s, 0 seconds passed
    ... 30%, 38944 KB, 57759 KB/s, 0 seconds passed
    ... 30%, 38976 KB, 57777 KB/s, 0 seconds passed
    ... 30%, 39008 KB, 57811 KB/s, 0 seconds passed
    ... 30%, 39040 KB, 57845 KB/s, 0 seconds passed
    ... 31%, 39072 KB, 57868 KB/s, 0 seconds passed
    ... 31%, 39104 KB, 57897 KB/s, 0 seconds passed
    ... 31%, 39136 KB, 57918 KB/s, 0 seconds passed
    ... 31%, 39168 KB, 57942 KB/s, 0 seconds passed
    ... 31%, 39200 KB, 57963 KB/s, 0 seconds passed
    ... 31%, 39232 KB, 57991 KB/s, 0 seconds passed
    ... 31%, 39264 KB, 58012 KB/s, 0 seconds passed
    ... 31%, 39296 KB, 58037 KB/s, 0 seconds passed
    ... 31%, 39328 KB, 58057 KB/s, 0 seconds passed
    ... 31%, 39360 KB, 58082 KB/s, 0 seconds passed
    ... 31%, 39392 KB, 58106 KB/s, 0 seconds passed
    ... 31%, 39424 KB, 58131 KB/s, 0 seconds passed
    ... 31%, 39456 KB, 58151 KB/s, 0 seconds passed
    ... 31%, 39488 KB, 58176 KB/s, 0 seconds passed
    ... 31%, 39520 KB, 58187 KB/s, 0 seconds passed
    ... 31%, 39552 KB, 58207 KB/s, 0 seconds passed
    ... 31%, 39584 KB, 58220 KB/s, 0 seconds passed
    ... 31%, 39616 KB, 58234 KB/s, 0 seconds passed
    ... 31%, 39648 KB, 58254 KB/s, 0 seconds passed
    ... 31%, 39680 KB, 58287 KB/s, 0 seconds passed
    ... 31%, 39712 KB, 58321 KB/s, 0 seconds passed
    ... 31%, 39744 KB, 58352 KB/s, 0 seconds passed
    ... 31%, 39776 KB, 58380 KB/s, 0 seconds passed
    ... 31%, 39808 KB, 58405 KB/s, 0 seconds passed
    ... 31%, 39840 KB, 58424 KB/s, 0 seconds passed
    ... 31%, 39872 KB, 58436 KB/s, 0 seconds passed
    ... 31%, 39904 KB, 58449 KB/s, 0 seconds passed
    ... 31%, 39936 KB, 58481 KB/s, 0 seconds passed
    ... 31%, 39968 KB, 58511 KB/s, 0 seconds passed
    ... 31%, 40000 KB, 58536 KB/s, 0 seconds passed
    ... 31%, 40032 KB, 58560 KB/s, 0 seconds passed
    ... 31%, 40064 KB, 58580 KB/s, 0 seconds passed
    ... 31%, 40096 KB, 58604 KB/s, 0 seconds passed
    ... 31%, 40128 KB, 58628 KB/s, 0 seconds passed
    ... 31%, 40160 KB, 58653 KB/s, 0 seconds passed
    ... 31%, 40192 KB, 58672 KB/s, 0 seconds passed
    ... 31%, 40224 KB, 58696 KB/s, 0 seconds passed
    ... 31%, 40256 KB, 58721 KB/s, 0 seconds passed
    ... 31%, 40288 KB, 58740 KB/s, 0 seconds passed
    ... 32%, 40320 KB, 58765 KB/s, 0 seconds passed
    ... 32%, 40352 KB, 58789 KB/s, 0 seconds passed
    ... 32%, 40384 KB, 58812 KB/s, 0 seconds passed
    ... 32%, 40416 KB, 58832 KB/s, 0 seconds passed
    ... 32%, 40448 KB, 58856 KB/s, 0 seconds passed
    ... 32%, 40480 KB, 58876 KB/s, 0 seconds passed
    ... 32%, 40512 KB, 58900 KB/s, 0 seconds passed
    ... 32%, 40544 KB, 58924 KB/s, 0 seconds passed
    ... 32%, 40576 KB, 58944 KB/s, 0 seconds passed
    ... 32%, 40608 KB, 58967 KB/s, 0 seconds passed
    ... 32%, 40640 KB, 58991 KB/s, 0 seconds passed
    ... 32%, 40672 KB, 59011 KB/s, 0 seconds passed
    ... 32%, 40704 KB, 59035 KB/s, 0 seconds passed
    ... 32%, 40736 KB, 59059 KB/s, 0 seconds passed
    ... 32%, 40768 KB, 59083 KB/s, 0 seconds passed
    ... 32%, 40800 KB, 59102 KB/s, 0 seconds passed
    ... 32%, 40832 KB, 59126 KB/s, 0 seconds passed
    ... 32%, 40864 KB, 59146 KB/s, 0 seconds passed
    ... 32%, 40896 KB, 59170 KB/s, 0 seconds passed
    ... 32%, 40928 KB, 59193 KB/s, 0 seconds passed

.. parsed-literal::

    ... 32%, 40960 KB, 57064 KB/s, 0 seconds passed
    ... 32%, 40992 KB, 57076 KB/s, 0 seconds passed
    ... 32%, 41024 KB, 57086 KB/s, 0 seconds passed
    ... 32%, 41056 KB, 57101 KB/s, 0 seconds passed
    ... 32%, 41088 KB, 57119 KB/s, 0 seconds passed
    ... 32%, 41120 KB, 57114 KB/s, 0 seconds passed
    ... 32%, 41152 KB, 57129 KB/s, 0 seconds passed
    ... 32%, 41184 KB, 57145 KB/s, 0 seconds passed
    ... 32%, 41216 KB, 57158 KB/s, 0 seconds passed
    ... 32%, 41248 KB, 57173 KB/s, 0 seconds passed
    ... 32%, 41280 KB, 57187 KB/s, 0 seconds passed
    ... 32%, 41312 KB, 57202 KB/s, 0 seconds passed
    ... 32%, 41344 KB, 57217 KB/s, 0 seconds passed
    ... 32%, 41376 KB, 57230 KB/s, 0 seconds passed
    ... 32%, 41408 KB, 57247 KB/s, 0 seconds passed
    ... 32%, 41440 KB, 57260 KB/s, 0 seconds passed
    ... 32%, 41472 KB, 57276 KB/s, 0 seconds passed
    ... 32%, 41504 KB, 57288 KB/s, 0 seconds passed
    ... 32%, 41536 KB, 57303 KB/s, 0 seconds passed
    ... 33%, 41568 KB, 57313 KB/s, 0 seconds passed
    ... 33%, 41600 KB, 57328 KB/s, 0 seconds passed
    ... 33%, 41632 KB, 57344 KB/s, 0 seconds passed
    ... 33%, 41664 KB, 57365 KB/s, 0 seconds passed
    ... 33%, 41696 KB, 57386 KB/s, 0 seconds passed
    ... 33%, 41728 KB, 57405 KB/s, 0 seconds passed
    ... 33%, 41760 KB, 57421 KB/s, 0 seconds passed
    ... 33%, 41792 KB, 57433 KB/s, 0 seconds passed
    ... 33%, 41824 KB, 57447 KB/s, 0 seconds passed
    ... 33%, 41856 KB, 57462 KB/s, 0 seconds passed
    ... 33%, 41888 KB, 57478 KB/s, 0 seconds passed
    ... 33%, 41920 KB, 57490 KB/s, 0 seconds passed
    ... 33%, 41952 KB, 57504 KB/s, 0 seconds passed
    ... 33%, 41984 KB, 57520 KB/s, 0 seconds passed
    ... 33%, 42016 KB, 57535 KB/s, 0 seconds passed
    ... 33%, 42048 KB, 57550 KB/s, 0 seconds passed
    ... 33%, 42080 KB, 57564 KB/s, 0 seconds passed
    ... 33%, 42112 KB, 57579 KB/s, 0 seconds passed
    ... 33%, 42144 KB, 57592 KB/s, 0 seconds passed
    ... 33%, 42176 KB, 57608 KB/s, 0 seconds passed
    ... 33%, 42208 KB, 57621 KB/s, 0 seconds passed
    ... 33%, 42240 KB, 57635 KB/s, 0 seconds passed
    ... 33%, 42272 KB, 57647 KB/s, 0 seconds passed
    ... 33%, 42304 KB, 57660 KB/s, 0 seconds passed
    ... 33%, 42336 KB, 57675 KB/s, 0 seconds passed
    ... 33%, 42368 KB, 57701 KB/s, 0 seconds passed
    ... 33%, 42400 KB, 57725 KB/s, 0 seconds passed
    ... 33%, 42432 KB, 57750 KB/s, 0 seconds passed
    ... 33%, 42464 KB, 57774 KB/s, 0 seconds passed
    ... 33%, 42496 KB, 57799 KB/s, 0 seconds passed
    ... 33%, 42528 KB, 57822 KB/s, 0 seconds passed
    ... 33%, 42560 KB, 57848 KB/s, 0 seconds passed
    ... 33%, 42592 KB, 57870 KB/s, 0 seconds passed
    ... 33%, 42624 KB, 57895 KB/s, 0 seconds passed
    ... 33%, 42656 KB, 57919 KB/s, 0 seconds passed
    ... 33%, 42688 KB, 57943 KB/s, 0 seconds passed
    ... 33%, 42720 KB, 57967 KB/s, 0 seconds passed
    ... 33%, 42752 KB, 57991 KB/s, 0 seconds passed
    ... 33%, 42784 KB, 58015 KB/s, 0 seconds passed
    ... 33%, 42816 KB, 58040 KB/s, 0 seconds passed
    ... 34%, 42848 KB, 58065 KB/s, 0 seconds passed
    ... 34%, 42880 KB, 58089 KB/s, 0 seconds passed
    ... 34%, 42912 KB, 58114 KB/s, 0 seconds passed
    ... 34%, 42944 KB, 58138 KB/s, 0 seconds passed
    ... 34%, 42976 KB, 58162 KB/s, 0 seconds passed
    ... 34%, 43008 KB, 58186 KB/s, 0 seconds passed
    ... 34%, 43040 KB, 58211 KB/s, 0 seconds passed
    ... 34%, 43072 KB, 58234 KB/s, 0 seconds passed
    ... 34%, 43104 KB, 58260 KB/s, 0 seconds passed
    ... 34%, 43136 KB, 58285 KB/s, 0 seconds passed
    ... 34%, 43168 KB, 58310 KB/s, 0 seconds passed
    ... 34%, 43200 KB, 58335 KB/s, 0 seconds passed
    ... 34%, 43232 KB, 58359 KB/s, 0 seconds passed
    ... 34%, 43264 KB, 58383 KB/s, 0 seconds passed
    ... 34%, 43296 KB, 58406 KB/s, 0 seconds passed
    ... 34%, 43328 KB, 58430 KB/s, 0 seconds passed
    ... 34%, 43360 KB, 58455 KB/s, 0 seconds passed
    ... 34%, 43392 KB, 58479 KB/s, 0 seconds passed
    ... 34%, 43424 KB, 58504 KB/s, 0 seconds passed
    ... 34%, 43456 KB, 58529 KB/s, 0 seconds passed
    ... 34%, 43488 KB, 58553 KB/s, 0 seconds passed
    ... 34%, 43520 KB, 58577 KB/s, 0 seconds passed
    ... 34%, 43552 KB, 58600 KB/s, 0 seconds passed
    ... 34%, 43584 KB, 58624 KB/s, 0 seconds passed
    ... 34%, 43616 KB, 58649 KB/s, 0 seconds passed
    ... 34%, 43648 KB, 58673 KB/s, 0 seconds passed
    ... 34%, 43680 KB, 58702 KB/s, 0 seconds passed
    ... 34%, 43712 KB, 58732 KB/s, 0 seconds passed
    ... 34%, 43744 KB, 58763 KB/s, 0 seconds passed
    ... 34%, 43776 KB, 58793 KB/s, 0 seconds passed
    ... 34%, 43808 KB, 58822 KB/s, 0 seconds passed
    ... 34%, 43840 KB, 58852 KB/s, 0 seconds passed
    ... 34%, 43872 KB, 58878 KB/s, 0 seconds passed

.. parsed-literal::

    ... 34%, 43904 KB, 57969 KB/s, 0 seconds passed
    ... 34%, 43936 KB, 57990 KB/s, 0 seconds passed
    ... 34%, 43968 KB, 58014 KB/s, 0 seconds passed
    ... 34%, 44000 KB, 58040 KB/s, 0 seconds passed
    ... 34%, 44032 KB, 58063 KB/s, 0 seconds passed
    ... 34%, 44064 KB, 58082 KB/s, 0 seconds passed
    ... 35%, 44096 KB, 58103 KB/s, 0 seconds passed
    ... 35%, 44128 KB, 58125 KB/s, 0 seconds passed
    ... 35%, 44160 KB, 58143 KB/s, 0 seconds passed
    ... 35%, 44192 KB, 58168 KB/s, 0 seconds passed
    ... 35%, 44224 KB, 58190 KB/s, 0 seconds passed
    ... 35%, 44256 KB, 58212 KB/s, 0 seconds passed
    ... 35%, 44288 KB, 58230 KB/s, 0 seconds passed
    ... 35%, 44320 KB, 58252 KB/s, 0 seconds passed
    ... 35%, 44352 KB, 58273 KB/s, 0 seconds passed
    ... 35%, 44384 KB, 58295 KB/s, 0 seconds passed
    ... 35%, 44416 KB, 58313 KB/s, 0 seconds passed
    ... 35%, 44448 KB, 58335 KB/s, 0 seconds passed
    ... 35%, 44480 KB, 58356 KB/s, 0 seconds passed
    ... 35%, 44512 KB, 58378 KB/s, 0 seconds passed
    ... 35%, 44544 KB, 58396 KB/s, 0 seconds passed
    ... 35%, 44576 KB, 58417 KB/s, 0 seconds passed
    ... 35%, 44608 KB, 58439 KB/s, 0 seconds passed
    ... 35%, 44640 KB, 58460 KB/s, 0 seconds passed
    ... 35%, 44672 KB, 58478 KB/s, 0 seconds passed
    ... 35%, 44704 KB, 58500 KB/s, 0 seconds passed
    ... 35%, 44736 KB, 58521 KB/s, 0 seconds passed
    ... 35%, 44768 KB, 58539 KB/s, 0 seconds passed
    ... 35%, 44800 KB, 58559 KB/s, 0 seconds passed
    ... 35%, 44832 KB, 58582 KB/s, 0 seconds passed
    ... 35%, 44864 KB, 58602 KB/s, 0 seconds passed
    ... 35%, 44896 KB, 58620 KB/s, 0 seconds passed
    ... 35%, 44928 KB, 58642 KB/s, 0 seconds passed
    ... 35%, 44960 KB, 58663 KB/s, 0 seconds passed
    ... 35%, 44992 KB, 58684 KB/s, 0 seconds passed
    ... 35%, 45024 KB, 58702 KB/s, 0 seconds passed
    ... 35%, 45056 KB, 58723 KB/s, 0 seconds passed
    ... 35%, 45088 KB, 58744 KB/s, 0 seconds passed
    ... 35%, 45120 KB, 58766 KB/s, 0 seconds passed
    ... 35%, 45152 KB, 58783 KB/s, 0 seconds passed
    ... 35%, 45184 KB, 58807 KB/s, 0 seconds passed
    ... 35%, 45216 KB, 58826 KB/s, 0 seconds passed
    ... 35%, 45248 KB, 58847 KB/s, 0 seconds passed
    ... 35%, 45280 KB, 58867 KB/s, 0 seconds passed
    ... 35%, 45312 KB, 58885 KB/s, 0 seconds passed
    ... 36%, 45344 KB, 58903 KB/s, 0 seconds passed
    ... 36%, 45376 KB, 58925 KB/s, 0 seconds passed
    ... 36%, 45408 KB, 58948 KB/s, 0 seconds passed
    ... 36%, 45440 KB, 58966 KB/s, 0 seconds passed
    ... 36%, 45472 KB, 58987 KB/s, 0 seconds passed
    ... 36%, 45504 KB, 59008 KB/s, 0 seconds passed
    ... 36%, 45536 KB, 59029 KB/s, 0 seconds passed
    ... 36%, 45568 KB, 59047 KB/s, 0 seconds passed
    ... 36%, 45600 KB, 59068 KB/s, 0 seconds passed
    ... 36%, 45632 KB, 59089 KB/s, 0 seconds passed
    ... 36%, 45664 KB, 59110 KB/s, 0 seconds passed
    ... 36%, 45696 KB, 59127 KB/s, 0 seconds passed
    ... 36%, 45728 KB, 59148 KB/s, 0 seconds passed
    ... 36%, 45760 KB, 59169 KB/s, 0 seconds passed
    ... 36%, 45792 KB, 59191 KB/s, 0 seconds passed
    ... 36%, 45824 KB, 59211 KB/s, 0 seconds passed
    ... 36%, 45856 KB, 59229 KB/s, 0 seconds passed
    ... 36%, 45888 KB, 59249 KB/s, 0 seconds passed
    ... 36%, 45920 KB, 59267 KB/s, 0 seconds passed
    ... 36%, 45952 KB, 59288 KB/s, 0 seconds passed
    ... 36%, 45984 KB, 59309 KB/s, 0 seconds passed
    ... 36%, 46016 KB, 59325 KB/s, 0 seconds passed
    ... 36%, 46048 KB, 59335 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 46080 KB, 57198 KB/s, 0 seconds passed
    ... 36%, 46112 KB, 57202 KB/s, 0 seconds passed
    ... 36%, 46144 KB, 57214 KB/s, 0 seconds passed
    ... 36%, 46176 KB, 57226 KB/s, 0 seconds passed
    ... 36%, 46208 KB, 57238 KB/s, 0 seconds passed
    ... 36%, 46240 KB, 57248 KB/s, 0 seconds passed
    ... 36%, 46272 KB, 57261 KB/s, 0 seconds passed
    ... 36%, 46304 KB, 57271 KB/s, 0 seconds passed
    ... 36%, 46336 KB, 57284 KB/s, 0 seconds passed
    ... 36%, 46368 KB, 57291 KB/s, 0 seconds passed
    ... 36%, 46400 KB, 57304 KB/s, 0 seconds passed
    ... 36%, 46432 KB, 57316 KB/s, 0 seconds passed
    ... 36%, 46464 KB, 57329 KB/s, 0 seconds passed
    ... 36%, 46496 KB, 57341 KB/s, 0 seconds passed
    ... 36%, 46528 KB, 57350 KB/s, 0 seconds passed
    ... 36%, 46560 KB, 57363 KB/s, 0 seconds passed
    ... 36%, 46592 KB, 57374 KB/s, 0 seconds passed
    ... 37%, 46624 KB, 57384 KB/s, 0 seconds passed
    ... 37%, 46656 KB, 57400 KB/s, 0 seconds passed
    ... 37%, 46688 KB, 57415 KB/s, 0 seconds passed
    ... 37%, 46720 KB, 57433 KB/s, 0 seconds passed
    ... 37%, 46752 KB, 57451 KB/s, 0 seconds passed
    ... 37%, 46784 KB, 57469 KB/s, 0 seconds passed
    ... 37%, 46816 KB, 57487 KB/s, 0 seconds passed
    ... 37%, 46848 KB, 57506 KB/s, 0 seconds passed
    ... 37%, 46880 KB, 57367 KB/s, 0 seconds passed
    ... 37%, 46912 KB, 57379 KB/s, 0 seconds passed
    ... 37%, 46944 KB, 57398 KB/s, 0 seconds passed
    ... 37%, 46976 KB, 57416 KB/s, 0 seconds passed
    ... 37%, 47008 KB, 57434 KB/s, 0 seconds passed
    ... 37%, 47040 KB, 57453 KB/s, 0 seconds passed
    ... 37%, 47072 KB, 57470 KB/s, 0 seconds passed
    ... 37%, 47104 KB, 57490 KB/s, 0 seconds passed
    ... 37%, 47136 KB, 57508 KB/s, 0 seconds passed
    ... 37%, 47168 KB, 57528 KB/s, 0 seconds passed
    ... 37%, 47200 KB, 57546 KB/s, 0 seconds passed
    ... 37%, 47232 KB, 57564 KB/s, 0 seconds passed
    ... 37%, 47264 KB, 57583 KB/s, 0 seconds passed
    ... 37%, 47296 KB, 57601 KB/s, 0 seconds passed
    ... 37%, 47328 KB, 57617 KB/s, 0 seconds passed
    ... 37%, 47360 KB, 57637 KB/s, 0 seconds passed
    ... 37%, 47392 KB, 57655 KB/s, 0 seconds passed
    ... 37%, 47424 KB, 57673 KB/s, 0 seconds passed
    ... 37%, 47456 KB, 57691 KB/s, 0 seconds passed
    ... 37%, 47488 KB, 57711 KB/s, 0 seconds passed
    ... 37%, 47520 KB, 57730 KB/s, 0 seconds passed
    ... 37%, 47552 KB, 57747 KB/s, 0 seconds passed
    ... 37%, 47584 KB, 57767 KB/s, 0 seconds passed
    ... 37%, 47616 KB, 57786 KB/s, 0 seconds passed
    ... 37%, 47648 KB, 57805 KB/s, 0 seconds passed
    ... 37%, 47680 KB, 57824 KB/s, 0 seconds passed
    ... 37%, 47712 KB, 57843 KB/s, 0 seconds passed
    ... 37%, 47744 KB, 57859 KB/s, 0 seconds passed
    ... 37%, 47776 KB, 57882 KB/s, 0 seconds passed
    ... 37%, 47808 KB, 57904 KB/s, 0 seconds passed
    ... 37%, 47840 KB, 57926 KB/s, 0 seconds passed
    ... 38%, 47872 KB, 57949 KB/s, 0 seconds passed
    ... 38%, 47904 KB, 57973 KB/s, 0 seconds passed
    ... 38%, 47936 KB, 57996 KB/s, 0 seconds passed
    ... 38%, 47968 KB, 58018 KB/s, 0 seconds passed
    ... 38%, 48000 KB, 58040 KB/s, 0 seconds passed
    ... 38%, 48032 KB, 58061 KB/s, 0 seconds passed
    ... 38%, 48064 KB, 58084 KB/s, 0 seconds passed
    ... 38%, 48096 KB, 58102 KB/s, 0 seconds passed
    ... 38%, 48128 KB, 58119 KB/s, 0 seconds passed
    ... 38%, 48160 KB, 58141 KB/s, 0 seconds passed
    ... 38%, 48192 KB, 58162 KB/s, 0 seconds passed
    ... 38%, 48224 KB, 58182 KB/s, 0 seconds passed
    ... 38%, 48256 KB, 58195 KB/s, 0 seconds passed
    ... 38%, 48288 KB, 58214 KB/s, 0 seconds passed
    ... 38%, 48320 KB, 58234 KB/s, 0 seconds passed
    ... 38%, 48352 KB, 58254 KB/s, 0 seconds passed
    ... 38%, 48384 KB, 58274 KB/s, 0 seconds passed
    ... 38%, 48416 KB, 58296 KB/s, 0 seconds passed
    ... 38%, 48448 KB, 58310 KB/s, 0 seconds passed
    ... 38%, 48480 KB, 58330 KB/s, 0 seconds passed
    ... 38%, 48512 KB, 58352 KB/s, 0 seconds passed
    ... 38%, 48544 KB, 58374 KB/s, 0 seconds passed
    ... 38%, 48576 KB, 58390 KB/s, 0 seconds passed
    ... 38%, 48608 KB, 58410 KB/s, 0 seconds passed
    ... 38%, 48640 KB, 58429 KB/s, 0 seconds passed
    ... 38%, 48672 KB, 58449 KB/s, 0 seconds passed
    ... 38%, 48704 KB, 58465 KB/s, 0 seconds passed
    ... 38%, 48736 KB, 58485 KB/s, 0 seconds passed
    ... 38%, 48768 KB, 58505 KB/s, 0 seconds passed
    ... 38%, 48800 KB, 58517 KB/s, 0 seconds passed
    ... 38%, 48832 KB, 58537 KB/s, 0 seconds passed
    ... 38%, 48864 KB, 58553 KB/s, 0 seconds passed
    ... 38%, 48896 KB, 58573 KB/s, 0 seconds passed
    ... 38%, 48928 KB, 58595 KB/s, 0 seconds passed
    ... 38%, 48960 KB, 58616 KB/s, 0 seconds passed
    ... 38%, 48992 KB, 58637 KB/s, 0 seconds passed
    ... 38%, 49024 KB, 58649 KB/s, 0 seconds passed
    ... 38%, 49056 KB, 58669 KB/s, 0 seconds passed
    ... 38%, 49088 KB, 58685 KB/s, 0 seconds passed
    ... 38%, 49120 KB, 58705 KB/s, 0 seconds passed
    ... 39%, 49152 KB, 58724 KB/s, 0 seconds passed
    ... 39%, 49184 KB, 58743 KB/s, 0 seconds passed
    ... 39%, 49216 KB, 58765 KB/s, 0 seconds passed
    ... 39%, 49248 KB, 58787 KB/s, 0 seconds passed
    ... 39%, 49280 KB, 58807 KB/s, 0 seconds passed
    ... 39%, 49312 KB, 58820 KB/s, 0 seconds passed
    ... 39%, 49344 KB, 58840 KB/s, 0 seconds passed
    ... 39%, 49376 KB, 58856 KB/s, 0 seconds passed
    ... 39%, 49408 KB, 58876 KB/s, 0 seconds passed
    ... 39%, 49440 KB, 58896 KB/s, 0 seconds passed
    ... 39%, 49472 KB, 58908 KB/s, 0 seconds passed
    ... 39%, 49504 KB, 58928 KB/s, 0 seconds passed
    ... 39%, 49536 KB, 58944 KB/s, 0 seconds passed

.. parsed-literal::

    ... 39%, 49568 KB, 58167 KB/s, 0 seconds passed
    ... 39%, 49600 KB, 58165 KB/s, 0 seconds passed
    ... 39%, 49632 KB, 58164 KB/s, 0 seconds passed
    ... 39%, 49664 KB, 58163 KB/s, 0 seconds passed
    ... 39%, 49696 KB, 58167 KB/s, 0 seconds passed
    ... 39%, 49728 KB, 58170 KB/s, 0 seconds passed
    ... 39%, 49760 KB, 58175 KB/s, 0 seconds passed
    ... 39%, 49792 KB, 58176 KB/s, 0 seconds passed
    ... 39%, 49824 KB, 58178 KB/s, 0 seconds passed
    ... 39%, 49856 KB, 58181 KB/s, 0 seconds passed
    ... 39%, 49888 KB, 58184 KB/s, 0 seconds passed
    ... 39%, 49920 KB, 58196 KB/s, 0 seconds passed
    ... 39%, 49952 KB, 58208 KB/s, 0 seconds passed
    ... 39%, 49984 KB, 58220 KB/s, 0 seconds passed
    ... 39%, 50016 KB, 58231 KB/s, 0 seconds passed
    ... 39%, 50048 KB, 58244 KB/s, 0 seconds passed
    ... 39%, 50080 KB, 58256 KB/s, 0 seconds passed
    ... 39%, 50112 KB, 58270 KB/s, 0 seconds passed
    ... 39%, 50144 KB, 58282 KB/s, 0 seconds passed
    ... 39%, 50176 KB, 58294 KB/s, 0 seconds passed
    ... 39%, 50208 KB, 58305 KB/s, 0 seconds passed
    ... 39%, 50240 KB, 58318 KB/s, 0 seconds passed
    ... 39%, 50272 KB, 58329 KB/s, 0 seconds passed
    ... 39%, 50304 KB, 58347 KB/s, 0 seconds passed
    ... 39%, 50336 KB, 58362 KB/s, 0 seconds passed
    ... 39%, 50368 KB, 58380 KB/s, 0 seconds passed
    ... 40%, 50400 KB, 58398 KB/s, 0 seconds passed
    ... 40%, 50432 KB, 58416 KB/s, 0 seconds passed
    ... 40%, 50464 KB, 58434 KB/s, 0 seconds passed
    ... 40%, 50496 KB, 58452 KB/s, 0 seconds passed
    ... 40%, 50528 KB, 58470 KB/s, 0 seconds passed
    ... 40%, 50560 KB, 58487 KB/s, 0 seconds passed
    ... 40%, 50592 KB, 58505 KB/s, 0 seconds passed
    ... 40%, 50624 KB, 58522 KB/s, 0 seconds passed
    ... 40%, 50656 KB, 58540 KB/s, 0 seconds passed
    ... 40%, 50688 KB, 58110 KB/s, 0 seconds passed
    ... 40%, 50720 KB, 58125 KB/s, 0 seconds passed
    ... 40%, 50752 KB, 58145 KB/s, 0 seconds passed
    ... 40%, 50784 KB, 58164 KB/s, 0 seconds passed
    ... 40%, 50816 KB, 58184 KB/s, 0 seconds passed
    ... 40%, 50848 KB, 58203 KB/s, 0 seconds passed
    ... 40%, 50880 KB, 58219 KB/s, 0 seconds passed
    ... 40%, 50912 KB, 58238 KB/s, 0 seconds passed
    ... 40%, 50944 KB, 58256 KB/s, 0 seconds passed
    ... 40%, 50976 KB, 58276 KB/s, 0 seconds passed
    ... 40%, 51008 KB, 58291 KB/s, 0 seconds passed
    ... 40%, 51040 KB, 58310 KB/s, 0 seconds passed
    ... 40%, 51072 KB, 58324 KB/s, 0 seconds passed
    ... 40%, 51104 KB, 58339 KB/s, 0 seconds passed
    ... 40%, 51136 KB, 58355 KB/s, 0 seconds passed
    ... 40%, 51168 KB, 58370 KB/s, 0 seconds passed
    ... 40%, 51200 KB, 58386 KB/s, 0 seconds passed
    ... 40%, 51232 KB, 58398 KB/s, 0 seconds passed
    ... 40%, 51264 KB, 58413 KB/s, 0 seconds passed
    ... 40%, 51296 KB, 58427 KB/s, 0 seconds passed
    ... 40%, 51328 KB, 58443 KB/s, 0 seconds passed
    ... 40%, 51360 KB, 58457 KB/s, 0 seconds passed
    ... 40%, 51392 KB, 58472 KB/s, 0 seconds passed
    ... 40%, 51424 KB, 58486 KB/s, 0 seconds passed
    ... 40%, 51456 KB, 58500 KB/s, 0 seconds passed
    ... 40%, 51488 KB, 58516 KB/s, 0 seconds passed
    ... 40%, 51520 KB, 58530 KB/s, 0 seconds passed
    ... 40%, 51552 KB, 58547 KB/s, 0 seconds passed
    ... 40%, 51584 KB, 58563 KB/s, 0 seconds passed
    ... 40%, 51616 KB, 58575 KB/s, 0 seconds passed
    ... 41%, 51648 KB, 58590 KB/s, 0 seconds passed
    ... 41%, 51680 KB, 58604 KB/s, 0 seconds passed
    ... 41%, 51712 KB, 58621 KB/s, 0 seconds passed
    ... 41%, 51744 KB, 58638 KB/s, 0 seconds passed
    ... 41%, 51776 KB, 58459 KB/s, 0 seconds passed
    ... 41%, 51808 KB, 58474 KB/s, 0 seconds passed
    ... 41%, 51840 KB, 58492 KB/s, 0 seconds passed
    ... 41%, 51872 KB, 58510 KB/s, 0 seconds passed
    ... 41%, 51904 KB, 58528 KB/s, 0 seconds passed
    ... 41%, 51936 KB, 58546 KB/s, 0 seconds passed
    ... 41%, 51968 KB, 58563 KB/s, 0 seconds passed
    ... 41%, 52000 KB, 58575 KB/s, 0 seconds passed
    ... 41%, 52032 KB, 58592 KB/s, 0 seconds passed
    ... 41%, 52064 KB, 58610 KB/s, 0 seconds passed
    ... 41%, 52096 KB, 58628 KB/s, 0 seconds passed
    ... 41%, 52128 KB, 58645 KB/s, 0 seconds passed
    ... 41%, 52160 KB, 58661 KB/s, 0 seconds passed
    ... 41%, 52192 KB, 58679 KB/s, 0 seconds passed
    ... 41%, 52224 KB, 58695 KB/s, 0 seconds passed
    ... 41%, 52256 KB, 58712 KB/s, 0 seconds passed
    ... 41%, 52288 KB, 58728 KB/s, 0 seconds passed
    ... 41%, 52320 KB, 58746 KB/s, 0 seconds passed
    ... 41%, 52352 KB, 58765 KB/s, 0 seconds passed
    ... 41%, 52384 KB, 58782 KB/s, 0 seconds passed
    ... 41%, 52416 KB, 58798 KB/s, 0 seconds passed
    ... 41%, 52448 KB, 58817 KB/s, 0 seconds passed
    ... 41%, 52480 KB, 58833 KB/s, 0 seconds passed
    ... 41%, 52512 KB, 58850 KB/s, 0 seconds passed
    ... 41%, 52544 KB, 58867 KB/s, 0 seconds passed
    ... 41%, 52576 KB, 58884 KB/s, 0 seconds passed
    ... 41%, 52608 KB, 58899 KB/s, 0 seconds passed
    ... 41%, 52640 KB, 58916 KB/s, 0 seconds passed
    ... 41%, 52672 KB, 58934 KB/s, 0 seconds passed
    ... 41%, 52704 KB, 58952 KB/s, 0 seconds passed
    ... 41%, 52736 KB, 58969 KB/s, 0 seconds passed
    ... 41%, 52768 KB, 58987 KB/s, 0 seconds passed
    ... 41%, 52800 KB, 59002 KB/s, 0 seconds passed
    ... 41%, 52832 KB, 59019 KB/s, 0 seconds passed
    ... 41%, 52864 KB, 59036 KB/s, 0 seconds passed
    ... 41%, 52896 KB, 59054 KB/s, 0 seconds passed
    ... 42%, 52928 KB, 59072 KB/s, 0 seconds passed
    ... 42%, 52960 KB, 59091 KB/s, 0 seconds passed
    ... 42%, 52992 KB, 59114 KB/s, 0 seconds passed
    ... 42%, 53024 KB, 59137 KB/s, 0 seconds passed
    ... 42%, 53056 KB, 59161 KB/s, 0 seconds passed
    ... 42%, 53088 KB, 59179 KB/s, 0 seconds passed
    ... 42%, 53120 KB, 59194 KB/s, 0 seconds passed
    ... 42%, 53152 KB, 59212 KB/s, 0 seconds passed
    ... 42%, 53184 KB, 59231 KB/s, 0 seconds passed
    ... 42%, 53216 KB, 59245 KB/s, 0 seconds passed
    ... 42%, 53248 KB, 59263 KB/s, 0 seconds passed
    ... 42%, 53280 KB, 59281 KB/s, 0 seconds passed
    ... 42%, 53312 KB, 59300 KB/s, 0 seconds passed
    ... 42%, 53344 KB, 59314 KB/s, 0 seconds passed

.. parsed-literal::

    ... 42%, 53376 KB, 59331 KB/s, 0 seconds passed
    ... 42%, 53408 KB, 59351 KB/s, 0 seconds passed
    ... 42%, 53440 KB, 59369 KB/s, 0 seconds passed
    ... 42%, 53472 KB, 59384 KB/s, 0 seconds passed
    ... 42%, 53504 KB, 59398 KB/s, 0 seconds passed
    ... 42%, 53536 KB, 59324 KB/s, 0 seconds passed
    ... 42%, 53568 KB, 59344 KB/s, 0 seconds passed
    ... 42%, 53600 KB, 59367 KB/s, 0 seconds passed
    ... 42%, 53632 KB, 59390 KB/s, 0 seconds passed
    ... 42%, 53664 KB, 59414 KB/s, 0 seconds passed
    ... 42%, 53696 KB, 59437 KB/s, 0 seconds passed
    ... 42%, 53728 KB, 59460 KB/s, 0 seconds passed
    ... 42%, 53760 KB, 59484 KB/s, 0 seconds passed
    ... 42%, 53792 KB, 59507 KB/s, 0 seconds passed
    ... 42%, 53824 KB, 59531 KB/s, 0 seconds passed
    ... 42%, 53856 KB, 59554 KB/s, 0 seconds passed
    ... 42%, 53888 KB, 59577 KB/s, 0 seconds passed
    ... 42%, 53920 KB, 59601 KB/s, 0 seconds passed
    ... 42%, 53952 KB, 59622 KB/s, 0 seconds passed
    ... 42%, 53984 KB, 59639 KB/s, 0 seconds passed
    ... 42%, 54016 KB, 59647 KB/s, 0 seconds passed
    ... 42%, 54048 KB, 59665 KB/s, 0 seconds passed
    ... 42%, 54080 KB, 59683 KB/s, 0 seconds passed
    ... 42%, 54112 KB, 59697 KB/s, 0 seconds passed
    ... 42%, 54144 KB, 59716 KB/s, 0 seconds passed
    ... 43%, 54176 KB, 59733 KB/s, 0 seconds passed
    ... 43%, 54208 KB, 59752 KB/s, 0 seconds passed
    ... 43%, 54240 KB, 59770 KB/s, 0 seconds passed
    ... 43%, 54272 KB, 59787 KB/s, 0 seconds passed
    ... 43%, 54304 KB, 59802 KB/s, 0 seconds passed
    ... 43%, 54336 KB, 59820 KB/s, 0 seconds passed
    ... 43%, 54368 KB, 59838 KB/s, 0 seconds passed
    ... 43%, 54400 KB, 59859 KB/s, 0 seconds passed
    ... 43%, 54432 KB, 59877 KB/s, 0 seconds passed
    ... 43%, 54464 KB, 59891 KB/s, 0 seconds passed
    ... 43%, 54496 KB, 59908 KB/s, 0 seconds passed
    ... 43%, 54528 KB, 59915 KB/s, 0 seconds passed
    ... 43%, 54560 KB, 59927 KB/s, 0 seconds passed
    ... 43%, 54592 KB, 59953 KB/s, 0 seconds passed
    ... 43%, 54624 KB, 59975 KB/s, 0 seconds passed
    ... 43%, 54656 KB, 59993 KB/s, 0 seconds passed
    ... 43%, 54688 KB, 60000 KB/s, 0 seconds passed
    ... 43%, 54720 KB, 60008 KB/s, 0 seconds passed
    ... 43%, 54752 KB, 60017 KB/s, 0 seconds passed
    ... 43%, 54784 KB, 60031 KB/s, 0 seconds passed
    ... 43%, 54816 KB, 60058 KB/s, 0 seconds passed
    ... 43%, 54848 KB, 60084 KB/s, 0 seconds passed
    ... 43%, 54880 KB, 60102 KB/s, 0 seconds passed
    ... 43%, 54912 KB, 60118 KB/s, 0 seconds passed
    ... 43%, 54944 KB, 60126 KB/s, 0 seconds passed
    ... 43%, 54976 KB, 60135 KB/s, 0 seconds passed
    ... 43%, 55008 KB, 60144 KB/s, 0 seconds passed
    ... 43%, 55040 KB, 60166 KB/s, 0 seconds passed
    ... 43%, 55072 KB, 60192 KB/s, 0 seconds passed
    ... 43%, 55104 KB, 60218 KB/s, 0 seconds passed
    ... 43%, 55136 KB, 59936 KB/s, 0 seconds passed
    ... 43%, 55168 KB, 59921 KB/s, 0 seconds passed
    ... 43%, 55200 KB, 59934 KB/s, 0 seconds passed
    ... 43%, 55232 KB, 59925 KB/s, 0 seconds passed
    ... 43%, 55264 KB, 59934 KB/s, 0 seconds passed
    ... 43%, 55296 KB, 59942 KB/s, 0 seconds passed
    ... 43%, 55328 KB, 59956 KB/s, 0 seconds passed
    ... 43%, 55360 KB, 59971 KB/s, 0 seconds passed
    ... 43%, 55392 KB, 59986 KB/s, 0 seconds passed
    ... 44%, 55424 KB, 60001 KB/s, 0 seconds passed
    ... 44%, 55456 KB, 60018 KB/s, 0 seconds passed
    ... 44%, 55488 KB, 60032 KB/s, 0 seconds passed
    ... 44%, 55520 KB, 60047 KB/s, 0 seconds passed
    ... 44%, 55552 KB, 60062 KB/s, 0 seconds passed
    ... 44%, 55584 KB, 60073 KB/s, 0 seconds passed
    ... 44%, 55616 KB, 60088 KB/s, 0 seconds passed
    ... 44%, 55648 KB, 60102 KB/s, 0 seconds passed
    ... 44%, 55680 KB, 60117 KB/s, 0 seconds passed
    ... 44%, 55712 KB, 60130 KB/s, 0 seconds passed
    ... 44%, 55744 KB, 60142 KB/s, 0 seconds passed
    ... 44%, 55776 KB, 60153 KB/s, 0 seconds passed
    ... 44%, 55808 KB, 60163 KB/s, 0 seconds passed
    ... 44%, 55840 KB, 60174 KB/s, 0 seconds passed
    ... 44%, 55872 KB, 60185 KB/s, 0 seconds passed
    ... 44%, 55904 KB, 60195 KB/s, 0 seconds passed
    ... 44%, 55936 KB, 60204 KB/s, 0 seconds passed
    ... 44%, 55968 KB, 60214 KB/s, 0 seconds passed
    ... 44%, 56000 KB, 60225 KB/s, 0 seconds passed
    ... 44%, 56032 KB, 60236 KB/s, 0 seconds passed
    ... 44%, 56064 KB, 60247 KB/s, 0 seconds passed
    ... 44%, 56096 KB, 60258 KB/s, 0 seconds passed
    ... 44%, 56128 KB, 60269 KB/s, 0 seconds passed
    ... 44%, 56160 KB, 60283 KB/s, 0 seconds passed
    ... 44%, 56192 KB, 60299 KB/s, 0 seconds passed
    ... 44%, 56224 KB, 60315 KB/s, 0 seconds passed
    ... 44%, 56256 KB, 60332 KB/s, 0 seconds passed
    ... 44%, 56288 KB, 60348 KB/s, 0 seconds passed

.. parsed-literal::

    ... 44%, 56320 KB, 56202 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 56352 KB, 56204 KB/s, 1 seconds passed
    ... 44%, 56384 KB, 56212 KB/s, 1 seconds passed
    ... 44%, 56416 KB, 56224 KB/s, 1 seconds passed
    ... 44%, 56448 KB, 56238 KB/s, 1 seconds passed
    ... 44%, 56480 KB, 56198 KB/s, 1 seconds passed
    ... 44%, 56512 KB, 56191 KB/s, 1 seconds passed
    ... 44%, 56544 KB, 56203 KB/s, 1 seconds passed
    ... 44%, 56576 KB, 56217 KB/s, 1 seconds passed
    ... 44%, 56608 KB, 56230 KB/s, 1 seconds passed
    ... 44%, 56640 KB, 56245 KB/s, 1 seconds passed
    ... 44%, 56672 KB, 56260 KB/s, 1 seconds passed
    ... 45%, 56704 KB, 56274 KB/s, 1 seconds passed
    ... 45%, 56736 KB, 56289 KB/s, 1 seconds passed
    ... 45%, 56768 KB, 56304 KB/s, 1 seconds passed
    ... 45%, 56800 KB, 56319 KB/s, 1 seconds passed
    ... 45%, 56832 KB, 56333 KB/s, 1 seconds passed
    ... 45%, 56864 KB, 56346 KB/s, 1 seconds passed
    ... 45%, 56896 KB, 56361 KB/s, 1 seconds passed
    ... 45%, 56928 KB, 56377 KB/s, 1 seconds passed
    ... 45%, 56960 KB, 56391 KB/s, 1 seconds passed
    ... 45%, 56992 KB, 56406 KB/s, 1 seconds passed
    ... 45%, 57024 KB, 56421 KB/s, 1 seconds passed
    ... 45%, 57056 KB, 56437 KB/s, 1 seconds passed
    ... 45%, 57088 KB, 56452 KB/s, 1 seconds passed
    ... 45%, 57120 KB, 56467 KB/s, 1 seconds passed
    ... 45%, 57152 KB, 56481 KB/s, 1 seconds passed
    ... 45%, 57184 KB, 56496 KB/s, 1 seconds passed
    ... 45%, 57216 KB, 56511 KB/s, 1 seconds passed
    ... 45%, 57248 KB, 56526 KB/s, 1 seconds passed
    ... 45%, 57280 KB, 56540 KB/s, 1 seconds passed
    ... 45%, 57312 KB, 56554 KB/s, 1 seconds passed
    ... 45%, 57344 KB, 56569 KB/s, 1 seconds passed
    ... 45%, 57376 KB, 56584 KB/s, 1 seconds passed
    ... 45%, 57408 KB, 56597 KB/s, 1 seconds passed
    ... 45%, 57440 KB, 56612 KB/s, 1 seconds passed
    ... 45%, 57472 KB, 56625 KB/s, 1 seconds passed
    ... 45%, 57504 KB, 56640 KB/s, 1 seconds passed
    ... 45%, 57536 KB, 56654 KB/s, 1 seconds passed
    ... 45%, 57568 KB, 56669 KB/s, 1 seconds passed
    ... 45%, 57600 KB, 56683 KB/s, 1 seconds passed
    ... 45%, 57632 KB, 56698 KB/s, 1 seconds passed
    ... 45%, 57664 KB, 56712 KB/s, 1 seconds passed
    ... 45%, 57696 KB, 56727 KB/s, 1 seconds passed
    ... 45%, 57728 KB, 56741 KB/s, 1 seconds passed
    ... 45%, 57760 KB, 56762 KB/s, 1 seconds passed
    ... 45%, 57792 KB, 56781 KB/s, 1 seconds passed
    ... 45%, 57824 KB, 56802 KB/s, 1 seconds passed
    ... 45%, 57856 KB, 56822 KB/s, 1 seconds passed
    ... 45%, 57888 KB, 56842 KB/s, 1 seconds passed
    ... 45%, 57920 KB, 56861 KB/s, 1 seconds passed
    ... 46%, 57952 KB, 56881 KB/s, 1 seconds passed
    ... 46%, 57984 KB, 56901 KB/s, 1 seconds passed
    ... 46%, 58016 KB, 56922 KB/s, 1 seconds passed
    ... 46%, 58048 KB, 56941 KB/s, 1 seconds passed
    ... 46%, 58080 KB, 56962 KB/s, 1 seconds passed
    ... 46%, 58112 KB, 56982 KB/s, 1 seconds passed
    ... 46%, 58144 KB, 57002 KB/s, 1 seconds passed
    ... 46%, 58176 KB, 57019 KB/s, 1 seconds passed
    ... 46%, 58208 KB, 57036 KB/s, 1 seconds passed
    ... 46%, 58240 KB, 57047 KB/s, 1 seconds passed
    ... 46%, 58272 KB, 57063 KB/s, 1 seconds passed
    ... 46%, 58304 KB, 57079 KB/s, 1 seconds passed
    ... 46%, 58336 KB, 57096 KB/s, 1 seconds passed
    ... 46%, 58368 KB, 57109 KB/s, 1 seconds passed
    ... 46%, 58400 KB, 57129 KB/s, 1 seconds passed
    ... 46%, 58432 KB, 57143 KB/s, 1 seconds passed
    ... 46%, 58464 KB, 57159 KB/s, 1 seconds passed
    ... 46%, 58496 KB, 57175 KB/s, 1 seconds passed
    ... 46%, 58528 KB, 57191 KB/s, 1 seconds passed
    ... 46%, 58560 KB, 57208 KB/s, 1 seconds passed
    ... 46%, 58592 KB, 57221 KB/s, 1 seconds passed
    ... 46%, 58624 KB, 57238 KB/s, 1 seconds passed
    ... 46%, 58656 KB, 57254 KB/s, 1 seconds passed
    ... 46%, 58688 KB, 57270 KB/s, 1 seconds passed
    ... 46%, 58720 KB, 57284 KB/s, 1 seconds passed
    ... 46%, 58752 KB, 57296 KB/s, 1 seconds passed
    ... 46%, 58784 KB, 57313 KB/s, 1 seconds passed
    ... 46%, 58816 KB, 57327 KB/s, 1 seconds passed
    ... 46%, 58848 KB, 57347 KB/s, 1 seconds passed
    ... 46%, 58880 KB, 57364 KB/s, 1 seconds passed
    ... 46%, 58912 KB, 57378 KB/s, 1 seconds passed
    ... 46%, 58944 KB, 57395 KB/s, 1 seconds passed
    ... 46%, 58976 KB, 57410 KB/s, 1 seconds passed
    ... 46%, 59008 KB, 57427 KB/s, 1 seconds passed
    ... 46%, 59040 KB, 57444 KB/s, 1 seconds passed
    ... 46%, 59072 KB, 57460 KB/s, 1 seconds passed
    ... 46%, 59104 KB, 57473 KB/s, 1 seconds passed
    ... 46%, 59136 KB, 57490 KB/s, 1 seconds passed
    ... 46%, 59168 KB, 57499 KB/s, 1 seconds passed
    ... 47%, 59200 KB, 57516 KB/s, 1 seconds passed
    ... 47%, 59232 KB, 57529 KB/s, 1 seconds passed
    ... 47%, 59264 KB, 57543 KB/s, 1 seconds passed
    ... 47%, 59296 KB, 57556 KB/s, 1 seconds passed
    ... 47%, 59328 KB, 57574 KB/s, 1 seconds passed
    ... 47%, 59360 KB, 57594 KB/s, 1 seconds passed
    ... 47%, 59392 KB, 57610 KB/s, 1 seconds passed
    ... 47%, 59424 KB, 57623 KB/s, 1 seconds passed
    ... 47%, 59456 KB, 57636 KB/s, 1 seconds passed
    ... 47%, 59488 KB, 57652 KB/s, 1 seconds passed
    ... 47%, 59520 KB, 57669 KB/s, 1 seconds passed
    ... 47%, 59552 KB, 57685 KB/s, 1 seconds passed
    ... 47%, 59584 KB, 57698 KB/s, 1 seconds passed
    ... 47%, 59616 KB, 57720 KB/s, 1 seconds passed
    ... 47%, 59648 KB, 57736 KB/s, 1 seconds passed
    ... 47%, 59680 KB, 57749 KB/s, 1 seconds passed
    ... 47%, 59712 KB, 57765 KB/s, 1 seconds passed
    ... 47%, 59744 KB, 57781 KB/s, 1 seconds passed
    ... 47%, 59776 KB, 57797 KB/s, 1 seconds passed
    ... 47%, 59808 KB, 57807 KB/s, 1 seconds passed
    ... 47%, 59840 KB, 57823 KB/s, 1 seconds passed
    ... 47%, 59872 KB, 57839 KB/s, 1 seconds passed
    ... 47%, 59904 KB, 57853 KB/s, 1 seconds passed
    ... 47%, 59936 KB, 57869 KB/s, 1 seconds passed
    ... 47%, 59968 KB, 57884 KB/s, 1 seconds passed
    ... 47%, 60000 KB, 57901 KB/s, 1 seconds passed
    ... 47%, 60032 KB, 57917 KB/s, 1 seconds passed
    ... 47%, 60064 KB, 57932 KB/s, 1 seconds passed
    ... 47%, 60096 KB, 57946 KB/s, 1 seconds passed
    ... 47%, 60128 KB, 57958 KB/s, 1 seconds passed
    ... 47%, 60160 KB, 57975 KB/s, 1 seconds passed
    ... 47%, 60192 KB, 57990 KB/s, 1 seconds passed
    ... 47%, 60224 KB, 57726 KB/s, 1 seconds passed
    ... 47%, 60256 KB, 57743 KB/s, 1 seconds passed
    ... 47%, 60288 KB, 57759 KB/s, 1 seconds passed
    ... 47%, 60320 KB, 57775 KB/s, 1 seconds passed
    ... 47%, 60352 KB, 57790 KB/s, 1 seconds passed
    ... 47%, 60384 KB, 57803 KB/s, 1 seconds passed
    ... 47%, 60416 KB, 57819 KB/s, 1 seconds passed
    ... 47%, 60448 KB, 57835 KB/s, 1 seconds passed
    ... 48%, 60480 KB, 57845 KB/s, 1 seconds passed
    ... 48%, 60512 KB, 57867 KB/s, 1 seconds passed
    ... 48%, 60544 KB, 57877 KB/s, 1 seconds passed
    ... 48%, 60576 KB, 57890 KB/s, 1 seconds passed
    ... 48%, 60608 KB, 57906 KB/s, 1 seconds passed
    ... 48%, 60640 KB, 57921 KB/s, 1 seconds passed
    ... 48%, 60672 KB, 57938 KB/s, 1 seconds passed
    ... 48%, 60704 KB, 57943 KB/s, 1 seconds passed
    ... 48%, 60736 KB, 57955 KB/s, 1 seconds passed
    ... 48%, 60768 KB, 57968 KB/s, 1 seconds passed
    ... 48%, 60800 KB, 57988 KB/s, 1 seconds passed
    ... 48%, 60832 KB, 58009 KB/s, 1 seconds passed
    ... 48%, 60864 KB, 58020 KB/s, 1 seconds passed
    ... 48%, 60896 KB, 58036 KB/s, 1 seconds passed
    ... 48%, 60928 KB, 58039 KB/s, 1 seconds passed
    ... 48%, 60960 KB, 58054 KB/s, 1 seconds passed
    ... 48%, 60992 KB, 58070 KB/s, 1 seconds passed
    ... 48%, 61024 KB, 58091 KB/s, 1 seconds passed
    ... 48%, 61056 KB, 58107 KB/s, 1 seconds passed
    ... 48%, 61088 KB, 58114 KB/s, 1 seconds passed
    ... 48%, 61120 KB, 58135 KB/s, 1 seconds passed
    ... 48%, 61152 KB, 58151 KB/s, 1 seconds passed
    ... 48%, 61184 KB, 58168 KB/s, 1 seconds passed
    ... 48%, 61216 KB, 58183 KB/s, 1 seconds passed
    ... 48%, 61248 KB, 58196 KB/s, 1 seconds passed
    ... 48%, 61280 KB, 58156 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 61312 KB, 58168 KB/s, 1 seconds passed
    ... 48%, 61344 KB, 58181 KB/s, 1 seconds passed
    ... 48%, 61376 KB, 58194 KB/s, 1 seconds passed
    ... 48%, 61408 KB, 58211 KB/s, 1 seconds passed
    ... 48%, 61440 KB, 56870 KB/s, 1 seconds passed
    ... 48%, 61472 KB, 56875 KB/s, 1 seconds passed
    ... 48%, 61504 KB, 56886 KB/s, 1 seconds passed
    ... 48%, 61536 KB, 56898 KB/s, 1 seconds passed
    ... 48%, 61568 KB, 56910 KB/s, 1 seconds passed
    ... 48%, 61600 KB, 56923 KB/s, 1 seconds passed
    ... 48%, 61632 KB, 56932 KB/s, 1 seconds passed
    ... 48%, 61664 KB, 56946 KB/s, 1 seconds passed
    ... 48%, 61696 KB, 56931 KB/s, 1 seconds passed
    ... 49%, 61728 KB, 56942 KB/s, 1 seconds passed
    ... 49%, 61760 KB, 56932 KB/s, 1 seconds passed
    ... 49%, 61792 KB, 56942 KB/s, 1 seconds passed
    ... 49%, 61824 KB, 56955 KB/s, 1 seconds passed
    ... 49%, 61856 KB, 56969 KB/s, 1 seconds passed
    ... 49%, 61888 KB, 56982 KB/s, 1 seconds passed
    ... 49%, 61920 KB, 56919 KB/s, 1 seconds passed
    ... 49%, 61952 KB, 56931 KB/s, 1 seconds passed
    ... 49%, 61984 KB, 56945 KB/s, 1 seconds passed
    ... 49%, 62016 KB, 56958 KB/s, 1 seconds passed
    ... 49%, 62048 KB, 56971 KB/s, 1 seconds passed
    ... 49%, 62080 KB, 56984 KB/s, 1 seconds passed
    ... 49%, 62112 KB, 56997 KB/s, 1 seconds passed
    ... 49%, 62144 KB, 57012 KB/s, 1 seconds passed
    ... 49%, 62176 KB, 57024 KB/s, 1 seconds passed
    ... 49%, 62208 KB, 57036 KB/s, 1 seconds passed
    ... 49%, 62240 KB, 57050 KB/s, 1 seconds passed
    ... 49%, 62272 KB, 57064 KB/s, 1 seconds passed
    ... 49%, 62304 KB, 57077 KB/s, 1 seconds passed
    ... 49%, 62336 KB, 57091 KB/s, 1 seconds passed
    ... 49%, 62368 KB, 57104 KB/s, 1 seconds passed
    ... 49%, 62400 KB, 57116 KB/s, 1 seconds passed
    ... 49%, 62432 KB, 57130 KB/s, 1 seconds passed
    ... 49%, 62464 KB, 57142 KB/s, 1 seconds passed
    ... 49%, 62496 KB, 57154 KB/s, 1 seconds passed
    ... 49%, 62528 KB, 57167 KB/s, 1 seconds passed
    ... 49%, 62560 KB, 57181 KB/s, 1 seconds passed
    ... 49%, 62592 KB, 57194 KB/s, 1 seconds passed
    ... 49%, 62624 KB, 57208 KB/s, 1 seconds passed
    ... 49%, 62656 KB, 57222 KB/s, 1 seconds passed
    ... 49%, 62688 KB, 57237 KB/s, 1 seconds passed
    ... 49%, 62720 KB, 57098 KB/s, 1 seconds passed
    ... 49%, 62752 KB, 57108 KB/s, 1 seconds passed
    ... 49%, 62784 KB, 57120 KB/s, 1 seconds passed
    ... 49%, 62816 KB, 57134 KB/s, 1 seconds passed
    ... 49%, 62848 KB, 57146 KB/s, 1 seconds passed
    ... 49%, 62880 KB, 57160 KB/s, 1 seconds passed
    ... 49%, 62912 KB, 57173 KB/s, 1 seconds passed
    ... 49%, 62944 KB, 57186 KB/s, 1 seconds passed
    ... 49%, 62976 KB, 57198 KB/s, 1 seconds passed
    ... 50%, 63008 KB, 57211 KB/s, 1 seconds passed
    ... 50%, 63040 KB, 57225 KB/s, 1 seconds passed
    ... 50%, 63072 KB, 57238 KB/s, 1 seconds passed
    ... 50%, 63104 KB, 57252 KB/s, 1 seconds passed
    ... 50%, 63136 KB, 57266 KB/s, 1 seconds passed
    ... 50%, 63168 KB, 57278 KB/s, 1 seconds passed
    ... 50%, 63200 KB, 57291 KB/s, 1 seconds passed
    ... 50%, 63232 KB, 57303 KB/s, 1 seconds passed
    ... 50%, 63264 KB, 57317 KB/s, 1 seconds passed
    ... 50%, 63296 KB, 57330 KB/s, 1 seconds passed
    ... 50%, 63328 KB, 57344 KB/s, 1 seconds passed
    ... 50%, 63360 KB, 57358 KB/s, 1 seconds passed
    ... 50%, 63392 KB, 57371 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 63424 KB, 57383 KB/s, 1 seconds passed
    ... 50%, 63456 KB, 57397 KB/s, 1 seconds passed
    ... 50%, 63488 KB, 57410 KB/s, 1 seconds passed
    ... 50%, 63520 KB, 57424 KB/s, 1 seconds passed
    ... 50%, 63552 KB, 57436 KB/s, 1 seconds passed
    ... 50%, 63584 KB, 57449 KB/s, 1 seconds passed
    ... 50%, 63616 KB, 57463 KB/s, 1 seconds passed
    ... 50%, 63648 KB, 57476 KB/s, 1 seconds passed
    ... 50%, 63680 KB, 57490 KB/s, 1 seconds passed
    ... 50%, 63712 KB, 57503 KB/s, 1 seconds passed
    ... 50%, 63744 KB, 57517 KB/s, 1 seconds passed
    ... 50%, 63776 KB, 57530 KB/s, 1 seconds passed
    ... 50%, 63808 KB, 57544 KB/s, 1 seconds passed
    ... 50%, 63840 KB, 57559 KB/s, 1 seconds passed
    ... 50%, 63872 KB, 57577 KB/s, 1 seconds passed
    ... 50%, 63904 KB, 57596 KB/s, 1 seconds passed
    ... 50%, 63936 KB, 57614 KB/s, 1 seconds passed
    ... 50%, 63968 KB, 57632 KB/s, 1 seconds passed
    ... 50%, 64000 KB, 57649 KB/s, 1 seconds passed
    ... 50%, 64032 KB, 57667 KB/s, 1 seconds passed
    ... 50%, 64064 KB, 57686 KB/s, 1 seconds passed
    ... 50%, 64096 KB, 57704 KB/s, 1 seconds passed
    ... 50%, 64128 KB, 57722 KB/s, 1 seconds passed
    ... 50%, 64160 KB, 57740 KB/s, 1 seconds passed
    ... 50%, 64192 KB, 57758 KB/s, 1 seconds passed
    ... 50%, 64224 KB, 57776 KB/s, 1 seconds passed
    ... 51%, 64256 KB, 57792 KB/s, 1 seconds passed
    ... 51%, 64288 KB, 57801 KB/s, 1 seconds passed
    ... 51%, 64320 KB, 57820 KB/s, 1 seconds passed
    ... 51%, 64352 KB, 57834 KB/s, 1 seconds passed
    ... 51%, 64384 KB, 57852 KB/s, 1 seconds passed
    ... 51%, 64416 KB, 57863 KB/s, 1 seconds passed
    ... 51%, 64448 KB, 57878 KB/s, 1 seconds passed
    ... 51%, 64480 KB, 57890 KB/s, 1 seconds passed
    ... 51%, 64512 KB, 57908 KB/s, 1 seconds passed
    ... 51%, 64544 KB, 57920 KB/s, 1 seconds passed
    ... 51%, 64576 KB, 57932 KB/s, 1 seconds passed
    ... 51%, 64608 KB, 57947 KB/s, 1 seconds passed
    ... 51%, 64640 KB, 57962 KB/s, 1 seconds passed
    ... 51%, 64672 KB, 57977 KB/s, 1 seconds passed
    ... 51%, 64704 KB, 57991 KB/s, 1 seconds passed
    ... 51%, 64736 KB, 58006 KB/s, 1 seconds passed
    ... 51%, 64768 KB, 58021 KB/s, 1 seconds passed
    ... 51%, 64800 KB, 58033 KB/s, 1 seconds passed
    ... 51%, 64832 KB, 58051 KB/s, 1 seconds passed
    ... 51%, 64864 KB, 58066 KB/s, 1 seconds passed
    ... 51%, 64896 KB, 58080 KB/s, 1 seconds passed
    ... 51%, 64928 KB, 58087 KB/s, 1 seconds passed
    ... 51%, 64960 KB, 58105 KB/s, 1 seconds passed
    ... 51%, 64992 KB, 58122 KB/s, 1 seconds passed
    ... 51%, 65024 KB, 58136 KB/s, 1 seconds passed
    ... 51%, 65056 KB, 58143 KB/s, 1 seconds passed
    ... 51%, 65088 KB, 58158 KB/s, 1 seconds passed
    ... 51%, 65120 KB, 58173 KB/s, 1 seconds passed
    ... 51%, 65152 KB, 58190 KB/s, 1 seconds passed
    ... 51%, 65184 KB, 58202 KB/s, 1 seconds passed
    ... 51%, 65216 KB, 58217 KB/s, 1 seconds passed
    ... 51%, 65248 KB, 58232 KB/s, 1 seconds passed
    ... 51%, 65280 KB, 58246 KB/s, 1 seconds passed
    ... 51%, 65312 KB, 58258 KB/s, 1 seconds passed
    ... 51%, 65344 KB, 58270 KB/s, 1 seconds passed
    ... 51%, 65376 KB, 58285 KB/s, 1 seconds passed
    ... 51%, 65408 KB, 58231 KB/s, 1 seconds passed
    ... 51%, 65440 KB, 58243 KB/s, 1 seconds passed
    ... 51%, 65472 KB, 58262 KB/s, 1 seconds passed
    ... 52%, 65504 KB, 58273 KB/s, 1 seconds passed
    ... 52%, 65536 KB, 58287 KB/s, 1 seconds passed
    ... 52%, 65568 KB, 58303 KB/s, 1 seconds passed
    ... 52%, 65600 KB, 58317 KB/s, 1 seconds passed
    ... 52%, 65632 KB, 58331 KB/s, 1 seconds passed
    ... 52%, 65664 KB, 58344 KB/s, 1 seconds passed
    ... 52%, 65696 KB, 58205 KB/s, 1 seconds passed
    ... 52%, 65728 KB, 58211 KB/s, 1 seconds passed
    ... 52%, 65760 KB, 58224 KB/s, 1 seconds passed
    ... 52%, 65792 KB, 58236 KB/s, 1 seconds passed
    ... 52%, 65824 KB, 58249 KB/s, 1 seconds passed
    ... 52%, 65856 KB, 58262 KB/s, 1 seconds passed
    ... 52%, 65888 KB, 58275 KB/s, 1 seconds passed
    ... 52%, 65920 KB, 58288 KB/s, 1 seconds passed
    ... 52%, 65952 KB, 58301 KB/s, 1 seconds passed
    ... 52%, 65984 KB, 58313 KB/s, 1 seconds passed
    ... 52%, 66016 KB, 58326 KB/s, 1 seconds passed
    ... 52%, 66048 KB, 58339 KB/s, 1 seconds passed
    ... 52%, 66080 KB, 58352 KB/s, 1 seconds passed
    ... 52%, 66112 KB, 58365 KB/s, 1 seconds passed
    ... 52%, 66144 KB, 58377 KB/s, 1 seconds passed
    ... 52%, 66176 KB, 58389 KB/s, 1 seconds passed
    ... 52%, 66208 KB, 58401 KB/s, 1 seconds passed
    ... 52%, 66240 KB, 58414 KB/s, 1 seconds passed
    ... 52%, 66272 KB, 58427 KB/s, 1 seconds passed
    ... 52%, 66304 KB, 58440 KB/s, 1 seconds passed
    ... 52%, 66336 KB, 58453 KB/s, 1 seconds passed
    ... 52%, 66368 KB, 58465 KB/s, 1 seconds passed
    ... 52%, 66400 KB, 58479 KB/s, 1 seconds passed
    ... 52%, 66432 KB, 58492 KB/s, 1 seconds passed
    ... 52%, 66464 KB, 58505 KB/s, 1 seconds passed
    ... 52%, 66496 KB, 58517 KB/s, 1 seconds passed
    ... 52%, 66528 KB, 58529 KB/s, 1 seconds passed

.. parsed-literal::

    ... 52%, 66560 KB, 54734 KB/s, 1 seconds passed
    ... 52%, 66592 KB, 54738 KB/s, 1 seconds passed
    ... 52%, 66624 KB, 54749 KB/s, 1 seconds passed
    ... 52%, 66656 KB, 54760 KB/s, 1 seconds passed
    ... 52%, 66688 KB, 54720 KB/s, 1 seconds passed
    ... 52%, 66720 KB, 54726 KB/s, 1 seconds passed
    ... 52%, 66752 KB, 54737 KB/s, 1 seconds passed
    ... 53%, 66784 KB, 54749 KB/s, 1 seconds passed
    ... 53%, 66816 KB, 54761 KB/s, 1 seconds passed
    ... 53%, 66848 KB, 54772 KB/s, 1 seconds passed
    ... 53%, 66880 KB, 54784 KB/s, 1 seconds passed
    ... 53%, 66912 KB, 54796 KB/s, 1 seconds passed
    ... 53%, 66944 KB, 54808 KB/s, 1 seconds passed
    ... 53%, 66976 KB, 54821 KB/s, 1 seconds passed
    ... 53%, 67008 KB, 54833 KB/s, 1 seconds passed
    ... 53%, 67040 KB, 54847 KB/s, 1 seconds passed
    ... 53%, 67072 KB, 54859 KB/s, 1 seconds passed
    ... 53%, 67104 KB, 54872 KB/s, 1 seconds passed
    ... 53%, 67136 KB, 54885 KB/s, 1 seconds passed
    ... 53%, 67168 KB, 54897 KB/s, 1 seconds passed
    ... 53%, 67200 KB, 54909 KB/s, 1 seconds passed
    ... 53%, 67232 KB, 54921 KB/s, 1 seconds passed
    ... 53%, 67264 KB, 54934 KB/s, 1 seconds passed
    ... 53%, 67296 KB, 54946 KB/s, 1 seconds passed
    ... 53%, 67328 KB, 54958 KB/s, 1 seconds passed
    ... 53%, 67360 KB, 54969 KB/s, 1 seconds passed
    ... 53%, 67392 KB, 54982 KB/s, 1 seconds passed
    ... 53%, 67424 KB, 54994 KB/s, 1 seconds passed
    ... 53%, 67456 KB, 55006 KB/s, 1 seconds passed
    ... 53%, 67488 KB, 55018 KB/s, 1 seconds passed
    ... 53%, 67520 KB, 55029 KB/s, 1 seconds passed
    ... 53%, 67552 KB, 55041 KB/s, 1 seconds passed
    ... 53%, 67584 KB, 55054 KB/s, 1 seconds passed
    ... 53%, 67616 KB, 55065 KB/s, 1 seconds passed
    ... 53%, 67648 KB, 55077 KB/s, 1 seconds passed
    ... 53%, 67680 KB, 55089 KB/s, 1 seconds passed
    ... 53%, 67712 KB, 55102 KB/s, 1 seconds passed
    ... 53%, 67744 KB, 55112 KB/s, 1 seconds passed
    ... 53%, 67776 KB, 55124 KB/s, 1 seconds passed
    ... 53%, 67808 KB, 55136 KB/s, 1 seconds passed
    ... 53%, 67840 KB, 55150 KB/s, 1 seconds passed
    ... 53%, 67872 KB, 55163 KB/s, 1 seconds passed
    ... 53%, 67904 KB, 55180 KB/s, 1 seconds passed
    ... 53%, 67936 KB, 55197 KB/s, 1 seconds passed
    ... 53%, 67968 KB, 55214 KB/s, 1 seconds passed
    ... 53%, 68000 KB, 55231 KB/s, 1 seconds passed
    ... 54%, 68032 KB, 55248 KB/s, 1 seconds passed
    ... 54%, 68064 KB, 55074 KB/s, 1 seconds passed
    ... 54%, 68096 KB, 55084 KB/s, 1 seconds passed
    ... 54%, 68128 KB, 55095 KB/s, 1 seconds passed
    ... 54%, 68160 KB, 55109 KB/s, 1 seconds passed
    ... 54%, 68192 KB, 55124 KB/s, 1 seconds passed
    ... 54%, 68224 KB, 55140 KB/s, 1 seconds passed
    ... 54%, 68256 KB, 55157 KB/s, 1 seconds passed
    ... 54%, 68288 KB, 55170 KB/s, 1 seconds passed
    ... 54%, 68320 KB, 55181 KB/s, 1 seconds passed
    ... 54%, 68352 KB, 55196 KB/s, 1 seconds passed
    ... 54%, 68384 KB, 55210 KB/s, 1 seconds passed
    ... 54%, 68416 KB, 55219 KB/s, 1 seconds passed

.. parsed-literal::

    ... 54%, 68448 KB, 54036 KB/s, 1 seconds passed
    ... 54%, 68480 KB, 54038 KB/s, 1 seconds passed
    ... 54%, 68512 KB, 54042 KB/s, 1 seconds passed
    ... 54%, 68544 KB, 54048 KB/s, 1 seconds passed
    ... 54%, 68576 KB, 54055 KB/s, 1 seconds passed
    ... 54%, 68608 KB, 54062 KB/s, 1 seconds passed
    ... 54%, 68640 KB, 54067 KB/s, 1 seconds passed
    ... 54%, 68672 KB, 54075 KB/s, 1 seconds passed
    ... 54%, 68704 KB, 54084 KB/s, 1 seconds passed
    ... 54%, 68736 KB, 54093 KB/s, 1 seconds passed
    ... 54%, 68768 KB, 54102 KB/s, 1 seconds passed
    ... 54%, 68800 KB, 54111 KB/s, 1 seconds passed
    ... 54%, 68832 KB, 54121 KB/s, 1 seconds passed
    ... 54%, 68864 KB, 54130 KB/s, 1 seconds passed
    ... 54%, 68896 KB, 54139 KB/s, 1 seconds passed
    ... 54%, 68928 KB, 54147 KB/s, 1 seconds passed
    ... 54%, 68960 KB, 54154 KB/s, 1 seconds passed
    ... 54%, 68992 KB, 54163 KB/s, 1 seconds passed
    ... 54%, 69024 KB, 54173 KB/s, 1 seconds passed
    ... 54%, 69056 KB, 54182 KB/s, 1 seconds passed
    ... 54%, 69088 KB, 54191 KB/s, 1 seconds passed
    ... 54%, 69120 KB, 54199 KB/s, 1 seconds passed
    ... 54%, 69152 KB, 54207 KB/s, 1 seconds passed
    ... 54%, 69184 KB, 54217 KB/s, 1 seconds passed
    ... 54%, 69216 KB, 54226 KB/s, 1 seconds passed
    ... 54%, 69248 KB, 54236 KB/s, 1 seconds passed
    ... 55%, 69280 KB, 54243 KB/s, 1 seconds passed
    ... 55%, 69312 KB, 54252 KB/s, 1 seconds passed
    ... 55%, 69344 KB, 54260 KB/s, 1 seconds passed
    ... 55%, 69376 KB, 54269 KB/s, 1 seconds passed
    ... 55%, 69408 KB, 54278 KB/s, 1 seconds passed
    ... 55%, 69440 KB, 54287 KB/s, 1 seconds passed
    ... 55%, 69472 KB, 54296 KB/s, 1 seconds passed
    ... 55%, 69504 KB, 54306 KB/s, 1 seconds passed
    ... 55%, 69536 KB, 54315 KB/s, 1 seconds passed
    ... 55%, 69568 KB, 54324 KB/s, 1 seconds passed
    ... 55%, 69600 KB, 54332 KB/s, 1 seconds passed
    ... 55%, 69632 KB, 54346 KB/s, 1 seconds passed
    ... 55%, 69664 KB, 54361 KB/s, 1 seconds passed
    ... 55%, 69696 KB, 54376 KB/s, 1 seconds passed
    ... 55%, 69728 KB, 54391 KB/s, 1 seconds passed
    ... 55%, 69760 KB, 54405 KB/s, 1 seconds passed
    ... 55%, 69792 KB, 54420 KB/s, 1 seconds passed
    ... 55%, 69824 KB, 54435 KB/s, 1 seconds passed
    ... 55%, 69856 KB, 54450 KB/s, 1 seconds passed
    ... 55%, 69888 KB, 54465 KB/s, 1 seconds passed
    ... 55%, 69920 KB, 54479 KB/s, 1 seconds passed
    ... 55%, 69952 KB, 54494 KB/s, 1 seconds passed
    ... 55%, 69984 KB, 54509 KB/s, 1 seconds passed
    ... 55%, 70016 KB, 54524 KB/s, 1 seconds passed
    ... 55%, 70048 KB, 54539 KB/s, 1 seconds passed
    ... 55%, 70080 KB, 54554 KB/s, 1 seconds passed
    ... 55%, 70112 KB, 54569 KB/s, 1 seconds passed
    ... 55%, 70144 KB, 54583 KB/s, 1 seconds passed
    ... 55%, 70176 KB, 54598 KB/s, 1 seconds passed
    ... 55%, 70208 KB, 54612 KB/s, 1 seconds passed
    ... 55%, 70240 KB, 54627 KB/s, 1 seconds passed
    ... 55%, 70272 KB, 54642 KB/s, 1 seconds passed
    ... 55%, 70304 KB, 54657 KB/s, 1 seconds passed
    ... 55%, 70336 KB, 54672 KB/s, 1 seconds passed
    ... 55%, 70368 KB, 54686 KB/s, 1 seconds passed
    ... 55%, 70400 KB, 54701 KB/s, 1 seconds passed
    ... 55%, 70432 KB, 54716 KB/s, 1 seconds passed
    ... 55%, 70464 KB, 54730 KB/s, 1 seconds passed
    ... 55%, 70496 KB, 54744 KB/s, 1 seconds passed
    ... 55%, 70528 KB, 54759 KB/s, 1 seconds passed
    ... 56%, 70560 KB, 54774 KB/s, 1 seconds passed
    ... 56%, 70592 KB, 54789 KB/s, 1 seconds passed
    ... 56%, 70624 KB, 54804 KB/s, 1 seconds passed
    ... 56%, 70656 KB, 54819 KB/s, 1 seconds passed
    ... 56%, 70688 KB, 54832 KB/s, 1 seconds passed
    ... 56%, 70720 KB, 54847 KB/s, 1 seconds passed
    ... 56%, 70752 KB, 54862 KB/s, 1 seconds passed
    ... 56%, 70784 KB, 54877 KB/s, 1 seconds passed
    ... 56%, 70816 KB, 54892 KB/s, 1 seconds passed
    ... 56%, 70848 KB, 54906 KB/s, 1 seconds passed
    ... 56%, 70880 KB, 54921 KB/s, 1 seconds passed
    ... 56%, 70912 KB, 54936 KB/s, 1 seconds passed
    ... 56%, 70944 KB, 54950 KB/s, 1 seconds passed
    ... 56%, 70976 KB, 54968 KB/s, 1 seconds passed
    ... 56%, 71008 KB, 54986 KB/s, 1 seconds passed
    ... 56%, 71040 KB, 55003 KB/s, 1 seconds passed
    ... 56%, 71072 KB, 55021 KB/s, 1 seconds passed
    ... 56%, 71104 KB, 55039 KB/s, 1 seconds passed
    ... 56%, 71136 KB, 55056 KB/s, 1 seconds passed
    ... 56%, 71168 KB, 55074 KB/s, 1 seconds passed
    ... 56%, 71200 KB, 55091 KB/s, 1 seconds passed
    ... 56%, 71232 KB, 55109 KB/s, 1 seconds passed
    ... 56%, 71264 KB, 55127 KB/s, 1 seconds passed
    ... 56%, 71296 KB, 55145 KB/s, 1 seconds passed
    ... 56%, 71328 KB, 55162 KB/s, 1 seconds passed
    ... 56%, 71360 KB, 55180 KB/s, 1 seconds passed
    ... 56%, 71392 KB, 55198 KB/s, 1 seconds passed
    ... 56%, 71424 KB, 55217 KB/s, 1 seconds passed
    ... 56%, 71456 KB, 55236 KB/s, 1 seconds passed
    ... 56%, 71488 KB, 55256 KB/s, 1 seconds passed
    ... 56%, 71520 KB, 55254 KB/s, 1 seconds passed
    ... 56%, 71552 KB, 55267 KB/s, 1 seconds passed
    ... 56%, 71584 KB, 55279 KB/s, 1 seconds passed
    ... 56%, 71616 KB, 55290 KB/s, 1 seconds passed
    ... 56%, 71648 KB, 55298 KB/s, 1 seconds passed

.. parsed-literal::

    ... 56%, 71680 KB, 54468 KB/s, 1 seconds passed
    ... 56%, 71712 KB, 54471 KB/s, 1 seconds passed
    ... 56%, 71744 KB, 54478 KB/s, 1 seconds passed
    ... 56%, 71776 KB, 54487 KB/s, 1 seconds passed
    ... 57%, 71808 KB, 54494 KB/s, 1 seconds passed
    ... 57%, 71840 KB, 54503 KB/s, 1 seconds passed
    ... 57%, 71872 KB, 54511 KB/s, 1 seconds passed
    ... 57%, 71904 KB, 54520 KB/s, 1 seconds passed
    ... 57%, 71936 KB, 54532 KB/s, 1 seconds passed
    ... 57%, 71968 KB, 54544 KB/s, 1 seconds passed
    ... 57%, 72000 KB, 54547 KB/s, 1 seconds passed
    ... 57%, 72032 KB, 54555 KB/s, 1 seconds passed
    ... 57%, 72064 KB, 54564 KB/s, 1 seconds passed
    ... 57%, 72096 KB, 54572 KB/s, 1 seconds passed
    ... 57%, 72128 KB, 54581 KB/s, 1 seconds passed
    ... 57%, 72160 KB, 54590 KB/s, 1 seconds passed
    ... 57%, 72192 KB, 54596 KB/s, 1 seconds passed
    ... 57%, 72224 KB, 54604 KB/s, 1 seconds passed
    ... 57%, 72256 KB, 54613 KB/s, 1 seconds passed
    ... 57%, 72288 KB, 54622 KB/s, 1 seconds passed
    ... 57%, 72320 KB, 54631 KB/s, 1 seconds passed
    ... 57%, 72352 KB, 54639 KB/s, 1 seconds passed
    ... 57%, 72384 KB, 54647 KB/s, 1 seconds passed
    ... 57%, 72416 KB, 54656 KB/s, 1 seconds passed
    ... 57%, 72448 KB, 54663 KB/s, 1 seconds passed
    ... 57%, 72480 KB, 54672 KB/s, 1 seconds passed
    ... 57%, 72512 KB, 54681 KB/s, 1 seconds passed
    ... 57%, 72544 KB, 54690 KB/s, 1 seconds passed
    ... 57%, 72576 KB, 54702 KB/s, 1 seconds passed
    ... 57%, 72608 KB, 54714 KB/s, 1 seconds passed
    ... 57%, 72640 KB, 54726 KB/s, 1 seconds passed
    ... 57%, 72672 KB, 54739 KB/s, 1 seconds passed
    ... 57%, 72704 KB, 54751 KB/s, 1 seconds passed
    ... 57%, 72736 KB, 54652 KB/s, 1 seconds passed
    ... 57%, 72768 KB, 54551 KB/s, 1 seconds passed
    ... 57%, 72800 KB, 54553 KB/s, 1 seconds passed
    ... 57%, 72832 KB, 54561 KB/s, 1 seconds passed
    ... 57%, 72864 KB, 54570 KB/s, 1 seconds passed
    ... 57%, 72896 KB, 54579 KB/s, 1 seconds passed
    ... 57%, 72928 KB, 54589 KB/s, 1 seconds passed
    ... 57%, 72960 KB, 54599 KB/s, 1 seconds passed
    ... 57%, 72992 KB, 54609 KB/s, 1 seconds passed
    ... 57%, 73024 KB, 54618 KB/s, 1 seconds passed
    ... 58%, 73056 KB, 54628 KB/s, 1 seconds passed
    ... 58%, 73088 KB, 54638 KB/s, 1 seconds passed
    ... 58%, 73120 KB, 54647 KB/s, 1 seconds passed
    ... 58%, 73152 KB, 54657 KB/s, 1 seconds passed
    ... 58%, 73184 KB, 54667 KB/s, 1 seconds passed
    ... 58%, 73216 KB, 54676 KB/s, 1 seconds passed
    ... 58%, 73248 KB, 54686 KB/s, 1 seconds passed
    ... 58%, 73280 KB, 54696 KB/s, 1 seconds passed
    ... 58%, 73312 KB, 54706 KB/s, 1 seconds passed
    ... 58%, 73344 KB, 54716 KB/s, 1 seconds passed
    ... 58%, 73376 KB, 54726 KB/s, 1 seconds passed
    ... 58%, 73408 KB, 54735 KB/s, 1 seconds passed
    ... 58%, 73440 KB, 54744 KB/s, 1 seconds passed
    ... 58%, 73472 KB, 54754 KB/s, 1 seconds passed
    ... 58%, 73504 KB, 54765 KB/s, 1 seconds passed
    ... 58%, 73536 KB, 54777 KB/s, 1 seconds passed
    ... 58%, 73568 KB, 54790 KB/s, 1 seconds passed
    ... 58%, 73600 KB, 54802 KB/s, 1 seconds passed
    ... 58%, 73632 KB, 54815 KB/s, 1 seconds passed
    ... 58%, 73664 KB, 54827 KB/s, 1 seconds passed
    ... 58%, 73696 KB, 54839 KB/s, 1 seconds passed
    ... 58%, 73728 KB, 54851 KB/s, 1 seconds passed
    ... 58%, 73760 KB, 54864 KB/s, 1 seconds passed
    ... 58%, 73792 KB, 54876 KB/s, 1 seconds passed
    ... 58%, 73824 KB, 54888 KB/s, 1 seconds passed
    ... 58%, 73856 KB, 54900 KB/s, 1 seconds passed
    ... 58%, 73888 KB, 54911 KB/s, 1 seconds passed
    ... 58%, 73920 KB, 54923 KB/s, 1 seconds passed
    ... 58%, 73952 KB, 54935 KB/s, 1 seconds passed
    ... 58%, 73984 KB, 54948 KB/s, 1 seconds passed
    ... 58%, 74016 KB, 54960 KB/s, 1 seconds passed
    ... 58%, 74048 KB, 54972 KB/s, 1 seconds passed
    ... 58%, 74080 KB, 54984 KB/s, 1 seconds passed
    ... 58%, 74112 KB, 54996 KB/s, 1 seconds passed
    ... 58%, 74144 KB, 55009 KB/s, 1 seconds passed
    ... 58%, 74176 KB, 55022 KB/s, 1 seconds passed
    ... 58%, 74208 KB, 55034 KB/s, 1 seconds passed
    ... 58%, 74240 KB, 55046 KB/s, 1 seconds passed
    ... 58%, 74272 KB, 55058 KB/s, 1 seconds passed
    ... 58%, 74304 KB, 55070 KB/s, 1 seconds passed
    ... 59%, 74336 KB, 55081 KB/s, 1 seconds passed
    ... 59%, 74368 KB, 55094 KB/s, 1 seconds passed
    ... 59%, 74400 KB, 55106 KB/s, 1 seconds passed
    ... 59%, 74432 KB, 55118 KB/s, 1 seconds passed
    ... 59%, 74464 KB, 55130 KB/s, 1 seconds passed
    ... 59%, 74496 KB, 55141 KB/s, 1 seconds passed
    ... 59%, 74528 KB, 55154 KB/s, 1 seconds passed
    ... 59%, 74560 KB, 55166 KB/s, 1 seconds passed
    ... 59%, 74592 KB, 55178 KB/s, 1 seconds passed
    ... 59%, 74624 KB, 55191 KB/s, 1 seconds passed
    ... 59%, 74656 KB, 55207 KB/s, 1 seconds passed
    ... 59%, 74688 KB, 55223 KB/s, 1 seconds passed
    ... 59%, 74720 KB, 55239 KB/s, 1 seconds passed
    ... 59%, 74752 KB, 55255 KB/s, 1 seconds passed
    ... 59%, 74784 KB, 55271 KB/s, 1 seconds passed
    ... 59%, 74816 KB, 55287 KB/s, 1 seconds passed
    ... 59%, 74848 KB, 55302 KB/s, 1 seconds passed
    ... 59%, 74880 KB, 55319 KB/s, 1 seconds passed
    ... 59%, 74912 KB, 55336 KB/s, 1 seconds passed
    ... 59%, 74944 KB, 55353 KB/s, 1 seconds passed
    ... 59%, 74976 KB, 55371 KB/s, 1 seconds passed
    ... 59%, 75008 KB, 55388 KB/s, 1 seconds passed

.. parsed-literal::

    ... 59%, 75040 KB, 55037 KB/s, 1 seconds passed
    ... 59%, 75072 KB, 55042 KB/s, 1 seconds passed
    ... 59%, 75104 KB, 55049 KB/s, 1 seconds passed
    ... 59%, 75136 KB, 55057 KB/s, 1 seconds passed
    ... 59%, 75168 KB, 55065 KB/s, 1 seconds passed
    ... 59%, 75200 KB, 55073 KB/s, 1 seconds passed
    ... 59%, 75232 KB, 55082 KB/s, 1 seconds passed
    ... 59%, 75264 KB, 55089 KB/s, 1 seconds passed
    ... 59%, 75296 KB, 55098 KB/s, 1 seconds passed
    ... 59%, 75328 KB, 55106 KB/s, 1 seconds passed
    ... 59%, 75360 KB, 55115 KB/s, 1 seconds passed
    ... 59%, 75392 KB, 55122 KB/s, 1 seconds passed
    ... 59%, 75424 KB, 55130 KB/s, 1 seconds passed
    ... 59%, 75456 KB, 55137 KB/s, 1 seconds passed
    ... 59%, 75488 KB, 55146 KB/s, 1 seconds passed
    ... 59%, 75520 KB, 55154 KB/s, 1 seconds passed
    ... 59%, 75552 KB, 55162 KB/s, 1 seconds passed
    ... 60%, 75584 KB, 55168 KB/s, 1 seconds passed
    ... 60%, 75616 KB, 55176 KB/s, 1 seconds passed
    ... 60%, 75648 KB, 55185 KB/s, 1 seconds passed
    ... 60%, 75680 KB, 55193 KB/s, 1 seconds passed
    ... 60%, 75712 KB, 55201 KB/s, 1 seconds passed
    ... 60%, 75744 KB, 55210 KB/s, 1 seconds passed
    ... 60%, 75776 KB, 55219 KB/s, 1 seconds passed
    ... 60%, 75808 KB, 55230 KB/s, 1 seconds passed
    ... 60%, 75840 KB, 55241 KB/s, 1 seconds passed
    ... 60%, 75872 KB, 55252 KB/s, 1 seconds passed
    ... 60%, 75904 KB, 55263 KB/s, 1 seconds passed
    ... 60%, 75936 KB, 55274 KB/s, 1 seconds passed
    ... 60%, 75968 KB, 55285 KB/s, 1 seconds passed
    ... 60%, 76000 KB, 55296 KB/s, 1 seconds passed
    ... 60%, 76032 KB, 55308 KB/s, 1 seconds passed
    ... 60%, 76064 KB, 55319 KB/s, 1 seconds passed
    ... 60%, 76096 KB, 55330 KB/s, 1 seconds passed
    ... 60%, 76128 KB, 55341 KB/s, 1 seconds passed
    ... 60%, 76160 KB, 55352 KB/s, 1 seconds passed
    ... 60%, 76192 KB, 55363 KB/s, 1 seconds passed
    ... 60%, 76224 KB, 55374 KB/s, 1 seconds passed
    ... 60%, 76256 KB, 55386 KB/s, 1 seconds passed
    ... 60%, 76288 KB, 55394 KB/s, 1 seconds passed
    ... 60%, 76320 KB, 55405 KB/s, 1 seconds passed
    ... 60%, 76352 KB, 55416 KB/s, 1 seconds passed
    ... 60%, 76384 KB, 55427 KB/s, 1 seconds passed
    ... 60%, 76416 KB, 55439 KB/s, 1 seconds passed
    ... 60%, 76448 KB, 55451 KB/s, 1 seconds passed
    ... 60%, 76480 KB, 55464 KB/s, 1 seconds passed
    ... 60%, 76512 KB, 55478 KB/s, 1 seconds passed
    ... 60%, 76544 KB, 55492 KB/s, 1 seconds passed
    ... 60%, 76576 KB, 55506 KB/s, 1 seconds passed
    ... 60%, 76608 KB, 55520 KB/s, 1 seconds passed
    ... 60%, 76640 KB, 55534 KB/s, 1 seconds passed
    ... 60%, 76672 KB, 55548 KB/s, 1 seconds passed
    ... 60%, 76704 KB, 55561 KB/s, 1 seconds passed
    ... 60%, 76736 KB, 55575 KB/s, 1 seconds passed
    ... 60%, 76768 KB, 55589 KB/s, 1 seconds passed
    ... 60%, 76800 KB, 54876 KB/s, 1 seconds passed
    ... 61%, 76832 KB, 54829 KB/s, 1 seconds passed
    ... 61%, 76864 KB, 54835 KB/s, 1 seconds passed
    ... 61%, 76896 KB, 54842 KB/s, 1 seconds passed
    ... 61%, 76928 KB, 54850 KB/s, 1 seconds passed
    ... 61%, 76960 KB, 54860 KB/s, 1 seconds passed
    ... 61%, 76992 KB, 54854 KB/s, 1 seconds passed
    ... 61%, 77024 KB, 54861 KB/s, 1 seconds passed
    ... 61%, 77056 KB, 54868 KB/s, 1 seconds passed
    ... 61%, 77088 KB, 54875 KB/s, 1 seconds passed
    ... 61%, 77120 KB, 54881 KB/s, 1 seconds passed
    ... 61%, 77152 KB, 54888 KB/s, 1 seconds passed
    ... 61%, 77184 KB, 54897 KB/s, 1 seconds passed
    ... 61%, 77216 KB, 54905 KB/s, 1 seconds passed
    ... 61%, 77248 KB, 54912 KB/s, 1 seconds passed
    ... 61%, 77280 KB, 54920 KB/s, 1 seconds passed
    ... 61%, 77312 KB, 54928 KB/s, 1 seconds passed
    ... 61%, 77344 KB, 54937 KB/s, 1 seconds passed
    ... 61%, 77376 KB, 54944 KB/s, 1 seconds passed
    ... 61%, 77408 KB, 54953 KB/s, 1 seconds passed
    ... 61%, 77440 KB, 54958 KB/s, 1 seconds passed
    ... 61%, 77472 KB, 54965 KB/s, 1 seconds passed
    ... 61%, 77504 KB, 54973 KB/s, 1 seconds passed
    ... 61%, 77536 KB, 54981 KB/s, 1 seconds passed
    ... 61%, 77568 KB, 54989 KB/s, 1 seconds passed
    ... 61%, 77600 KB, 54997 KB/s, 1 seconds passed
    ... 61%, 77632 KB, 55005 KB/s, 1 seconds passed
    ... 61%, 77664 KB, 55012 KB/s, 1 seconds passed
    ... 61%, 77696 KB, 55020 KB/s, 1 seconds passed
    ... 61%, 77728 KB, 55028 KB/s, 1 seconds passed
    ... 61%, 77760 KB, 55036 KB/s, 1 seconds passed
    ... 61%, 77792 KB, 55048 KB/s, 1 seconds passed

.. parsed-literal::

    ... 61%, 77824 KB, 55061 KB/s, 1 seconds passed
    ... 61%, 77856 KB, 55074 KB/s, 1 seconds passed
    ... 61%, 77888 KB, 55087 KB/s, 1 seconds passed
    ... 61%, 77920 KB, 55100 KB/s, 1 seconds passed
    ... 61%, 77952 KB, 55113 KB/s, 1 seconds passed
    ... 61%, 77984 KB, 55125 KB/s, 1 seconds passed
    ... 61%, 78016 KB, 55138 KB/s, 1 seconds passed
    ... 61%, 78048 KB, 55151 KB/s, 1 seconds passed
    ... 61%, 78080 KB, 55163 KB/s, 1 seconds passed
    ... 62%, 78112 KB, 55176 KB/s, 1 seconds passed
    ... 62%, 78144 KB, 55189 KB/s, 1 seconds passed
    ... 62%, 78176 KB, 55201 KB/s, 1 seconds passed
    ... 62%, 78208 KB, 55213 KB/s, 1 seconds passed
    ... 62%, 78240 KB, 55226 KB/s, 1 seconds passed
    ... 62%, 78272 KB, 55238 KB/s, 1 seconds passed
    ... 62%, 78304 KB, 55250 KB/s, 1 seconds passed
    ... 62%, 78336 KB, 55262 KB/s, 1 seconds passed
    ... 62%, 78368 KB, 55275 KB/s, 1 seconds passed
    ... 62%, 78400 KB, 55288 KB/s, 1 seconds passed
    ... 62%, 78432 KB, 55301 KB/s, 1 seconds passed
    ... 62%, 78464 KB, 55314 KB/s, 1 seconds passed
    ... 62%, 78496 KB, 55326 KB/s, 1 seconds passed
    ... 62%, 78528 KB, 55338 KB/s, 1 seconds passed
    ... 62%, 78560 KB, 55351 KB/s, 1 seconds passed
    ... 62%, 78592 KB, 55364 KB/s, 1 seconds passed
    ... 62%, 78624 KB, 55376 KB/s, 1 seconds passed
    ... 62%, 78656 KB, 55389 KB/s, 1 seconds passed
    ... 62%, 78688 KB, 55402 KB/s, 1 seconds passed
    ... 62%, 78720 KB, 55415 KB/s, 1 seconds passed
    ... 62%, 78752 KB, 55427 KB/s, 1 seconds passed
    ... 62%, 78784 KB, 55440 KB/s, 1 seconds passed
    ... 62%, 78816 KB, 55452 KB/s, 1 seconds passed
    ... 62%, 78848 KB, 55465 KB/s, 1 seconds passed
    ... 62%, 78880 KB, 55477 KB/s, 1 seconds passed
    ... 62%, 78912 KB, 55490 KB/s, 1 seconds passed
    ... 62%, 78944 KB, 55503 KB/s, 1 seconds passed
    ... 62%, 78976 KB, 55515 KB/s, 1 seconds passed
    ... 62%, 79008 KB, 55527 KB/s, 1 seconds passed
    ... 62%, 79040 KB, 55540 KB/s, 1 seconds passed
    ... 62%, 79072 KB, 55553 KB/s, 1 seconds passed
    ... 62%, 79104 KB, 55567 KB/s, 1 seconds passed
    ... 62%, 79136 KB, 55583 KB/s, 1 seconds passed
    ... 62%, 79168 KB, 55599 KB/s, 1 seconds passed
    ... 62%, 79200 KB, 55615 KB/s, 1 seconds passed
    ... 62%, 79232 KB, 55630 KB/s, 1 seconds passed
    ... 62%, 79264 KB, 55646 KB/s, 1 seconds passed
    ... 62%, 79296 KB, 55662 KB/s, 1 seconds passed
    ... 62%, 79328 KB, 55678 KB/s, 1 seconds passed
    ... 63%, 79360 KB, 55694 KB/s, 1 seconds passed
    ... 63%, 79392 KB, 55709 KB/s, 1 seconds passed
    ... 63%, 79424 KB, 55725 KB/s, 1 seconds passed
    ... 63%, 79456 KB, 55740 KB/s, 1 seconds passed
    ... 63%, 79488 KB, 55756 KB/s, 1 seconds passed
    ... 63%, 79520 KB, 55772 KB/s, 1 seconds passed
    ... 63%, 79552 KB, 55788 KB/s, 1 seconds passed
    ... 63%, 79584 KB, 55804 KB/s, 1 seconds passed
    ... 63%, 79616 KB, 55820 KB/s, 1 seconds passed
    ... 63%, 79648 KB, 55831 KB/s, 1 seconds passed
    ... 63%, 79680 KB, 55840 KB/s, 1 seconds passed
    ... 63%, 79712 KB, 55852 KB/s, 1 seconds passed
    ... 63%, 79744 KB, 55860 KB/s, 1 seconds passed
    ... 63%, 79776 KB, 55867 KB/s, 1 seconds passed
    ... 63%, 79808 KB, 55881 KB/s, 1 seconds passed
    ... 63%, 79840 KB, 55897 KB/s, 1 seconds passed
    ... 63%, 79872 KB, 55910 KB/s, 1 seconds passed
    ... 63%, 79904 KB, 55922 KB/s, 1 seconds passed
    ... 63%, 79936 KB, 55932 KB/s, 1 seconds passed
    ... 63%, 79968 KB, 55945 KB/s, 1 seconds passed
    ... 63%, 80000 KB, 55957 KB/s, 1 seconds passed
    ... 63%, 80032 KB, 55969 KB/s, 1 seconds passed
    ... 63%, 80064 KB, 55981 KB/s, 1 seconds passed
    ... 63%, 80096 KB, 55993 KB/s, 1 seconds passed
    ... 63%, 80128 KB, 56003 KB/s, 1 seconds passed
    ... 63%, 80160 KB, 56015 KB/s, 1 seconds passed
    ... 63%, 80192 KB, 56027 KB/s, 1 seconds passed
    ... 63%, 80224 KB, 56033 KB/s, 1 seconds passed
    ... 63%, 80256 KB, 56043 KB/s, 1 seconds passed
    ... 63%, 80288 KB, 56050 KB/s, 1 seconds passed
    ... 63%, 80320 KB, 56063 KB/s, 1 seconds passed
    ... 63%, 80352 KB, 56079 KB/s, 1 seconds passed
    ... 63%, 80384 KB, 56092 KB/s, 1 seconds passed
    ... 63%, 80416 KB, 56104 KB/s, 1 seconds passed
    ... 63%, 80448 KB, 56113 KB/s, 1 seconds passed
    ... 63%, 80480 KB, 56124 KB/s, 1 seconds passed
    ... 63%, 80512 KB, 56135 KB/s, 1 seconds passed
    ... 63%, 80544 KB, 56146 KB/s, 1 seconds passed
    ... 63%, 80576 KB, 55707 KB/s, 1 seconds passed
    ... 63%, 80608 KB, 55711 KB/s, 1 seconds passed
    ... 64%, 80640 KB, 55718 KB/s, 1 seconds passed
    ... 64%, 80672 KB, 55725 KB/s, 1 seconds passed
    ... 64%, 80704 KB, 55733 KB/s, 1 seconds passed
    ... 64%, 80736 KB, 55741 KB/s, 1 seconds passed
    ... 64%, 80768 KB, 55748 KB/s, 1 seconds passed
    ... 64%, 80800 KB, 55754 KB/s, 1 seconds passed
    ... 64%, 80832 KB, 55761 KB/s, 1 seconds passed
    ... 64%, 80864 KB, 55769 KB/s, 1 seconds passed
    ... 64%, 80896 KB, 55777 KB/s, 1 seconds passed
    ... 64%, 80928 KB, 55785 KB/s, 1 seconds passed
    ... 64%, 80960 KB, 55792 KB/s, 1 seconds passed
    ... 64%, 80992 KB, 55799 KB/s, 1 seconds passed
    ... 64%, 81024 KB, 55807 KB/s, 1 seconds passed
    ... 64%, 81056 KB, 55815 KB/s, 1 seconds passed
    ... 64%, 81088 KB, 55823 KB/s, 1 seconds passed
    ... 64%, 81120 KB, 55831 KB/s, 1 seconds passed
    ... 64%, 81152 KB, 55838 KB/s, 1 seconds passed
    ... 64%, 81184 KB, 55846 KB/s, 1 seconds passed
    ... 64%, 81216 KB, 55853 KB/s, 1 seconds passed
    ... 64%, 81248 KB, 55861 KB/s, 1 seconds passed
    ... 64%, 81280 KB, 55869 KB/s, 1 seconds passed
    ... 64%, 81312 KB, 55877 KB/s, 1 seconds passed
    ... 64%, 81344 KB, 55885 KB/s, 1 seconds passed
    ... 64%, 81376 KB, 55892 KB/s, 1 seconds passed
    ... 64%, 81408 KB, 55900 KB/s, 1 seconds passed
    ... 64%, 81440 KB, 55908 KB/s, 1 seconds passed
    ... 64%, 81472 KB, 55916 KB/s, 1 seconds passed
    ... 64%, 81504 KB, 55926 KB/s, 1 seconds passed
    ... 64%, 81536 KB, 55937 KB/s, 1 seconds passed
    ... 64%, 81568 KB, 55947 KB/s, 1 seconds passed
    ... 64%, 81600 KB, 55958 KB/s, 1 seconds passed
    ... 64%, 81632 KB, 55969 KB/s, 1 seconds passed
    ... 64%, 81664 KB, 55981 KB/s, 1 seconds passed
    ... 64%, 81696 KB, 55991 KB/s, 1 seconds passed
    ... 64%, 81728 KB, 56003 KB/s, 1 seconds passed
    ... 64%, 81760 KB, 56014 KB/s, 1 seconds passed

.. parsed-literal::

    ... 64%, 81792 KB, 55397 KB/s, 1 seconds passed
    ... 64%, 81824 KB, 55258 KB/s, 1 seconds passed
    ... 64%, 81856 KB, 55258 KB/s, 1 seconds passed
    ... 65%, 81888 KB, 55264 KB/s, 1 seconds passed
    ... 65%, 81920 KB, 55270 KB/s, 1 seconds passed
    ... 65%, 81952 KB, 55277 KB/s, 1 seconds passed
    ... 65%, 81984 KB, 55284 KB/s, 1 seconds passed
    ... 65%, 82016 KB, 55291 KB/s, 1 seconds passed
    ... 65%, 82048 KB, 55298 KB/s, 1 seconds passed
    ... 65%, 82080 KB, 55305 KB/s, 1 seconds passed
    ... 65%, 82112 KB, 55312 KB/s, 1 seconds passed
    ... 65%, 82144 KB, 55319 KB/s, 1 seconds passed
    ... 65%, 82176 KB, 55325 KB/s, 1 seconds passed
    ... 65%, 82208 KB, 55333 KB/s, 1 seconds passed
    ... 65%, 82240 KB, 55340 KB/s, 1 seconds passed
    ... 65%, 82272 KB, 55348 KB/s, 1 seconds passed
    ... 65%, 82304 KB, 55355 KB/s, 1 seconds passed
    ... 65%, 82336 KB, 55362 KB/s, 1 seconds passed
    ... 65%, 82368 KB, 55370 KB/s, 1 seconds passed
    ... 65%, 82400 KB, 55377 KB/s, 1 seconds passed
    ... 65%, 82432 KB, 55385 KB/s, 1 seconds passed
    ... 65%, 82464 KB, 55393 KB/s, 1 seconds passed
    ... 65%, 82496 KB, 55399 KB/s, 1 seconds passed
    ... 65%, 82528 KB, 55406 KB/s, 1 seconds passed
    ... 65%, 82560 KB, 55415 KB/s, 1 seconds passed
    ... 65%, 82592 KB, 55422 KB/s, 1 seconds passed
    ... 65%, 82624 KB, 55429 KB/s, 1 seconds passed
    ... 65%, 82656 KB, 55437 KB/s, 1 seconds passed
    ... 65%, 82688 KB, 55445 KB/s, 1 seconds passed
    ... 65%, 82720 KB, 55453 KB/s, 1 seconds passed
    ... 65%, 82752 KB, 55461 KB/s, 1 seconds passed
    ... 65%, 82784 KB, 55467 KB/s, 1 seconds passed
    ... 65%, 82816 KB, 55475 KB/s, 1 seconds passed
    ... 65%, 82848 KB, 55482 KB/s, 1 seconds passed
    ... 65%, 82880 KB, 55490 KB/s, 1 seconds passed
    ... 65%, 82912 KB, 55499 KB/s, 1 seconds passed
    ... 65%, 82944 KB, 55511 KB/s, 1 seconds passed
    ... 65%, 82976 KB, 55522 KB/s, 1 seconds passed
    ... 65%, 83008 KB, 55535 KB/s, 1 seconds passed
    ... 65%, 83040 KB, 55547 KB/s, 1 seconds passed
    ... 65%, 83072 KB, 55560 KB/s, 1 seconds passed
    ... 65%, 83104 KB, 55573 KB/s, 1 seconds passed
    ... 66%, 83136 KB, 55585 KB/s, 1 seconds passed
    ... 66%, 83168 KB, 55597 KB/s, 1 seconds passed
    ... 66%, 83200 KB, 55610 KB/s, 1 seconds passed
    ... 66%, 83232 KB, 55622 KB/s, 1 seconds passed
    ... 66%, 83264 KB, 55635 KB/s, 1 seconds passed
    ... 66%, 83296 KB, 55647 KB/s, 1 seconds passed
    ... 66%, 83328 KB, 55659 KB/s, 1 seconds passed
    ... 66%, 83360 KB, 55671 KB/s, 1 seconds passed
    ... 66%, 83392 KB, 55684 KB/s, 1 seconds passed
    ... 66%, 83424 KB, 55696 KB/s, 1 seconds passed
    ... 66%, 83456 KB, 55709 KB/s, 1 seconds passed
    ... 66%, 83488 KB, 55720 KB/s, 1 seconds passed
    ... 66%, 83520 KB, 55733 KB/s, 1 seconds passed
    ... 66%, 83552 KB, 55745 KB/s, 1 seconds passed
    ... 66%, 83584 KB, 55757 KB/s, 1 seconds passed
    ... 66%, 83616 KB, 55769 KB/s, 1 seconds passed
    ... 66%, 83648 KB, 55781 KB/s, 1 seconds passed
    ... 66%, 83680 KB, 55794 KB/s, 1 seconds passed
    ... 66%, 83712 KB, 55806 KB/s, 1 seconds passed
    ... 66%, 83744 KB, 55819 KB/s, 1 seconds passed
    ... 66%, 83776 KB, 55831 KB/s, 1 seconds passed
    ... 66%, 83808 KB, 55844 KB/s, 1 seconds passed
    ... 66%, 83840 KB, 55855 KB/s, 1 seconds passed
    ... 66%, 83872 KB, 55868 KB/s, 1 seconds passed
    ... 66%, 83904 KB, 55880 KB/s, 1 seconds passed
    ... 66%, 83936 KB, 55893 KB/s, 1 seconds passed
    ... 66%, 83968 KB, 55905 KB/s, 1 seconds passed
    ... 66%, 84000 KB, 55918 KB/s, 1 seconds passed
    ... 66%, 84032 KB, 55930 KB/s, 1 seconds passed
    ... 66%, 84064 KB, 55943 KB/s, 1 seconds passed
    ... 66%, 84096 KB, 55955 KB/s, 1 seconds passed
    ... 66%, 84128 KB, 55968 KB/s, 1 seconds passed
    ... 66%, 84160 KB, 55980 KB/s, 1 seconds passed
    ... 66%, 84192 KB, 55992 KB/s, 1 seconds passed
    ... 66%, 84224 KB, 56005 KB/s, 1 seconds passed
    ... 66%, 84256 KB, 56020 KB/s, 1 seconds passed
    ... 66%, 84288 KB, 56035 KB/s, 1 seconds passed
    ... 66%, 84320 KB, 56050 KB/s, 1 seconds passed
    ... 66%, 84352 KB, 56064 KB/s, 1 seconds passed
    ... 66%, 84384 KB, 56079 KB/s, 1 seconds passed
    ... 67%, 84416 KB, 56094 KB/s, 1 seconds passed
    ... 67%, 84448 KB, 56109 KB/s, 1 seconds passed
    ... 67%, 84480 KB, 56124 KB/s, 1 seconds passed
    ... 67%, 84512 KB, 56139 KB/s, 1 seconds passed
    ... 67%, 84544 KB, 56155 KB/s, 1 seconds passed
    ... 67%, 84576 KB, 56170 KB/s, 1 seconds passed
    ... 67%, 84608 KB, 56185 KB/s, 1 seconds passed
    ... 67%, 84640 KB, 56200 KB/s, 1 seconds passed
    ... 67%, 84672 KB, 56215 KB/s, 1 seconds passed
    ... 67%, 84704 KB, 56230 KB/s, 1 seconds passed
    ... 67%, 84736 KB, 56245 KB/s, 1 seconds passed
    ... 67%, 84768 KB, 56260 KB/s, 1 seconds passed
    ... 67%, 84800 KB, 56275 KB/s, 1 seconds passed
    ... 67%, 84832 KB, 56290 KB/s, 1 seconds passed
    ... 67%, 84864 KB, 56305 KB/s, 1 seconds passed
    ... 67%, 84896 KB, 56315 KB/s, 1 seconds passed
    ... 67%, 84928 KB, 56321 KB/s, 1 seconds passed
    ... 67%, 84960 KB, 56327 KB/s, 1 seconds passed
    ... 67%, 84992 KB, 56341 KB/s, 1 seconds passed
    ... 67%, 85024 KB, 56356 KB/s, 1 seconds passed
    ... 67%, 85056 KB, 56370 KB/s, 1 seconds passed
    ... 67%, 85088 KB, 56382 KB/s, 1 seconds passed
    ... 67%, 85120 KB, 56393 KB/s, 1 seconds passed
    ... 67%, 85152 KB, 56406 KB/s, 1 seconds passed
    ... 67%, 85184 KB, 56410 KB/s, 1 seconds passed
    ... 67%, 85216 KB, 56416 KB/s, 1 seconds passed
    ... 67%, 85248 KB, 56429 KB/s, 1 seconds passed
    ... 67%, 85280 KB, 56444 KB/s, 1 seconds passed
    ... 67%, 85312 KB, 56457 KB/s, 1 seconds passed
    ... 67%, 85344 KB, 56468 KB/s, 1 seconds passed
    ... 67%, 85376 KB, 56475 KB/s, 1 seconds passed
    ... 67%, 85408 KB, 56490 KB/s, 1 seconds passed
    ... 67%, 85440 KB, 56502 KB/s, 1 seconds passed
    ... 67%, 85472 KB, 56513 KB/s, 1 seconds passed
    ... 67%, 85504 KB, 56522 KB/s, 1 seconds passed
    ... 67%, 85536 KB, 56532 KB/s, 1 seconds passed
    ... 67%, 85568 KB, 56543 KB/s, 1 seconds passed
    ... 67%, 85600 KB, 56554 KB/s, 1 seconds passed
    ... 67%, 85632 KB, 56566 KB/s, 1 seconds passed
    ... 68%, 85664 KB, 56575 KB/s, 1 seconds passed
    ... 68%, 85696 KB, 56588 KB/s, 1 seconds passed
    ... 68%, 85728 KB, 56599 KB/s, 1 seconds passed
    ... 68%, 85760 KB, 56607 KB/s, 1 seconds passed
    ... 68%, 85792 KB, 56614 KB/s, 1 seconds passed
    ... 68%, 85824 KB, 56630 KB/s, 1 seconds passed
    ... 68%, 85856 KB, 56642 KB/s, 1 seconds passed

.. parsed-literal::

    ... 68%, 85888 KB, 56651 KB/s, 1 seconds passed
    ... 68%, 85920 KB, 56663 KB/s, 1 seconds passed
    ... 68%, 85952 KB, 56675 KB/s, 1 seconds passed
    ... 68%, 85984 KB, 56684 KB/s, 1 seconds passed
    ... 68%, 86016 KB, 56692 KB/s, 1 seconds passed
    ... 68%, 86048 KB, 56703 KB/s, 1 seconds passed
    ... 68%, 86080 KB, 56714 KB/s, 1 seconds passed
    ... 68%, 86112 KB, 56724 KB/s, 1 seconds passed
    ... 68%, 86144 KB, 56733 KB/s, 1 seconds passed
    ... 68%, 86176 KB, 56744 KB/s, 1 seconds passed
    ... 68%, 86208 KB, 56756 KB/s, 1 seconds passed
    ... 68%, 86240 KB, 56767 KB/s, 1 seconds passed
    ... 68%, 86272 KB, 56776 KB/s, 1 seconds passed
    ... 68%, 86304 KB, 56787 KB/s, 1 seconds passed
    ... 68%, 86336 KB, 56799 KB/s, 1 seconds passed
    ... 68%, 86368 KB, 56808 KB/s, 1 seconds passed
    ... 68%, 86400 KB, 56819 KB/s, 1 seconds passed
    ... 68%, 86432 KB, 56831 KB/s, 1 seconds passed
    ... 68%, 86464 KB, 56840 KB/s, 1 seconds passed
    ... 68%, 86496 KB, 56851 KB/s, 1 seconds passed
    ... 68%, 86528 KB, 56862 KB/s, 1 seconds passed
    ... 68%, 86560 KB, 56872 KB/s, 1 seconds passed
    ... 68%, 86592 KB, 56883 KB/s, 1 seconds passed
    ... 68%, 86624 KB, 56894 KB/s, 1 seconds passed
    ... 68%, 86656 KB, 56905 KB/s, 1 seconds passed
    ... 68%, 86688 KB, 56917 KB/s, 1 seconds passed
    ... 68%, 86720 KB, 56926 KB/s, 1 seconds passed
    ... 68%, 86752 KB, 56937 KB/s, 1 seconds passed
    ... 68%, 86784 KB, 56946 KB/s, 1 seconds passed
    ... 68%, 86816 KB, 56957 KB/s, 1 seconds passed
    ... 68%, 86848 KB, 56967 KB/s, 1 seconds passed
    ... 68%, 86880 KB, 56978 KB/s, 1 seconds passed
    ... 69%, 86912 KB, 56989 KB/s, 1 seconds passed
    ... 69%, 86944 KB, 56998 KB/s, 1 seconds passed
    ... 69%, 86976 KB, 57009 KB/s, 1 seconds passed
    ... 69%, 87008 KB, 57021 KB/s, 1 seconds passed
    ... 69%, 87040 KB, 56633 KB/s, 1 seconds passed
    ... 69%, 87072 KB, 56638 KB/s, 1 seconds passed
    ... 69%, 87104 KB, 56644 KB/s, 1 seconds passed
    ... 69%, 87136 KB, 56651 KB/s, 1 seconds passed
    ... 69%, 87168 KB, 56658 KB/s, 1 seconds passed
    ... 69%, 87200 KB, 56666 KB/s, 1 seconds passed
    ... 69%, 87232 KB, 56674 KB/s, 1 seconds passed
    ... 69%, 87264 KB, 56681 KB/s, 1 seconds passed
    ... 69%, 87296 KB, 56689 KB/s, 1 seconds passed
    ... 69%, 87328 KB, 56696 KB/s, 1 seconds passed
    ... 69%, 87360 KB, 56702 KB/s, 1 seconds passed
    ... 69%, 87392 KB, 56708 KB/s, 1 seconds passed
    ... 69%, 87424 KB, 56715 KB/s, 1 seconds passed
    ... 69%, 87456 KB, 56723 KB/s, 1 seconds passed
    ... 69%, 87488 KB, 56730 KB/s, 1 seconds passed
    ... 69%, 87520 KB, 56737 KB/s, 1 seconds passed
    ... 69%, 87552 KB, 56744 KB/s, 1 seconds passed
    ... 69%, 87584 KB, 56752 KB/s, 1 seconds passed
    ... 69%, 87616 KB, 56759 KB/s, 1 seconds passed
    ... 69%, 87648 KB, 56766 KB/s, 1 seconds passed
    ... 69%, 87680 KB, 56774 KB/s, 1 seconds passed
    ... 69%, 87712 KB, 56781 KB/s, 1 seconds passed
    ... 69%, 87744 KB, 56789 KB/s, 1 seconds passed
    ... 69%, 87776 KB, 56797 KB/s, 1 seconds passed
    ... 69%, 87808 KB, 56805 KB/s, 1 seconds passed
    ... 69%, 87840 KB, 56813 KB/s, 1 seconds passed
    ... 69%, 87872 KB, 56820 KB/s, 1 seconds passed
    ... 69%, 87904 KB, 56829 KB/s, 1 seconds passed
    ... 69%, 87936 KB, 56837 KB/s, 1 seconds passed
    ... 69%, 87968 KB, 56845 KB/s, 1 seconds passed
    ... 69%, 88000 KB, 56853 KB/s, 1 seconds passed
    ... 69%, 88032 KB, 56863 KB/s, 1 seconds passed
    ... 69%, 88064 KB, 56873 KB/s, 1 seconds passed
    ... 69%, 88096 KB, 56884 KB/s, 1 seconds passed
    ... 69%, 88128 KB, 56895 KB/s, 1 seconds passed
    ... 69%, 88160 KB, 56903 KB/s, 1 seconds passed
    ... 70%, 88192 KB, 56911 KB/s, 1 seconds passed
    ... 70%, 88224 KB, 56920 KB/s, 1 seconds passed
    ... 70%, 88256 KB, 56929 KB/s, 1 seconds passed
    ... 70%, 88288 KB, 56937 KB/s, 1 seconds passed
    ... 70%, 88320 KB, 56945 KB/s, 1 seconds passed
    ... 70%, 88352 KB, 56954 KB/s, 1 seconds passed
    ... 70%, 88384 KB, 56962 KB/s, 1 seconds passed
    ... 70%, 88416 KB, 56971 KB/s, 1 seconds passed
    ... 70%, 88448 KB, 56979 KB/s, 1 seconds passed
    ... 70%, 88480 KB, 56987 KB/s, 1 seconds passed
    ... 70%, 88512 KB, 56994 KB/s, 1 seconds passed
    ... 70%, 88544 KB, 57003 KB/s, 1 seconds passed
    ... 70%, 88576 KB, 57010 KB/s, 1 seconds passed
    ... 70%, 88608 KB, 57018 KB/s, 1 seconds passed
    ... 70%, 88640 KB, 57026 KB/s, 1 seconds passed
    ... 70%, 88672 KB, 57034 KB/s, 1 seconds passed
    ... 70%, 88704 KB, 57045 KB/s, 1 seconds passed
    ... 70%, 88736 KB, 57057 KB/s, 1 seconds passed
    ... 70%, 88768 KB, 57070 KB/s, 1 seconds passed
    ... 70%, 88800 KB, 57083 KB/s, 1 seconds passed
    ... 70%, 88832 KB, 57096 KB/s, 1 seconds passed
    ... 70%, 88864 KB, 57110 KB/s, 1 seconds passed
    ... 70%, 88896 KB, 57124 KB/s, 1 seconds passed
    ... 70%, 88928 KB, 57138 KB/s, 1 seconds passed
    ... 70%, 88960 KB, 57152 KB/s, 1 seconds passed
    ... 70%, 88992 KB, 57166 KB/s, 1 seconds passed
    ... 70%, 89024 KB, 57180 KB/s, 1 seconds passed

.. parsed-literal::

    ... 70%, 89056 KB, 56075 KB/s, 1 seconds passed
    ... 70%, 89088 KB, 56076 KB/s, 1 seconds passed
    ... 70%, 89120 KB, 56081 KB/s, 1 seconds passed
    ... 70%, 89152 KB, 56090 KB/s, 1 seconds passed
    ... 70%, 89184 KB, 56088 KB/s, 1 seconds passed
    ... 70%, 89216 KB, 56091 KB/s, 1 seconds passed
    ... 70%, 89248 KB, 56098 KB/s, 1 seconds passed
    ... 70%, 89280 KB, 56105 KB/s, 1 seconds passed
    ... 70%, 89312 KB, 56112 KB/s, 1 seconds passed
    ... 70%, 89344 KB, 56119 KB/s, 1 seconds passed
    ... 70%, 89376 KB, 56125 KB/s, 1 seconds passed
    ... 70%, 89408 KB, 56132 KB/s, 1 seconds passed
    ... 71%, 89440 KB, 56138 KB/s, 1 seconds passed
    ... 71%, 89472 KB, 56145 KB/s, 1 seconds passed
    ... 71%, 89504 KB, 56151 KB/s, 1 seconds passed
    ... 71%, 89536 KB, 56158 KB/s, 1 seconds passed
    ... 71%, 89568 KB, 56165 KB/s, 1 seconds passed
    ... 71%, 89600 KB, 56172 KB/s, 1 seconds passed
    ... 71%, 89632 KB, 56178 KB/s, 1 seconds passed
    ... 71%, 89664 KB, 56184 KB/s, 1 seconds passed
    ... 71%, 89696 KB, 56191 KB/s, 1 seconds passed
    ... 71%, 89728 KB, 56199 KB/s, 1 seconds passed
    ... 71%, 89760 KB, 56208 KB/s, 1 seconds passed
    ... 71%, 89792 KB, 56217 KB/s, 1 seconds passed
    ... 71%, 89824 KB, 56226 KB/s, 1 seconds passed
    ... 71%, 89856 KB, 56235 KB/s, 1 seconds passed
    ... 71%, 89888 KB, 56245 KB/s, 1 seconds passed

.. parsed-literal::

    ... 71%, 89920 KB, 55471 KB/s, 1 seconds passed
    ... 71%, 89952 KB, 55457 KB/s, 1 seconds passed
    ... 71%, 89984 KB, 55443 KB/s, 1 seconds passed
    ... 71%, 90016 KB, 55442 KB/s, 1 seconds passed
    ... 71%, 90048 KB, 55447 KB/s, 1 seconds passed
    ... 71%, 90080 KB, 55453 KB/s, 1 seconds passed
    ... 71%, 90112 KB, 55457 KB/s, 1 seconds passed
    ... 71%, 90144 KB, 55459 KB/s, 1 seconds passed
    ... 71%, 90176 KB, 55465 KB/s, 1 seconds passed
    ... 71%, 90208 KB, 55471 KB/s, 1 seconds passed
    ... 71%, 90240 KB, 55478 KB/s, 1 seconds passed
    ... 71%, 90272 KB, 55484 KB/s, 1 seconds passed
    ... 71%, 90304 KB, 55492 KB/s, 1 seconds passed
    ... 71%, 90336 KB, 55498 KB/s, 1 seconds passed
    ... 71%, 90368 KB, 55506 KB/s, 1 seconds passed
    ... 71%, 90400 KB, 55516 KB/s, 1 seconds passed
    ... 71%, 90432 KB, 55525 KB/s, 1 seconds passed
    ... 71%, 90464 KB, 55534 KB/s, 1 seconds passed
    ... 71%, 90496 KB, 55543 KB/s, 1 seconds passed
    ... 71%, 90528 KB, 55553 KB/s, 1 seconds passed
    ... 71%, 90560 KB, 55562 KB/s, 1 seconds passed
    ... 71%, 90592 KB, 55572 KB/s, 1 seconds passed
    ... 71%, 90624 KB, 55582 KB/s, 1 seconds passed
    ... 71%, 90656 KB, 55592 KB/s, 1 seconds passed
    ... 72%, 90688 KB, 55602 KB/s, 1 seconds passed
    ... 72%, 90720 KB, 55613 KB/s, 1 seconds passed
    ... 72%, 90752 KB, 55150 KB/s, 1 seconds passed
    ... 72%, 90784 KB, 55152 KB/s, 1 seconds passed
    ... 72%, 90816 KB, 55156 KB/s, 1 seconds passed
    ... 72%, 90848 KB, 55162 KB/s, 1 seconds passed
    ... 72%, 90880 KB, 55168 KB/s, 1 seconds passed
    ... 72%, 90912 KB, 55174 KB/s, 1 seconds passed
    ... 72%, 90944 KB, 55181 KB/s, 1 seconds passed
    ... 72%, 90976 KB, 55188 KB/s, 1 seconds passed
    ... 72%, 91008 KB, 55194 KB/s, 1 seconds passed
    ... 72%, 91040 KB, 55199 KB/s, 1 seconds passed
    ... 72%, 91072 KB, 55207 KB/s, 1 seconds passed
    ... 72%, 91104 KB, 55213 KB/s, 1 seconds passed
    ... 72%, 91136 KB, 55220 KB/s, 1 seconds passed
    ... 72%, 91168 KB, 55227 KB/s, 1 seconds passed
    ... 72%, 91200 KB, 55234 KB/s, 1 seconds passed
    ... 72%, 91232 KB, 55241 KB/s, 1 seconds passed
    ... 72%, 91264 KB, 55248 KB/s, 1 seconds passed
    ... 72%, 91296 KB, 55254 KB/s, 1 seconds passed
    ... 72%, 91328 KB, 55261 KB/s, 1 seconds passed
    ... 72%, 91360 KB, 55267 KB/s, 1 seconds passed
    ... 72%, 91392 KB, 55273 KB/s, 1 seconds passed
    ... 72%, 91424 KB, 55280 KB/s, 1 seconds passed
    ... 72%, 91456 KB, 55287 KB/s, 1 seconds passed
    ... 72%, 91488 KB, 55294 KB/s, 1 seconds passed
    ... 72%, 91520 KB, 55300 KB/s, 1 seconds passed
    ... 72%, 91552 KB, 55306 KB/s, 1 seconds passed
    ... 72%, 91584 KB, 55313 KB/s, 1 seconds passed
    ... 72%, 91616 KB, 55322 KB/s, 1 seconds passed
    ... 72%, 91648 KB, 55333 KB/s, 1 seconds passed
    ... 72%, 91680 KB, 55344 KB/s, 1 seconds passed
    ... 72%, 91712 KB, 55355 KB/s, 1 seconds passed
    ... 72%, 91744 KB, 55365 KB/s, 1 seconds passed
    ... 72%, 91776 KB, 55375 KB/s, 1 seconds passed
    ... 72%, 91808 KB, 55386 KB/s, 1 seconds passed
    ... 72%, 91840 KB, 55397 KB/s, 1 seconds passed
    ... 72%, 91872 KB, 55408 KB/s, 1 seconds passed
    ... 72%, 91904 KB, 55419 KB/s, 1 seconds passed
    ... 72%, 91936 KB, 55430 KB/s, 1 seconds passed
    ... 73%, 91968 KB, 55440 KB/s, 1 seconds passed
    ... 73%, 92000 KB, 55451 KB/s, 1 seconds passed
    ... 73%, 92032 KB, 55462 KB/s, 1 seconds passed
    ... 73%, 92064 KB, 55473 KB/s, 1 seconds passed
    ... 73%, 92096 KB, 55484 KB/s, 1 seconds passed
    ... 73%, 92128 KB, 55495 KB/s, 1 seconds passed
    ... 73%, 92160 KB, 55505 KB/s, 1 seconds passed
    ... 73%, 92192 KB, 55516 KB/s, 1 seconds passed
    ... 73%, 92224 KB, 55527 KB/s, 1 seconds passed
    ... 73%, 92256 KB, 55537 KB/s, 1 seconds passed
    ... 73%, 92288 KB, 55548 KB/s, 1 seconds passed
    ... 73%, 92320 KB, 55559 KB/s, 1 seconds passed
    ... 73%, 92352 KB, 55570 KB/s, 1 seconds passed
    ... 73%, 92384 KB, 55581 KB/s, 1 seconds passed
    ... 73%, 92416 KB, 55591 KB/s, 1 seconds passed
    ... 73%, 92448 KB, 55602 KB/s, 1 seconds passed
    ... 73%, 92480 KB, 55613 KB/s, 1 seconds passed
    ... 73%, 92512 KB, 55624 KB/s, 1 seconds passed
    ... 73%, 92544 KB, 55635 KB/s, 1 seconds passed
    ... 73%, 92576 KB, 55647 KB/s, 1 seconds passed
    ... 73%, 92608 KB, 55657 KB/s, 1 seconds passed
    ... 73%, 92640 KB, 55668 KB/s, 1 seconds passed
    ... 73%, 92672 KB, 55679 KB/s, 1 seconds passed
    ... 73%, 92704 KB, 55690 KB/s, 1 seconds passed
    ... 73%, 92736 KB, 55701 KB/s, 1 seconds passed
    ... 73%, 92768 KB, 55710 KB/s, 1 seconds passed
    ... 73%, 92800 KB, 55720 KB/s, 1 seconds passed
    ... 73%, 92832 KB, 55731 KB/s, 1 seconds passed
    ... 73%, 92864 KB, 55742 KB/s, 1 seconds passed
    ... 73%, 92896 KB, 55755 KB/s, 1 seconds passed
    ... 73%, 92928 KB, 55768 KB/s, 1 seconds passed
    ... 73%, 92960 KB, 55781 KB/s, 1 seconds passed
    ... 73%, 92992 KB, 55794 KB/s, 1 seconds passed
    ... 73%, 93024 KB, 55808 KB/s, 1 seconds passed
    ... 73%, 93056 KB, 55821 KB/s, 1 seconds passed
    ... 73%, 93088 KB, 55835 KB/s, 1 seconds passed
    ... 73%, 93120 KB, 55847 KB/s, 1 seconds passed
    ... 73%, 93152 KB, 55861 KB/s, 1 seconds passed
    ... 73%, 93184 KB, 55874 KB/s, 1 seconds passed
    ... 74%, 93216 KB, 55887 KB/s, 1 seconds passed
    ... 74%, 93248 KB, 55901 KB/s, 1 seconds passed
    ... 74%, 93280 KB, 55914 KB/s, 1 seconds passed
    ... 74%, 93312 KB, 55928 KB/s, 1 seconds passed
    ... 74%, 93344 KB, 55940 KB/s, 1 seconds passed
    ... 74%, 93376 KB, 55954 KB/s, 1 seconds passed
    ... 74%, 93408 KB, 55967 KB/s, 1 seconds passed
    ... 74%, 93440 KB, 55980 KB/s, 1 seconds passed
    ... 74%, 93472 KB, 55993 KB/s, 1 seconds passed
    ... 74%, 93504 KB, 56007 KB/s, 1 seconds passed

.. parsed-literal::

    ... 74%, 93536 KB, 56017 KB/s, 1 seconds passed
    ... 74%, 93568 KB, 56026 KB/s, 1 seconds passed
    ... 74%, 93600 KB, 56036 KB/s, 1 seconds passed
    ... 74%, 93632 KB, 56046 KB/s, 1 seconds passed
    ... 74%, 93664 KB, 56055 KB/s, 1 seconds passed
    ... 74%, 93696 KB, 56065 KB/s, 1 seconds passed
    ... 74%, 93728 KB, 56075 KB/s, 1 seconds passed
    ... 74%, 93760 KB, 56086 KB/s, 1 seconds passed
    ... 74%, 93792 KB, 56096 KB/s, 1 seconds passed
    ... 74%, 93824 KB, 56105 KB/s, 1 seconds passed
    ... 74%, 93856 KB, 56115 KB/s, 1 seconds passed
    ... 74%, 93888 KB, 56123 KB/s, 1 seconds passed
    ... 74%, 93920 KB, 56134 KB/s, 1 seconds passed
    ... 74%, 93952 KB, 56142 KB/s, 1 seconds passed
    ... 74%, 93984 KB, 56152 KB/s, 1 seconds passed
    ... 74%, 94016 KB, 56163 KB/s, 1 seconds passed
    ... 74%, 94048 KB, 56173 KB/s, 1 seconds passed
    ... 74%, 94080 KB, 56181 KB/s, 1 seconds passed
    ... 74%, 94112 KB, 56194 KB/s, 1 seconds passed
    ... 74%, 94144 KB, 56202 KB/s, 1 seconds passed
    ... 74%, 94176 KB, 56212 KB/s, 1 seconds passed
    ... 74%, 94208 KB, 56223 KB/s, 1 seconds passed
    ... 74%, 94240 KB, 56231 KB/s, 1 seconds passed
    ... 74%, 94272 KB, 56241 KB/s, 1 seconds passed
    ... 74%, 94304 KB, 56251 KB/s, 1 seconds passed
    ... 74%, 94336 KB, 56262 KB/s, 1 seconds passed
    ... 74%, 94368 KB, 56272 KB/s, 1 seconds passed
    ... 74%, 94400 KB, 56282 KB/s, 1 seconds passed
    ... 74%, 94432 KB, 56290 KB/s, 1 seconds passed
    ... 74%, 94464 KB, 56301 KB/s, 1 seconds passed
    ... 75%, 94496 KB, 56309 KB/s, 1 seconds passed
    ... 75%, 94528 KB, 56320 KB/s, 1 seconds passed
    ... 75%, 94560 KB, 56330 KB/s, 1 seconds passed
    ... 75%, 94592 KB, 56340 KB/s, 1 seconds passed
    ... 75%, 94624 KB, 56348 KB/s, 1 seconds passed
    ... 75%, 94656 KB, 56359 KB/s, 1 seconds passed
    ... 75%, 94688 KB, 56366 KB/s, 1 seconds passed
    ... 75%, 94720 KB, 56376 KB/s, 1 seconds passed
    ... 75%, 94752 KB, 56386 KB/s, 1 seconds passed
    ... 75%, 94784 KB, 56395 KB/s, 1 seconds passed
    ... 75%, 94816 KB, 56404 KB/s, 1 seconds passed
    ... 75%, 94848 KB, 56415 KB/s, 1 seconds passed
    ... 75%, 94880 KB, 56424 KB/s, 1 seconds passed
    ... 75%, 94912 KB, 56434 KB/s, 1 seconds passed
    ... 75%, 94944 KB, 56444 KB/s, 1 seconds passed
    ... 75%, 94976 KB, 56455 KB/s, 1 seconds passed
    ... 75%, 95008 KB, 56465 KB/s, 1 seconds passed
    ... 75%, 95040 KB, 56473 KB/s, 1 seconds passed
    ... 75%, 95072 KB, 56483 KB/s, 1 seconds passed
    ... 75%, 95104 KB, 56495 KB/s, 1 seconds passed
    ... 75%, 95136 KB, 56506 KB/s, 1 seconds passed
    ... 75%, 95168 KB, 56514 KB/s, 1 seconds passed
    ... 75%, 95200 KB, 56524 KB/s, 1 seconds passed
    ... 75%, 95232 KB, 56534 KB/s, 1 seconds passed
    ... 75%, 95264 KB, 56545 KB/s, 1 seconds passed
    ... 75%, 95296 KB, 56553 KB/s, 1 seconds passed
    ... 75%, 95328 KB, 56563 KB/s, 1 seconds passed
    ... 75%, 95360 KB, 56573 KB/s, 1 seconds passed
    ... 75%, 95392 KB, 56584 KB/s, 1 seconds passed
    ... 75%, 95424 KB, 56592 KB/s, 1 seconds passed
    ... 75%, 95456 KB, 56600 KB/s, 1 seconds passed
    ... 75%, 95488 KB, 56610 KB/s, 1 seconds passed
    ... 75%, 95520 KB, 56620 KB/s, 1 seconds passed
    ... 75%, 95552 KB, 56629 KB/s, 1 seconds passed
    ... 75%, 95584 KB, 56639 KB/s, 1 seconds passed
    ... 75%, 95616 KB, 56649 KB/s, 1 seconds passed
    ... 75%, 95648 KB, 56657 KB/s, 1 seconds passed
    ... 75%, 95680 KB, 56667 KB/s, 1 seconds passed
    ... 75%, 95712 KB, 56678 KB/s, 1 seconds passed
    ... 76%, 95744 KB, 56688 KB/s, 1 seconds passed
    ... 76%, 95776 KB, 56698 KB/s, 1 seconds passed
    ... 76%, 95808 KB, 56706 KB/s, 1 seconds passed
    ... 76%, 95840 KB, 56716 KB/s, 1 seconds passed
    ... 76%, 95872 KB, 56727 KB/s, 1 seconds passed
    ... 76%, 95904 KB, 56735 KB/s, 1 seconds passed
    ... 76%, 95936 KB, 56745 KB/s, 1 seconds passed
    ... 76%, 95968 KB, 56750 KB/s, 1 seconds passed
    ... 76%, 96000 KB, 56755 KB/s, 1 seconds passed
    ... 76%, 96032 KB, 56760 KB/s, 1 seconds passed
    ... 76%, 96064 KB, 56774 KB/s, 1 seconds passed
    ... 76%, 96096 KB, 56788 KB/s, 1 seconds passed
    ... 76%, 96128 KB, 56801 KB/s, 1 seconds passed
    ... 76%, 96160 KB, 56810 KB/s, 1 seconds passed
    ... 76%, 96192 KB, 56820 KB/s, 1 seconds passed
    ... 76%, 96224 KB, 56830 KB/s, 1 seconds passed
    ... 76%, 96256 KB, 56840 KB/s, 1 seconds passed
    ... 76%, 96288 KB, 56848 KB/s, 1 seconds passed
    ... 76%, 96320 KB, 56858 KB/s, 1 seconds passed
    ... 76%, 96352 KB, 56869 KB/s, 1 seconds passed
    ... 76%, 96384 KB, 56877 KB/s, 1 seconds passed
    ... 76%, 96416 KB, 56887 KB/s, 1 seconds passed
    ... 76%, 96448 KB, 56897 KB/s, 1 seconds passed
    ... 76%, 96480 KB, 56907 KB/s, 1 seconds passed
    ... 76%, 96512 KB, 56915 KB/s, 1 seconds passed
    ... 76%, 96544 KB, 56925 KB/s, 1 seconds passed
    ... 76%, 96576 KB, 56935 KB/s, 1 seconds passed
    ... 76%, 96608 KB, 56945 KB/s, 1 seconds passed
    ... 76%, 96640 KB, 56954 KB/s, 1 seconds passed
    ... 76%, 96672 KB, 56963 KB/s, 1 seconds passed
    ... 76%, 96704 KB, 56972 KB/s, 1 seconds passed
    ... 76%, 96736 KB, 56982 KB/s, 1 seconds passed
    ... 76%, 96768 KB, 56992 KB/s, 1 seconds passed
    ... 76%, 96800 KB, 57002 KB/s, 1 seconds passed
    ... 76%, 96832 KB, 57010 KB/s, 1 seconds passed
    ... 76%, 96864 KB, 57020 KB/s, 1 seconds passed
    ... 76%, 96896 KB, 57029 KB/s, 1 seconds passed
    ... 76%, 96928 KB, 57039 KB/s, 1 seconds passed
    ... 76%, 96960 KB, 57048 KB/s, 1 seconds passed
    ... 77%, 96992 KB, 57058 KB/s, 1 seconds passed
    ... 77%, 97024 KB, 57067 KB/s, 1 seconds passed
    ... 77%, 97056 KB, 57077 KB/s, 1 seconds passed
    ... 77%, 97088 KB, 57087 KB/s, 1 seconds passed
    ... 77%, 97120 KB, 57094 KB/s, 1 seconds passed
    ... 77%, 97152 KB, 57105 KB/s, 1 seconds passed
    ... 77%, 97184 KB, 57115 KB/s, 1 seconds passed
    ... 77%, 97216 KB, 57123 KB/s, 1 seconds passed
    ... 77%, 97248 KB, 57132 KB/s, 1 seconds passed
    ... 77%, 97280 KB, 57144 KB/s, 1 seconds passed
    ... 77%, 97312 KB, 57152 KB/s, 1 seconds passed
    ... 77%, 97344 KB, 57160 KB/s, 1 seconds passed
    ... 77%, 97376 KB, 57171 KB/s, 1 seconds passed
    ... 77%, 97408 KB, 57179 KB/s, 1 seconds passed
    ... 77%, 97440 KB, 57189 KB/s, 1 seconds passed
    ... 77%, 97472 KB, 57201 KB/s, 1 seconds passed
    ... 77%, 97504 KB, 57209 KB/s, 1 seconds passed
    ... 77%, 97536 KB, 57219 KB/s, 1 seconds passed
    ... 77%, 97568 KB, 57229 KB/s, 1 seconds passed
    ... 77%, 97600 KB, 57239 KB/s, 1 seconds passed
    ... 77%, 97632 KB, 57247 KB/s, 1 seconds passed
    ... 77%, 97664 KB, 57254 KB/s, 1 seconds passed
    ... 77%, 97696 KB, 57262 KB/s, 1 seconds passed
    ... 77%, 97728 KB, 57272 KB/s, 1 seconds passed
    ... 77%, 97760 KB, 57284 KB/s, 1 seconds passed
    ... 77%, 97792 KB, 57295 KB/s, 1 seconds passed
    ... 77%, 97824 KB, 57301 KB/s, 1 seconds passed
    ... 77%, 97856 KB, 57310 KB/s, 1 seconds passed
    ... 77%, 97888 KB, 57319 KB/s, 1 seconds passed
    ... 77%, 97920 KB, 57329 KB/s, 1 seconds passed
    ... 77%, 97952 KB, 57338 KB/s, 1 seconds passed
    ... 77%, 97984 KB, 57346 KB/s, 1 seconds passed
    ... 77%, 98016 KB, 57357 KB/s, 1 seconds passed
    ... 77%, 98048 KB, 57367 KB/s, 1 seconds passed
    ... 77%, 98080 KB, 57375 KB/s, 1 seconds passed
    ... 77%, 98112 KB, 57385 KB/s, 1 seconds passed
    ... 77%, 98144 KB, 57395 KB/s, 1 seconds passed
    ... 77%, 98176 KB, 57403 KB/s, 1 seconds passed
    ... 77%, 98208 KB, 57415 KB/s, 1 seconds passed
    ... 77%, 98240 KB, 57423 KB/s, 1 seconds passed
    ... 78%, 98272 KB, 57433 KB/s, 1 seconds passed
    ... 78%, 98304 KB, 57443 KB/s, 1 seconds passed
    ... 78%, 98336 KB, 57451 KB/s, 1 seconds passed
    ... 78%, 98368 KB, 57459 KB/s, 1 seconds passed
    ... 78%, 98400 KB, 57469 KB/s, 1 seconds passed
    ... 78%, 98432 KB, 57481 KB/s, 1 seconds passed
    ... 78%, 98464 KB, 57490 KB/s, 1 seconds passed
    ... 78%, 98496 KB, 57500 KB/s, 1 seconds passed
    ... 78%, 98528 KB, 57508 KB/s, 1 seconds passed
    ... 78%, 98560 KB, 57518 KB/s, 1 seconds passed
    ... 78%, 98592 KB, 57526 KB/s, 1 seconds passed
    ... 78%, 98624 KB, 57536 KB/s, 1 seconds passed
    ... 78%, 98656 KB, 57546 KB/s, 1 seconds passed
    ... 78%, 98688 KB, 57554 KB/s, 1 seconds passed
    ... 78%, 98720 KB, 57562 KB/s, 1 seconds passed
    ... 78%, 98752 KB, 57572 KB/s, 1 seconds passed
    ... 78%, 98784 KB, 57582 KB/s, 1 seconds passed
    ... 78%, 98816 KB, 57593 KB/s, 1 seconds passed
    ... 78%, 98848 KB, 57601 KB/s, 1 seconds passed
    ... 78%, 98880 KB, 57609 KB/s, 1 seconds passed
    ... 78%, 98912 KB, 57618 KB/s, 1 seconds passed
    ... 78%, 98944 KB, 57631 KB/s, 1 seconds passed
    ... 78%, 98976 KB, 57639 KB/s, 1 seconds passed
    ... 78%, 99008 KB, 57649 KB/s, 1 seconds passed
    ... 78%, 99040 KB, 57657 KB/s, 1 seconds passed
    ... 78%, 99072 KB, 57665 KB/s, 1 seconds passed
    ... 78%, 99104 KB, 57672 KB/s, 1 seconds passed
    ... 78%, 99136 KB, 57677 KB/s, 1 seconds passed
    ... 78%, 99168 KB, 57683 KB/s, 1 seconds passed
    ... 78%, 99200 KB, 57690 KB/s, 1 seconds passed
    ... 78%, 99232 KB, 57703 KB/s, 1 seconds passed
    ... 78%, 99264 KB, 57716 KB/s, 1 seconds passed
    ... 78%, 99296 KB, 57728 KB/s, 1 seconds passed
    ... 78%, 99328 KB, 57741 KB/s, 1 seconds passed
    ... 78%, 99360 KB, 57751 KB/s, 1 seconds passed
    ... 78%, 99392 KB, 57760 KB/s, 1 seconds passed

.. parsed-literal::

    ... 78%, 99424 KB, 57768 KB/s, 1 seconds passed
    ... 78%, 99456 KB, 57776 KB/s, 1 seconds passed
    ... 78%, 99488 KB, 57788 KB/s, 1 seconds passed
    ... 79%, 99520 KB, 57796 KB/s, 1 seconds passed
    ... 79%, 99552 KB, 57806 KB/s, 1 seconds passed
    ... 79%, 99584 KB, 57816 KB/s, 1 seconds passed
    ... 79%, 99616 KB, 57824 KB/s, 1 seconds passed
    ... 79%, 99648 KB, 57833 KB/s, 1 seconds passed
    ... 79%, 99680 KB, 57843 KB/s, 1 seconds passed
    ... 79%, 99712 KB, 57848 KB/s, 1 seconds passed
    ... 79%, 99744 KB, 57857 KB/s, 1 seconds passed
    ... 79%, 99776 KB, 57869 KB/s, 1 seconds passed
    ... 79%, 99808 KB, 57879 KB/s, 1 seconds passed
    ... 79%, 99840 KB, 57888 KB/s, 1 seconds passed
    ... 79%, 99872 KB, 57898 KB/s, 1 seconds passed
    ... 79%, 99904 KB, 57906 KB/s, 1 seconds passed
    ... 79%, 99936 KB, 57914 KB/s, 1 seconds passed
    ... 79%, 99968 KB, 57924 KB/s, 1 seconds passed
    ... 79%, 100000 KB, 57930 KB/s, 1 seconds passed
    ... 79%, 100032 KB, 57940 KB/s, 1 seconds passed
    ... 79%, 100064 KB, 57950 KB/s, 1 seconds passed
    ... 79%, 100096 KB, 57961 KB/s, 1 seconds passed
    ... 79%, 100128 KB, 57971 KB/s, 1 seconds passed
    ... 79%, 100160 KB, 57977 KB/s, 1 seconds passed
    ... 79%, 100192 KB, 57987 KB/s, 1 seconds passed
    ... 79%, 100224 KB, 57995 KB/s, 1 seconds passed
    ... 79%, 100256 KB, 58005 KB/s, 1 seconds passed
    ... 79%, 100288 KB, 58014 KB/s, 1 seconds passed
    ... 79%, 100320 KB, 58024 KB/s, 1 seconds passed
    ... 79%, 100352 KB, 58034 KB/s, 1 seconds passed
    ... 79%, 100384 KB, 58042 KB/s, 1 seconds passed
    ... 79%, 100416 KB, 58050 KB/s, 1 seconds passed
    ... 79%, 100448 KB, 58059 KB/s, 1 seconds passed
    ... 79%, 100480 KB, 58073 KB/s, 1 seconds passed
    ... 79%, 100512 KB, 58079 KB/s, 1 seconds passed
    ... 79%, 100544 KB, 58087 KB/s, 1 seconds passed
    ... 79%, 100576 KB, 58097 KB/s, 1 seconds passed
    ... 79%, 100608 KB, 58106 KB/s, 1 seconds passed
    ... 79%, 100640 KB, 58114 KB/s, 1 seconds passed
    ... 79%, 100672 KB, 58124 KB/s, 1 seconds passed
    ... 79%, 100704 KB, 58132 KB/s, 1 seconds passed
    ... 79%, 100736 KB, 58143 KB/s, 1 seconds passed
    ... 80%, 100768 KB, 58151 KB/s, 1 seconds passed
    ... 80%, 100800 KB, 58161 KB/s, 1 seconds passed
    ... 80%, 100832 KB, 58171 KB/s, 1 seconds passed
    ... 80%, 100864 KB, 58180 KB/s, 1 seconds passed
    ... 80%, 100896 KB, 58188 KB/s, 1 seconds passed
    ... 80%, 100928 KB, 58200 KB/s, 1 seconds passed
    ... 80%, 100960 KB, 58208 KB/s, 1 seconds passed
    ... 80%, 100992 KB, 58215 KB/s, 1 seconds passed
    ... 80%, 101024 KB, 58224 KB/s, 1 seconds passed
    ... 80%, 101056 KB, 58236 KB/s, 1 seconds passed
    ... 80%, 101088 KB, 58244 KB/s, 1 seconds passed
    ... 80%, 101120 KB, 58253 KB/s, 1 seconds passed
    ... 80%, 101152 KB, 58262 KB/s, 1 seconds passed
    ... 80%, 101184 KB, 58267 KB/s, 1 seconds passed
    ... 80%, 101216 KB, 58271 KB/s, 1 seconds passed
    ... 80%, 101248 KB, 58276 KB/s, 1 seconds passed
    ... 80%, 101280 KB, 58284 KB/s, 1 seconds passed
    ... 80%, 101312 KB, 58298 KB/s, 1 seconds passed
    ... 80%, 101344 KB, 58312 KB/s, 1 seconds passed
    ... 80%, 101376 KB, 58325 KB/s, 1 seconds passed
    ... 80%, 101408 KB, 58333 KB/s, 1 seconds passed
    ... 80%, 101440 KB, 58343 KB/s, 1 seconds passed
    ... 80%, 101472 KB, 58351 KB/s, 1 seconds passed
    ... 80%, 101504 KB, 58360 KB/s, 1 seconds passed
    ... 80%, 101536 KB, 58370 KB/s, 1 seconds passed
    ... 80%, 101568 KB, 58378 KB/s, 1 seconds passed
    ... 80%, 101600 KB, 58387 KB/s, 1 seconds passed
    ... 80%, 101632 KB, 58397 KB/s, 1 seconds passed
    ... 80%, 101664 KB, 58405 KB/s, 1 seconds passed
    ... 80%, 101696 KB, 58414 KB/s, 1 seconds passed
    ... 80%, 101728 KB, 58424 KB/s, 1 seconds passed
    ... 80%, 101760 KB, 58434 KB/s, 1 seconds passed
    ... 80%, 101792 KB, 58442 KB/s, 1 seconds passed
    ... 80%, 101824 KB, 58451 KB/s, 1 seconds passed
    ... 80%, 101856 KB, 58459 KB/s, 1 seconds passed
    ... 80%, 101888 KB, 58469 KB/s, 1 seconds passed
    ... 80%, 101920 KB, 58478 KB/s, 1 seconds passed
    ... 80%, 101952 KB, 58488 KB/s, 1 seconds passed
    ... 80%, 101984 KB, 58496 KB/s, 1 seconds passed
    ... 80%, 102016 KB, 58505 KB/s, 1 seconds passed
    ... 81%, 102048 KB, 58513 KB/s, 1 seconds passed
    ... 81%, 102080 KB, 58523 KB/s, 1 seconds passed
    ... 81%, 102112 KB, 58532 KB/s, 1 seconds passed
    ... 81%, 102144 KB, 58540 KB/s, 1 seconds passed
    ... 81%, 102176 KB, 58550 KB/s, 1 seconds passed
    ... 81%, 102208 KB, 58559 KB/s, 1 seconds passed
    ... 81%, 102240 KB, 58567 KB/s, 1 seconds passed
    ... 81%, 102272 KB, 58577 KB/s, 1 seconds passed
    ... 81%, 102304 KB, 58585 KB/s, 1 seconds passed
    ... 81%, 102336 KB, 58594 KB/s, 1 seconds passed
    ... 81%, 102368 KB, 58602 KB/s, 1 seconds passed
    ... 81%, 102400 KB, 58081 KB/s, 1 seconds passed
    ... 81%, 102432 KB, 58083 KB/s, 1 seconds passed
    ... 81%, 102464 KB, 58089 KB/s, 1 seconds passed
    ... 81%, 102496 KB, 58095 KB/s, 1 seconds passed
    ... 81%, 102528 KB, 58101 KB/s, 1 seconds passed
    ... 81%, 102560 KB, 58106 KB/s, 1 seconds passed
    ... 81%, 102592 KB, 58112 KB/s, 1 seconds passed
    ... 81%, 102624 KB, 58118 KB/s, 1 seconds passed
    ... 81%, 102656 KB, 58125 KB/s, 1 seconds passed
    ... 81%, 102688 KB, 58131 KB/s, 1 seconds passed
    ... 81%, 102720 KB, 58137 KB/s, 1 seconds passed
    ... 81%, 102752 KB, 58144 KB/s, 1 seconds passed
    ... 81%, 102784 KB, 58150 KB/s, 1 seconds passed
    ... 81%, 102816 KB, 58156 KB/s, 1 seconds passed
    ... 81%, 102848 KB, 58162 KB/s, 1 seconds passed
    ... 81%, 102880 KB, 58168 KB/s, 1 seconds passed
    ... 81%, 102912 KB, 58173 KB/s, 1 seconds passed
    ... 81%, 102944 KB, 58180 KB/s, 1 seconds passed
    ... 81%, 102976 KB, 58185 KB/s, 1 seconds passed
    ... 81%, 103008 KB, 58191 KB/s, 1 seconds passed
    ... 81%, 103040 KB, 58196 KB/s, 1 seconds passed
    ... 81%, 103072 KB, 58203 KB/s, 1 seconds passed
    ... 81%, 103104 KB, 58208 KB/s, 1 seconds passed
    ... 81%, 103136 KB, 58214 KB/s, 1 seconds passed
    ... 81%, 103168 KB, 58221 KB/s, 1 seconds passed

.. parsed-literal::

    ... 81%, 103200 KB, 58227 KB/s, 1 seconds passed
    ... 81%, 103232 KB, 58233 KB/s, 1 seconds passed
    ... 81%, 103264 KB, 58239 KB/s, 1 seconds passed
    ... 82%, 103296 KB, 58245 KB/s, 1 seconds passed
    ... 82%, 103328 KB, 58251 KB/s, 1 seconds passed
    ... 82%, 103360 KB, 58256 KB/s, 1 seconds passed
    ... 82%, 103392 KB, 58262 KB/s, 1 seconds passed
    ... 82%, 103424 KB, 58268 KB/s, 1 seconds passed
    ... 82%, 103456 KB, 58275 KB/s, 1 seconds passed
    ... 82%, 103488 KB, 58280 KB/s, 1 seconds passed
    ... 82%, 103520 KB, 58286 KB/s, 1 seconds passed
    ... 82%, 103552 KB, 58293 KB/s, 1 seconds passed
    ... 82%, 103584 KB, 58298 KB/s, 1 seconds passed
    ... 82%, 103616 KB, 58307 KB/s, 1 seconds passed
    ... 82%, 103648 KB, 58317 KB/s, 1 seconds passed
    ... 82%, 103680 KB, 58327 KB/s, 1 seconds passed
    ... 82%, 103712 KB, 58338 KB/s, 1 seconds passed
    ... 82%, 103744 KB, 58348 KB/s, 1 seconds passed
    ... 82%, 103776 KB, 58358 KB/s, 1 seconds passed
    ... 82%, 103808 KB, 58369 KB/s, 1 seconds passed
    ... 82%, 103840 KB, 58379 KB/s, 1 seconds passed
    ... 82%, 103872 KB, 58389 KB/s, 1 seconds passed
    ... 82%, 103904 KB, 58399 KB/s, 1 seconds passed
    ... 82%, 103936 KB, 58410 KB/s, 1 seconds passed
    ... 82%, 103968 KB, 58419 KB/s, 1 seconds passed
    ... 82%, 104000 KB, 58429 KB/s, 1 seconds passed
    ... 82%, 104032 KB, 58440 KB/s, 1 seconds passed
    ... 82%, 104064 KB, 58450 KB/s, 1 seconds passed
    ... 82%, 104096 KB, 58460 KB/s, 1 seconds passed
    ... 82%, 104128 KB, 58470 KB/s, 1 seconds passed
    ... 82%, 104160 KB, 58481 KB/s, 1 seconds passed
    ... 82%, 104192 KB, 58491 KB/s, 1 seconds passed
    ... 82%, 104224 KB, 58502 KB/s, 1 seconds passed
    ... 82%, 104256 KB, 58512 KB/s, 1 seconds passed
    ... 82%, 104288 KB, 58522 KB/s, 1 seconds passed
    ... 82%, 104320 KB, 58532 KB/s, 1 seconds passed
    ... 82%, 104352 KB, 58543 KB/s, 1 seconds passed
    ... 82%, 104384 KB, 58554 KB/s, 1 seconds passed
    ... 82%, 104416 KB, 58564 KB/s, 1 seconds passed
    ... 82%, 104448 KB, 58574 KB/s, 1 seconds passed
    ... 82%, 104480 KB, 58584 KB/s, 1 seconds passed
    ... 82%, 104512 KB, 58595 KB/s, 1 seconds passed
    ... 83%, 104544 KB, 58605 KB/s, 1 seconds passed
    ... 83%, 104576 KB, 58615 KB/s, 1 seconds passed
    ... 83%, 104608 KB, 58626 KB/s, 1 seconds passed
    ... 83%, 104640 KB, 58637 KB/s, 1 seconds passed
    ... 83%, 104672 KB, 58647 KB/s, 1 seconds passed
    ... 83%, 104704 KB, 58657 KB/s, 1 seconds passed
    ... 83%, 104736 KB, 58667 KB/s, 1 seconds passed
    ... 83%, 104768 KB, 58677 KB/s, 1 seconds passed
    ... 83%, 104800 KB, 58687 KB/s, 1 seconds passed
    ... 83%, 104832 KB, 58698 KB/s, 1 seconds passed
    ... 83%, 104864 KB, 58708 KB/s, 1 seconds passed
    ... 83%, 104896 KB, 58718 KB/s, 1 seconds passed
    ... 83%, 104928 KB, 58728 KB/s, 1 seconds passed
    ... 83%, 104960 KB, 58739 KB/s, 1 seconds passed
    ... 83%, 104992 KB, 58751 KB/s, 1 seconds passed
    ... 83%, 105024 KB, 58763 KB/s, 1 seconds passed
    ... 83%, 105056 KB, 58776 KB/s, 1 seconds passed
    ... 83%, 105088 KB, 58789 KB/s, 1 seconds passed
    ... 83%, 105120 KB, 58802 KB/s, 1 seconds passed
    ... 83%, 105152 KB, 58814 KB/s, 1 seconds passed
    ... 83%, 105184 KB, 58827 KB/s, 1 seconds passed
    ... 83%, 105216 KB, 58840 KB/s, 1 seconds passed
    ... 83%, 105248 KB, 58852 KB/s, 1 seconds passed
    ... 83%, 105280 KB, 58865 KB/s, 1 seconds passed
    ... 83%, 105312 KB, 58878 KB/s, 1 seconds passed
    ... 83%, 105344 KB, 58891 KB/s, 1 seconds passed
    ... 83%, 105376 KB, 58903 KB/s, 1 seconds passed
    ... 83%, 105408 KB, 58913 KB/s, 1 seconds passed
    ... 83%, 105440 KB, 58922 KB/s, 1 seconds passed
    ... 83%, 105472 KB, 58931 KB/s, 1 seconds passed
    ... 83%, 105504 KB, 58941 KB/s, 1 seconds passed
    ... 83%, 105536 KB, 58950 KB/s, 1 seconds passed
    ... 83%, 105568 KB, 58959 KB/s, 1 seconds passed
    ... 83%, 105600 KB, 58968 KB/s, 1 seconds passed
    ... 83%, 105632 KB, 58978 KB/s, 1 seconds passed
    ... 83%, 105664 KB, 58986 KB/s, 1 seconds passed
    ... 83%, 105696 KB, 58994 KB/s, 1 seconds passed
    ... 83%, 105728 KB, 59003 KB/s, 1 seconds passed
    ... 83%, 105760 KB, 59011 KB/s, 1 seconds passed
    ... 83%, 105792 KB, 59019 KB/s, 1 seconds passed
    ... 84%, 105824 KB, 59028 KB/s, 1 seconds passed
    ... 84%, 105856 KB, 59037 KB/s, 1 seconds passed
    ... 84%, 105888 KB, 59044 KB/s, 1 seconds passed
    ... 84%, 105920 KB, 59054 KB/s, 1 seconds passed
    ... 84%, 105952 KB, 59065 KB/s, 1 seconds passed
    ... 84%, 105984 KB, 59072 KB/s, 1 seconds passed
    ... 84%, 106016 KB, 59082 KB/s, 1 seconds passed
    ... 84%, 106048 KB, 59091 KB/s, 1 seconds passed
    ... 84%, 106080 KB, 59098 KB/s, 1 seconds passed
    ... 84%, 106112 KB, 59106 KB/s, 1 seconds passed
    ... 84%, 106144 KB, 59115 KB/s, 1 seconds passed
    ... 84%, 106176 KB, 59122 KB/s, 1 seconds passed
    ... 84%, 106208 KB, 59131 KB/s, 1 seconds passed
    ... 84%, 106240 KB, 59138 KB/s, 1 seconds passed
    ... 84%, 106272 KB, 59143 KB/s, 1 seconds passed
    ... 84%, 106304 KB, 59156 KB/s, 1 seconds passed
    ... 84%, 106336 KB, 59036 KB/s, 1 seconds passed
    ... 84%, 106368 KB, 59041 KB/s, 1 seconds passed
    ... 84%, 106400 KB, 59049 KB/s, 1 seconds passed
    ... 84%, 106432 KB, 59062 KB/s, 1 seconds passed
    ... 84%, 106464 KB, 59074 KB/s, 1 seconds passed
    ... 84%, 106496 KB, 59081 KB/s, 1 seconds passed
    ... 84%, 106528 KB, 59089 KB/s, 1 seconds passed
    ... 84%, 106560 KB, 59094 KB/s, 1 seconds passed
    ... 84%, 106592 KB, 59031 KB/s, 1 seconds passed
    ... 84%, 106624 KB, 59035 KB/s, 1 seconds passed
    ... 84%, 106656 KB, 59042 KB/s, 1 seconds passed
    ... 84%, 106688 KB, 59055 KB/s, 1 seconds passed
    ... 84%, 106720 KB, 59067 KB/s, 1 seconds passed
    ... 84%, 106752 KB, 59076 KB/s, 1 seconds passed
    ... 84%, 106784 KB, 59085 KB/s, 1 seconds passed
    ... 84%, 106816 KB, 59092 KB/s, 1 seconds passed
    ... 84%, 106848 KB, 59101 KB/s, 1 seconds passed
    ... 84%, 106880 KB, 59110 KB/s, 1 seconds passed
    ... 84%, 106912 KB, 59120 KB/s, 1 seconds passed
    ... 84%, 106944 KB, 59129 KB/s, 1 seconds passed
    ... 84%, 106976 KB, 59136 KB/s, 1 seconds passed
    ... 84%, 107008 KB, 59145 KB/s, 1 seconds passed
    ... 84%, 107040 KB, 59154 KB/s, 1 seconds passed
    ... 85%, 107072 KB, 59162 KB/s, 1 seconds passed
    ... 85%, 107104 KB, 59171 KB/s, 1 seconds passed
    ... 85%, 107136 KB, 59178 KB/s, 1 seconds passed
    ... 85%, 107168 KB, 59187 KB/s, 1 seconds passed
    ... 85%, 107200 KB, 59196 KB/s, 1 seconds passed
    ... 85%, 107232 KB, 59204 KB/s, 1 seconds passed
    ... 85%, 107264 KB, 59213 KB/s, 1 seconds passed
    ... 85%, 107296 KB, 59222 KB/s, 1 seconds passed
    ... 85%, 107328 KB, 59229 KB/s, 1 seconds passed
    ... 85%, 107360 KB, 59238 KB/s, 1 seconds passed
    ... 85%, 107392 KB, 59247 KB/s, 1 seconds passed
    ... 85%, 107424 KB, 59255 KB/s, 1 seconds passed
    ... 85%, 107456 KB, 59264 KB/s, 1 seconds passed
    ... 85%, 107488 KB, 59271 KB/s, 1 seconds passed

.. parsed-literal::

    ... 85%, 107520 KB, 57976 KB/s, 1 seconds passed
    ... 85%, 107552 KB, 57978 KB/s, 1 seconds passed
    ... 85%, 107584 KB, 57984 KB/s, 1 seconds passed
    ... 85%, 107616 KB, 57989 KB/s, 1 seconds passed
    ... 85%, 107648 KB, 57995 KB/s, 1 seconds passed
    ... 85%, 107680 KB, 58001 KB/s, 1 seconds passed
    ... 85%, 107712 KB, 58006 KB/s, 1 seconds passed
    ... 85%, 107744 KB, 58011 KB/s, 1 seconds passed
    ... 85%, 107776 KB, 58017 KB/s, 1 seconds passed
    ... 85%, 107808 KB, 58022 KB/s, 1 seconds passed
    ... 85%, 107840 KB, 58028 KB/s, 1 seconds passed
    ... 85%, 107872 KB, 58033 KB/s, 1 seconds passed
    ... 85%, 107904 KB, 58040 KB/s, 1 seconds passed
    ... 85%, 107936 KB, 58045 KB/s, 1 seconds passed
    ... 85%, 107968 KB, 58051 KB/s, 1 seconds passed
    ... 85%, 108000 KB, 58057 KB/s, 1 seconds passed
    ... 85%, 108032 KB, 58063 KB/s, 1 seconds passed
    ... 85%, 108064 KB, 58070 KB/s, 1 seconds passed
    ... 85%, 108096 KB, 58076 KB/s, 1 seconds passed
    ... 85%, 108128 KB, 58081 KB/s, 1 seconds passed
    ... 85%, 108160 KB, 58087 KB/s, 1 seconds passed
    ... 85%, 108192 KB, 58092 KB/s, 1 seconds passed
    ... 85%, 108224 KB, 58098 KB/s, 1 seconds passed
    ... 85%, 108256 KB, 58103 KB/s, 1 seconds passed
    ... 85%, 108288 KB, 58109 KB/s, 1 seconds passed
    ... 86%, 108320 KB, 58115 KB/s, 1 seconds passed
    ... 86%, 108352 KB, 58121 KB/s, 1 seconds passed
    ... 86%, 108384 KB, 58127 KB/s, 1 seconds passed
    ... 86%, 108416 KB, 58132 KB/s, 1 seconds passed
    ... 86%, 108448 KB, 58138 KB/s, 1 seconds passed
    ... 86%, 108480 KB, 58144 KB/s, 1 seconds passed
    ... 86%, 108512 KB, 58151 KB/s, 1 seconds passed
    ... 86%, 108544 KB, 58158 KB/s, 1 seconds passed
    ... 86%, 108576 KB, 58164 KB/s, 1 seconds passed
    ... 86%, 108608 KB, 58171 KB/s, 1 seconds passed
    ... 86%, 108640 KB, 58178 KB/s, 1 seconds passed
    ... 86%, 108672 KB, 58185 KB/s, 1 seconds passed
    ... 86%, 108704 KB, 58191 KB/s, 1 seconds passed
    ... 86%, 108736 KB, 58198 KB/s, 1 seconds passed
    ... 86%, 108768 KB, 58208 KB/s, 1 seconds passed
    ... 86%, 108800 KB, 58218 KB/s, 1 seconds passed
    ... 86%, 108832 KB, 58228 KB/s, 1 seconds passed
    ... 86%, 108864 KB, 58238 KB/s, 1 seconds passed
    ... 86%, 108896 KB, 58249 KB/s, 1 seconds passed
    ... 86%, 108928 KB, 58259 KB/s, 1 seconds passed
    ... 86%, 108960 KB, 58269 KB/s, 1 seconds passed
    ... 86%, 108992 KB, 58279 KB/s, 1 seconds passed
    ... 86%, 109024 KB, 58289 KB/s, 1 seconds passed
    ... 86%, 109056 KB, 58299 KB/s, 1 seconds passed
    ... 86%, 109088 KB, 58309 KB/s, 1 seconds passed
    ... 86%, 109120 KB, 58319 KB/s, 1 seconds passed
    ... 86%, 109152 KB, 58330 KB/s, 1 seconds passed
    ... 86%, 109184 KB, 58340 KB/s, 1 seconds passed
    ... 86%, 109216 KB, 58351 KB/s, 1 seconds passed
    ... 86%, 109248 KB, 58362 KB/s, 1 seconds passed
    ... 86%, 109280 KB, 58373 KB/s, 1 seconds passed
    ... 86%, 109312 KB, 58384 KB/s, 1 seconds passed
    ... 86%, 109344 KB, 58395 KB/s, 1 seconds passed
    ... 86%, 109376 KB, 58407 KB/s, 1 seconds passed

.. parsed-literal::

    ... 86%, 109408 KB, 57427 KB/s, 1 seconds passed
    ... 86%, 109440 KB, 57429 KB/s, 1 seconds passed
    ... 86%, 109472 KB, 57433 KB/s, 1 seconds passed
    ... 86%, 109504 KB, 57437 KB/s, 1 seconds passed
    ... 86%, 109536 KB, 57442 KB/s, 1 seconds passed
    ... 86%, 109568 KB, 57448 KB/s, 1 seconds passed
    ... 87%, 109600 KB, 57453 KB/s, 1 seconds passed
    ... 87%, 109632 KB, 57459 KB/s, 1 seconds passed
    ... 87%, 109664 KB, 57464 KB/s, 1 seconds passed
    ... 87%, 109696 KB, 57469 KB/s, 1 seconds passed
    ... 87%, 109728 KB, 57474 KB/s, 1 seconds passed
    ... 87%, 109760 KB, 57480 KB/s, 1 seconds passed
    ... 87%, 109792 KB, 57485 KB/s, 1 seconds passed
    ... 87%, 109824 KB, 57490 KB/s, 1 seconds passed
    ... 87%, 109856 KB, 57496 KB/s, 1 seconds passed
    ... 87%, 109888 KB, 57501 KB/s, 1 seconds passed
    ... 87%, 109920 KB, 57507 KB/s, 1 seconds passed
    ... 87%, 109952 KB, 57513 KB/s, 1 seconds passed
    ... 87%, 109984 KB, 57521 KB/s, 1 seconds passed
    ... 87%, 110016 KB, 57529 KB/s, 1 seconds passed
    ... 87%, 110048 KB, 57537 KB/s, 1 seconds passed
    ... 87%, 110080 KB, 57546 KB/s, 1 seconds passed
    ... 87%, 110112 KB, 57553 KB/s, 1 seconds passed
    ... 87%, 110144 KB, 57561 KB/s, 1 seconds passed

.. parsed-literal::

    ... 87%, 110176 KB, 57037 KB/s, 1 seconds passed
    ... 87%, 110208 KB, 57039 KB/s, 1 seconds passed
    ... 87%, 110240 KB, 57044 KB/s, 1 seconds passed
    ... 87%, 110272 KB, 57049 KB/s, 1 seconds passed
    ... 87%, 110304 KB, 57053 KB/s, 1 seconds passed
    ... 87%, 110336 KB, 57058 KB/s, 1 seconds passed
    ... 87%, 110368 KB, 57064 KB/s, 1 seconds passed
    ... 87%, 110400 KB, 57070 KB/s, 1 seconds passed
    ... 87%, 110432 KB, 57075 KB/s, 1 seconds passed
    ... 87%, 110464 KB, 57081 KB/s, 1 seconds passed
    ... 87%, 110496 KB, 57087 KB/s, 1 seconds passed
    ... 87%, 110528 KB, 57092 KB/s, 1 seconds passed
    ... 87%, 110560 KB, 57097 KB/s, 1 seconds passed
    ... 87%, 110592 KB, 57102 KB/s, 1 seconds passed
    ... 87%, 110624 KB, 57108 KB/s, 1 seconds passed
    ... 87%, 110656 KB, 57113 KB/s, 1 seconds passed
    ... 87%, 110688 KB, 57118 KB/s, 1 seconds passed
    ... 87%, 110720 KB, 57124 KB/s, 1 seconds passed
    ... 87%, 110752 KB, 57129 KB/s, 1 seconds passed
    ... 87%, 110784 KB, 57135 KB/s, 1 seconds passed
    ... 87%, 110816 KB, 57140 KB/s, 1 seconds passed
    ... 88%, 110848 KB, 57147 KB/s, 1 seconds passed
    ... 88%, 110880 KB, 57153 KB/s, 1 seconds passed
    ... 88%, 110912 KB, 57159 KB/s, 1 seconds passed
    ... 88%, 110944 KB, 57165 KB/s, 1 seconds passed
    ... 88%, 110976 KB, 57172 KB/s, 1 seconds passed
    ... 88%, 111008 KB, 57178 KB/s, 1 seconds passed
    ... 88%, 111040 KB, 57184 KB/s, 1 seconds passed
    ... 88%, 111072 KB, 57191 KB/s, 1 seconds passed
    ... 88%, 111104 KB, 57197 KB/s, 1 seconds passed
    ... 88%, 111136 KB, 57204 KB/s, 1 seconds passed
    ... 88%, 111168 KB, 57210 KB/s, 1 seconds passed
    ... 88%, 111200 KB, 57217 KB/s, 1 seconds passed
    ... 88%, 111232 KB, 57222 KB/s, 1 seconds passed
    ... 88%, 111264 KB, 57229 KB/s, 1 seconds passed
    ... 88%, 111296 KB, 57235 KB/s, 1 seconds passed
    ... 88%, 111328 KB, 57241 KB/s, 1 seconds passed
    ... 88%, 111360 KB, 57247 KB/s, 1 seconds passed
    ... 88%, 111392 KB, 57252 KB/s, 1 seconds passed
    ... 88%, 111424 KB, 57259 KB/s, 1 seconds passed
    ... 88%, 111456 KB, 57265 KB/s, 1 seconds passed
    ... 88%, 111488 KB, 57271 KB/s, 1 seconds passed
    ... 88%, 111520 KB, 57277 KB/s, 1 seconds passed
    ... 88%, 111552 KB, 57284 KB/s, 1 seconds passed
    ... 88%, 111584 KB, 57290 KB/s, 1 seconds passed
    ... 88%, 111616 KB, 57297 KB/s, 1 seconds passed
    ... 88%, 111648 KB, 57303 KB/s, 1 seconds passed
    ... 88%, 111680 KB, 57309 KB/s, 1 seconds passed
    ... 88%, 111712 KB, 57316 KB/s, 1 seconds passed
    ... 88%, 111744 KB, 57322 KB/s, 1 seconds passed
    ... 88%, 111776 KB, 57331 KB/s, 1 seconds passed
    ... 88%, 111808 KB, 57340 KB/s, 1 seconds passed
    ... 88%, 111840 KB, 57350 KB/s, 1 seconds passed
    ... 88%, 111872 KB, 57360 KB/s, 1 seconds passed
    ... 88%, 111904 KB, 57369 KB/s, 1 seconds passed
    ... 88%, 111936 KB, 57379 KB/s, 1 seconds passed
    ... 88%, 111968 KB, 57389 KB/s, 1 seconds passed
    ... 88%, 112000 KB, 57399 KB/s, 1 seconds passed
    ... 88%, 112032 KB, 57409 KB/s, 1 seconds passed
    ... 88%, 112064 KB, 57418 KB/s, 1 seconds passed
    ... 88%, 112096 KB, 57428 KB/s, 1 seconds passed
    ... 89%, 112128 KB, 57438 KB/s, 1 seconds passed
    ... 89%, 112160 KB, 57448 KB/s, 1 seconds passed
    ... 89%, 112192 KB, 57457 KB/s, 1 seconds passed
    ... 89%, 112224 KB, 57467 KB/s, 1 seconds passed
    ... 89%, 112256 KB, 57476 KB/s, 1 seconds passed
    ... 89%, 112288 KB, 57486 KB/s, 1 seconds passed
    ... 89%, 112320 KB, 57495 KB/s, 1 seconds passed
    ... 89%, 112352 KB, 57505 KB/s, 1 seconds passed
    ... 89%, 112384 KB, 57515 KB/s, 1 seconds passed
    ... 89%, 112416 KB, 57525 KB/s, 1 seconds passed
    ... 89%, 112448 KB, 57534 KB/s, 1 seconds passed
    ... 89%, 112480 KB, 57544 KB/s, 1 seconds passed
    ... 89%, 112512 KB, 57553 KB/s, 1 seconds passed
    ... 89%, 112544 KB, 57563 KB/s, 1 seconds passed
    ... 89%, 112576 KB, 57572 KB/s, 1 seconds passed
    ... 89%, 112608 KB, 57582 KB/s, 1 seconds passed
    ... 89%, 112640 KB, 57591 KB/s, 1 seconds passed
    ... 89%, 112672 KB, 57601 KB/s, 1 seconds passed
    ... 89%, 112704 KB, 57610 KB/s, 1 seconds passed
    ... 89%, 112736 KB, 57620 KB/s, 1 seconds passed
    ... 89%, 112768 KB, 57630 KB/s, 1 seconds passed
    ... 89%, 112800 KB, 57639 KB/s, 1 seconds passed
    ... 89%, 112832 KB, 57649 KB/s, 1 seconds passed
    ... 89%, 112864 KB, 57658 KB/s, 1 seconds passed
    ... 89%, 112896 KB, 57667 KB/s, 1 seconds passed
    ... 89%, 112928 KB, 57677 KB/s, 1 seconds passed
    ... 89%, 112960 KB, 57687 KB/s, 1 seconds passed
    ... 89%, 112992 KB, 57696 KB/s, 1 seconds passed
    ... 89%, 113024 KB, 57706 KB/s, 1 seconds passed
    ... 89%, 113056 KB, 57715 KB/s, 1 seconds passed
    ... 89%, 113088 KB, 57725 KB/s, 1 seconds passed
    ... 89%, 113120 KB, 57734 KB/s, 1 seconds passed
    ... 89%, 113152 KB, 57745 KB/s, 1 seconds passed
    ... 89%, 113184 KB, 57757 KB/s, 1 seconds passed
    ... 89%, 113216 KB, 57769 KB/s, 1 seconds passed
    ... 89%, 113248 KB, 57780 KB/s, 1 seconds passed
    ... 89%, 113280 KB, 57792 KB/s, 1 seconds passed
    ... 89%, 113312 KB, 57804 KB/s, 1 seconds passed
    ... 89%, 113344 KB, 57816 KB/s, 1 seconds passed
    ... 90%, 113376 KB, 57828 KB/s, 1 seconds passed
    ... 90%, 113408 KB, 57839 KB/s, 1 seconds passed
    ... 90%, 113440 KB, 57851 KB/s, 1 seconds passed
    ... 90%, 113472 KB, 57862 KB/s, 1 seconds passed
    ... 90%, 113504 KB, 57874 KB/s, 1 seconds passed
    ... 90%, 113536 KB, 57886 KB/s, 1 seconds passed
    ... 90%, 113568 KB, 57897 KB/s, 1 seconds passed
    ... 90%, 113600 KB, 57909 KB/s, 1 seconds passed
    ... 90%, 113632 KB, 57921 KB/s, 1 seconds passed
    ... 90%, 113664 KB, 57932 KB/s, 1 seconds passed
    ... 90%, 113696 KB, 57941 KB/s, 1 seconds passed
    ... 90%, 113728 KB, 57950 KB/s, 1 seconds passed
    ... 90%, 113760 KB, 57957 KB/s, 1 seconds passed
    ... 90%, 113792 KB, 57965 KB/s, 1 seconds passed
    ... 90%, 113824 KB, 57973 KB/s, 1 seconds passed
    ... 90%, 113856 KB, 57977 KB/s, 1 seconds passed
    ... 90%, 113888 KB, 57982 KB/s, 1 seconds passed
    ... 90%, 113920 KB, 57986 KB/s, 1 seconds passed
    ... 90%, 113952 KB, 57996 KB/s, 1 seconds passed
    ... 90%, 113984 KB, 58007 KB/s, 1 seconds passed
    ... 90%, 114016 KB, 58019 KB/s, 1 seconds passed
    ... 90%, 114048 KB, 58031 KB/s, 1 seconds passed
    ... 90%, 114080 KB, 58040 KB/s, 1 seconds passed
    ... 90%, 114112 KB, 58048 KB/s, 1 seconds passed
    ... 90%, 114144 KB, 58053 KB/s, 1 seconds passed
    ... 90%, 114176 KB, 58062 KB/s, 1 seconds passed
    ... 90%, 114208 KB, 58070 KB/s, 1 seconds passed
    ... 90%, 114240 KB, 58077 KB/s, 1 seconds passed
    ... 90%, 114272 KB, 58086 KB/s, 1 seconds passed
    ... 90%, 114304 KB, 58094 KB/s, 1 seconds passed
    ... 90%, 114336 KB, 58101 KB/s, 1 seconds passed
    ... 90%, 114368 KB, 58110 KB/s, 1 seconds passed
    ... 90%, 114400 KB, 58118 KB/s, 1 seconds passed
    ... 90%, 114432 KB, 58127 KB/s, 1 seconds passed
    ... 90%, 114464 KB, 58133 KB/s, 1 seconds passed
    ... 90%, 114496 KB, 58142 KB/s, 1 seconds passed
    ... 90%, 114528 KB, 58149 KB/s, 1 seconds passed
    ... 90%, 114560 KB, 58157 KB/s, 1 seconds passed
    ... 90%, 114592 KB, 58166 KB/s, 1 seconds passed
    ... 91%, 114624 KB, 58175 KB/s, 1 seconds passed
    ... 91%, 114656 KB, 58181 KB/s, 1 seconds passed
    ... 91%, 114688 KB, 58190 KB/s, 1 seconds passed
    ... 91%, 114720 KB, 58197 KB/s, 1 seconds passed
    ... 91%, 114752 KB, 58205 KB/s, 1 seconds passed
    ... 91%, 114784 KB, 58212 KB/s, 1 seconds passed
    ... 91%, 114816 KB, 58216 KB/s, 1 seconds passed
    ... 91%, 114848 KB, 58224 KB/s, 1 seconds passed
    ... 91%, 114880 KB, 58236 KB/s, 1 seconds passed
    ... 91%, 114912 KB, 58246 KB/s, 1 seconds passed
    ... 91%, 114944 KB, 58254 KB/s, 1 seconds passed
    ... 91%, 114976 KB, 58260 KB/s, 1 seconds passed
    ... 91%, 115008 KB, 58264 KB/s, 1 seconds passed
    ... 91%, 115040 KB, 58276 KB/s, 1 seconds passed
    ... 91%, 115072 KB, 58285 KB/s, 1 seconds passed

.. parsed-literal::

    ... 91%, 115104 KB, 58042 KB/s, 1 seconds passed
   ... 91%, 115136 KB, 58043 KB/s, 1 seconds passed
   ... 91%, 115168 KB, 58048 KB/s, 1 seconds passed
   ... 91%, 115200 KB, 58053 KB/s, 1 seconds passed
   ... 91%, 115232 KB, 58059 KB/s, 1 seconds passed
   ... 91%, 115264 KB, 58064 KB/s, 1 seconds passed
   ... 91%, 115296 KB, 58069 KB/s, 1 seconds passed
   ... 91%, 115328 KB, 58074 KB/s, 1 seconds passed
   ... 91%, 115360 KB, 58080 KB/s, 1 seconds passed
   ... 91%, 115392 KB, 58085 KB/s, 1 seconds passed
   ... 91%, 115424 KB, 58090 KB/s, 1 seconds passed
   ... 91%, 115456 KB, 58096 KB/s, 1 seconds passed
   ... 91%, 115488 KB, 58101 KB/s, 1 seconds passed
   ... 91%, 115520 KB, 58107 KB/s, 1 seconds passed
   ... 91%, 115552 KB, 58112 KB/s, 1 seconds passed
   ... 91%, 115584 KB, 58118 KB/s, 1 seconds passed
   ... 91%, 115616 KB, 58123 KB/s, 1 seconds passed
   ... 91%, 115648 KB, 58129 KB/s, 1 seconds passed
   ... 91%, 115680 KB, 58134 KB/s, 1 seconds passed
   ... 91%, 115712 KB, 58141 KB/s, 1 seconds passed
   ... 91%, 115744 KB, 58147 KB/s, 1 seconds passed
   ... 91%, 115776 KB, 58153 KB/s, 1 seconds passed
   ... 91%, 115808 KB, 58159 KB/s, 1 seconds passed
   ... 91%, 115840 KB, 58164 KB/s, 1 seconds passed
   ... 91%, 115872 KB, 58171 KB/s, 1 seconds passed
   ... 92%, 115904 KB, 58177 KB/s, 1 seconds passed
   ... 92%, 115936 KB, 58182 KB/s, 1 seconds passed
   ... 92%, 115968 KB, 58189 KB/s, 1 seconds passed
   ... 92%, 116000 KB, 58195 KB/s, 1 seconds passed
   ... 92%, 116032 KB, 58200 KB/s, 1 seconds passed
   ... 92%, 116064 KB, 58207 KB/s, 1 seconds passed
   ... 92%, 116096 KB, 58214 KB/s, 1 seconds passed
   ... 92%, 116128 KB, 58220 KB/s, 1 seconds passed
   ... 92%, 116160 KB, 58226 KB/s, 1 seconds passed
   ... 92%, 116192 KB, 58232 KB/s, 1 seconds passed
   ... 92%, 116224 KB, 58239 KB/s, 1 seconds passed
   ... 92%, 116256 KB, 58245 KB/s, 1 seconds passed
   ... 92%, 116288 KB, 58251 KB/s, 1 seconds passed
   ... 92%, 116320 KB, 58257 KB/s, 1 seconds passed
   ... 92%, 116352 KB, 58263 KB/s, 1 seconds passed
   ... 92%, 116384 KB, 58269 KB/s, 1 seconds passed
   ... 92%, 116416 KB, 58275 KB/s, 1 seconds passed
   ... 92%, 116448 KB, 58282 KB/s, 1 seconds passed
   ... 92%, 116480 KB, 58287 KB/s, 1 seconds passed
   ... 92%, 116512 KB, 58294 KB/s, 1 seconds passed
   ... 92%, 116544 KB, 58300 KB/s, 1 seconds passed
   ... 92%, 116576 KB, 58307 KB/s, 1 seconds passed
   ... 92%, 116608 KB, 58313 KB/s, 1 seconds passed
   ... 92%, 116640 KB, 58322 KB/s, 1 seconds passed
   ... 92%, 116672 KB, 58332 KB/s, 2 seconds passed
   ... 92%, 116704 KB, 58342 KB/s, 2 seconds passed
   ... 92%, 116736 KB, 58350 KB/s, 2 seconds passed
   ... 92%, 116768 KB, 58360 KB/s, 2 seconds passed
   ... 92%, 116800 KB, 58369 KB/s, 2 seconds passed
   ... 92%, 116832 KB, 58378 KB/s, 2 seconds passed
   ... 92%, 116864 KB, 58387 KB/s, 2 seconds passed
   ... 92%, 116896 KB, 58397 KB/s, 2 seconds passed
   ... 92%, 116928 KB, 58406 KB/s, 2 seconds passed
   ... 92%, 116960 KB, 58416 KB/s, 2 seconds passed
   ... 92%, 116992 KB, 58425 KB/s, 2 seconds passed
   ... 92%, 117024 KB, 58435 KB/s, 2 seconds passed
   ... 92%, 117056 KB, 58444 KB/s, 2 seconds passed
   ... 92%, 117088 KB, 58453 KB/s, 2 seconds passed
   ... 92%, 117120 KB, 58462 KB/s, 2 seconds passed
   ... 93%, 117152 KB, 58472 KB/s, 2 seconds passed
   ... 93%, 117184 KB, 58481 KB/s, 2 seconds passed
   ... 93%, 117216 KB, 58490 KB/s, 2 seconds passed
   ... 93%, 117248 KB, 58499 KB/s, 2 seconds passed
   ... 93%, 117280 KB, 58509 KB/s, 2 seconds passed
   ... 93%, 117312 KB, 58518 KB/s, 2 seconds passed
   ... 93%, 117344 KB, 58527 KB/s, 2 seconds passed
   ... 93%, 117376 KB, 58536 KB/s, 2 seconds passed
   ... 93%, 117408 KB, 58546 KB/s, 2 seconds passed
   ... 93%, 117440 KB, 58556 KB/s, 2 seconds passed
   ... 93%, 117472 KB, 58566 KB/s, 2 seconds passed
   ... 93%, 117504 KB, 58577 KB/s, 2 seconds passed
   ... 93%, 117536 KB, 58587 KB/s, 2 seconds passed
   ... 93%, 117568 KB, 58598 KB/s, 2 seconds passed
   ... 93%, 117600 KB, 58609 KB/s, 2 seconds passed
   ... 93%, 117632 KB, 58264 KB/s, 2 seconds passed
   ... 93%, 117664 KB, 58265 KB/s, 2 seconds passed
   ... 93%, 117696 KB, 58269 KB/s, 2 seconds passed
   ... 93%, 117728 KB, 58277 KB/s, 2 seconds passed
   ... 93%, 117760 KB, 58127 KB/s, 2 seconds passed
   ... 93%, 117792 KB, 58129 KB/s, 2 seconds passed
   ... 93%, 117824 KB, 58134 KB/s, 2 seconds passed
   ... 93%, 117856 KB, 58141 KB/s, 2 seconds passed

.. parsed-literal::

    ... 93%, 117888 KB, 57855 KB/s, 2 seconds passed
    ... 93%, 117920 KB, 57857 KB/s, 2 seconds passed
    ... 93%, 117952 KB, 57861 KB/s, 2 seconds passed
    ... 93%, 117984 KB, 57866 KB/s, 2 seconds passed
    ... 93%, 118016 KB, 57871 KB/s, 2 seconds passed
    ... 93%, 118048 KB, 57876 KB/s, 2 seconds passed
    ... 93%, 118080 KB, 57880 KB/s, 2 seconds passed
    ... 93%, 118112 KB, 57885 KB/s, 2 seconds passed
    ... 93%, 118144 KB, 57890 KB/s, 2 seconds passed
    ... 93%, 118176 KB, 57894 KB/s, 2 seconds passed
    ... 93%, 118208 KB, 57898 KB/s, 2 seconds passed
    ... 93%, 118240 KB, 57904 KB/s, 2 seconds passed
    ... 93%, 118272 KB, 57908 KB/s, 2 seconds passed
    ... 93%, 118304 KB, 57913 KB/s, 2 seconds passed
    ... 93%, 118336 KB, 57918 KB/s, 2 seconds passed
    ... 93%, 118368 KB, 57923 KB/s, 2 seconds passed
    ... 94%, 118400 KB, 57928 KB/s, 2 seconds passed
    ... 94%, 118432 KB, 57933 KB/s, 2 seconds passed
    ... 94%, 118464 KB, 57937 KB/s, 2 seconds passed
    ... 94%, 118496 KB, 57942 KB/s, 2 seconds passed
    ... 94%, 118528 KB, 57947 KB/s, 2 seconds passed
    ... 94%, 118560 KB, 57952 KB/s, 2 seconds passed
    ... 94%, 118592 KB, 57956 KB/s, 2 seconds passed
    ... 94%, 118624 KB, 57961 KB/s, 2 seconds passed
    ... 94%, 118656 KB, 57967 KB/s, 2 seconds passed
    ... 94%, 118688 KB, 57972 KB/s, 2 seconds passed
    ... 94%, 118720 KB, 57976 KB/s, 2 seconds passed
    ... 94%, 118752 KB, 57981 KB/s, 2 seconds passed
    ... 94%, 118784 KB, 57986 KB/s, 2 seconds passed
    ... 94%, 118816 KB, 57990 KB/s, 2 seconds passed
    ... 94%, 118848 KB, 57993 KB/s, 2 seconds passed
    ... 94%, 118880 KB, 57998 KB/s, 2 seconds passed
    ... 94%, 118912 KB, 58006 KB/s, 2 seconds passed
    ... 94%, 118944 KB, 58015 KB/s, 2 seconds passed
    ... 94%, 118976 KB, 58023 KB/s, 2 seconds passed
    ... 94%, 119008 KB, 58032 KB/s, 2 seconds passed
    ... 94%, 119040 KB, 58041 KB/s, 2 seconds passed
    ... 94%, 119072 KB, 58049 KB/s, 2 seconds passed
    ... 94%, 119104 KB, 58057 KB/s, 2 seconds passed
    ... 94%, 119136 KB, 58066 KB/s, 2 seconds passed
    ... 94%, 119168 KB, 58075 KB/s, 2 seconds passed
    ... 94%, 119200 KB, 58083 KB/s, 2 seconds passed
    ... 94%, 119232 KB, 58092 KB/s, 2 seconds passed
    ... 94%, 119264 KB, 58100 KB/s, 2 seconds passed
    ... 94%, 119296 KB, 58109 KB/s, 2 seconds passed
    ... 94%, 119328 KB, 58117 KB/s, 2 seconds passed
    ... 94%, 119360 KB, 58127 KB/s, 2 seconds passed
    ... 94%, 119392 KB, 58136 KB/s, 2 seconds passed
    ... 94%, 119424 KB, 58147 KB/s, 2 seconds passed
    ... 94%, 119456 KB, 58157 KB/s, 2 seconds passed
    ... 94%, 119488 KB, 58167 KB/s, 2 seconds passed
    ... 94%, 119520 KB, 58177 KB/s, 2 seconds passed
    ... 94%, 119552 KB, 58187 KB/s, 2 seconds passed
    ... 94%, 119584 KB, 58197 KB/s, 2 seconds passed
    ... 94%, 119616 KB, 58207 KB/s, 2 seconds passed
    ... 94%, 119648 KB, 58218 KB/s, 2 seconds passed
    ... 95%, 119680 KB, 58227 KB/s, 2 seconds passed
    ... 95%, 119712 KB, 57830 KB/s, 2 seconds passed
    ... 95%, 119744 KB, 57832 KB/s, 2 seconds passed
    ... 95%, 119776 KB, 57837 KB/s, 2 seconds passed
    ... 95%, 119808 KB, 57841 KB/s, 2 seconds passed
    ... 95%, 119840 KB, 57846 KB/s, 2 seconds passed
    ... 95%, 119872 KB, 57853 KB/s, 2 seconds passed
    ... 95%, 119904 KB, 57728 KB/s, 2 seconds passed
    ... 95%, 119936 KB, 57728 KB/s, 2 seconds passed
    ... 95%, 119968 KB, 57732 KB/s, 2 seconds passed
    ... 95%, 120000 KB, 57737 KB/s, 2 seconds passed
    ... 95%, 120032 KB, 57742 KB/s, 2 seconds passed
    ... 95%, 120064 KB, 57747 KB/s, 2 seconds passed
    ... 95%, 120096 KB, 57751 KB/s, 2 seconds passed
    ... 95%, 120128 KB, 57756 KB/s, 2 seconds passed

.. parsed-literal::

    ... 95%, 120160 KB, 57761 KB/s, 2 seconds passed
    ... 95%, 120192 KB, 57766 KB/s, 2 seconds passed
    ... 95%, 120224 KB, 57770 KB/s, 2 seconds passed
    ... 95%, 120256 KB, 57775 KB/s, 2 seconds passed
    ... 95%, 120288 KB, 57780 KB/s, 2 seconds passed
    ... 95%, 120320 KB, 57785 KB/s, 2 seconds passed
    ... 95%, 120352 KB, 57791 KB/s, 2 seconds passed
    ... 95%, 120384 KB, 57799 KB/s, 2 seconds passed
    ... 95%, 120416 KB, 57806 KB/s, 2 seconds passed
    ... 95%, 120448 KB, 57813 KB/s, 2 seconds passed
    ... 95%, 120480 KB, 57821 KB/s, 2 seconds passed
    ... 95%, 120512 KB, 57829 KB/s, 2 seconds passed
    ... 95%, 120544 KB, 57686 KB/s, 2 seconds passed
    ... 95%, 120576 KB, 57687 KB/s, 2 seconds passed
    ... 95%, 120608 KB, 57691 KB/s, 2 seconds passed
    ... 95%, 120640 KB, 57696 KB/s, 2 seconds passed
    ... 95%, 120672 KB, 57700 KB/s, 2 seconds passed
    ... 95%, 120704 KB, 57705 KB/s, 2 seconds passed
    ... 95%, 120736 KB, 57711 KB/s, 2 seconds passed
    ... 95%, 120768 KB, 57716 KB/s, 2 seconds passed
    ... 95%, 120800 KB, 57721 KB/s, 2 seconds passed
    ... 95%, 120832 KB, 57725 KB/s, 2 seconds passed
    ... 95%, 120864 KB, 57729 KB/s, 2 seconds passed
    ... 95%, 120896 KB, 57734 KB/s, 2 seconds passed
    ... 96%, 120928 KB, 57740 KB/s, 2 seconds passed
    ... 96%, 120960 KB, 57744 KB/s, 2 seconds passed
    ... 96%, 120992 KB, 57750 KB/s, 2 seconds passed
    ... 96%, 121024 KB, 57754 KB/s, 2 seconds passed
    ... 96%, 121056 KB, 57760 KB/s, 2 seconds passed
    ... 96%, 121088 KB, 57764 KB/s, 2 seconds passed
    ... 96%, 121120 KB, 57769 KB/s, 2 seconds passed
    ... 96%, 121152 KB, 57774 KB/s, 2 seconds passed
    ... 96%, 121184 KB, 57778 KB/s, 2 seconds passed
    ... 96%, 121216 KB, 57783 KB/s, 2 seconds passed
    ... 96%, 121248 KB, 57789 KB/s, 2 seconds passed
    ... 96%, 121280 KB, 57794 KB/s, 2 seconds passed
    ... 96%, 121312 KB, 57799 KB/s, 2 seconds passed
    ... 96%, 121344 KB, 57804 KB/s, 2 seconds passed
    ... 96%, 121376 KB, 57809 KB/s, 2 seconds passed
    ... 96%, 121408 KB, 57814 KB/s, 2 seconds passed
    ... 96%, 121440 KB, 57819 KB/s, 2 seconds passed
    ... 96%, 121472 KB, 57825 KB/s, 2 seconds passed
    ... 96%, 121504 KB, 57833 KB/s, 2 seconds passed
    ... 96%, 121536 KB, 57841 KB/s, 2 seconds passed
    ... 96%, 121568 KB, 57850 KB/s, 2 seconds passed
    ... 96%, 121600 KB, 57858 KB/s, 2 seconds passed
    ... 96%, 121632 KB, 57867 KB/s, 2 seconds passed
    ... 96%, 121664 KB, 57875 KB/s, 2 seconds passed
    ... 96%, 121696 KB, 57883 KB/s, 2 seconds passed
    ... 96%, 121728 KB, 57892 KB/s, 2 seconds passed
    ... 96%, 121760 KB, 57900 KB/s, 2 seconds passed
    ... 96%, 121792 KB, 57909 KB/s, 2 seconds passed
    ... 96%, 121824 KB, 57917 KB/s, 2 seconds passed
    ... 96%, 121856 KB, 57926 KB/s, 2 seconds passed
    ... 96%, 121888 KB, 57935 KB/s, 2 seconds passed
    ... 96%, 121920 KB, 57944 KB/s, 2 seconds passed
    ... 96%, 121952 KB, 57952 KB/s, 2 seconds passed
    ... 96%, 121984 KB, 57961 KB/s, 2 seconds passed
    ... 96%, 122016 KB, 57970 KB/s, 2 seconds passed
    ... 96%, 122048 KB, 57978 KB/s, 2 seconds passed
    ... 96%, 122080 KB, 57986 KB/s, 2 seconds passed
    ... 96%, 122112 KB, 57995 KB/s, 2 seconds passed
    ... 96%, 122144 KB, 58003 KB/s, 2 seconds passed
    ... 97%, 122176 KB, 58012 KB/s, 2 seconds passed
    ... 97%, 122208 KB, 58020 KB/s, 2 seconds passed
    ... 97%, 122240 KB, 58029 KB/s, 2 seconds passed
    ... 97%, 122272 KB, 58038 KB/s, 2 seconds passed
    ... 97%, 122304 KB, 58046 KB/s, 2 seconds passed
    ... 97%, 122336 KB, 58055 KB/s, 2 seconds passed
    ... 97%, 122368 KB, 58064 KB/s, 2 seconds passed
    ... 97%, 122400 KB, 58072 KB/s, 2 seconds passed
    ... 97%, 122432 KB, 58081 KB/s, 2 seconds passed
    ... 97%, 122464 KB, 58090 KB/s, 2 seconds passed
    ... 97%, 122496 KB, 58098 KB/s, 2 seconds passed
    ... 97%, 122528 KB, 58107 KB/s, 2 seconds passed
    ... 97%, 122560 KB, 58115 KB/s, 2 seconds passed
    ... 97%, 122592 KB, 58123 KB/s, 2 seconds passed
    ... 97%, 122624 KB, 58131 KB/s, 2 seconds passed
    ... 97%, 122656 KB, 58140 KB/s, 2 seconds passed
    ... 97%, 122688 KB, 58149 KB/s, 2 seconds passed
    ... 97%, 122720 KB, 58159 KB/s, 2 seconds passed
    ... 97%, 122752 KB, 58169 KB/s, 2 seconds passed
    ... 97%, 122784 KB, 58179 KB/s, 2 seconds passed
    ... 97%, 122816 KB, 58190 KB/s, 2 seconds passed
    ... 97%, 122848 KB, 58202 KB/s, 2 seconds passed
    ... 97%, 122880 KB, 57982 KB/s, 2 seconds passed
    ... 97%, 122912 KB, 57983 KB/s, 2 seconds passed
    ... 97%, 122944 KB, 57991 KB/s, 2 seconds passed
    ... 97%, 122976 KB, 58001 KB/s, 2 seconds passed
    ... 97%, 123008 KB, 58012 KB/s, 2 seconds passed
    ... 97%, 123040 KB, 58021 KB/s, 2 seconds passed
    ... 97%, 123072 KB, 58029 KB/s, 2 seconds passed
    ... 97%, 123104 KB, 58037 KB/s, 2 seconds passed
    ... 97%, 123136 KB, 58045 KB/s, 2 seconds passed
    ... 97%, 123168 KB, 58050 KB/s, 2 seconds passed
    ... 97%, 123200 KB, 58057 KB/s, 2 seconds passed
    ... 97%, 123232 KB, 58065 KB/s, 2 seconds passed
    ... 97%, 123264 KB, 58075 KB/s, 2 seconds passed
    ... 97%, 123296 KB, 58083 KB/s, 2 seconds passed
    ... 97%, 123328 KB, 58089 KB/s, 2 seconds passed
    ... 97%, 123360 KB, 58094 KB/s, 2 seconds passed
    ... 97%, 123392 KB, 58102 KB/s, 2 seconds passed
    ... 97%, 123424 KB, 58109 KB/s, 2 seconds passed
    ... 98%, 123456 KB, 58118 KB/s, 2 seconds passed
    ... 98%, 123488 KB, 58126 KB/s, 2 seconds passed
    ... 98%, 123520 KB, 58134 KB/s, 2 seconds passed
    ... 98%, 123552 KB, 58142 KB/s, 2 seconds passed
    ... 98%, 123584 KB, 58150 KB/s, 2 seconds passed
    ... 98%, 123616 KB, 58153 KB/s, 2 seconds passed
    ... 98%, 123648 KB, 58161 KB/s, 2 seconds passed
    ... 98%, 123680 KB, 58170 KB/s, 2 seconds passed
    ... 98%, 123712 KB, 58178 KB/s, 2 seconds passed
    ... 98%, 123744 KB, 58186 KB/s, 2 seconds passed
    ... 98%, 123776 KB, 58194 KB/s, 2 seconds passed
    ... 98%, 123808 KB, 58200 KB/s, 2 seconds passed
    ... 98%, 123840 KB, 58208 KB/s, 2 seconds passed
    ... 98%, 123872 KB, 58215 KB/s, 2 seconds passed
    ... 98%, 123904 KB, 58223 KB/s, 2 seconds passed
    ... 98%, 123936 KB, 58230 KB/s, 2 seconds passed
    ... 98%, 123968 KB, 58237 KB/s, 2 seconds passed

.. parsed-literal::

    ... 98%, 124000 KB, 58002 KB/s, 2 seconds passed
    ... 98%, 124032 KB, 58005 KB/s, 2 seconds passed
    ... 98%, 124064 KB, 58009 KB/s, 2 seconds passed
    ... 98%, 124096 KB, 58013 KB/s, 2 seconds passed
    ... 98%, 124128 KB, 58019 KB/s, 2 seconds passed
    ... 98%, 124160 KB, 58024 KB/s, 2 seconds passed
    ... 98%, 124192 KB, 58030 KB/s, 2 seconds passed
    ... 98%, 124224 KB, 58035 KB/s, 2 seconds passed
    ... 98%, 124256 KB, 58040 KB/s, 2 seconds passed
    ... 98%, 124288 KB, 58045 KB/s, 2 seconds passed
    ... 98%, 124320 KB, 58050 KB/s, 2 seconds passed
    ... 98%, 124352 KB, 58054 KB/s, 2 seconds passed
    ... 98%, 124384 KB, 58058 KB/s, 2 seconds passed
    ... 98%, 124416 KB, 58064 KB/s, 2 seconds passed
    ... 98%, 124448 KB, 58068 KB/s, 2 seconds passed
    ... 98%, 124480 KB, 58072 KB/s, 2 seconds passed
    ... 98%, 124512 KB, 58077 KB/s, 2 seconds passed
    ... 98%, 124544 KB, 58082 KB/s, 2 seconds passed
    ... 98%, 124576 KB, 58086 KB/s, 2 seconds passed
    ... 98%, 124608 KB, 58090 KB/s, 2 seconds passed
    ... 98%, 124640 KB, 58094 KB/s, 2 seconds passed
    ... 98%, 124672 KB, 58099 KB/s, 2 seconds passed
    ... 99%, 124704 KB, 58104 KB/s, 2 seconds passed
    ... 99%, 124736 KB, 58109 KB/s, 2 seconds passed
    ... 99%, 124768 KB, 58113 KB/s, 2 seconds passed
    ... 99%, 124800 KB, 58118 KB/s, 2 seconds passed
    ... 99%, 124832 KB, 58122 KB/s, 2 seconds passed
    ... 99%, 124864 KB, 58127 KB/s, 2 seconds passed
    ... 99%, 124896 KB, 58132 KB/s, 2 seconds passed
    ... 99%, 124928 KB, 58137 KB/s, 2 seconds passed
    ... 99%, 124960 KB, 58141 KB/s, 2 seconds passed
    ... 99%, 124992 KB, 58145 KB/s, 2 seconds passed
    ... 99%, 125024 KB, 58149 KB/s, 2 seconds passed
    ... 99%, 125056 KB, 58154 KB/s, 2 seconds passed
    ... 99%, 125088 KB, 58159 KB/s, 2 seconds passed
    ... 99%, 125120 KB, 58164 KB/s, 2 seconds passed
    ... 99%, 125152 KB, 58173 KB/s, 2 seconds passed
    ... 99%, 125184 KB, 58181 KB/s, 2 seconds passed
    ... 99%, 125216 KB, 58189 KB/s, 2 seconds passed
    ... 99%, 125248 KB, 58197 KB/s, 2 seconds passed
    ... 99%, 125280 KB, 58205 KB/s, 2 seconds passed
    ... 99%, 125312 KB, 58213 KB/s, 2 seconds passed
    ... 99%, 125344 KB, 58221 KB/s, 2 seconds passed
    ... 99%, 125376 KB, 58228 KB/s, 2 seconds passed
    ... 99%, 125408 KB, 58237 KB/s, 2 seconds passed
    ... 99%, 125440 KB, 58245 KB/s, 2 seconds passed
    ... 99%, 125472 KB, 58254 KB/s, 2 seconds passed
    ... 99%, 125504 KB, 58262 KB/s, 2 seconds passed
    ... 99%, 125536 KB, 58271 KB/s, 2 seconds passed
    ... 99%, 125568 KB, 58279 KB/s, 2 seconds passed
    ... 99%, 125600 KB, 58287 KB/s, 2 seconds passed
    ... 99%, 125632 KB, 58296 KB/s, 2 seconds passed
    ... 99%, 125664 KB, 58306 KB/s, 2 seconds passed
    ... 99%, 125696 KB, 58315 KB/s, 2 seconds passed
    ... 99%, 125728 KB, 58325 KB/s, 2 seconds passed
    ... 99%, 125760 KB, 58334 KB/s, 2 seconds passed
    ... 99%, 125792 KB, 58344 KB/s, 2 seconds passed
    ... 99%, 125824 KB, 58354 KB/s, 2 seconds passed
    ... 99%, 125856 KB, 58363 KB/s, 2 seconds passed
    ... 99%, 125888 KB, 58373 KB/s, 2 seconds passed
    ... 99%, 125920 KB, 58382 KB/s, 2 seconds passed
    ... 99%, 125952 KB, 58392 KB/s, 2 seconds passed
    ... 100%, 125953 KB, 58386 KB/s, 2 seconds passed



.. parsed-literal::


    ========== Downloading models/public/colorization-v2/model/__init__.py


.. parsed-literal::

    ... 100%, 0 KB, 310 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/base_color.py


.. parsed-literal::

    ... 100%, 0 KB, 1857 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/eccv16.py


.. parsed-literal::

    ... 100%, 4 KB, 17595 KB/s, 0 seconds passed


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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=models/public/colorization-v2 --model-name=ECCVGenerator --weights=models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth --import-module=model --input-shape=1,1,256,256 --output-file=models/public/colorization-v2/colorization-v2-eccv16.onnx --input-names=data_l --output-names=color_ab



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting colorization-v2 to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=models/public/colorization-v2/FP16 --model_name=colorization-v2 --input=data_l --output=color_ab --input_model=models/public/colorization-v2/colorization-v2-eccv16.onnx '--layout=data_l(NCHW)' '--input_shape=[1, 1, 256, 256]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-642/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.bin


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

