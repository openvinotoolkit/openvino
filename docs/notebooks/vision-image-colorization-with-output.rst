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

    import platform

    %pip install "openvino-dev>=2024.0.0" opencv-python tqdm

    if platform.system() != "Windows":
        %pip install -q "matplotlib>=3.4"
    else:
        %pip install -q "matplotlib>=3.4,<3.7"


.. parsed-literal::

    Collecting openvino-dev>=2024.0.0
      Using cached openvino_dev-2024.0.0-14509-py3-none-any.whl.metadata (16 kB)
    Requirement already satisfied: opencv-python in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.9.0.80)
    Requirement already satisfied: tqdm in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.66.2)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (3.1)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.1.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.7)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


.. parsed-literal::

    Using cached openvino_dev-2024.0.0-14509-py3-none-any.whl (4.7 MB)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Installing collected packages: openvino-dev


.. parsed-literal::

    Successfully installed openvino-dev-2024.0.0


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import os
    from pathlib import Path

    import cv2
    import matplotlib.pyplot as plt
    import numpy as np
    import openvino as ov

    # Fetch `notebook_utils` module
    import requests

    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )

    open("notebook_utils.py", "w").write(r.text)

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
        value="AUTO",
        description="Device:",
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

    download_command = f"omz_downloader " f"--name {MODEL_NAME} " f"--output_dir {MODEL_DIR} " f"--cache_dir {MODEL_DIR}"
    ! $download_command


.. parsed-literal::

    ################|| Downloading colorization-v2 ||################

    ========== Downloading models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth


.. parsed-literal::

    ... 0%, 32 KB, 969 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 972 KB/s, 0 seconds passed
... 0%, 96 KB, 1391 KB/s, 0 seconds passed
... 0%, 128 KB, 1294 KB/s, 0 seconds passed
... 0%, 160 KB, 1589 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 192 KB, 1871 KB/s, 0 seconds passed
... 0%, 224 KB, 2144 KB/s, 0 seconds passed
... 0%, 256 KB, 2405 KB/s, 0 seconds passed
... 0%, 288 KB, 2185 KB/s, 0 seconds passed
... 0%, 320 KB, 2418 KB/s, 0 seconds passed
... 0%, 352 KB, 2651 KB/s, 0 seconds passed
... 0%, 384 KB, 2879 KB/s, 0 seconds passed
... 0%, 416 KB, 3098 KB/s, 0 seconds passed
... 0%, 448 KB, 3316 KB/s, 0 seconds passed
... 0%, 480 KB, 3460 KB/s, 0 seconds passed
... 0%, 512 KB, 3657 KB/s, 0 seconds passed
... 0%, 544 KB, 3874 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 576 KB, 3482 KB/s, 0 seconds passed
... 0%, 608 KB, 3665 KB/s, 0 seconds passed
... 0%, 640 KB, 3847 KB/s, 0 seconds passed
... 0%, 672 KB, 4030 KB/s, 0 seconds passed
... 0%, 704 KB, 4208 KB/s, 0 seconds passed
... 0%, 736 KB, 4389 KB/s, 0 seconds passed
... 0%, 768 KB, 4566 KB/s, 0 seconds passed
... 0%, 800 KB, 4745 KB/s, 0 seconds passed
... 0%, 832 KB, 4920 KB/s, 0 seconds passed
... 0%, 864 KB, 5099 KB/s, 0 seconds passed
... 0%, 896 KB, 5275 KB/s, 0 seconds passed
... 0%, 928 KB, 5451 KB/s, 0 seconds passed
... 0%, 960 KB, 5619 KB/s, 0 seconds passed
... 0%, 992 KB, 5789 KB/s, 0 seconds passed
... 0%, 1024 KB, 5960 KB/s, 0 seconds passed
... 0%, 1056 KB, 6130 KB/s, 0 seconds passed
... 0%, 1088 KB, 6302 KB/s, 0 seconds passed
... 0%, 1120 KB, 6461 KB/s, 0 seconds passed
... 0%, 1152 KB, 5741 KB/s, 0 seconds passed
... 0%, 1184 KB, 5885 KB/s, 0 seconds passed
... 0%, 1216 KB, 6023 KB/s, 0 seconds passed
... 0%, 1248 KB, 6169 KB/s, 0 seconds passed
... 1%, 1280 KB, 6314 KB/s, 0 seconds passed
... 1%, 1312 KB, 6460 KB/s, 0 seconds passed
... 1%, 1344 KB, 6602 KB/s, 0 seconds passed
... 1%, 1376 KB, 6745 KB/s, 0 seconds passed
... 1%, 1408 KB, 6881 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1440 KB, 7024 KB/s, 0 seconds passed
... 1%, 1472 KB, 7166 KB/s, 0 seconds passed
... 1%, 1504 KB, 7307 KB/s, 0 seconds passed
... 1%, 1536 KB, 7448 KB/s, 0 seconds passed
... 1%, 1568 KB, 7588 KB/s, 0 seconds passed
... 1%, 1600 KB, 7729 KB/s, 0 seconds passed
... 1%, 1632 KB, 7869 KB/s, 0 seconds passed
... 1%, 1664 KB, 8009 KB/s, 0 seconds passed
... 1%, 1696 KB, 8147 KB/s, 0 seconds passed
... 1%, 1728 KB, 8285 KB/s, 0 seconds passed
... 1%, 1760 KB, 8423 KB/s, 0 seconds passed
... 1%, 1792 KB, 8561 KB/s, 0 seconds passed
... 1%, 1824 KB, 8698 KB/s, 0 seconds passed
... 1%, 1856 KB, 8833 KB/s, 0 seconds passed
... 1%, 1888 KB, 8970 KB/s, 0 seconds passed
... 1%, 1920 KB, 9105 KB/s, 0 seconds passed
... 1%, 1952 KB, 9240 KB/s, 0 seconds passed
... 1%, 1984 KB, 9374 KB/s, 0 seconds passed
... 1%, 2016 KB, 9507 KB/s, 0 seconds passed
... 1%, 2048 KB, 9643 KB/s, 0 seconds passed
... 1%, 2080 KB, 9780 KB/s, 0 seconds passed
... 1%, 2112 KB, 9917 KB/s, 0 seconds passed
... 1%, 2144 KB, 10053 KB/s, 0 seconds passed
... 1%, 2176 KB, 10189 KB/s, 0 seconds passed
... 1%, 2208 KB, 10325 KB/s, 0 seconds passed
... 1%, 2240 KB, 10461 KB/s, 0 seconds passed
... 1%, 2272 KB, 10596 KB/s, 0 seconds passed
... 1%, 2304 KB, 9916 KB/s, 0 seconds passed
... 1%, 2336 KB, 10033 KB/s, 0 seconds passed
... 1%, 2368 KB, 10152 KB/s, 0 seconds passed
... 1%, 2400 KB, 10272 KB/s, 0 seconds passed
... 1%, 2432 KB, 10393 KB/s, 0 seconds passed
... 1%, 2464 KB, 10516 KB/s, 0 seconds passed
... 1%, 2496 KB, 10448 KB/s, 0 seconds passed
... 2%, 2528 KB, 10559 KB/s, 0 seconds passed
... 2%, 2560 KB, 10672 KB/s, 0 seconds passed
... 2%, 2592 KB, 10787 KB/s, 0 seconds passed
... 2%, 2624 KB, 10902 KB/s, 0 seconds passed
... 2%, 2656 KB, 11018 KB/s, 0 seconds passed
... 2%, 2688 KB, 11132 KB/s, 0 seconds passed
... 2%, 2720 KB, 11247 KB/s, 0 seconds passed
... 2%, 2752 KB, 11362 KB/s, 0 seconds passed
... 2%, 2784 KB, 11476 KB/s, 0 seconds passed
... 2%, 2816 KB, 11590 KB/s, 0 seconds passed
... 2%, 2848 KB, 11703 KB/s, 0 seconds passed
... 2%, 2880 KB, 11815 KB/s, 0 seconds passed
... 2%, 2912 KB, 11928 KB/s, 0 seconds passed
... 2%, 2944 KB, 12040 KB/s, 0 seconds passed
... 2%, 2976 KB, 12151 KB/s, 0 seconds passed
... 2%, 3008 KB, 12263 KB/s, 0 seconds passed
... 2%, 3040 KB, 12375 KB/s, 0 seconds passed
... 2%, 3072 KB, 12484 KB/s, 0 seconds passed
... 2%, 3104 KB, 12594 KB/s, 0 seconds passed
... 2%, 3136 KB, 12705 KB/s, 0 seconds passed
... 2%, 3168 KB, 12814 KB/s, 0 seconds passed
... 2%, 3200 KB, 12923 KB/s, 0 seconds passed
... 2%, 3232 KB, 13031 KB/s, 0 seconds passed
... 2%, 3264 KB, 13140 KB/s, 0 seconds passed
... 2%, 3296 KB, 13249 KB/s, 0 seconds passed
... 2%, 3328 KB, 13362 KB/s, 0 seconds passed
... 2%, 3360 KB, 13475 KB/s, 0 seconds passed
... 2%, 3392 KB, 13587 KB/s, 0 seconds passed
... 2%, 3424 KB, 13700 KB/s, 0 seconds passed
... 2%, 3456 KB, 13812 KB/s, 0 seconds passed
... 2%, 3488 KB, 13924 KB/s, 0 seconds passed
... 2%, 3520 KB, 14035 KB/s, 0 seconds passed
... 2%, 3552 KB, 14147 KB/s, 0 seconds passed
... 2%, 3584 KB, 14258 KB/s, 0 seconds passed
... 2%, 3616 KB, 14369 KB/s, 0 seconds passed
... 2%, 3648 KB, 14480 KB/s, 0 seconds passed
... 2%, 3680 KB, 14589 KB/s, 0 seconds passed
... 2%, 3712 KB, 14699 KB/s, 0 seconds passed
... 2%, 3744 KB, 14809 KB/s, 0 seconds passed
... 2%, 3776 KB, 14918 KB/s, 0 seconds passed
... 3%, 3808 KB, 15027 KB/s, 0 seconds passed
... 3%, 3840 KB, 15136 KB/s, 0 seconds passed
... 3%, 3872 KB, 15245 KB/s, 0 seconds passed
... 3%, 3904 KB, 15354 KB/s, 0 seconds passed
... 3%, 3936 KB, 15462 KB/s, 0 seconds passed
... 3%, 3968 KB, 15571 KB/s, 0 seconds passed
... 3%, 4000 KB, 15679 KB/s, 0 seconds passed
... 3%, 4032 KB, 15787 KB/s, 0 seconds passed
... 3%, 4064 KB, 15895 KB/s, 0 seconds passed
... 3%, 4096 KB, 16002 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 4128 KB, 16108 KB/s, 0 seconds passed
... 3%, 4160 KB, 16214 KB/s, 0 seconds passed
... 3%, 4192 KB, 16320 KB/s, 0 seconds passed
... 3%, 4224 KB, 16427 KB/s, 0 seconds passed
... 3%, 4256 KB, 16534 KB/s, 0 seconds passed
... 3%, 4288 KB, 16643 KB/s, 0 seconds passed
... 3%, 4320 KB, 16752 KB/s, 0 seconds passed
... 3%, 4352 KB, 16861 KB/s, 0 seconds passed
... 3%, 4384 KB, 16970 KB/s, 0 seconds passed
... 3%, 4416 KB, 17079 KB/s, 0 seconds passed
... 3%, 4448 KB, 17189 KB/s, 0 seconds passed
... 3%, 4480 KB, 17302 KB/s, 0 seconds passed
... 3%, 4512 KB, 17415 KB/s, 0 seconds passed
... 3%, 4544 KB, 17529 KB/s, 0 seconds passed
... 3%, 4576 KB, 17642 KB/s, 0 seconds passed
... 3%, 4608 KB, 17755 KB/s, 0 seconds passed
... 3%, 4640 KB, 17473 KB/s, 0 seconds passed
... 3%, 4672 KB, 17575 KB/s, 0 seconds passed
... 3%, 4704 KB, 17678 KB/s, 0 seconds passed
... 3%, 4736 KB, 17781 KB/s, 0 seconds passed
... 3%, 4768 KB, 17883 KB/s, 0 seconds passed
... 3%, 4800 KB, 17985 KB/s, 0 seconds passed
... 3%, 4832 KB, 18084 KB/s, 0 seconds passed
... 3%, 4864 KB, 18186 KB/s, 0 seconds passed
... 3%, 4896 KB, 18287 KB/s, 0 seconds passed
... 3%, 4928 KB, 18015 KB/s, 0 seconds passed
... 3%, 4960 KB, 18110 KB/s, 0 seconds passed
... 3%, 4992 KB, 18209 KB/s, 0 seconds passed
... 3%, 5024 KB, 18309 KB/s, 0 seconds passed
... 4%, 5056 KB, 18409 KB/s, 0 seconds passed
... 4%, 5088 KB, 18508 KB/s, 0 seconds passed
... 4%, 5120 KB, 18607 KB/s, 0 seconds passed
... 4%, 5152 KB, 18701 KB/s, 0 seconds passed
... 4%, 5184 KB, 18798 KB/s, 0 seconds passed
... 4%, 5216 KB, 18895 KB/s, 0 seconds passed
... 4%, 5248 KB, 18993 KB/s, 0 seconds passed
... 4%, 5280 KB, 19089 KB/s, 0 seconds passed
... 4%, 5312 KB, 19186 KB/s, 0 seconds passed
... 4%, 5344 KB, 19283 KB/s, 0 seconds passed
... 4%, 5376 KB, 19379 KB/s, 0 seconds passed
... 4%, 5408 KB, 19472 KB/s, 0 seconds passed
... 4%, 5440 KB, 19569 KB/s, 0 seconds passed
... 4%, 5472 KB, 19665 KB/s, 0 seconds passed
... 4%, 5504 KB, 19761 KB/s, 0 seconds passed
... 4%, 5536 KB, 19854 KB/s, 0 seconds passed
... 4%, 5568 KB, 19948 KB/s, 0 seconds passed
... 4%, 5600 KB, 20037 KB/s, 0 seconds passed
... 4%, 5632 KB, 20125 KB/s, 0 seconds passed
... 4%, 5664 KB, 20213 KB/s, 0 seconds passed
... 4%, 5696 KB, 20302 KB/s, 0 seconds passed
... 4%, 5728 KB, 20391 KB/s, 0 seconds passed
... 4%, 5760 KB, 20479 KB/s, 0 seconds passed
... 4%, 5792 KB, 20567 KB/s, 0 seconds passed
... 4%, 5824 KB, 20656 KB/s, 0 seconds passed
... 4%, 5856 KB, 20744 KB/s, 0 seconds passed
... 4%, 5888 KB, 20832 KB/s, 0 seconds passed
... 4%, 5920 KB, 20919 KB/s, 0 seconds passed
... 4%, 5952 KB, 21006 KB/s, 0 seconds passed
... 4%, 5984 KB, 21093 KB/s, 0 seconds passed
... 4%, 6016 KB, 21178 KB/s, 0 seconds passed
... 4%, 6048 KB, 21264 KB/s, 0 seconds passed
... 4%, 6080 KB, 21350 KB/s, 0 seconds passed
... 4%, 6112 KB, 21435 KB/s, 0 seconds passed
... 4%, 6144 KB, 21522 KB/s, 0 seconds passed
... 4%, 6176 KB, 21607 KB/s, 0 seconds passed
... 4%, 6208 KB, 21693 KB/s, 0 seconds passed
... 4%, 6240 KB, 21776 KB/s, 0 seconds passed
... 4%, 6272 KB, 21860 KB/s, 0 seconds passed
... 5%, 6304 KB, 21946 KB/s, 0 seconds passed
... 5%, 6336 KB, 22029 KB/s, 0 seconds passed
... 5%, 6368 KB, 22111 KB/s, 0 seconds passed
... 5%, 6400 KB, 22194 KB/s, 0 seconds passed
... 5%, 6432 KB, 22279 KB/s, 0 seconds passed
... 5%, 6464 KB, 22362 KB/s, 0 seconds passed
... 5%, 6496 KB, 22451 KB/s, 0 seconds passed
... 5%, 6528 KB, 22543 KB/s, 0 seconds passed
... 5%, 6560 KB, 22637 KB/s, 0 seconds passed
... 5%, 6592 KB, 22729 KB/s, 0 seconds passed
... 5%, 6624 KB, 22822 KB/s, 0 seconds passed
... 5%, 6656 KB, 22915 KB/s, 0 seconds passed
... 5%, 6688 KB, 23008 KB/s, 0 seconds passed
... 5%, 6720 KB, 23101 KB/s, 0 seconds passed
... 5%, 6752 KB, 23196 KB/s, 0 seconds passed
... 5%, 6784 KB, 23291 KB/s, 0 seconds passed
... 5%, 6816 KB, 23387 KB/s, 0 seconds passed
... 5%, 6848 KB, 23482 KB/s, 0 seconds passed
... 5%, 6880 KB, 23576 KB/s, 0 seconds passed
... 5%, 6912 KB, 23667 KB/s, 0 seconds passed
... 5%, 6944 KB, 23759 KB/s, 0 seconds passed
... 5%, 6976 KB, 23850 KB/s, 0 seconds passed
... 5%, 7008 KB, 23938 KB/s, 0 seconds passed
... 5%, 7040 KB, 24025 KB/s, 0 seconds passed
... 5%, 7072 KB, 23978 KB/s, 0 seconds passed
... 5%, 7104 KB, 24051 KB/s, 0 seconds passed
... 5%, 7136 KB, 24133 KB/s, 0 seconds passed
... 5%, 7168 KB, 24224 KB/s, 0 seconds passed
... 5%, 7200 KB, 24312 KB/s, 0 seconds passed
... 5%, 7232 KB, 24399 KB/s, 0 seconds passed
... 5%, 7264 KB, 24485 KB/s, 0 seconds passed
... 5%, 7296 KB, 24567 KB/s, 0 seconds passed
... 5%, 7328 KB, 24652 KB/s, 0 seconds passed
... 5%, 7360 KB, 24738 KB/s, 0 seconds passed
... 5%, 7392 KB, 24823 KB/s, 0 seconds passed
... 5%, 7424 KB, 24905 KB/s, 0 seconds passed
... 5%, 7456 KB, 24874 KB/s, 0 seconds passed
... 5%, 7488 KB, 24952 KB/s, 0 seconds passed
... 5%, 7520 KB, 25031 KB/s, 0 seconds passed
... 5%, 7552 KB, 25110 KB/s, 0 seconds passed
... 6%, 7584 KB, 25190 KB/s, 0 seconds passed
... 6%, 7616 KB, 25270 KB/s, 0 seconds passed
... 6%, 7648 KB, 25349 KB/s, 0 seconds passed
... 6%, 7680 KB, 25428 KB/s, 0 seconds passed
... 6%, 7712 KB, 25507 KB/s, 0 seconds passed
... 6%, 7744 KB, 25587 KB/s, 0 seconds passed
... 6%, 7776 KB, 25664 KB/s, 0 seconds passed
... 6%, 7808 KB, 25742 KB/s, 0 seconds passed
... 6%, 7840 KB, 25820 KB/s, 0 seconds passed
... 6%, 7872 KB, 25899 KB/s, 0 seconds passed
... 6%, 7904 KB, 25974 KB/s, 0 seconds passed
... 6%, 7936 KB, 26051 KB/s, 0 seconds passed
... 6%, 7968 KB, 26129 KB/s, 0 seconds passed
... 6%, 8000 KB, 26206 KB/s, 0 seconds passed
... 6%, 8032 KB, 26282 KB/s, 0 seconds passed
... 6%, 8064 KB, 26363 KB/s, 0 seconds passed
... 6%, 8096 KB, 26445 KB/s, 0 seconds passed
... 6%, 8128 KB, 26528 KB/s, 0 seconds passed
... 6%, 8160 KB, 26599 KB/s, 0 seconds passed
... 6%, 8192 KB, 26675 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 8224 KB, 26751 KB/s, 0 seconds passed
... 6%, 8256 KB, 26825 KB/s, 0 seconds passed
... 6%, 8288 KB, 26899 KB/s, 0 seconds passed
... 6%, 8320 KB, 26974 KB/s, 0 seconds passed
... 6%, 8352 KB, 27051 KB/s, 0 seconds passed
... 6%, 8384 KB, 27072 KB/s, 0 seconds passed
... 6%, 8416 KB, 27151 KB/s, 0 seconds passed
... 6%, 8448 KB, 27227 KB/s, 0 seconds passed
... 6%, 8480 KB, 27307 KB/s, 0 seconds passed
... 6%, 8512 KB, 27386 KB/s, 0 seconds passed
... 6%, 8544 KB, 27466 KB/s, 0 seconds passed
... 6%, 8576 KB, 27519 KB/s, 0 seconds passed
... 6%, 8608 KB, 27593 KB/s, 0 seconds passed
... 6%, 8640 KB, 27672 KB/s, 0 seconds passed
... 6%, 8672 KB, 27751 KB/s, 0 seconds passed
... 6%, 8704 KB, 27830 KB/s, 0 seconds passed
... 6%, 8736 KB, 27909 KB/s, 0 seconds passed
... 6%, 8768 KB, 27982 KB/s, 0 seconds passed
... 6%, 8800 KB, 28061 KB/s, 0 seconds passed
... 7%, 8832 KB, 28139 KB/s, 0 seconds passed
... 7%, 8864 KB, 28212 KB/s, 0 seconds passed
... 7%, 8896 KB, 28290 KB/s, 0 seconds passed
... 7%, 8928 KB, 28368 KB/s, 0 seconds passed
... 7%, 8960 KB, 28446 KB/s, 0 seconds passed
... 7%, 8992 KB, 28524 KB/s, 0 seconds passed
... 7%, 9024 KB, 28596 KB/s, 0 seconds passed
... 7%, 9056 KB, 28674 KB/s, 0 seconds passed
... 7%, 9088 KB, 28751 KB/s, 0 seconds passed
... 7%, 9120 KB, 28828 KB/s, 0 seconds passed
... 7%, 9152 KB, 28900 KB/s, 0 seconds passed
... 7%, 9184 KB, 28980 KB/s, 0 seconds passed
... 7%, 9216 KB, 29054 KB/s, 0 seconds passed
... 7%, 9248 KB, 29130 KB/s, 0 seconds passed
... 7%, 9280 KB, 29202 KB/s, 0 seconds passed
... 7%, 9312 KB, 29278 KB/s, 0 seconds passed
... 7%, 9344 KB, 29354 KB/s, 0 seconds passed
... 7%, 9376 KB, 29426 KB/s, 0 seconds passed
... 7%, 9408 KB, 29501 KB/s, 0 seconds passed
... 7%, 9440 KB, 29576 KB/s, 0 seconds passed
... 7%, 9472 KB, 29651 KB/s, 0 seconds passed
... 7%, 9504 KB, 29724 KB/s, 0 seconds passed
... 7%, 9536 KB, 29799 KB/s, 0 seconds passed
... 7%, 9568 KB, 29874 KB/s, 0 seconds passed
... 7%, 9600 KB, 29945 KB/s, 0 seconds passed
... 7%, 9632 KB, 30020 KB/s, 0 seconds passed
... 7%, 9664 KB, 30095 KB/s, 0 seconds passed
... 7%, 9696 KB, 30169 KB/s, 0 seconds passed
... 7%, 9728 KB, 30239 KB/s, 0 seconds passed
... 7%, 9760 KB, 30314 KB/s, 0 seconds passed
... 7%, 9792 KB, 30389 KB/s, 0 seconds passed
... 7%, 9824 KB, 30463 KB/s, 0 seconds passed
... 7%, 9856 KB, 30532 KB/s, 0 seconds passed
... 7%, 9888 KB, 30606 KB/s, 0 seconds passed
... 7%, 9920 KB, 30681 KB/s, 0 seconds passed
... 7%, 9952 KB, 30750 KB/s, 0 seconds passed
... 7%, 9984 KB, 30823 KB/s, 0 seconds passed
... 7%, 10016 KB, 30892 KB/s, 0 seconds passed
... 7%, 10048 KB, 30951 KB/s, 0 seconds passed
... 8%, 10080 KB, 31028 KB/s, 0 seconds passed
... 8%, 10112 KB, 31110 KB/s, 0 seconds passed
... 8%, 10144 KB, 31184 KB/s, 0 seconds passed
... 8%, 10176 KB, 31253 KB/s, 0 seconds passed
... 8%, 10208 KB, 31330 KB/s, 0 seconds passed
... 8%, 10240 KB, 31398 KB/s, 0 seconds passed
... 8%, 10272 KB, 31471 KB/s, 0 seconds passed
... 8%, 10304 KB, 31538 KB/s, 0 seconds passed
... 8%, 10336 KB, 31611 KB/s, 0 seconds passed
... 8%, 10368 KB, 31683 KB/s, 0 seconds passed
... 8%, 10400 KB, 31750 KB/s, 0 seconds passed
... 8%, 10432 KB, 31822 KB/s, 0 seconds passed
... 8%, 10464 KB, 31894 KB/s, 0 seconds passed
... 8%, 10496 KB, 31961 KB/s, 0 seconds passed
... 8%, 10528 KB, 32038 KB/s, 0 seconds passed
... 8%, 10560 KB, 32104 KB/s, 0 seconds passed
... 8%, 10592 KB, 32176 KB/s, 0 seconds passed
... 8%, 10624 KB, 32242 KB/s, 0 seconds passed
... 8%, 10656 KB, 32314 KB/s, 0 seconds passed
... 8%, 10688 KB, 32385 KB/s, 0 seconds passed
... 8%, 10720 KB, 32451 KB/s, 0 seconds passed
... 8%, 10752 KB, 32522 KB/s, 0 seconds passed
... 8%, 10784 KB, 32593 KB/s, 0 seconds passed
... 8%, 10816 KB, 32659 KB/s, 0 seconds passed
... 8%, 10848 KB, 32729 KB/s, 0 seconds passed
... 8%, 10880 KB, 31678 KB/s, 0 seconds passed
... 8%, 10912 KB, 31724 KB/s, 0 seconds passed
... 8%, 10944 KB, 31753 KB/s, 0 seconds passed
... 8%, 10976 KB, 31807 KB/s, 0 seconds passed
... 8%, 11008 KB, 31866 KB/s, 0 seconds passed
... 8%, 11040 KB, 31891 KB/s, 0 seconds passed
... 8%, 11072 KB, 31947 KB/s, 0 seconds passed
... 8%, 11104 KB, 31989 KB/s, 0 seconds passed
... 8%, 11136 KB, 31997 KB/s, 0 seconds passed
... 8%, 11168 KB, 32051 KB/s, 0 seconds passed
... 8%, 11200 KB, 32108 KB/s, 0 seconds passed
... 8%, 11232 KB, 32164 KB/s, 0 seconds passed
... 8%, 11264 KB, 32222 KB/s, 0 seconds passed
... 8%, 11296 KB, 32280 KB/s, 0 seconds passed
... 8%, 11328 KB, 32337 KB/s, 0 seconds passed
... 9%, 11360 KB, 32394 KB/s, 0 seconds passed
... 9%, 11392 KB, 32448 KB/s, 0 seconds passed
... 9%, 11424 KB, 32505 KB/s, 0 seconds passed
... 9%, 11456 KB, 32563 KB/s, 0 seconds passed
... 9%, 11488 KB, 32624 KB/s, 0 seconds passed
... 9%, 11520 KB, 32681 KB/s, 0 seconds passed
... 9%, 11552 KB, 32737 KB/s, 0 seconds passed
... 9%, 11584 KB, 32793 KB/s, 0 seconds passed
... 9%, 11616 KB, 32848 KB/s, 0 seconds passed
... 9%, 11648 KB, 32905 KB/s, 0 seconds passed
... 9%, 11680 KB, 32961 KB/s, 0 seconds passed
... 9%, 11712 KB, 33017 KB/s, 0 seconds passed
... 9%, 11744 KB, 33072 KB/s, 0 seconds passed
... 9%, 11776 KB, 33127 KB/s, 0 seconds passed
... 9%, 11808 KB, 33182 KB/s, 0 seconds passed
... 9%, 11840 KB, 33232 KB/s, 0 seconds passed
... 9%, 11872 KB, 33286 KB/s, 0 seconds passed
... 9%, 11904 KB, 33340 KB/s, 0 seconds passed
... 9%, 11936 KB, 33394 KB/s, 0 seconds passed
... 9%, 11968 KB, 33447 KB/s, 0 seconds passed
... 9%, 12000 KB, 33502 KB/s, 0 seconds passed

.. parsed-literal::

    ... 9%, 12032 KB, 33557 KB/s, 0 seconds passed
... 9%, 12064 KB, 33611 KB/s, 0 seconds passed
... 9%, 12096 KB, 33664 KB/s, 0 seconds passed
... 9%, 12128 KB, 33719 KB/s, 0 seconds passed
... 9%, 12160 KB, 33770 KB/s, 0 seconds passed
... 9%, 12192 KB, 33832 KB/s, 0 seconds passed
... 9%, 12224 KB, 33897 KB/s, 0 seconds passed
... 9%, 12256 KB, 33963 KB/s, 0 seconds passed
... 9%, 12288 KB, 34028 KB/s, 0 seconds passed
... 9%, 12320 KB, 34093 KB/s, 0 seconds passed
... 9%, 12352 KB, 34158 KB/s, 0 seconds passed
... 9%, 12384 KB, 34224 KB/s, 0 seconds passed
... 9%, 12416 KB, 34290 KB/s, 0 seconds passed
... 9%, 12448 KB, 34354 KB/s, 0 seconds passed
... 9%, 12480 KB, 34419 KB/s, 0 seconds passed
... 9%, 12512 KB, 34485 KB/s, 0 seconds passed
... 9%, 12544 KB, 34550 KB/s, 0 seconds passed
... 9%, 12576 KB, 34615 KB/s, 0 seconds passed
... 10%, 12608 KB, 34680 KB/s, 0 seconds passed
... 10%, 12640 KB, 34745 KB/s, 0 seconds passed
... 10%, 12672 KB, 34808 KB/s, 0 seconds passed
... 10%, 12704 KB, 34871 KB/s, 0 seconds passed
... 10%, 12736 KB, 34936 KB/s, 0 seconds passed
... 10%, 12768 KB, 35001 KB/s, 0 seconds passed
... 10%, 12800 KB, 35065 KB/s, 0 seconds passed
... 10%, 12832 KB, 35129 KB/s, 0 seconds passed
... 10%, 12864 KB, 35194 KB/s, 0 seconds passed
... 10%, 12896 KB, 35257 KB/s, 0 seconds passed
... 10%, 12928 KB, 35319 KB/s, 0 seconds passed
... 10%, 12960 KB, 35383 KB/s, 0 seconds passed
... 10%, 12992 KB, 35447 KB/s, 0 seconds passed
... 10%, 13024 KB, 35512 KB/s, 0 seconds passed
... 10%, 13056 KB, 35575 KB/s, 0 seconds passed
... 10%, 13088 KB, 35638 KB/s, 0 seconds passed
... 10%, 13120 KB, 35702 KB/s, 0 seconds passed
... 10%, 13152 KB, 35765 KB/s, 0 seconds passed
... 10%, 13184 KB, 35828 KB/s, 0 seconds passed
... 10%, 13216 KB, 35891 KB/s, 0 seconds passed
... 10%, 13248 KB, 35954 KB/s, 0 seconds passed
... 10%, 13280 KB, 36017 KB/s, 0 seconds passed
... 10%, 13312 KB, 36080 KB/s, 0 seconds passed
... 10%, 13344 KB, 36143 KB/s, 0 seconds passed
... 10%, 13376 KB, 36206 KB/s, 0 seconds passed
... 10%, 13408 KB, 36269 KB/s, 0 seconds passed
... 10%, 13440 KB, 36331 KB/s, 0 seconds passed
... 10%, 13472 KB, 36393 KB/s, 0 seconds passed
... 10%, 13504 KB, 36463 KB/s, 0 seconds passed
... 10%, 13536 KB, 36532 KB/s, 0 seconds passed
... 10%, 13568 KB, 36602 KB/s, 0 seconds passed
... 10%, 13600 KB, 36671 KB/s, 0 seconds passed
... 10%, 13632 KB, 36740 KB/s, 0 seconds passed
... 10%, 13664 KB, 36810 KB/s, 0 seconds passed
... 10%, 13696 KB, 36880 KB/s, 0 seconds passed
... 10%, 13728 KB, 36949 KB/s, 0 seconds passed
... 10%, 13760 KB, 37019 KB/s, 0 seconds passed
... 10%, 13792 KB, 37089 KB/s, 0 seconds passed
... 10%, 13824 KB, 37157 KB/s, 0 seconds passed
... 11%, 13856 KB, 37226 KB/s, 0 seconds passed
... 11%, 13888 KB, 37287 KB/s, 0 seconds passed
... 11%, 13920 KB, 37346 KB/s, 0 seconds passed
... 11%, 13952 KB, 37400 KB/s, 0 seconds passed
... 11%, 13984 KB, 37458 KB/s, 0 seconds passed
... 11%, 14016 KB, 37517 KB/s, 0 seconds passed
... 11%, 14048 KB, 37571 KB/s, 0 seconds passed
... 11%, 14080 KB, 37631 KB/s, 0 seconds passed
... 11%, 14112 KB, 37690 KB/s, 0 seconds passed
... 11%, 14144 KB, 37749 KB/s, 0 seconds passed
... 11%, 14176 KB, 37802 KB/s, 0 seconds passed
... 11%, 14208 KB, 37861 KB/s, 0 seconds passed
... 11%, 14240 KB, 37920 KB/s, 0 seconds passed
... 11%, 14272 KB, 37978 KB/s, 0 seconds passed
... 11%, 14304 KB, 38031 KB/s, 0 seconds passed
... 11%, 14336 KB, 38090 KB/s, 0 seconds passed
... 11%, 14368 KB, 38143 KB/s, 0 seconds passed
... 11%, 14400 KB, 38201 KB/s, 0 seconds passed
... 11%, 14432 KB, 38259 KB/s, 0 seconds passed
... 11%, 14464 KB, 38312 KB/s, 0 seconds passed
... 11%, 14496 KB, 38370 KB/s, 0 seconds passed
... 11%, 14528 KB, 38428 KB/s, 0 seconds passed
... 11%, 14560 KB, 38486 KB/s, 0 seconds passed
... 11%, 14592 KB, 38539 KB/s, 0 seconds passed
... 11%, 14624 KB, 38596 KB/s, 0 seconds passed
... 11%, 14656 KB, 38654 KB/s, 0 seconds passed
... 11%, 14688 KB, 38706 KB/s, 0 seconds passed
... 11%, 14720 KB, 38763 KB/s, 0 seconds passed
... 11%, 14752 KB, 38820 KB/s, 0 seconds passed
... 11%, 14784 KB, 38883 KB/s, 0 seconds passed
... 11%, 14816 KB, 38930 KB/s, 0 seconds passed
... 11%, 14848 KB, 38987 KB/s, 0 seconds passed
... 11%, 14880 KB, 39044 KB/s, 0 seconds passed
... 11%, 14912 KB, 39096 KB/s, 0 seconds passed
... 11%, 14944 KB, 39153 KB/s, 0 seconds passed
... 11%, 14976 KB, 39209 KB/s, 0 seconds passed
... 11%, 15008 KB, 39261 KB/s, 0 seconds passed
... 11%, 15040 KB, 39318 KB/s, 0 seconds passed
... 11%, 15072 KB, 39374 KB/s, 0 seconds passed
... 11%, 15104 KB, 39425 KB/s, 0 seconds passed
... 12%, 15136 KB, 39482 KB/s, 0 seconds passed
... 12%, 15168 KB, 39538 KB/s, 0 seconds passed
... 12%, 15200 KB, 39589 KB/s, 0 seconds passed
... 12%, 15232 KB, 39645 KB/s, 0 seconds passed
... 12%, 15264 KB, 39702 KB/s, 0 seconds passed
... 12%, 15296 KB, 39758 KB/s, 0 seconds passed
... 12%, 15328 KB, 39808 KB/s, 0 seconds passed
... 12%, 15360 KB, 39864 KB/s, 0 seconds passed
... 12%, 15392 KB, 39920 KB/s, 0 seconds passed
... 12%, 15424 KB, 39970 KB/s, 0 seconds passed
... 12%, 15456 KB, 40026 KB/s, 0 seconds passed
... 12%, 15488 KB, 40082 KB/s, 0 seconds passed
... 12%, 15520 KB, 40137 KB/s, 0 seconds passed
... 12%, 15552 KB, 40187 KB/s, 0 seconds passed
... 12%, 15584 KB, 40243 KB/s, 0 seconds passed
... 12%, 15616 KB, 40292 KB/s, 0 seconds passed
... 12%, 15648 KB, 40348 KB/s, 0 seconds passed
... 12%, 15680 KB, 40403 KB/s, 0 seconds passed
... 12%, 15712 KB, 40458 KB/s, 0 seconds passed
... 12%, 15744 KB, 40508 KB/s, 0 seconds passed
... 12%, 15776 KB, 40563 KB/s, 0 seconds passed
... 12%, 15808 KB, 40612 KB/s, 0 seconds passed
... 12%, 15840 KB, 40667 KB/s, 0 seconds passed
... 12%, 15872 KB, 40722 KB/s, 0 seconds passed
... 12%, 15904 KB, 40776 KB/s, 0 seconds passed
... 12%, 15936 KB, 40825 KB/s, 0 seconds passed
... 12%, 15968 KB, 40880 KB/s, 0 seconds passed
... 12%, 16000 KB, 40934 KB/s, 0 seconds passed
... 12%, 16032 KB, 40983 KB/s, 0 seconds passed
... 12%, 16064 KB, 41037 KB/s, 0 seconds passed
... 12%, 16096 KB, 41092 KB/s, 0 seconds passed
... 12%, 16128 KB, 41140 KB/s, 0 seconds passed
... 12%, 16160 KB, 41195 KB/s, 0 seconds passed
... 12%, 16192 KB, 41248 KB/s, 0 seconds passed
... 12%, 16224 KB, 41297 KB/s, 0 seconds passed
... 12%, 16256 KB, 41351 KB/s, 0 seconds passed
... 12%, 16288 KB, 41404 KB/s, 0 seconds passed
... 12%, 16320 KB, 41453 KB/s, 0 seconds passed
... 12%, 16352 KB, 41518 KB/s, 0 seconds passed
... 13%, 16384 KB, 41566 KB/s, 0 seconds passed
... 13%, 16416 KB, 41619 KB/s, 0 seconds passed
... 13%, 16448 KB, 41673 KB/s, 0 seconds passed
... 13%, 16480 KB, 41726 KB/s, 0 seconds passed
... 13%, 16512 KB, 41762 KB/s, 0 seconds passed
... 13%, 16544 KB, 41800 KB/s, 0 seconds passed
... 13%, 16576 KB, 41859 KB/s, 0 seconds passed
... 13%, 16608 KB, 41920 KB/s, 0 seconds passed
... 13%, 16640 KB, 41967 KB/s, 0 seconds passed
... 13%, 16672 KB, 42015 KB/s, 0 seconds passed
... 13%, 16704 KB, 42052 KB/s, 0 seconds passed
... 13%, 16736 KB, 42099 KB/s, 0 seconds passed
... 13%, 16768 KB, 42166 KB/s, 0 seconds passed
... 13%, 16800 KB, 42223 KB/s, 0 seconds passed
... 13%, 16832 KB, 42276 KB/s, 0 seconds passed
... 13%, 16864 KB, 42323 KB/s, 0 seconds passed
... 13%, 16896 KB, 42381 KB/s, 0 seconds passed
... 13%, 16928 KB, 42427 KB/s, 0 seconds passed
... 13%, 16960 KB, 42474 KB/s, 0 seconds passed
... 13%, 16992 KB, 42511 KB/s, 0 seconds passed
... 13%, 17024 KB, 42569 KB/s, 0 seconds passed
... 13%, 17056 KB, 42629 KB/s, 0 seconds passed
... 13%, 17088 KB, 42681 KB/s, 0 seconds passed
... 13%, 17120 KB, 42728 KB/s, 0 seconds passed
... 13%, 17152 KB, 42780 KB/s, 0 seconds passed
... 13%, 17184 KB, 42831 KB/s, 0 seconds passed
... 13%, 17216 KB, 42883 KB/s, 0 seconds passed
... 13%, 17248 KB, 42929 KB/s, 0 seconds passed
... 13%, 17280 KB, 42981 KB/s, 0 seconds passed
... 13%, 17312 KB, 43027 KB/s, 0 seconds passed
... 13%, 17344 KB, 43073 KB/s, 0 seconds passed
... 13%, 17376 KB, 43107 KB/s, 0 seconds passed
... 13%, 17408 KB, 43143 KB/s, 0 seconds passed
... 13%, 17440 KB, 43190 KB/s, 0 seconds passed
... 13%, 17472 KB, 43255 KB/s, 0 seconds passed
... 13%, 17504 KB, 43320 KB/s, 0 seconds passed
... 13%, 17536 KB, 43377 KB/s, 0 seconds passed
... 13%, 17568 KB, 43422 KB/s, 0 seconds passed
... 13%, 17600 KB, 43473 KB/s, 0 seconds passed
... 13%, 17632 KB, 43524 KB/s, 0 seconds passed
... 14%, 17664 KB, 43574 KB/s, 0 seconds passed
... 14%, 17696 KB, 43620 KB/s, 0 seconds passed
... 14%, 17728 KB, 43670 KB/s, 0 seconds passed
... 14%, 17760 KB, 43721 KB/s, 0 seconds passed
... 14%, 17792 KB, 43766 KB/s, 0 seconds passed
... 14%, 17824 KB, 43816 KB/s, 0 seconds passed
... 14%, 17856 KB, 43867 KB/s, 0 seconds passed
... 14%, 17888 KB, 43912 KB/s, 0 seconds passed
... 14%, 17920 KB, 43962 KB/s, 0 seconds passed
... 14%, 17952 KB, 44007 KB/s, 0 seconds passed
... 14%, 17984 KB, 44057 KB/s, 0 seconds passed
... 14%, 18016 KB, 44097 KB/s, 0 seconds passed
... 14%, 18048 KB, 44132 KB/s, 0 seconds passed
... 14%, 18080 KB, 44167 KB/s, 0 seconds passed
... 14%, 18112 KB, 44201 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 18144 KB, 44263 KB/s, 0 seconds passed
... 14%, 18176 KB, 44314 KB/s, 0 seconds passed
... 14%, 18208 KB, 44354 KB/s, 0 seconds passed
... 14%, 18240 KB, 44394 KB/s, 0 seconds passed
... 14%, 18272 KB, 44433 KB/s, 0 seconds passed
... 14%, 18304 KB, 44473 KB/s, 0 seconds passed
... 14%, 18336 KB, 44508 KB/s, 0 seconds passed
... 14%, 18368 KB, 44545 KB/s, 0 seconds passed
... 14%, 18400 KB, 44583 KB/s, 0 seconds passed
... 14%, 18432 KB, 44623 KB/s, 0 seconds passed
... 14%, 18464 KB, 44662 KB/s, 0 seconds passed
... 14%, 18496 KB, 44701 KB/s, 0 seconds passed
... 14%, 18528 KB, 44736 KB/s, 0 seconds passed
... 14%, 18560 KB, 44774 KB/s, 0 seconds passed
... 14%, 18592 KB, 44812 KB/s, 0 seconds passed
... 14%, 18624 KB, 44849 KB/s, 0 seconds passed
... 14%, 18656 KB, 44886 KB/s, 0 seconds passed
... 14%, 18688 KB, 44923 KB/s, 0 seconds passed
... 14%, 18720 KB, 44961 KB/s, 0 seconds passed
... 14%, 18752 KB, 44998 KB/s, 0 seconds passed
... 14%, 18784 KB, 45037 KB/s, 0 seconds passed
... 14%, 18816 KB, 45074 KB/s, 0 seconds passed
... 14%, 18848 KB, 45111 KB/s, 0 seconds passed
... 14%, 18880 KB, 45149 KB/s, 0 seconds passed
... 15%, 18912 KB, 45188 KB/s, 0 seconds passed
... 15%, 18944 KB, 45226 KB/s, 0 seconds passed
... 15%, 18976 KB, 45263 KB/s, 0 seconds passed
... 15%, 19008 KB, 45300 KB/s, 0 seconds passed
... 15%, 19040 KB, 45337 KB/s, 0 seconds passed
... 15%, 19072 KB, 45379 KB/s, 0 seconds passed
... 15%, 19104 KB, 45431 KB/s, 0 seconds passed
... 15%, 19136 KB, 45482 KB/s, 0 seconds passed
... 15%, 19168 KB, 45532 KB/s, 0 seconds passed
... 15%, 19200 KB, 45584 KB/s, 0 seconds passed
... 15%, 19232 KB, 45634 KB/s, 0 seconds passed
... 15%, 19264 KB, 45685 KB/s, 0 seconds passed
... 15%, 19296 KB, 45737 KB/s, 0 seconds passed
... 15%, 19328 KB, 45786 KB/s, 0 seconds passed
... 15%, 19360 KB, 45837 KB/s, 0 seconds passed
... 15%, 19392 KB, 45888 KB/s, 0 seconds passed
... 15%, 19424 KB, 45937 KB/s, 0 seconds passed
... 15%, 19456 KB, 45988 KB/s, 0 seconds passed
... 15%, 19488 KB, 46038 KB/s, 0 seconds passed
... 15%, 19520 KB, 46088 KB/s, 0 seconds passed
... 15%, 19552 KB, 46138 KB/s, 0 seconds passed
... 15%, 19584 KB, 46187 KB/s, 0 seconds passed
... 15%, 19616 KB, 46236 KB/s, 0 seconds passed
... 15%, 19648 KB, 46285 KB/s, 0 seconds passed
... 15%, 19680 KB, 46335 KB/s, 0 seconds passed
... 15%, 19712 KB, 46384 KB/s, 0 seconds passed
... 15%, 19744 KB, 46434 KB/s, 0 seconds passed
... 15%, 19776 KB, 46484 KB/s, 0 seconds passed
... 15%, 19808 KB, 46534 KB/s, 0 seconds passed
... 15%, 19840 KB, 46582 KB/s, 0 seconds passed
... 15%, 19872 KB, 46631 KB/s, 0 seconds passed
... 15%, 19904 KB, 46680 KB/s, 0 seconds passed
... 15%, 19936 KB, 46729 KB/s, 0 seconds passed
... 15%, 19968 KB, 46778 KB/s, 0 seconds passed
... 15%, 20000 KB, 46827 KB/s, 0 seconds passed
... 15%, 20032 KB, 46876 KB/s, 0 seconds passed
... 15%, 20064 KB, 46925 KB/s, 0 seconds passed
... 15%, 20096 KB, 46973 KB/s, 0 seconds passed
... 15%, 20128 KB, 47021 KB/s, 0 seconds passed
... 16%, 20160 KB, 47069 KB/s, 0 seconds passed
... 16%, 20192 KB, 47119 KB/s, 0 seconds passed
... 16%, 20224 KB, 47167 KB/s, 0 seconds passed
... 16%, 20256 KB, 47217 KB/s, 0 seconds passed
... 16%, 20288 KB, 47265 KB/s, 0 seconds passed
... 16%, 20320 KB, 47314 KB/s, 0 seconds passed
... 16%, 20352 KB, 47362 KB/s, 0 seconds passed
... 16%, 20384 KB, 47411 KB/s, 0 seconds passed
... 16%, 20416 KB, 47463 KB/s, 0 seconds passed
... 16%, 20448 KB, 47516 KB/s, 0 seconds passed
... 16%, 20480 KB, 47577 KB/s, 0 seconds passed
... 16%, 20512 KB, 47522 KB/s, 0 seconds passed
... 16%, 20544 KB, 47548 KB/s, 0 seconds passed
... 16%, 20576 KB, 47577 KB/s, 0 seconds passed
... 16%, 20608 KB, 47615 KB/s, 0 seconds passed
... 16%, 20640 KB, 47672 KB/s, 0 seconds passed
... 16%, 20672 KB, 47728 KB/s, 0 seconds passed
... 16%, 20704 KB, 47770 KB/s, 0 seconds passed
... 16%, 20736 KB, 47800 KB/s, 0 seconds passed
... 16%, 20768 KB, 47831 KB/s, 0 seconds passed
... 16%, 20800 KB, 47861 KB/s, 0 seconds passed
... 16%, 20832 KB, 47898 KB/s, 0 seconds passed
... 16%, 20864 KB, 47954 KB/s, 0 seconds passed
... 16%, 20896 KB, 48011 KB/s, 0 seconds passed
... 16%, 20928 KB, 48067 KB/s, 0 seconds passed
... 16%, 20960 KB, 48124 KB/s, 0 seconds passed
... 16%, 20992 KB, 48168 KB/s, 0 seconds passed
... 16%, 21024 KB, 48206 KB/s, 0 seconds passed
... 16%, 21056 KB, 48251 KB/s, 0 seconds passed
... 16%, 21088 KB, 48295 KB/s, 0 seconds passed
... 16%, 21120 KB, 48334 KB/s, 0 seconds passed
... 16%, 21152 KB, 48378 KB/s, 0 seconds passed
... 16%, 21184 KB, 48422 KB/s, 0 seconds passed
... 16%, 21216 KB, 48460 KB/s, 0 seconds passed
... 16%, 21248 KB, 48505 KB/s, 0 seconds passed
... 16%, 21280 KB, 48543 KB/s, 0 seconds passed
... 16%, 21312 KB, 48587 KB/s, 0 seconds passed
... 16%, 21344 KB, 48630 KB/s, 0 seconds passed
... 16%, 21376 KB, 48674 KB/s, 0 seconds passed
... 16%, 21408 KB, 48712 KB/s, 0 seconds passed
... 17%, 21440 KB, 48756 KB/s, 0 seconds passed
... 17%, 21472 KB, 48799 KB/s, 0 seconds passed
... 17%, 21504 KB, 48843 KB/s, 0 seconds passed
... 17%, 21536 KB, 48881 KB/s, 0 seconds passed
... 17%, 21568 KB, 48925 KB/s, 0 seconds passed
... 17%, 21600 KB, 48962 KB/s, 0 seconds passed
... 17%, 21632 KB, 49006 KB/s, 0 seconds passed
... 17%, 21664 KB, 49043 KB/s, 0 seconds passed
... 17%, 21696 KB, 49086 KB/s, 0 seconds passed
... 17%, 21728 KB, 49113 KB/s, 0 seconds passed
... 17%, 21760 KB, 49161 KB/s, 0 seconds passed
... 17%, 21792 KB, 49214 KB/s, 0 seconds passed
... 17%, 21824 KB, 49252 KB/s, 0 seconds passed
... 17%, 21856 KB, 49295 KB/s, 0 seconds passed
... 17%, 21888 KB, 49338 KB/s, 0 seconds passed
... 17%, 21920 KB, 49375 KB/s, 0 seconds passed
... 17%, 21952 KB, 49418 KB/s, 0 seconds passed
... 17%, 21984 KB, 49461 KB/s, 0 seconds passed
... 17%, 22016 KB, 49504 KB/s, 0 seconds passed
... 17%, 22048 KB, 49541 KB/s, 0 seconds passed
... 17%, 22080 KB, 49584 KB/s, 0 seconds passed
... 17%, 22112 KB, 49620 KB/s, 0 seconds passed
... 17%, 22144 KB, 49663 KB/s, 0 seconds passed
... 17%, 22176 KB, 49706 KB/s, 0 seconds passed
... 17%, 22208 KB, 49742 KB/s, 0 seconds passed
... 17%, 22240 KB, 49785 KB/s, 0 seconds passed
... 17%, 22272 KB, 49827 KB/s, 0 seconds passed
... 17%, 22304 KB, 49870 KB/s, 0 seconds passed
... 17%, 22336 KB, 49904 KB/s, 0 seconds passed
... 17%, 22368 KB, 49947 KB/s, 0 seconds passed
... 17%, 22400 KB, 49989 KB/s, 0 seconds passed
... 17%, 22432 KB, 50031 KB/s, 0 seconds passed
... 17%, 22464 KB, 50061 KB/s, 0 seconds passed
... 17%, 22496 KB, 50086 KB/s, 0 seconds passed
... 17%, 22528 KB, 50136 KB/s, 0 seconds passed
... 17%, 22560 KB, 50191 KB/s, 0 seconds passed
... 17%, 22592 KB, 50219 KB/s, 0 seconds passed
... 17%, 22624 KB, 50254 KB/s, 0 seconds passed
... 17%, 22656 KB, 50308 KB/s, 0 seconds passed
... 18%, 22688 KB, 50349 KB/s, 0 seconds passed
... 18%, 22720 KB, 50386 KB/s, 0 seconds passed
... 18%, 22752 KB, 50428 KB/s, 0 seconds passed
... 18%, 22784 KB, 50469 KB/s, 0 seconds passed
... 18%, 22816 KB, 50505 KB/s, 0 seconds passed
... 18%, 22848 KB, 50547 KB/s, 0 seconds passed
... 18%, 22880 KB, 50588 KB/s, 0 seconds passed
... 18%, 22912 KB, 50624 KB/s, 0 seconds passed
... 18%, 22944 KB, 50666 KB/s, 0 seconds passed
... 18%, 22976 KB, 50701 KB/s, 0 seconds passed
... 18%, 23008 KB, 50743 KB/s, 0 seconds passed
... 18%, 23040 KB, 50784 KB/s, 0 seconds passed
... 18%, 23072 KB, 50825 KB/s, 0 seconds passed
... 18%, 23104 KB, 50861 KB/s, 0 seconds passed
... 18%, 23136 KB, 50896 KB/s, 0 seconds passed
... 18%, 23168 KB, 50920 KB/s, 0 seconds passed
... 18%, 23200 KB, 50946 KB/s, 0 seconds passed
... 18%, 23232 KB, 50981 KB/s, 0 seconds passed
... 18%, 23264 KB, 51038 KB/s, 0 seconds passed
... 18%, 23296 KB, 51094 KB/s, 0 seconds passed
... 18%, 23328 KB, 51133 KB/s, 0 seconds passed
... 18%, 23360 KB, 51174 KB/s, 0 seconds passed
... 18%, 23392 KB, 51214 KB/s, 0 seconds passed
... 18%, 23424 KB, 51255 KB/s, 0 seconds passed
... 18%, 23456 KB, 51290 KB/s, 0 seconds passed
... 18%, 23488 KB, 51331 KB/s, 0 seconds passed
... 18%, 23520 KB, 51365 KB/s, 0 seconds passed
... 18%, 23552 KB, 51406 KB/s, 0 seconds passed
... 18%, 23584 KB, 51446 KB/s, 0 seconds passed
... 18%, 23616 KB, 51487 KB/s, 0 seconds passed
... 18%, 23648 KB, 51521 KB/s, 0 seconds passed
... 18%, 23680 KB, 51562 KB/s, 0 seconds passed
... 18%, 23712 KB, 51596 KB/s, 0 seconds passed
... 18%, 23744 KB, 51637 KB/s, 0 seconds passed
... 18%, 23776 KB, 51677 KB/s, 0 seconds passed
... 18%, 23808 KB, 51717 KB/s, 0 seconds passed
... 18%, 23840 KB, 51751 KB/s, 0 seconds passed
... 18%, 23872 KB, 51792 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 23904 KB, 51832 KB/s, 0 seconds passed
... 19%, 23936 KB, 51866 KB/s, 0 seconds passed
... 19%, 23968 KB, 51906 KB/s, 0 seconds passed
... 19%, 24000 KB, 51940 KB/s, 0 seconds passed
... 19%, 24032 KB, 51976 KB/s, 0 seconds passed
... 19%, 24064 KB, 52018 KB/s, 0 seconds passed
... 19%, 24096 KB, 52051 KB/s, 0 seconds passed
... 19%, 24128 KB, 52087 KB/s, 0 seconds passed
... 19%, 24160 KB, 52126 KB/s, 0 seconds passed
... 19%, 24192 KB, 52163 KB/s, 0 seconds passed
... 19%, 24224 KB, 52207 KB/s, 0 seconds passed
... 19%, 24256 KB, 52247 KB/s, 0 seconds passed
... 19%, 24288 KB, 52281 KB/s, 0 seconds passed
... 19%, 24320 KB, 52321 KB/s, 0 seconds passed
... 19%, 24352 KB, 52360 KB/s, 0 seconds passed
... 19%, 24384 KB, 52394 KB/s, 0 seconds passed
... 19%, 24416 KB, 52433 KB/s, 0 seconds passed
... 19%, 24448 KB, 52467 KB/s, 0 seconds passed
... 19%, 24480 KB, 52506 KB/s, 0 seconds passed
... 19%, 24512 KB, 52540 KB/s, 0 seconds passed
... 19%, 24544 KB, 52579 KB/s, 0 seconds passed
... 19%, 24576 KB, 52618 KB/s, 0 seconds passed
... 19%, 24608 KB, 52656 KB/s, 0 seconds passed
... 19%, 24640 KB, 52689 KB/s, 0 seconds passed
... 19%, 24672 KB, 52728 KB/s, 0 seconds passed
... 19%, 24704 KB, 52761 KB/s, 0 seconds passed
... 19%, 24736 KB, 52800 KB/s, 0 seconds passed
... 19%, 24768 KB, 52839 KB/s, 0 seconds passed
... 19%, 24800 KB, 52872 KB/s, 0 seconds passed
... 19%, 24832 KB, 52910 KB/s, 0 seconds passed
... 19%, 24864 KB, 52949 KB/s, 0 seconds passed
... 19%, 24896 KB, 52988 KB/s, 0 seconds passed
... 19%, 24928 KB, 53019 KB/s, 0 seconds passed
... 19%, 24960 KB, 53058 KB/s, 0 seconds passed
... 19%, 24992 KB, 53096 KB/s, 0 seconds passed
... 19%, 25024 KB, 53129 KB/s, 0 seconds passed
... 19%, 25056 KB, 53167 KB/s, 0 seconds passed
... 19%, 25088 KB, 53199 KB/s, 0 seconds passed
... 19%, 25120 KB, 53238 KB/s, 0 seconds passed
... 19%, 25152 KB, 53276 KB/s, 0 seconds passed
... 19%, 25184 KB, 53314 KB/s, 0 seconds passed
... 20%, 25216 KB, 53346 KB/s, 0 seconds passed
... 20%, 25248 KB, 53385 KB/s, 0 seconds passed
... 20%, 25280 KB, 53417 KB/s, 0 seconds passed
... 20%, 25312 KB, 53455 KB/s, 0 seconds passed
... 20%, 25344 KB, 53493 KB/s, 0 seconds passed
... 20%, 25376 KB, 53531 KB/s, 0 seconds passed
... 20%, 25408 KB, 53563 KB/s, 0 seconds passed
... 20%, 25440 KB, 53601 KB/s, 0 seconds passed
... 20%, 25472 KB, 53639 KB/s, 0 seconds passed
... 20%, 25504 KB, 53670 KB/s, 0 seconds passed
... 20%, 25536 KB, 53708 KB/s, 0 seconds passed
... 20%, 25568 KB, 53741 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 25600 KB, 49582 KB/s, 0 seconds passed
... 20%, 25632 KB, 49523 KB/s, 0 seconds passed
... 20%, 25664 KB, 49544 KB/s, 0 seconds passed
... 20%, 25696 KB, 49569 KB/s, 0 seconds passed
... 20%, 25728 KB, 49597 KB/s, 0 seconds passed
... 20%, 25760 KB, 49621 KB/s, 0 seconds passed
... 20%, 25792 KB, 49647 KB/s, 0 seconds passed
... 20%, 25824 KB, 49674 KB/s, 0 seconds passed
... 20%, 25856 KB, 49699 KB/s, 0 seconds passed
... 20%, 25888 KB, 49722 KB/s, 0 seconds passed
... 20%, 25920 KB, 49748 KB/s, 0 seconds passed
... 20%, 25952 KB, 49775 KB/s, 0 seconds passed
... 20%, 25984 KB, 49800 KB/s, 0 seconds passed
... 20%, 26016 KB, 49828 KB/s, 0 seconds passed
... 20%, 26048 KB, 49854 KB/s, 0 seconds passed
... 20%, 26080 KB, 49881 KB/s, 0 seconds passed
... 20%, 26112 KB, 49907 KB/s, 0 seconds passed
... 20%, 26144 KB, 49930 KB/s, 0 seconds passed
... 20%, 26176 KB, 49956 KB/s, 0 seconds passed
... 20%, 26208 KB, 49979 KB/s, 0 seconds passed
... 20%, 26240 KB, 50003 KB/s, 0 seconds passed
... 20%, 26272 KB, 50030 KB/s, 0 seconds passed
... 20%, 26304 KB, 50055 KB/s, 0 seconds passed
... 20%, 26336 KB, 50080 KB/s, 0 seconds passed
... 20%, 26368 KB, 50104 KB/s, 0 seconds passed
... 20%, 26400 KB, 50129 KB/s, 0 seconds passed
... 20%, 26432 KB, 50156 KB/s, 0 seconds passed
... 21%, 26464 KB, 50181 KB/s, 0 seconds passed
... 21%, 26496 KB, 50206 KB/s, 0 seconds passed
... 21%, 26528 KB, 50227 KB/s, 0 seconds passed
... 21%, 26560 KB, 50252 KB/s, 0 seconds passed
... 21%, 26592 KB, 50276 KB/s, 0 seconds passed
... 21%, 26624 KB, 50299 KB/s, 0 seconds passed
... 21%, 26656 KB, 50323 KB/s, 0 seconds passed
... 21%, 26688 KB, 50346 KB/s, 0 seconds passed
... 21%, 26720 KB, 50370 KB/s, 0 seconds passed
... 21%, 26752 KB, 50392 KB/s, 0 seconds passed
... 21%, 26784 KB, 50415 KB/s, 0 seconds passed
... 21%, 26816 KB, 50452 KB/s, 0 seconds passed
... 21%, 26848 KB, 50489 KB/s, 0 seconds passed
... 21%, 26880 KB, 50522 KB/s, 0 seconds passed
... 21%, 26912 KB, 50559 KB/s, 0 seconds passed
... 21%, 26944 KB, 50597 KB/s, 0 seconds passed
... 21%, 26976 KB, 50634 KB/s, 0 seconds passed
... 21%, 27008 KB, 50671 KB/s, 0 seconds passed
... 21%, 27040 KB, 50707 KB/s, 0 seconds passed
... 21%, 27072 KB, 50744 KB/s, 0 seconds passed
... 21%, 27104 KB, 50780 KB/s, 0 seconds passed
... 21%, 27136 KB, 50818 KB/s, 0 seconds passed
... 21%, 27168 KB, 50855 KB/s, 0 seconds passed
... 21%, 27200 KB, 50890 KB/s, 0 seconds passed
... 21%, 27232 KB, 50926 KB/s, 0 seconds passed
... 21%, 27264 KB, 50963 KB/s, 0 seconds passed
... 21%, 27296 KB, 50999 KB/s, 0 seconds passed
... 21%, 27328 KB, 51035 KB/s, 0 seconds passed
... 21%, 27360 KB, 51073 KB/s, 0 seconds passed
... 21%, 27392 KB, 51110 KB/s, 0 seconds passed
... 21%, 27424 KB, 51143 KB/s, 0 seconds passed
... 21%, 27456 KB, 51179 KB/s, 0 seconds passed
... 21%, 27488 KB, 51216 KB/s, 0 seconds passed
... 21%, 27520 KB, 51249 KB/s, 0 seconds passed
... 21%, 27552 KB, 51286 KB/s, 0 seconds passed
... 21%, 27584 KB, 51323 KB/s, 0 seconds passed
... 21%, 27616 KB, 51359 KB/s, 0 seconds passed
... 21%, 27648 KB, 51396 KB/s, 0 seconds passed
... 21%, 27680 KB, 51433 KB/s, 0 seconds passed
... 22%, 27712 KB, 51469 KB/s, 0 seconds passed
... 22%, 27744 KB, 51506 KB/s, 0 seconds passed
... 22%, 27776 KB, 51541 KB/s, 0 seconds passed
... 22%, 27808 KB, 51577 KB/s, 0 seconds passed
... 22%, 27840 KB, 51613 KB/s, 0 seconds passed
... 22%, 27872 KB, 51649 KB/s, 0 seconds passed
... 22%, 27904 KB, 51684 KB/s, 0 seconds passed
... 22%, 27936 KB, 51719 KB/s, 0 seconds passed
... 22%, 27968 KB, 51752 KB/s, 0 seconds passed
... 22%, 28000 KB, 51788 KB/s, 0 seconds passed
... 22%, 28032 KB, 51825 KB/s, 0 seconds passed
... 22%, 28064 KB, 51860 KB/s, 0 seconds passed
... 22%, 28096 KB, 51895 KB/s, 0 seconds passed
... 22%, 28128 KB, 51935 KB/s, 0 seconds passed
... 22%, 28160 KB, 51977 KB/s, 0 seconds passed
... 22%, 28192 KB, 52021 KB/s, 0 seconds passed
... 22%, 28224 KB, 52064 KB/s, 0 seconds passed
... 22%, 28256 KB, 52107 KB/s, 0 seconds passed
... 22%, 28288 KB, 52151 KB/s, 0 seconds passed
... 22%, 28320 KB, 52193 KB/s, 0 seconds passed
... 22%, 28352 KB, 52235 KB/s, 0 seconds passed
... 22%, 28384 KB, 52280 KB/s, 0 seconds passed
... 22%, 28416 KB, 52321 KB/s, 0 seconds passed
... 22%, 28448 KB, 52364 KB/s, 0 seconds passed
... 22%, 28480 KB, 52408 KB/s, 0 seconds passed
... 22%, 28512 KB, 52451 KB/s, 0 seconds passed
... 22%, 28544 KB, 52494 KB/s, 0 seconds passed
... 22%, 28576 KB, 52537 KB/s, 0 seconds passed
... 22%, 28608 KB, 52578 KB/s, 0 seconds passed
... 22%, 28640 KB, 52621 KB/s, 0 seconds passed
... 22%, 28672 KB, 52663 KB/s, 0 seconds passed
... 22%, 28704 KB, 52706 KB/s, 0 seconds passed
... 22%, 28736 KB, 52750 KB/s, 0 seconds passed
... 22%, 28768 KB, 52793 KB/s, 0 seconds passed
... 22%, 28800 KB, 52822 KB/s, 0 seconds passed
... 22%, 28832 KB, 52856 KB/s, 0 seconds passed
... 22%, 28864 KB, 52889 KB/s, 0 seconds passed
... 22%, 28896 KB, 52922 KB/s, 0 seconds passed
... 22%, 28928 KB, 52950 KB/s, 0 seconds passed
... 22%, 28960 KB, 52983 KB/s, 0 seconds passed
... 23%, 28992 KB, 53011 KB/s, 0 seconds passed
... 23%, 29024 KB, 53044 KB/s, 0 seconds passed
... 23%, 29056 KB, 53077 KB/s, 0 seconds passed
... 23%, 29088 KB, 53110 KB/s, 0 seconds passed
... 23%, 29120 KB, 53138 KB/s, 0 seconds passed
... 23%, 29152 KB, 53170 KB/s, 0 seconds passed
... 23%, 29184 KB, 53204 KB/s, 0 seconds passed
... 23%, 29216 KB, 53231 KB/s, 0 seconds passed
... 23%, 29248 KB, 53264 KB/s, 0 seconds passed
... 23%, 29280 KB, 53297 KB/s, 0 seconds passed
... 23%, 29312 KB, 53324 KB/s, 0 seconds passed
... 23%, 29344 KB, 53357 KB/s, 0 seconds passed
... 23%, 29376 KB, 53390 KB/s, 0 seconds passed
... 23%, 29408 KB, 53422 KB/s, 0 seconds passed
... 23%, 29440 KB, 53450 KB/s, 0 seconds passed
... 23%, 29472 KB, 52522 KB/s, 0 seconds passed
... 23%, 29504 KB, 52474 KB/s, 0 seconds passed
... 23%, 29536 KB, 52500 KB/s, 0 seconds passed
... 23%, 29568 KB, 52528 KB/s, 0 seconds passed

.. parsed-literal::

    ... 23%, 29600 KB, 52500 KB/s, 0 seconds passed
... 23%, 29632 KB, 52532 KB/s, 0 seconds passed
... 23%, 29664 KB, 52560 KB/s, 0 seconds passed
... 23%, 29696 KB, 52592 KB/s, 0 seconds passed
... 23%, 29728 KB, 52624 KB/s, 0 seconds passed
... 23%, 29760 KB, 52656 KB/s, 0 seconds passed
... 23%, 29792 KB, 52684 KB/s, 0 seconds passed
... 23%, 29824 KB, 52716 KB/s, 0 seconds passed
... 23%, 29856 KB, 52748 KB/s, 0 seconds passed
... 23%, 29888 KB, 52775 KB/s, 0 seconds passed
... 23%, 29920 KB, 52807 KB/s, 0 seconds passed
... 23%, 29952 KB, 52839 KB/s, 0 seconds passed
... 23%, 29984 KB, 52871 KB/s, 0 seconds passed
... 23%, 30016 KB, 52898 KB/s, 0 seconds passed
... 23%, 30048 KB, 52930 KB/s, 0 seconds passed
... 23%, 30080 KB, 52961 KB/s, 0 seconds passed
... 23%, 30112 KB, 52988 KB/s, 0 seconds passed
... 23%, 30144 KB, 53020 KB/s, 0 seconds passed
... 23%, 30176 KB, 53052 KB/s, 0 seconds passed
... 23%, 30208 KB, 53079 KB/s, 0 seconds passed
... 24%, 30240 KB, 53111 KB/s, 0 seconds passed
... 24%, 30272 KB, 53142 KB/s, 0 seconds passed
... 24%, 30304 KB, 53169 KB/s, 0 seconds passed
... 24%, 30336 KB, 53201 KB/s, 0 seconds passed
... 24%, 30368 KB, 53233 KB/s, 0 seconds passed
... 24%, 30400 KB, 53259 KB/s, 0 seconds passed
... 24%, 30432 KB, 53291 KB/s, 0 seconds passed
... 24%, 30464 KB, 53322 KB/s, 0 seconds passed
... 24%, 30496 KB, 53353 KB/s, 0 seconds passed
... 24%, 30528 KB, 53379 KB/s, 0 seconds passed
... 24%, 30560 KB, 53411 KB/s, 0 seconds passed
... 24%, 30592 KB, 53442 KB/s, 0 seconds passed
... 24%, 30624 KB, 53473 KB/s, 0 seconds passed
... 24%, 30656 KB, 53499 KB/s, 0 seconds passed
... 24%, 30688 KB, 53531 KB/s, 0 seconds passed

.. parsed-literal::

    ... 24%, 30720 KB, 49857 KB/s, 0 seconds passed
... 24%, 30752 KB, 49868 KB/s, 0 seconds passed
... 24%, 30784 KB, 49887 KB/s, 0 seconds passed
... 24%, 30816 KB, 49913 KB/s, 0 seconds passed
... 24%, 30848 KB, 49380 KB/s, 0 seconds passed
... 24%, 30880 KB, 49299 KB/s, 0 seconds passed
... 24%, 30912 KB, 49306 KB/s, 0 seconds passed
... 24%, 30944 KB, 49326 KB/s, 0 seconds passed
... 24%, 30976 KB, 49345 KB/s, 0 seconds passed
... 24%, 31008 KB, 49362 KB/s, 0 seconds passed
... 24%, 31040 KB, 49382 KB/s, 0 seconds passed
... 24%, 31072 KB, 49403 KB/s, 0 seconds passed
... 24%, 31104 KB, 49424 KB/s, 0 seconds passed
... 24%, 31136 KB, 49446 KB/s, 0 seconds passed
... 24%, 31168 KB, 49466 KB/s, 0 seconds passed
... 24%, 31200 KB, 49488 KB/s, 0 seconds passed
... 24%, 31232 KB, 49511 KB/s, 0 seconds passed
... 24%, 31264 KB, 49533 KB/s, 0 seconds passed
... 24%, 31296 KB, 49553 KB/s, 0 seconds passed
... 24%, 31328 KB, 49574 KB/s, 0 seconds passed
... 24%, 31360 KB, 49593 KB/s, 0 seconds passed
... 24%, 31392 KB, 49613 KB/s, 0 seconds passed
... 24%, 31424 KB, 49639 KB/s, 0 seconds passed
... 24%, 31456 KB, 49660 KB/s, 0 seconds passed
... 24%, 31488 KB, 49679 KB/s, 0 seconds passed
... 25%, 31520 KB, 49699 KB/s, 0 seconds passed
... 25%, 31552 KB, 49718 KB/s, 0 seconds passed
... 25%, 31584 KB, 49740 KB/s, 0 seconds passed
... 25%, 31616 KB, 49761 KB/s, 0 seconds passed
... 25%, 31648 KB, 49782 KB/s, 0 seconds passed
... 25%, 31680 KB, 49800 KB/s, 0 seconds passed
... 25%, 31712 KB, 49821 KB/s, 0 seconds passed
... 25%, 31744 KB, 49842 KB/s, 0 seconds passed
... 25%, 31776 KB, 49864 KB/s, 0 seconds passed
... 25%, 31808 KB, 49884 KB/s, 0 seconds passed
... 25%, 31840 KB, 49903 KB/s, 0 seconds passed
... 25%, 31872 KB, 49924 KB/s, 0 seconds passed
... 25%, 31904 KB, 49945 KB/s, 0 seconds passed
... 25%, 31936 KB, 49965 KB/s, 0 seconds passed
... 25%, 31968 KB, 49986 KB/s, 0 seconds passed
... 25%, 32000 KB, 50004 KB/s, 0 seconds passed
... 25%, 32032 KB, 50023 KB/s, 0 seconds passed
... 25%, 32064 KB, 50044 KB/s, 0 seconds passed
... 25%, 32096 KB, 50065 KB/s, 0 seconds passed
... 25%, 32128 KB, 50087 KB/s, 0 seconds passed
... 25%, 32160 KB, 50108 KB/s, 0 seconds passed
... 25%, 32192 KB, 50129 KB/s, 0 seconds passed
... 25%, 32224 KB, 50152 KB/s, 0 seconds passed
... 25%, 32256 KB, 50181 KB/s, 0 seconds passed
... 25%, 32288 KB, 50215 KB/s, 0 seconds passed
... 25%, 32320 KB, 50250 KB/s, 0 seconds passed
... 25%, 32352 KB, 50285 KB/s, 0 seconds passed
... 25%, 32384 KB, 50320 KB/s, 0 seconds passed
... 25%, 32416 KB, 50356 KB/s, 0 seconds passed
... 25%, 32448 KB, 50390 KB/s, 0 seconds passed
... 25%, 32480 KB, 49121 KB/s, 0 seconds passed
... 25%, 32512 KB, 49132 KB/s, 0 seconds passed
... 25%, 32544 KB, 49150 KB/s, 0 seconds passed
... 25%, 32576 KB, 49169 KB/s, 0 seconds passed
... 25%, 32608 KB, 49187 KB/s, 0 seconds passed
... 25%, 32640 KB, 49206 KB/s, 0 seconds passed
... 25%, 32672 KB, 49226 KB/s, 0 seconds passed
... 25%, 32704 KB, 49245 KB/s, 0 seconds passed
... 25%, 32736 KB, 49265 KB/s, 0 seconds passed
... 26%, 32768 KB, 49284 KB/s, 0 seconds passed
... 26%, 32800 KB, 49302 KB/s, 0 seconds passed
... 26%, 32832 KB, 49322 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 32864 KB, 49343 KB/s, 0 seconds passed
... 26%, 32896 KB, 49362 KB/s, 0 seconds passed
... 26%, 32928 KB, 49382 KB/s, 0 seconds passed
... 26%, 32960 KB, 49401 KB/s, 0 seconds passed
... 26%, 32992 KB, 49420 KB/s, 0 seconds passed
... 26%, 33024 KB, 49439 KB/s, 0 seconds passed
... 26%, 33056 KB, 49459 KB/s, 0 seconds passed
... 26%, 33088 KB, 49477 KB/s, 0 seconds passed
... 26%, 33120 KB, 49497 KB/s, 0 seconds passed
... 26%, 33152 KB, 49517 KB/s, 0 seconds passed
... 26%, 33184 KB, 49537 KB/s, 0 seconds passed
... 26%, 33216 KB, 49557 KB/s, 0 seconds passed
... 26%, 33248 KB, 49577 KB/s, 0 seconds passed
... 26%, 33280 KB, 49597 KB/s, 0 seconds passed
... 26%, 33312 KB, 49617 KB/s, 0 seconds passed
... 26%, 33344 KB, 49636 KB/s, 0 seconds passed
... 26%, 33376 KB, 49655 KB/s, 0 seconds passed
... 26%, 33408 KB, 49674 KB/s, 0 seconds passed
... 26%, 33440 KB, 49698 KB/s, 0 seconds passed
... 26%, 33472 KB, 49728 KB/s, 0 seconds passed
... 26%, 33504 KB, 49758 KB/s, 0 seconds passed
... 26%, 33536 KB, 49787 KB/s, 0 seconds passed
... 26%, 33568 KB, 49817 KB/s, 0 seconds passed
... 26%, 33600 KB, 49846 KB/s, 0 seconds passed
... 26%, 33632 KB, 49876 KB/s, 0 seconds passed
... 26%, 33664 KB, 49905 KB/s, 0 seconds passed
... 26%, 33696 KB, 49933 KB/s, 0 seconds passed
... 26%, 33728 KB, 49963 KB/s, 0 seconds passed
... 26%, 33760 KB, 49992 KB/s, 0 seconds passed
... 26%, 33792 KB, 50022 KB/s, 0 seconds passed
... 26%, 33824 KB, 50052 KB/s, 0 seconds passed
... 26%, 33856 KB, 50079 KB/s, 0 seconds passed
... 26%, 33888 KB, 50108 KB/s, 0 seconds passed
... 26%, 33920 KB, 50137 KB/s, 0 seconds passed
... 26%, 33952 KB, 50167 KB/s, 0 seconds passed
... 26%, 33984 KB, 50196 KB/s, 0 seconds passed
... 27%, 34016 KB, 50226 KB/s, 0 seconds passed
... 27%, 34048 KB, 50255 KB/s, 0 seconds passed
... 27%, 34080 KB, 50284 KB/s, 0 seconds passed
... 27%, 34112 KB, 50313 KB/s, 0 seconds passed
... 27%, 34144 KB, 50343 KB/s, 0 seconds passed
... 27%, 34176 KB, 50372 KB/s, 0 seconds passed
... 27%, 34208 KB, 50400 KB/s, 0 seconds passed
... 27%, 34240 KB, 50430 KB/s, 0 seconds passed
... 27%, 34272 KB, 50459 KB/s, 0 seconds passed
... 27%, 34304 KB, 50487 KB/s, 0 seconds passed
... 27%, 34336 KB, 50516 KB/s, 0 seconds passed
... 27%, 34368 KB, 50543 KB/s, 0 seconds passed
... 27%, 34400 KB, 50572 KB/s, 0 seconds passed
... 27%, 34432 KB, 50601 KB/s, 0 seconds passed
... 27%, 34464 KB, 50630 KB/s, 0 seconds passed
... 27%, 34496 KB, 50659 KB/s, 0 seconds passed
... 27%, 34528 KB, 50689 KB/s, 0 seconds passed
... 27%, 34560 KB, 50719 KB/s, 0 seconds passed
... 27%, 34592 KB, 50746 KB/s, 0 seconds passed
... 27%, 34624 KB, 50775 KB/s, 0 seconds passed
... 27%, 34656 KB, 50804 KB/s, 0 seconds passed
... 27%, 34688 KB, 50833 KB/s, 0 seconds passed
... 27%, 34720 KB, 50863 KB/s, 0 seconds passed
... 27%, 34752 KB, 50891 KB/s, 0 seconds passed
... 27%, 34784 KB, 50921 KB/s, 0 seconds passed
... 27%, 34816 KB, 50952 KB/s, 0 seconds passed
... 27%, 34848 KB, 50987 KB/s, 0 seconds passed
... 27%, 34880 KB, 51021 KB/s, 0 seconds passed
... 27%, 34912 KB, 51055 KB/s, 0 seconds passed
... 27%, 34944 KB, 51088 KB/s, 0 seconds passed
... 27%, 34976 KB, 51122 KB/s, 0 seconds passed
... 27%, 35008 KB, 51156 KB/s, 0 seconds passed
... 27%, 35040 KB, 51190 KB/s, 0 seconds passed
... 27%, 35072 KB, 51225 KB/s, 0 seconds passed
... 27%, 35104 KB, 51259 KB/s, 0 seconds passed
... 27%, 35136 KB, 51293 KB/s, 0 seconds passed
... 27%, 35168 KB, 51326 KB/s, 0 seconds passed
... 27%, 35200 KB, 51359 KB/s, 0 seconds passed
... 27%, 35232 KB, 51394 KB/s, 0 seconds passed
... 27%, 35264 KB, 51429 KB/s, 0 seconds passed
... 28%, 35296 KB, 51463 KB/s, 0 seconds passed
... 28%, 35328 KB, 51497 KB/s, 0 seconds passed
... 28%, 35360 KB, 51532 KB/s, 0 seconds passed
... 28%, 35392 KB, 51559 KB/s, 0 seconds passed
... 28%, 35424 KB, 51582 KB/s, 0 seconds passed
... 28%, 35456 KB, 51609 KB/s, 0 seconds passed
... 28%, 35488 KB, 51635 KB/s, 0 seconds passed
... 28%, 35520 KB, 51662 KB/s, 0 seconds passed
... 28%, 35552 KB, 51685 KB/s, 0 seconds passed
... 28%, 35584 KB, 51712 KB/s, 0 seconds passed
... 28%, 35616 KB, 51739 KB/s, 0 seconds passed
... 28%, 35648 KB, 51761 KB/s, 0 seconds passed
... 28%, 35680 KB, 51788 KB/s, 0 seconds passed
... 28%, 35712 KB, 51815 KB/s, 0 seconds passed
... 28%, 35744 KB, 51826 KB/s, 0 seconds passed
... 28%, 35776 KB, 51841 KB/s, 0 seconds passed
... 28%, 35808 KB, 51861 KB/s, 0 seconds passed
... 28%, 35840 KB, 51319 KB/s, 0 seconds passed
... 28%, 35872 KB, 51334 KB/s, 0 seconds passed
... 28%, 35904 KB, 51349 KB/s, 0 seconds passed
... 28%, 35936 KB, 51366 KB/s, 0 seconds passed
... 28%, 35968 KB, 51385 KB/s, 0 seconds passed
... 28%, 36000 KB, 51405 KB/s, 0 seconds passed
... 28%, 36032 KB, 51426 KB/s, 0 seconds passed
... 28%, 36064 KB, 51447 KB/s, 0 seconds passed
... 28%, 36096 KB, 51467 KB/s, 0 seconds passed
... 28%, 36128 KB, 51487 KB/s, 0 seconds passed
... 28%, 36160 KB, 51506 KB/s, 0 seconds passed
... 28%, 36192 KB, 51526 KB/s, 0 seconds passed
... 28%, 36224 KB, 51548 KB/s, 0 seconds passed
... 28%, 36256 KB, 51569 KB/s, 0 seconds passed
... 28%, 36288 KB, 51590 KB/s, 0 seconds passed
... 28%, 36320 KB, 51611 KB/s, 0 seconds passed
... 28%, 36352 KB, 51627 KB/s, 0 seconds passed
... 28%, 36384 KB, 51642 KB/s, 0 seconds passed
... 28%, 36416 KB, 51661 KB/s, 0 seconds passed
... 28%, 36448 KB, 51679 KB/s, 0 seconds passed
... 28%, 36480 KB, 51698 KB/s, 0 seconds passed
... 28%, 36512 KB, 51716 KB/s, 0 seconds passed
... 29%, 36544 KB, 51734 KB/s, 0 seconds passed
... 29%, 36576 KB, 51751 KB/s, 0 seconds passed
... 29%, 36608 KB, 51767 KB/s, 0 seconds passed
... 29%, 36640 KB, 51785 KB/s, 0 seconds passed
... 29%, 36672 KB, 51802 KB/s, 0 seconds passed
... 29%, 36704 KB, 51817 KB/s, 0 seconds passed
... 29%, 36736 KB, 51833 KB/s, 0 seconds passed
... 29%, 36768 KB, 51851 KB/s, 0 seconds passed
... 29%, 36800 KB, 51867 KB/s, 0 seconds passed
... 29%, 36832 KB, 51884 KB/s, 0 seconds passed
... 29%, 36864 KB, 51902 KB/s, 0 seconds passed
... 29%, 36896 KB, 51921 KB/s, 0 seconds passed
... 29%, 36928 KB, 51938 KB/s, 0 seconds passed
... 29%, 36960 KB, 51957 KB/s, 0 seconds passed
... 29%, 36992 KB, 51975 KB/s, 0 seconds passed
... 29%, 37024 KB, 51987 KB/s, 0 seconds passed
... 29%, 37056 KB, 51993 KB/s, 0 seconds passed
... 29%, 37088 KB, 52006 KB/s, 0 seconds passed
... 29%, 37120 KB, 52022 KB/s, 0 seconds passed
... 29%, 37152 KB, 52041 KB/s, 0 seconds passed
... 29%, 37184 KB, 52067 KB/s, 0 seconds passed
... 29%, 37216 KB, 52094 KB/s, 0 seconds passed
... 29%, 37248 KB, 52120 KB/s, 0 seconds passed
... 29%, 37280 KB, 52147 KB/s, 0 seconds passed
... 29%, 37312 KB, 52175 KB/s, 0 seconds passed
... 29%, 37344 KB, 52201 KB/s, 0 seconds passed
... 29%, 37376 KB, 52228 KB/s, 0 seconds passed
... 29%, 37408 KB, 52256 KB/s, 0 seconds passed
... 29%, 37440 KB, 52279 KB/s, 0 seconds passed
... 29%, 37472 KB, 52306 KB/s, 0 seconds passed
... 29%, 37504 KB, 52334 KB/s, 0 seconds passed
... 29%, 37536 KB, 52361 KB/s, 0 seconds passed
... 29%, 37568 KB, 52388 KB/s, 0 seconds passed

.. parsed-literal::

    ... 29%, 37600 KB, 52416 KB/s, 0 seconds passed
... 29%, 37632 KB, 52441 KB/s, 0 seconds passed
... 29%, 37664 KB, 52469 KB/s, 0 seconds passed
... 29%, 37696 KB, 52495 KB/s, 0 seconds passed
... 29%, 37728 KB, 52521 KB/s, 0 seconds passed
... 29%, 37760 KB, 52546 KB/s, 0 seconds passed
... 30%, 37792 KB, 52574 KB/s, 0 seconds passed
... 30%, 37824 KB, 52601 KB/s, 0 seconds passed
... 30%, 37856 KB, 52628 KB/s, 0 seconds passed
... 30%, 37888 KB, 52653 KB/s, 0 seconds passed
... 30%, 37920 KB, 52680 KB/s, 0 seconds passed
... 30%, 37952 KB, 52703 KB/s, 0 seconds passed
... 30%, 37984 KB, 52730 KB/s, 0 seconds passed
... 30%, 38016 KB, 52755 KB/s, 0 seconds passed
... 30%, 38048 KB, 52782 KB/s, 0 seconds passed
... 30%, 38080 KB, 52807 KB/s, 0 seconds passed
... 30%, 38112 KB, 52833 KB/s, 0 seconds passed
... 30%, 38144 KB, 52859 KB/s, 0 seconds passed
... 30%, 38176 KB, 52886 KB/s, 0 seconds passed
... 30%, 38208 KB, 52911 KB/s, 0 seconds passed
... 30%, 38240 KB, 52937 KB/s, 0 seconds passed
... 30%, 38272 KB, 52963 KB/s, 0 seconds passed
... 30%, 38304 KB, 52990 KB/s, 0 seconds passed
... 30%, 38336 KB, 53017 KB/s, 0 seconds passed
... 30%, 38368 KB, 53044 KB/s, 0 seconds passed
... 30%, 38400 KB, 53070 KB/s, 0 seconds passed
... 30%, 38432 KB, 53099 KB/s, 0 seconds passed
... 30%, 38464 KB, 53129 KB/s, 0 seconds passed
... 30%, 38496 KB, 53160 KB/s, 0 seconds passed
... 30%, 38528 KB, 53192 KB/s, 0 seconds passed
... 30%, 38560 KB, 53224 KB/s, 0 seconds passed
... 30%, 38592 KB, 53256 KB/s, 0 seconds passed
... 30%, 38624 KB, 53287 KB/s, 0 seconds passed
... 30%, 38656 KB, 53318 KB/s, 0 seconds passed
... 30%, 38688 KB, 53350 KB/s, 0 seconds passed
... 30%, 38720 KB, 53380 KB/s, 0 seconds passed
... 30%, 38752 KB, 53412 KB/s, 0 seconds passed
... 30%, 38784 KB, 53443 KB/s, 0 seconds passed
... 30%, 38816 KB, 53475 KB/s, 0 seconds passed
... 30%, 38848 KB, 53507 KB/s, 0 seconds passed
... 30%, 38880 KB, 53538 KB/s, 0 seconds passed
... 30%, 38912 KB, 53570 KB/s, 0 seconds passed
... 30%, 38944 KB, 53602 KB/s, 0 seconds passed
... 30%, 38976 KB, 53632 KB/s, 0 seconds passed
... 30%, 39008 KB, 53663 KB/s, 0 seconds passed
... 30%, 39040 KB, 53695 KB/s, 0 seconds passed
... 31%, 39072 KB, 53727 KB/s, 0 seconds passed
... 31%, 39104 KB, 53759 KB/s, 0 seconds passed
... 31%, 39136 KB, 53790 KB/s, 0 seconds passed
... 31%, 39168 KB, 53822 KB/s, 0 seconds passed
... 31%, 39200 KB, 53854 KB/s, 0 seconds passed
... 31%, 39232 KB, 53885 KB/s, 0 seconds passed
... 31%, 39264 KB, 53916 KB/s, 0 seconds passed
... 31%, 39296 KB, 53942 KB/s, 0 seconds passed
... 31%, 39328 KB, 53966 KB/s, 0 seconds passed
... 31%, 39360 KB, 53991 KB/s, 0 seconds passed
... 31%, 39392 KB, 54011 KB/s, 0 seconds passed
... 31%, 39424 KB, 54040 KB/s, 0 seconds passed
... 31%, 39456 KB, 54064 KB/s, 0 seconds passed
... 31%, 39488 KB, 54088 KB/s, 0 seconds passed
... 31%, 39520 KB, 54109 KB/s, 0 seconds passed
... 31%, 39552 KB, 54133 KB/s, 0 seconds passed
... 31%, 39584 KB, 54154 KB/s, 0 seconds passed
... 31%, 39616 KB, 54178 KB/s, 0 seconds passed
... 31%, 39648 KB, 53798 KB/s, 0 seconds passed
... 31%, 39680 KB, 53812 KB/s, 0 seconds passed
... 31%, 39712 KB, 53833 KB/s, 0 seconds passed
... 31%, 39744 KB, 53854 KB/s, 0 seconds passed
... 31%, 39776 KB, 53811 KB/s, 0 seconds passed
... 31%, 39808 KB, 53819 KB/s, 0 seconds passed
... 31%, 39840 KB, 53834 KB/s, 0 seconds passed
... 31%, 39872 KB, 53853 KB/s, 0 seconds passed
... 31%, 39904 KB, 53870 KB/s, 0 seconds passed
... 31%, 39936 KB, 53891 KB/s, 0 seconds passed
... 31%, 39968 KB, 53910 KB/s, 0 seconds passed
... 31%, 40000 KB, 53931 KB/s, 0 seconds passed
... 31%, 40032 KB, 53951 KB/s, 0 seconds passed
... 31%, 40064 KB, 53971 KB/s, 0 seconds passed
... 31%, 40096 KB, 53995 KB/s, 0 seconds passed
... 31%, 40128 KB, 54011 KB/s, 0 seconds passed
... 31%, 40160 KB, 54025 KB/s, 0 seconds passed
... 31%, 40192 KB, 54043 KB/s, 0 seconds passed
... 31%, 40224 KB, 54058 KB/s, 0 seconds passed
... 31%, 40256 KB, 54073 KB/s, 0 seconds passed
... 31%, 40288 KB, 54085 KB/s, 0 seconds passed
... 32%, 40320 KB, 54101 KB/s, 0 seconds passed
... 32%, 40352 KB, 54117 KB/s, 0 seconds passed
... 32%, 40384 KB, 54133 KB/s, 0 seconds passed
... 32%, 40416 KB, 54149 KB/s, 0 seconds passed
... 32%, 40448 KB, 54164 KB/s, 0 seconds passed
... 32%, 40480 KB, 54181 KB/s, 0 seconds passed
... 32%, 40512 KB, 54196 KB/s, 0 seconds passed
... 32%, 40544 KB, 54211 KB/s, 0 seconds passed
... 32%, 40576 KB, 54224 KB/s, 0 seconds passed
... 32%, 40608 KB, 54240 KB/s, 0 seconds passed
... 32%, 40640 KB, 54255 KB/s, 0 seconds passed
... 32%, 40672 KB, 54271 KB/s, 0 seconds passed
... 32%, 40704 KB, 54286 KB/s, 0 seconds passed
... 32%, 40736 KB, 54302 KB/s, 0 seconds passed
... 32%, 40768 KB, 54323 KB/s, 0 seconds passed
... 32%, 40800 KB, 54346 KB/s, 0 seconds passed
... 32%, 40832 KB, 54368 KB/s, 0 seconds passed
... 32%, 40864 KB, 54390 KB/s, 0 seconds passed
... 32%, 40896 KB, 54412 KB/s, 0 seconds passed
... 32%, 40928 KB, 54434 KB/s, 0 seconds passed

.. parsed-literal::

    ... 32%, 40960 KB, 52719 KB/s, 0 seconds passed
... 32%, 40992 KB, 52726 KB/s, 0 seconds passed
... 32%, 41024 KB, 52740 KB/s, 0 seconds passed
... 32%, 41056 KB, 52755 KB/s, 0 seconds passed
... 32%, 41088 KB, 52768 KB/s, 0 seconds passed
... 32%, 41120 KB, 52782 KB/s, 0 seconds passed
... 32%, 41152 KB, 52797 KB/s, 0 seconds passed
... 32%, 41184 KB, 52810 KB/s, 0 seconds passed
... 32%, 41216 KB, 52822 KB/s, 0 seconds passed
... 32%, 41248 KB, 52838 KB/s, 0 seconds passed
... 32%, 41280 KB, 52853 KB/s, 0 seconds passed
... 32%, 41312 KB, 52868 KB/s, 0 seconds passed
... 32%, 41344 KB, 52881 KB/s, 0 seconds passed
... 32%, 41376 KB, 52897 KB/s, 0 seconds passed
... 32%, 41408 KB, 52913 KB/s, 0 seconds passed
... 32%, 41440 KB, 52927 KB/s, 0 seconds passed
... 32%, 41472 KB, 52941 KB/s, 0 seconds passed
... 32%, 41504 KB, 52957 KB/s, 0 seconds passed
... 32%, 41536 KB, 52969 KB/s, 0 seconds passed
... 33%, 41568 KB, 52985 KB/s, 0 seconds passed
... 33%, 41600 KB, 52999 KB/s, 0 seconds passed
... 33%, 41632 KB, 53016 KB/s, 0 seconds passed
... 33%, 41664 KB, 53029 KB/s, 0 seconds passed
... 33%, 41696 KB, 53044 KB/s, 0 seconds passed
... 33%, 41728 KB, 53059 KB/s, 0 seconds passed
... 33%, 41760 KB, 53076 KB/s, 0 seconds passed
... 33%, 41792 KB, 53092 KB/s, 0 seconds passed
... 33%, 41824 KB, 53108 KB/s, 0 seconds passed
... 33%, 41856 KB, 53123 KB/s, 0 seconds passed
... 33%, 41888 KB, 53137 KB/s, 0 seconds passed
... 33%, 41920 KB, 53150 KB/s, 0 seconds passed
... 33%, 41952 KB, 53166 KB/s, 0 seconds passed
... 33%, 41984 KB, 53182 KB/s, 0 seconds passed
... 33%, 42016 KB, 53198 KB/s, 0 seconds passed
... 33%, 42048 KB, 53210 KB/s, 0 seconds passed
... 33%, 42080 KB, 53223 KB/s, 0 seconds passed
... 33%, 42112 KB, 53238 KB/s, 0 seconds passed
... 33%, 42144 KB, 53253 KB/s, 0 seconds passed
... 33%, 42176 KB, 53267 KB/s, 0 seconds passed
... 33%, 42208 KB, 53281 KB/s, 0 seconds passed
... 33%, 42240 KB, 53296 KB/s, 0 seconds passed
... 33%, 42272 KB, 53311 KB/s, 0 seconds passed
... 33%, 42304 KB, 53326 KB/s, 0 seconds passed
... 33%, 42336 KB, 53341 KB/s, 0 seconds passed
... 33%, 42368 KB, 53356 KB/s, 0 seconds passed
... 33%, 42400 KB, 53372 KB/s, 0 seconds passed
... 33%, 42432 KB, 53396 KB/s, 0 seconds passed
... 33%, 42464 KB, 53420 KB/s, 0 seconds passed
... 33%, 42496 KB, 53445 KB/s, 0 seconds passed
... 33%, 42528 KB, 53469 KB/s, 0 seconds passed
... 33%, 42560 KB, 53493 KB/s, 0 seconds passed
... 33%, 42592 KB, 53518 KB/s, 0 seconds passed
... 33%, 42624 KB, 53539 KB/s, 0 seconds passed
... 33%, 42656 KB, 53563 KB/s, 0 seconds passed
... 33%, 42688 KB, 53587 KB/s, 0 seconds passed
... 33%, 42720 KB, 53611 KB/s, 0 seconds passed
... 33%, 42752 KB, 53635 KB/s, 0 seconds passed
... 33%, 42784 KB, 53658 KB/s, 0 seconds passed
... 33%, 42816 KB, 53682 KB/s, 0 seconds passed
... 34%, 42848 KB, 53705 KB/s, 0 seconds passed
... 34%, 42880 KB, 53729 KB/s, 0 seconds passed
... 34%, 42912 KB, 53753 KB/s, 0 seconds passed
... 34%, 42944 KB, 53777 KB/s, 0 seconds passed
... 34%, 42976 KB, 53801 KB/s, 0 seconds passed
... 34%, 43008 KB, 53824 KB/s, 0 seconds passed
... 34%, 43040 KB, 53848 KB/s, 0 seconds passed
... 34%, 43072 KB, 53872 KB/s, 0 seconds passed
... 34%, 43104 KB, 53896 KB/s, 0 seconds passed
... 34%, 43136 KB, 53919 KB/s, 0 seconds passed
... 34%, 43168 KB, 53942 KB/s, 0 seconds passed
... 34%, 43200 KB, 53965 KB/s, 0 seconds passed
... 34%, 43232 KB, 53989 KB/s, 0 seconds passed
... 34%, 43264 KB, 54013 KB/s, 0 seconds passed
... 34%, 43296 KB, 54036 KB/s, 0 seconds passed
... 34%, 43328 KB, 54060 KB/s, 0 seconds passed
... 34%, 43360 KB, 54083 KB/s, 0 seconds passed
... 34%, 43392 KB, 54106 KB/s, 0 seconds passed
... 34%, 43424 KB, 54130 KB/s, 0 seconds passed
... 34%, 43456 KB, 54154 KB/s, 0 seconds passed
... 34%, 43488 KB, 54178 KB/s, 0 seconds passed
... 34%, 43520 KB, 54201 KB/s, 0 seconds passed
... 34%, 43552 KB, 54224 KB/s, 0 seconds passed
... 34%, 43584 KB, 54247 KB/s, 0 seconds passed
... 34%, 43616 KB, 54271 KB/s, 0 seconds passed
... 34%, 43648 KB, 54296 KB/s, 0 seconds passed
... 34%, 43680 KB, 54315 KB/s, 0 seconds passed
... 34%, 43712 KB, 54338 KB/s, 0 seconds passed
... 34%, 43744 KB, 54362 KB/s, 0 seconds passed
... 34%, 43776 KB, 54386 KB/s, 0 seconds passed
... 34%, 43808 KB, 54409 KB/s, 0 seconds passed
... 34%, 43840 KB, 54434 KB/s, 0 seconds passed
... 34%, 43872 KB, 54462 KB/s, 0 seconds passed
... 34%, 43904 KB, 54491 KB/s, 0 seconds passed
... 34%, 43936 KB, 54519 KB/s, 0 seconds passed
... 34%, 43968 KB, 54547 KB/s, 0 seconds passed
... 34%, 44000 KB, 54575 KB/s, 0 seconds passed
... 34%, 44032 KB, 54604 KB/s, 0 seconds passed
... 34%, 44064 KB, 54633 KB/s, 0 seconds passed
... 35%, 44096 KB, 54661 KB/s, 0 seconds passed
... 35%, 44128 KB, 54690 KB/s, 0 seconds passed
... 35%, 44160 KB, 54717 KB/s, 0 seconds passed
... 35%, 44192 KB, 54745 KB/s, 0 seconds passed
... 35%, 44224 KB, 54773 KB/s, 0 seconds passed
... 35%, 44256 KB, 54801 KB/s, 0 seconds passed
... 35%, 44288 KB, 54830 KB/s, 0 seconds passed
... 35%, 44320 KB, 54859 KB/s, 0 seconds passed
... 35%, 44352 KB, 54887 KB/s, 0 seconds passed
... 35%, 44384 KB, 54915 KB/s, 0 seconds passed
... 35%, 44416 KB, 54944 KB/s, 0 seconds passed
... 35%, 44448 KB, 54972 KB/s, 0 seconds passed
... 35%, 44480 KB, 55000 KB/s, 0 seconds passed
... 35%, 44512 KB, 55029 KB/s, 0 seconds passed
... 35%, 44544 KB, 55058 KB/s, 0 seconds passed
... 35%, 44576 KB, 55086 KB/s, 0 seconds passed
... 35%, 44608 KB, 55115 KB/s, 0 seconds passed
... 35%, 44640 KB, 55144 KB/s, 0 seconds passed
... 35%, 44672 KB, 55172 KB/s, 0 seconds passed
... 35%, 44704 KB, 55201 KB/s, 0 seconds passed
... 35%, 44736 KB, 55227 KB/s, 0 seconds passed
... 35%, 44768 KB, 55248 KB/s, 0 seconds passed
... 35%, 44800 KB, 55270 KB/s, 0 seconds passed
... 35%, 44832 KB, 54983 KB/s, 0 seconds passed
... 35%, 44864 KB, 54995 KB/s, 0 seconds passed
... 35%, 44896 KB, 55005 KB/s, 0 seconds passed
... 35%, 44928 KB, 55018 KB/s, 0 seconds passed
... 35%, 44960 KB, 55032 KB/s, 0 seconds passed
... 35%, 44992 KB, 55047 KB/s, 0 seconds passed
... 35%, 45024 KB, 55061 KB/s, 0 seconds passed
... 35%, 45056 KB, 55078 KB/s, 0 seconds passed
... 35%, 45088 KB, 55097 KB/s, 0 seconds passed

.. parsed-literal::

    ... 35%, 45120 KB, 54913 KB/s, 0 seconds passed
... 35%, 45152 KB, 54876 KB/s, 0 seconds passed
... 35%, 45184 KB, 54889 KB/s, 0 seconds passed
... 35%, 45216 KB, 54904 KB/s, 0 seconds passed
... 35%, 45248 KB, 54917 KB/s, 0 seconds passed
... 35%, 45280 KB, 54929 KB/s, 0 seconds passed
... 35%, 45312 KB, 54943 KB/s, 0 seconds passed
... 36%, 45344 KB, 54958 KB/s, 0 seconds passed
... 36%, 45376 KB, 54972 KB/s, 0 seconds passed
... 36%, 45408 KB, 54986 KB/s, 0 seconds passed
... 36%, 45440 KB, 54999 KB/s, 0 seconds passed
... 36%, 45472 KB, 55014 KB/s, 0 seconds passed
... 36%, 45504 KB, 55028 KB/s, 0 seconds passed
... 36%, 45536 KB, 55043 KB/s, 0 seconds passed
... 36%, 45568 KB, 55056 KB/s, 0 seconds passed
... 36%, 45600 KB, 55070 KB/s, 0 seconds passed
... 36%, 45632 KB, 55082 KB/s, 0 seconds passed
... 36%, 45664 KB, 55097 KB/s, 0 seconds passed
... 36%, 45696 KB, 55110 KB/s, 0 seconds passed
... 36%, 45728 KB, 55124 KB/s, 0 seconds passed
... 36%, 45760 KB, 55139 KB/s, 0 seconds passed
... 36%, 45792 KB, 55153 KB/s, 0 seconds passed
... 36%, 45824 KB, 55167 KB/s, 0 seconds passed
... 36%, 45856 KB, 55182 KB/s, 0 seconds passed
... 36%, 45888 KB, 55200 KB/s, 0 seconds passed
... 36%, 45920 KB, 55219 KB/s, 0 seconds passed
... 36%, 45952 KB, 55238 KB/s, 0 seconds passed
... 36%, 45984 KB, 55257 KB/s, 0 seconds passed
... 36%, 46016 KB, 55277 KB/s, 0 seconds passed
... 36%, 46048 KB, 55296 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 46080 KB, 49752 KB/s, 0 seconds passed
... 36%, 46112 KB, 49759 KB/s, 0 seconds passed
... 36%, 46144 KB, 49771 KB/s, 0 seconds passed
... 36%, 46176 KB, 49785 KB/s, 0 seconds passed
... 36%, 46208 KB, 49797 KB/s, 0 seconds passed
... 36%, 46240 KB, 49810 KB/s, 0 seconds passed
... 36%, 46272 KB, 49824 KB/s, 0 seconds passed
... 36%, 46304 KB, 49838 KB/s, 0 seconds passed
... 36%, 46336 KB, 49852 KB/s, 0 seconds passed
... 36%, 46368 KB, 49867 KB/s, 0 seconds passed
... 36%, 46400 KB, 49881 KB/s, 0 seconds passed
... 36%, 46432 KB, 49895 KB/s, 0 seconds passed
... 36%, 46464 KB, 49909 KB/s, 0 seconds passed
... 36%, 46496 KB, 49923 KB/s, 0 seconds passed
... 36%, 46528 KB, 49937 KB/s, 0 seconds passed
... 36%, 46560 KB, 49948 KB/s, 0 seconds passed
... 36%, 46592 KB, 49962 KB/s, 0 seconds passed
... 37%, 46624 KB, 49977 KB/s, 0 seconds passed
... 37%, 46656 KB, 49990 KB/s, 0 seconds passed
... 37%, 46688 KB, 50002 KB/s, 0 seconds passed
... 37%, 46720 KB, 50017 KB/s, 0 seconds passed
... 37%, 46752 KB, 50031 KB/s, 0 seconds passed
... 37%, 46784 KB, 50046 KB/s, 0 seconds passed
... 37%, 46816 KB, 50060 KB/s, 0 seconds passed
... 37%, 46848 KB, 50078 KB/s, 0 seconds passed
... 37%, 46880 KB, 50096 KB/s, 0 seconds passed
... 37%, 46912 KB, 50113 KB/s, 0 seconds passed
... 37%, 46944 KB, 50131 KB/s, 0 seconds passed
... 37%, 46976 KB, 50149 KB/s, 0 seconds passed
... 37%, 47008 KB, 50167 KB/s, 0 seconds passed
... 37%, 47040 KB, 50184 KB/s, 0 seconds passed
... 37%, 47072 KB, 50202 KB/s, 0 seconds passed
... 37%, 47104 KB, 50220 KB/s, 0 seconds passed
... 37%, 47136 KB, 50238 KB/s, 0 seconds passed
... 37%, 47168 KB, 50256 KB/s, 0 seconds passed
... 37%, 47200 KB, 50274 KB/s, 0 seconds passed
... 37%, 47232 KB, 50290 KB/s, 0 seconds passed
... 37%, 47264 KB, 50308 KB/s, 0 seconds passed
... 37%, 47296 KB, 50326 KB/s, 0 seconds passed
... 37%, 47328 KB, 50343 KB/s, 0 seconds passed
... 37%, 47360 KB, 50361 KB/s, 0 seconds passed
... 37%, 47392 KB, 50380 KB/s, 0 seconds passed
... 37%, 47424 KB, 50397 KB/s, 0 seconds passed
... 37%, 47456 KB, 50415 KB/s, 0 seconds passed
... 37%, 47488 KB, 50433 KB/s, 0 seconds passed
... 37%, 47520 KB, 50450 KB/s, 0 seconds passed
... 37%, 47552 KB, 50467 KB/s, 0 seconds passed
... 37%, 47584 KB, 50485 KB/s, 0 seconds passed
... 37%, 47616 KB, 50503 KB/s, 0 seconds passed
... 37%, 47648 KB, 50520 KB/s, 0 seconds passed
... 37%, 47680 KB, 50538 KB/s, 0 seconds passed
... 37%, 47712 KB, 50555 KB/s, 0 seconds passed
... 37%, 47744 KB, 50572 KB/s, 0 seconds passed
... 37%, 47776 KB, 50589 KB/s, 0 seconds passed
... 37%, 47808 KB, 50608 KB/s, 0 seconds passed
... 37%, 47840 KB, 50625 KB/s, 0 seconds passed
... 38%, 47872 KB, 50643 KB/s, 0 seconds passed
... 38%, 47904 KB, 50666 KB/s, 0 seconds passed
... 38%, 47936 KB, 50689 KB/s, 0 seconds passed
... 38%, 47968 KB, 50712 KB/s, 0 seconds passed
... 38%, 48000 KB, 50735 KB/s, 0 seconds passed
... 38%, 48032 KB, 50758 KB/s, 0 seconds passed
... 38%, 48064 KB, 50781 KB/s, 0 seconds passed
... 38%, 48096 KB, 50804 KB/s, 0 seconds passed
... 38%, 48128 KB, 50826 KB/s, 0 seconds passed
... 38%, 48160 KB, 50849 KB/s, 0 seconds passed
... 38%, 48192 KB, 50872 KB/s, 0 seconds passed
... 38%, 48224 KB, 50894 KB/s, 0 seconds passed
... 38%, 48256 KB, 50917 KB/s, 0 seconds passed
... 38%, 48288 KB, 50940 KB/s, 0 seconds passed
... 38%, 48320 KB, 50962 KB/s, 0 seconds passed
... 38%, 48352 KB, 50985 KB/s, 0 seconds passed
... 38%, 48384 KB, 51008 KB/s, 0 seconds passed
... 38%, 48416 KB, 51030 KB/s, 0 seconds passed
... 38%, 48448 KB, 51052 KB/s, 0 seconds passed
... 38%, 48480 KB, 51075 KB/s, 0 seconds passed
... 38%, 48512 KB, 51098 KB/s, 0 seconds passed
... 38%, 48544 KB, 51119 KB/s, 0 seconds passed
... 38%, 48576 KB, 51142 KB/s, 0 seconds passed
... 38%, 48608 KB, 51164 KB/s, 0 seconds passed
... 38%, 48640 KB, 51187 KB/s, 0 seconds passed
... 38%, 48672 KB, 51211 KB/s, 0 seconds passed
... 38%, 48704 KB, 51234 KB/s, 0 seconds passed
... 38%, 48736 KB, 51256 KB/s, 0 seconds passed
... 38%, 48768 KB, 51279 KB/s, 0 seconds passed
... 38%, 48800 KB, 51301 KB/s, 0 seconds passed
... 38%, 48832 KB, 51324 KB/s, 0 seconds passed
... 38%, 48864 KB, 51346 KB/s, 0 seconds passed
... 38%, 48896 KB, 51370 KB/s, 0 seconds passed
... 38%, 48928 KB, 51392 KB/s, 0 seconds passed
... 38%, 48960 KB, 51414 KB/s, 0 seconds passed
... 38%, 48992 KB, 51437 KB/s, 0 seconds passed
... 38%, 49024 KB, 51459 KB/s, 0 seconds passed
... 38%, 49056 KB, 51481 KB/s, 0 seconds passed
... 38%, 49088 KB, 51504 KB/s, 0 seconds passed
... 38%, 49120 KB, 51527 KB/s, 0 seconds passed
... 39%, 49152 KB, 51549 KB/s, 0 seconds passed
... 39%, 49184 KB, 51572 KB/s, 0 seconds passed
... 39%, 49216 KB, 51595 KB/s, 0 seconds passed
... 39%, 49248 KB, 51617 KB/s, 0 seconds passed
... 39%, 49280 KB, 51639 KB/s, 0 seconds passed
... 39%, 49312 KB, 51658 KB/s, 0 seconds passed
... 39%, 49344 KB, 51675 KB/s, 0 seconds passed
... 39%, 49376 KB, 51694 KB/s, 0 seconds passed
... 39%, 49408 KB, 51713 KB/s, 0 seconds passed
... 39%, 49440 KB, 51730 KB/s, 0 seconds passed
... 39%, 49472 KB, 51752 KB/s, 0 seconds passed
... 39%, 49504 KB, 51771 KB/s, 0 seconds passed
... 39%, 49536 KB, 51788 KB/s, 0 seconds passed
... 39%, 49568 KB, 51807 KB/s, 0 seconds passed
... 39%, 49600 KB, 51823 KB/s, 0 seconds passed
... 39%, 49632 KB, 51843 KB/s, 0 seconds passed
... 39%, 49664 KB, 51862 KB/s, 0 seconds passed
... 39%, 49696 KB, 51881 KB/s, 0 seconds passed
... 39%, 49728 KB, 51897 KB/s, 0 seconds passed
... 39%, 49760 KB, 51914 KB/s, 0 seconds passed
... 39%, 49792 KB, 51933 KB/s, 0 seconds passed
... 39%, 49824 KB, 51952 KB/s, 0 seconds passed
... 39%, 49856 KB, 51971 KB/s, 0 seconds passed
... 39%, 49888 KB, 51990 KB/s, 0 seconds passed
... 39%, 49920 KB, 52009 KB/s, 0 seconds passed
... 39%, 49952 KB, 51919 KB/s, 0 seconds passed
... 39%, 49984 KB, 51921 KB/s, 0 seconds passed
... 39%, 50016 KB, 51940 KB/s, 0 seconds passed
... 39%, 50048 KB, 51957 KB/s, 0 seconds passed
... 39%, 50080 KB, 51976 KB/s, 0 seconds passed
... 39%, 50112 KB, 51992 KB/s, 0 seconds passed
... 39%, 50144 KB, 52011 KB/s, 0 seconds passed
... 39%, 50176 KB, 52030 KB/s, 0 seconds passed
... 39%, 50208 KB, 51982 KB/s, 0 seconds passed
... 39%, 50240 KB, 51992 KB/s, 0 seconds passed
... 39%, 50272 KB, 52005 KB/s, 0 seconds passed
... 39%, 50304 KB, 52019 KB/s, 0 seconds passed
... 39%, 50336 KB, 52033 KB/s, 0 seconds passed
... 39%, 50368 KB, 52046 KB/s, 0 seconds passed
... 40%, 50400 KB, 52059 KB/s, 0 seconds passed
... 40%, 50432 KB, 52073 KB/s, 0 seconds passed
... 40%, 50464 KB, 52086 KB/s, 0 seconds passed
... 40%, 50496 KB, 52100 KB/s, 0 seconds passed
... 40%, 50528 KB, 52113 KB/s, 0 seconds passed
... 40%, 50560 KB, 52126 KB/s, 0 seconds passed
... 40%, 50592 KB, 52139 KB/s, 0 seconds passed
... 40%, 50624 KB, 52154 KB/s, 0 seconds passed
... 40%, 50656 KB, 52167 KB/s, 0 seconds passed
... 40%, 50688 KB, 52180 KB/s, 0 seconds passed
... 40%, 50720 KB, 52194 KB/s, 0 seconds passed
... 40%, 50752 KB, 52206 KB/s, 0 seconds passed
... 40%, 50784 KB, 52219 KB/s, 0 seconds passed
... 40%, 50816 KB, 52233 KB/s, 0 seconds passed
... 40%, 50848 KB, 52246 KB/s, 0 seconds passed
... 40%, 50880 KB, 52259 KB/s, 0 seconds passed
... 40%, 50912 KB, 52273 KB/s, 0 seconds passed

.. parsed-literal::

    ... 40%, 50944 KB, 52286 KB/s, 0 seconds passed
... 40%, 50976 KB, 52303 KB/s, 0 seconds passed
... 40%, 51008 KB, 52320 KB/s, 0 seconds passed
... 40%, 51040 KB, 52338 KB/s, 0 seconds passed
... 40%, 51072 KB, 52356 KB/s, 0 seconds passed
... 40%, 51104 KB, 52374 KB/s, 0 seconds passed
... 40%, 51136 KB, 52391 KB/s, 0 seconds passed
... 40%, 51168 KB, 52409 KB/s, 0 seconds passed
... 40%, 51200 KB, 51274 KB/s, 0 seconds passed
... 40%, 51232 KB, 51281 KB/s, 0 seconds passed
... 40%, 51264 KB, 51293 KB/s, 0 seconds passed
... 40%, 51296 KB, 51307 KB/s, 0 seconds passed
... 40%, 51328 KB, 51300 KB/s, 1 seconds passed
... 40%, 51360 KB, 51312 KB/s, 1 seconds passed
... 40%, 51392 KB, 51323 KB/s, 1 seconds passed
... 40%, 51424 KB, 51337 KB/s, 1 seconds passed
... 40%, 51456 KB, 51352 KB/s, 1 seconds passed
... 40%, 51488 KB, 51336 KB/s, 1 seconds passed
... 40%, 51520 KB, 51228 KB/s, 1 seconds passed
... 40%, 51552 KB, 51237 KB/s, 1 seconds passed
... 40%, 51584 KB, 51248 KB/s, 1 seconds passed
... 40%, 51616 KB, 51260 KB/s, 1 seconds passed
... 41%, 51648 KB, 51273 KB/s, 1 seconds passed
... 41%, 51680 KB, 51285 KB/s, 1 seconds passed
... 41%, 51712 KB, 51297 KB/s, 1 seconds passed
... 41%, 51744 KB, 51309 KB/s, 1 seconds passed
... 41%, 51776 KB, 51321 KB/s, 1 seconds passed
... 41%, 51808 KB, 51333 KB/s, 1 seconds passed
... 41%, 51840 KB, 51346 KB/s, 1 seconds passed
... 41%, 51872 KB, 51358 KB/s, 1 seconds passed
... 41%, 51904 KB, 51372 KB/s, 1 seconds passed
... 41%, 51936 KB, 51385 KB/s, 1 seconds passed
... 41%, 51968 KB, 51396 KB/s, 1 seconds passed
... 41%, 52000 KB, 51408 KB/s, 1 seconds passed
... 41%, 52032 KB, 51421 KB/s, 1 seconds passed
... 41%, 52064 KB, 51433 KB/s, 1 seconds passed
... 41%, 52096 KB, 51446 KB/s, 1 seconds passed
... 41%, 52128 KB, 51458 KB/s, 1 seconds passed
... 41%, 52160 KB, 51470 KB/s, 1 seconds passed
... 41%, 52192 KB, 51483 KB/s, 1 seconds passed
... 41%, 52224 KB, 51495 KB/s, 1 seconds passed
... 41%, 52256 KB, 51508 KB/s, 1 seconds passed
... 41%, 52288 KB, 51520 KB/s, 1 seconds passed
... 41%, 52320 KB, 51533 KB/s, 1 seconds passed
... 41%, 52352 KB, 51546 KB/s, 1 seconds passed
... 41%, 52384 KB, 51556 KB/s, 1 seconds passed
... 41%, 52416 KB, 51569 KB/s, 1 seconds passed
... 41%, 52448 KB, 51588 KB/s, 1 seconds passed
... 41%, 52480 KB, 51607 KB/s, 1 seconds passed
... 41%, 52512 KB, 51626 KB/s, 1 seconds passed
... 41%, 52544 KB, 51645 KB/s, 1 seconds passed
... 41%, 52576 KB, 51664 KB/s, 1 seconds passed
... 41%, 52608 KB, 51682 KB/s, 1 seconds passed
... 41%, 52640 KB, 51701 KB/s, 1 seconds passed
... 41%, 52672 KB, 51720 KB/s, 1 seconds passed
... 41%, 52704 KB, 51738 KB/s, 1 seconds passed
... 41%, 52736 KB, 51757 KB/s, 1 seconds passed
... 41%, 52768 KB, 51775 KB/s, 1 seconds passed
... 41%, 52800 KB, 51794 KB/s, 1 seconds passed
... 41%, 52832 KB, 51813 KB/s, 1 seconds passed
... 41%, 52864 KB, 51833 KB/s, 1 seconds passed
... 41%, 52896 KB, 51850 KB/s, 1 seconds passed
... 42%, 52928 KB, 51870 KB/s, 1 seconds passed
... 42%, 52960 KB, 51888 KB/s, 1 seconds passed
... 42%, 52992 KB, 51907 KB/s, 1 seconds passed
... 42%, 53024 KB, 51926 KB/s, 1 seconds passed
... 42%, 53056 KB, 51945 KB/s, 1 seconds passed
... 42%, 53088 KB, 51964 KB/s, 1 seconds passed
... 42%, 53120 KB, 51983 KB/s, 1 seconds passed
... 42%, 53152 KB, 52003 KB/s, 1 seconds passed
... 42%, 53184 KB, 52021 KB/s, 1 seconds passed
... 42%, 53216 KB, 52041 KB/s, 1 seconds passed
... 42%, 53248 KB, 52060 KB/s, 1 seconds passed
... 42%, 53280 KB, 52079 KB/s, 1 seconds passed
... 42%, 53312 KB, 52098 KB/s, 1 seconds passed
... 42%, 53344 KB, 52117 KB/s, 1 seconds passed
... 42%, 53376 KB, 52136 KB/s, 1 seconds passed
... 42%, 53408 KB, 52154 KB/s, 1 seconds passed
... 42%, 53440 KB, 52173 KB/s, 1 seconds passed
... 42%, 53472 KB, 52192 KB/s, 1 seconds passed
... 42%, 53504 KB, 52210 KB/s, 1 seconds passed
... 42%, 53536 KB, 52230 KB/s, 1 seconds passed
... 42%, 53568 KB, 52249 KB/s, 1 seconds passed
... 42%, 53600 KB, 52268 KB/s, 1 seconds passed

.. parsed-literal::

    ... 42%, 53632 KB, 52287 KB/s, 1 seconds passed
... 42%, 53664 KB, 52306 KB/s, 1 seconds passed
... 42%, 53696 KB, 52325 KB/s, 1 seconds passed
... 42%, 53728 KB, 52348 KB/s, 1 seconds passed
... 42%, 53760 KB, 52370 KB/s, 1 seconds passed
... 42%, 53792 KB, 52392 KB/s, 1 seconds passed
... 42%, 53824 KB, 52415 KB/s, 1 seconds passed
... 42%, 53856 KB, 52438 KB/s, 1 seconds passed
... 42%, 53888 KB, 52461 KB/s, 1 seconds passed
... 42%, 53920 KB, 52483 KB/s, 1 seconds passed
... 42%, 53952 KB, 52506 KB/s, 1 seconds passed
... 42%, 53984 KB, 52528 KB/s, 1 seconds passed
... 42%, 54016 KB, 52551 KB/s, 1 seconds passed
... 42%, 54048 KB, 52572 KB/s, 1 seconds passed
... 42%, 54080 KB, 52595 KB/s, 1 seconds passed
... 42%, 54112 KB, 52617 KB/s, 1 seconds passed
... 42%, 54144 KB, 52640 KB/s, 1 seconds passed
... 43%, 54176 KB, 52658 KB/s, 1 seconds passed
... 43%, 54208 KB, 52676 KB/s, 1 seconds passed
... 43%, 54240 KB, 52693 KB/s, 1 seconds passed
... 43%, 54272 KB, 52711 KB/s, 1 seconds passed
... 43%, 54304 KB, 52725 KB/s, 1 seconds passed
... 43%, 54336 KB, 52748 KB/s, 1 seconds passed
... 43%, 54368 KB, 52764 KB/s, 1 seconds passed
... 43%, 54400 KB, 52774 KB/s, 1 seconds passed
... 43%, 54432 KB, 52784 KB/s, 1 seconds passed
... 43%, 54464 KB, 52795 KB/s, 1 seconds passed
... 43%, 54496 KB, 52817 KB/s, 1 seconds passed
... 43%, 54528 KB, 52840 KB/s, 1 seconds passed
... 43%, 54560 KB, 52859 KB/s, 1 seconds passed
... 43%, 54592 KB, 52881 KB/s, 1 seconds passed
... 43%, 54624 KB, 52899 KB/s, 1 seconds passed
... 43%, 54656 KB, 52917 KB/s, 1 seconds passed
... 43%, 54688 KB, 52934 KB/s, 1 seconds passed
... 43%, 54720 KB, 52952 KB/s, 1 seconds passed
... 43%, 54752 KB, 52966 KB/s, 1 seconds passed
... 43%, 54784 KB, 52984 KB/s, 1 seconds passed
... 43%, 54816 KB, 53001 KB/s, 1 seconds passed
... 43%, 54848 KB, 53016 KB/s, 1 seconds passed
... 43%, 54880 KB, 53034 KB/s, 1 seconds passed
... 43%, 54912 KB, 53048 KB/s, 1 seconds passed
... 43%, 54944 KB, 53066 KB/s, 1 seconds passed
... 43%, 54976 KB, 53083 KB/s, 1 seconds passed
... 43%, 55008 KB, 53098 KB/s, 1 seconds passed
... 43%, 55040 KB, 53115 KB/s, 1 seconds passed
... 43%, 55072 KB, 53130 KB/s, 1 seconds passed
... 43%, 55104 KB, 53147 KB/s, 1 seconds passed
... 43%, 55136 KB, 52987 KB/s, 1 seconds passed
... 43%, 55168 KB, 52996 KB/s, 1 seconds passed
... 43%, 55200 KB, 53007 KB/s, 1 seconds passed
... 43%, 55232 KB, 53017 KB/s, 1 seconds passed
... 43%, 55264 KB, 53041 KB/s, 1 seconds passed
... 43%, 55296 KB, 53065 KB/s, 1 seconds passed
... 43%, 55328 KB, 53082 KB/s, 1 seconds passed
... 43%, 55360 KB, 53100 KB/s, 1 seconds passed
... 43%, 55392 KB, 53117 KB/s, 1 seconds passed
... 44%, 55424 KB, 53134 KB/s, 1 seconds passed
... 44%, 55456 KB, 53152 KB/s, 1 seconds passed
... 44%, 55488 KB, 51653 KB/s, 1 seconds passed
... 44%, 55520 KB, 51658 KB/s, 1 seconds passed
... 44%, 55552 KB, 51669 KB/s, 1 seconds passed
... 44%, 55584 KB, 51682 KB/s, 1 seconds passed
... 44%, 55616 KB, 51666 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 55648 KB, 51677 KB/s, 1 seconds passed
... 44%, 55680 KB, 51689 KB/s, 1 seconds passed
... 44%, 55712 KB, 51701 KB/s, 1 seconds passed
... 44%, 55744 KB, 51712 KB/s, 1 seconds passed
... 44%, 55776 KB, 51724 KB/s, 1 seconds passed
... 44%, 55808 KB, 51736 KB/s, 1 seconds passed
... 44%, 55840 KB, 51748 KB/s, 1 seconds passed
... 44%, 55872 KB, 51760 KB/s, 1 seconds passed
... 44%, 55904 KB, 51772 KB/s, 1 seconds passed
... 44%, 55936 KB, 51779 KB/s, 1 seconds passed
... 44%, 55968 KB, 51790 KB/s, 1 seconds passed
... 44%, 56000 KB, 51802 KB/s, 1 seconds passed
... 44%, 56032 KB, 51813 KB/s, 1 seconds passed
... 44%, 56064 KB, 51824 KB/s, 1 seconds passed
... 44%, 56096 KB, 51836 KB/s, 1 seconds passed
... 44%, 56128 KB, 51847 KB/s, 1 seconds passed
... 44%, 56160 KB, 51861 KB/s, 1 seconds passed
... 44%, 56192 KB, 51877 KB/s, 1 seconds passed
... 44%, 56224 KB, 51892 KB/s, 1 seconds passed
... 44%, 56256 KB, 51909 KB/s, 1 seconds passed
... 44%, 56288 KB, 51923 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 56320 KB, 49866 KB/s, 1 seconds passed
... 44%, 56352 KB, 49873 KB/s, 1 seconds passed
... 44%, 56384 KB, 49880 KB/s, 1 seconds passed
... 44%, 56416 KB, 49891 KB/s, 1 seconds passed
... 44%, 56448 KB, 49904 KB/s, 1 seconds passed
... 44%, 56480 KB, 49917 KB/s, 1 seconds passed
... 44%, 56512 KB, 49929 KB/s, 1 seconds passed
... 44%, 56544 KB, 49939 KB/s, 1 seconds passed
... 44%, 56576 KB, 49951 KB/s, 1 seconds passed
... 44%, 56608 KB, 49962 KB/s, 1 seconds passed
... 44%, 56640 KB, 49972 KB/s, 1 seconds passed
... 44%, 56672 KB, 49984 KB/s, 1 seconds passed
... 45%, 56704 KB, 49995 KB/s, 1 seconds passed
... 45%, 56736 KB, 50006 KB/s, 1 seconds passed
... 45%, 56768 KB, 50018 KB/s, 1 seconds passed
... 45%, 56800 KB, 50029 KB/s, 1 seconds passed
... 45%, 56832 KB, 50040 KB/s, 1 seconds passed
... 45%, 56864 KB, 50050 KB/s, 1 seconds passed
... 45%, 56896 KB, 50061 KB/s, 1 seconds passed
... 45%, 56928 KB, 50073 KB/s, 1 seconds passed
... 45%, 56960 KB, 50084 KB/s, 1 seconds passed
... 45%, 56992 KB, 50095 KB/s, 1 seconds passed
... 45%, 57024 KB, 50108 KB/s, 1 seconds passed
... 45%, 57056 KB, 50121 KB/s, 1 seconds passed
... 45%, 57088 KB, 50134 KB/s, 1 seconds passed
... 45%, 57120 KB, 50146 KB/s, 1 seconds passed
... 45%, 57152 KB, 50159 KB/s, 1 seconds passed
... 45%, 57184 KB, 50171 KB/s, 1 seconds passed
... 45%, 57216 KB, 50183 KB/s, 1 seconds passed
... 45%, 57248 KB, 50194 KB/s, 1 seconds passed
... 45%, 57280 KB, 50207 KB/s, 1 seconds passed
... 45%, 57312 KB, 50220 KB/s, 1 seconds passed
... 45%, 57344 KB, 50232 KB/s, 1 seconds passed
... 45%, 57376 KB, 50244 KB/s, 1 seconds passed
... 45%, 57408 KB, 50257 KB/s, 1 seconds passed
... 45%, 57440 KB, 50270 KB/s, 1 seconds passed
... 45%, 57472 KB, 50283 KB/s, 1 seconds passed
... 45%, 57504 KB, 50296 KB/s, 1 seconds passed
... 45%, 57536 KB, 50309 KB/s, 1 seconds passed
... 45%, 57568 KB, 50321 KB/s, 1 seconds passed
... 45%, 57600 KB, 50333 KB/s, 1 seconds passed
... 45%, 57632 KB, 50347 KB/s, 1 seconds passed
... 45%, 57664 KB, 50360 KB/s, 1 seconds passed
... 45%, 57696 KB, 50373 KB/s, 1 seconds passed
... 45%, 57728 KB, 50385 KB/s, 1 seconds passed
... 45%, 57760 KB, 50398 KB/s, 1 seconds passed
... 45%, 57792 KB, 50411 KB/s, 1 seconds passed
... 45%, 57824 KB, 50424 KB/s, 1 seconds passed
... 45%, 57856 KB, 50436 KB/s, 1 seconds passed
... 45%, 57888 KB, 50449 KB/s, 1 seconds passed
... 45%, 57920 KB, 50465 KB/s, 1 seconds passed
... 46%, 57952 KB, 50482 KB/s, 1 seconds passed
... 46%, 57984 KB, 50499 KB/s, 1 seconds passed
... 46%, 58016 KB, 50517 KB/s, 1 seconds passed
... 46%, 58048 KB, 50535 KB/s, 1 seconds passed
... 46%, 58080 KB, 50553 KB/s, 1 seconds passed
... 46%, 58112 KB, 50570 KB/s, 1 seconds passed
... 46%, 58144 KB, 50588 KB/s, 1 seconds passed
... 46%, 58176 KB, 50606 KB/s, 1 seconds passed
... 46%, 58208 KB, 50624 KB/s, 1 seconds passed
... 46%, 58240 KB, 50642 KB/s, 1 seconds passed
... 46%, 58272 KB, 50660 KB/s, 1 seconds passed
... 46%, 58304 KB, 50677 KB/s, 1 seconds passed
... 46%, 58336 KB, 50695 KB/s, 1 seconds passed
... 46%, 58368 KB, 50712 KB/s, 1 seconds passed
... 46%, 58400 KB, 50729 KB/s, 1 seconds passed
... 46%, 58432 KB, 50747 KB/s, 1 seconds passed
... 46%, 58464 KB, 50765 KB/s, 1 seconds passed
... 46%, 58496 KB, 50782 KB/s, 1 seconds passed
... 46%, 58528 KB, 50799 KB/s, 1 seconds passed
... 46%, 58560 KB, 50817 KB/s, 1 seconds passed
... 46%, 58592 KB, 50834 KB/s, 1 seconds passed
... 46%, 58624 KB, 50852 KB/s, 1 seconds passed
... 46%, 58656 KB, 50869 KB/s, 1 seconds passed
... 46%, 58688 KB, 50887 KB/s, 1 seconds passed
... 46%, 58720 KB, 50905 KB/s, 1 seconds passed
... 46%, 58752 KB, 50923 KB/s, 1 seconds passed
... 46%, 58784 KB, 50940 KB/s, 1 seconds passed
... 46%, 58816 KB, 50958 KB/s, 1 seconds passed
... 46%, 58848 KB, 50974 KB/s, 1 seconds passed
... 46%, 58880 KB, 50992 KB/s, 1 seconds passed
... 46%, 58912 KB, 51010 KB/s, 1 seconds passed
... 46%, 58944 KB, 51027 KB/s, 1 seconds passed
... 46%, 58976 KB, 51045 KB/s, 1 seconds passed
... 46%, 59008 KB, 51063 KB/s, 1 seconds passed
... 46%, 59040 KB, 51080 KB/s, 1 seconds passed
... 46%, 59072 KB, 51097 KB/s, 1 seconds passed
... 46%, 59104 KB, 51114 KB/s, 1 seconds passed
... 46%, 59136 KB, 51132 KB/s, 1 seconds passed
... 46%, 59168 KB, 51149 KB/s, 1 seconds passed
... 47%, 59200 KB, 51167 KB/s, 1 seconds passed
... 47%, 59232 KB, 51185 KB/s, 1 seconds passed
... 47%, 59264 KB, 51202 KB/s, 1 seconds passed
... 47%, 59296 KB, 51218 KB/s, 1 seconds passed
... 47%, 59328 KB, 51239 KB/s, 1 seconds passed
... 47%, 59360 KB, 51260 KB/s, 1 seconds passed
... 47%, 59392 KB, 51281 KB/s, 1 seconds passed
... 47%, 59424 KB, 51302 KB/s, 1 seconds passed
... 47%, 59456 KB, 51323 KB/s, 1 seconds passed
... 47%, 59488 KB, 51344 KB/s, 1 seconds passed
... 47%, 59520 KB, 51365 KB/s, 1 seconds passed
... 47%, 59552 KB, 51385 KB/s, 1 seconds passed
... 47%, 59584 KB, 51406 KB/s, 1 seconds passed
... 47%, 59616 KB, 51427 KB/s, 1 seconds passed
... 47%, 59648 KB, 51447 KB/s, 1 seconds passed
... 47%, 59680 KB, 51468 KB/s, 1 seconds passed
... 47%, 59712 KB, 51489 KB/s, 1 seconds passed
... 47%, 59744 KB, 51509 KB/s, 1 seconds passed
... 47%, 59776 KB, 51529 KB/s, 1 seconds passed
... 47%, 59808 KB, 51546 KB/s, 1 seconds passed
... 47%, 59840 KB, 51562 KB/s, 1 seconds passed
... 47%, 59872 KB, 51577 KB/s, 1 seconds passed
... 47%, 59904 KB, 51591 KB/s, 1 seconds passed
... 47%, 59936 KB, 51602 KB/s, 1 seconds passed
... 47%, 59968 KB, 51622 KB/s, 1 seconds passed
... 47%, 60000 KB, 51636 KB/s, 1 seconds passed
... 47%, 60032 KB, 51652 KB/s, 1 seconds passed
... 47%, 60064 KB, 51663 KB/s, 1 seconds passed
... 47%, 60096 KB, 51679 KB/s, 1 seconds passed
... 47%, 60128 KB, 51697 KB/s, 1 seconds passed
... 47%, 60160 KB, 51710 KB/s, 1 seconds passed
... 47%, 60192 KB, 51729 KB/s, 1 seconds passed
... 47%, 60224 KB, 51744 KB/s, 1 seconds passed
... 47%, 60256 KB, 51621 KB/s, 1 seconds passed
... 47%, 60288 KB, 51637 KB/s, 1 seconds passed
... 47%, 60320 KB, 51651 KB/s, 1 seconds passed
... 47%, 60352 KB, 51661 KB/s, 1 seconds passed
... 47%, 60384 KB, 51671 KB/s, 1 seconds passed
... 47%, 60416 KB, 51682 KB/s, 1 seconds passed
... 47%, 60448 KB, 51694 KB/s, 1 seconds passed
... 48%, 60480 KB, 51694 KB/s, 1 seconds passed
... 48%, 60512 KB, 51706 KB/s, 1 seconds passed
... 48%, 60544 KB, 51716 KB/s, 1 seconds passed
... 48%, 60576 KB, 51727 KB/s, 1 seconds passed
... 48%, 60608 KB, 51738 KB/s, 1 seconds passed
... 48%, 60640 KB, 51749 KB/s, 1 seconds passed
... 48%, 60672 KB, 51758 KB/s, 1 seconds passed
... 48%, 60704 KB, 51769 KB/s, 1 seconds passed
... 48%, 60736 KB, 51780 KB/s, 1 seconds passed
... 48%, 60768 KB, 51791 KB/s, 1 seconds passed
... 48%, 60800 KB, 51803 KB/s, 1 seconds passed
... 48%, 60832 KB, 51814 KB/s, 1 seconds passed
... 48%, 60864 KB, 51825 KB/s, 1 seconds passed
... 48%, 60896 KB, 51836 KB/s, 1 seconds passed
... 48%, 60928 KB, 51847 KB/s, 1 seconds passed
... 48%, 60960 KB, 51857 KB/s, 1 seconds passed
... 48%, 60992 KB, 51867 KB/s, 1 seconds passed
... 48%, 61024 KB, 51877 KB/s, 1 seconds passed
... 48%, 61056 KB, 51888 KB/s, 1 seconds passed
... 48%, 61088 KB, 51898 KB/s, 1 seconds passed
... 48%, 61120 KB, 51909 KB/s, 1 seconds passed
... 48%, 61152 KB, 51921 KB/s, 1 seconds passed
... 48%, 61184 KB, 51938 KB/s, 1 seconds passed
... 48%, 61216 KB, 51954 KB/s, 1 seconds passed
... 48%, 61248 KB, 51972 KB/s, 1 seconds passed
... 48%, 61280 KB, 51990 KB/s, 1 seconds passed
... 48%, 61312 KB, 52008 KB/s, 1 seconds passed
... 48%, 61344 KB, 52027 KB/s, 1 seconds passed
... 48%, 61376 KB, 52046 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 61408 KB, 52065 KB/s, 1 seconds passed
... 48%, 61440 KB, 51476 KB/s, 1 seconds passed
... 48%, 61472 KB, 51469 KB/s, 1 seconds passed
... 48%, 61504 KB, 51477 KB/s, 1 seconds passed
... 48%, 61536 KB, 51486 KB/s, 1 seconds passed
... 48%, 61568 KB, 51495 KB/s, 1 seconds passed
... 48%, 61600 KB, 51508 KB/s, 1 seconds passed
... 48%, 61632 KB, 51495 KB/s, 1 seconds passed
... 48%, 61664 KB, 51506 KB/s, 1 seconds passed
... 48%, 61696 KB, 51517 KB/s, 1 seconds passed
... 49%, 61728 KB, 51528 KB/s, 1 seconds passed
... 49%, 61760 KB, 51538 KB/s, 1 seconds passed
... 49%, 61792 KB, 51548 KB/s, 1 seconds passed
... 49%, 61824 KB, 51558 KB/s, 1 seconds passed
... 49%, 61856 KB, 51569 KB/s, 1 seconds passed
... 49%, 61888 KB, 51579 KB/s, 1 seconds passed
... 49%, 61920 KB, 51587 KB/s, 1 seconds passed
... 49%, 61952 KB, 51597 KB/s, 1 seconds passed
... 49%, 61984 KB, 51608 KB/s, 1 seconds passed
... 49%, 62016 KB, 51618 KB/s, 1 seconds passed
... 49%, 62048 KB, 51630 KB/s, 1 seconds passed
... 49%, 62080 KB, 51639 KB/s, 1 seconds passed
... 49%, 62112 KB, 51650 KB/s, 1 seconds passed
... 49%, 62144 KB, 51660 KB/s, 1 seconds passed
... 49%, 62176 KB, 51670 KB/s, 1 seconds passed
... 49%, 62208 KB, 51682 KB/s, 1 seconds passed
... 49%, 62240 KB, 51690 KB/s, 1 seconds passed
... 49%, 62272 KB, 51700 KB/s, 1 seconds passed
... 49%, 62304 KB, 51711 KB/s, 1 seconds passed
... 49%, 62336 KB, 51721 KB/s, 1 seconds passed
... 49%, 62368 KB, 51732 KB/s, 1 seconds passed
... 49%, 62400 KB, 51743 KB/s, 1 seconds passed
... 49%, 62432 KB, 51754 KB/s, 1 seconds passed
... 49%, 62464 KB, 51764 KB/s, 1 seconds passed
... 49%, 62496 KB, 51775 KB/s, 1 seconds passed
... 49%, 62528 KB, 51785 KB/s, 1 seconds passed
... 49%, 62560 KB, 51795 KB/s, 1 seconds passed
... 49%, 62592 KB, 51804 KB/s, 1 seconds passed
... 49%, 62624 KB, 51815 KB/s, 1 seconds passed
... 49%, 62656 KB, 51829 KB/s, 1 seconds passed
... 49%, 62688 KB, 51845 KB/s, 1 seconds passed
... 49%, 62720 KB, 51861 KB/s, 1 seconds passed
... 49%, 62752 KB, 51877 KB/s, 1 seconds passed
... 49%, 62784 KB, 51893 KB/s, 1 seconds passed
... 49%, 62816 KB, 51909 KB/s, 1 seconds passed
... 49%, 62848 KB, 51925 KB/s, 1 seconds passed
... 49%, 62880 KB, 51940 KB/s, 1 seconds passed
... 49%, 62912 KB, 51956 KB/s, 1 seconds passed
... 49%, 62944 KB, 51973 KB/s, 1 seconds passed
... 49%, 62976 KB, 51989 KB/s, 1 seconds passed
... 50%, 63008 KB, 52005 KB/s, 1 seconds passed
... 50%, 63040 KB, 52022 KB/s, 1 seconds passed
... 50%, 63072 KB, 52036 KB/s, 1 seconds passed
... 50%, 63104 KB, 52052 KB/s, 1 seconds passed
... 50%, 63136 KB, 52068 KB/s, 1 seconds passed
... 50%, 63168 KB, 52084 KB/s, 1 seconds passed
... 50%, 63200 KB, 52100 KB/s, 1 seconds passed
... 50%, 63232 KB, 52116 KB/s, 1 seconds passed
... 50%, 63264 KB, 52132 KB/s, 1 seconds passed
... 50%, 63296 KB, 52148 KB/s, 1 seconds passed
... 50%, 63328 KB, 52164 KB/s, 1 seconds passed
... 50%, 63360 KB, 52180 KB/s, 1 seconds passed
... 50%, 63392 KB, 52196 KB/s, 1 seconds passed
... 50%, 63424 KB, 52212 KB/s, 1 seconds passed
... 50%, 63456 KB, 52228 KB/s, 1 seconds passed
... 50%, 63488 KB, 52244 KB/s, 1 seconds passed
... 50%, 63520 KB, 52261 KB/s, 1 seconds passed
... 50%, 63552 KB, 52277 KB/s, 1 seconds passed
... 50%, 63584 KB, 52293 KB/s, 1 seconds passed
... 50%, 63616 KB, 52308 KB/s, 1 seconds passed
... 50%, 63648 KB, 52325 KB/s, 1 seconds passed
... 50%, 63680 KB, 52341 KB/s, 1 seconds passed
... 50%, 63712 KB, 52357 KB/s, 1 seconds passed
... 50%, 63744 KB, 52372 KB/s, 1 seconds passed
... 50%, 63776 KB, 52389 KB/s, 1 seconds passed
... 50%, 63808 KB, 52405 KB/s, 1 seconds passed
... 50%, 63840 KB, 52421 KB/s, 1 seconds passed
... 50%, 63872 KB, 52437 KB/s, 1 seconds passed
... 50%, 63904 KB, 52453 KB/s, 1 seconds passed
... 50%, 63936 KB, 52469 KB/s, 1 seconds passed
... 50%, 63968 KB, 52486 KB/s, 1 seconds passed
... 50%, 64000 KB, 52504 KB/s, 1 seconds passed
... 50%, 64032 KB, 52523 KB/s, 1 seconds passed
... 50%, 64064 KB, 52543 KB/s, 1 seconds passed
... 50%, 64096 KB, 52562 KB/s, 1 seconds passed
... 50%, 64128 KB, 52581 KB/s, 1 seconds passed
... 50%, 64160 KB, 52600 KB/s, 1 seconds passed
... 50%, 64192 KB, 52620 KB/s, 1 seconds passed
... 50%, 64224 KB, 52637 KB/s, 1 seconds passed
... 51%, 64256 KB, 52656 KB/s, 1 seconds passed
... 51%, 64288 KB, 52675 KB/s, 1 seconds passed
... 51%, 64320 KB, 52694 KB/s, 1 seconds passed
... 51%, 64352 KB, 52713 KB/s, 1 seconds passed
... 51%, 64384 KB, 52732 KB/s, 1 seconds passed
... 51%, 64416 KB, 52751 KB/s, 1 seconds passed
... 51%, 64448 KB, 52769 KB/s, 1 seconds passed
... 51%, 64480 KB, 52781 KB/s, 1 seconds passed
... 51%, 64512 KB, 52799 KB/s, 1 seconds passed
... 51%, 64544 KB, 52813 KB/s, 1 seconds passed
... 51%, 64576 KB, 52826 KB/s, 1 seconds passed
... 51%, 64608 KB, 52841 KB/s, 1 seconds passed
... 51%, 64640 KB, 52855 KB/s, 1 seconds passed
... 51%, 64672 KB, 52868 KB/s, 1 seconds passed
... 51%, 64704 KB, 52883 KB/s, 1 seconds passed
... 51%, 64736 KB, 52895 KB/s, 1 seconds passed
... 51%, 64768 KB, 52910 KB/s, 1 seconds passed
... 51%, 64800 KB, 52925 KB/s, 1 seconds passed
... 51%, 64832 KB, 52940 KB/s, 1 seconds passed
... 51%, 64864 KB, 52952 KB/s, 1 seconds passed
... 51%, 64896 KB, 52967 KB/s, 1 seconds passed
... 51%, 64928 KB, 52979 KB/s, 1 seconds passed
... 51%, 64960 KB, 52994 KB/s, 1 seconds passed
... 51%, 64992 KB, 53009 KB/s, 1 seconds passed
... 51%, 65024 KB, 53026 KB/s, 1 seconds passed
... 51%, 65056 KB, 53038 KB/s, 1 seconds passed
... 51%, 65088 KB, 53053 KB/s, 1 seconds passed
... 51%, 65120 KB, 53068 KB/s, 1 seconds passed
... 51%, 65152 KB, 53080 KB/s, 1 seconds passed
... 51%, 65184 KB, 53095 KB/s, 1 seconds passed
... 51%, 65216 KB, 53103 KB/s, 1 seconds passed
... 51%, 65248 KB, 53119 KB/s, 1 seconds passed
... 51%, 65280 KB, 53134 KB/s, 1 seconds passed
... 51%, 65312 KB, 53150 KB/s, 1 seconds passed

.. parsed-literal::

    ... 51%, 65344 KB, 53065 KB/s, 1 seconds passed
... 51%, 65376 KB, 53075 KB/s, 1 seconds passed
... 51%, 65408 KB, 53083 KB/s, 1 seconds passed
... 51%, 65440 KB, 53095 KB/s, 1 seconds passed
... 51%, 65472 KB, 53110 KB/s, 1 seconds passed
... 52%, 65504 KB, 53124 KB/s, 1 seconds passed
... 52%, 65536 KB, 53137 KB/s, 1 seconds passed
... 52%, 65568 KB, 53124 KB/s, 1 seconds passed
... 52%, 65600 KB, 53136 KB/s, 1 seconds passed
... 52%, 65632 KB, 53151 KB/s, 1 seconds passed
... 52%, 65664 KB, 53166 KB/s, 1 seconds passed
... 52%, 65696 KB, 53178 KB/s, 1 seconds passed
... 52%, 65728 KB, 53193 KB/s, 1 seconds passed
... 52%, 65760 KB, 53208 KB/s, 1 seconds passed
... 52%, 65792 KB, 53220 KB/s, 1 seconds passed
... 52%, 65824 KB, 53235 KB/s, 1 seconds passed
... 52%, 65856 KB, 53249 KB/s, 1 seconds passed
... 52%, 65888 KB, 53262 KB/s, 1 seconds passed
... 52%, 65920 KB, 53279 KB/s, 1 seconds passed
... 52%, 65952 KB, 53291 KB/s, 1 seconds passed
... 52%, 65984 KB, 53305 KB/s, 1 seconds passed
... 52%, 66016 KB, 53318 KB/s, 1 seconds passed
... 52%, 66048 KB, 53332 KB/s, 1 seconds passed
... 52%, 66080 KB, 53347 KB/s, 1 seconds passed
... 52%, 66112 KB, 53360 KB/s, 1 seconds passed
... 52%, 66144 KB, 53373 KB/s, 1 seconds passed
... 52%, 66176 KB, 53383 KB/s, 1 seconds passed
... 52%, 66208 KB, 53397 KB/s, 1 seconds passed
... 52%, 66240 KB, 53410 KB/s, 1 seconds passed
... 52%, 66272 KB, 53424 KB/s, 1 seconds passed
... 52%, 66304 KB, 53441 KB/s, 1 seconds passed
... 52%, 66336 KB, 53455 KB/s, 1 seconds passed
... 52%, 66368 KB, 53468 KB/s, 1 seconds passed
... 52%, 66400 KB, 53482 KB/s, 1 seconds passed
... 52%, 66432 KB, 53498 KB/s, 1 seconds passed
... 52%, 66464 KB, 53510 KB/s, 1 seconds passed
... 52%, 66496 KB, 53525 KB/s, 1 seconds passed
... 52%, 66528 KB, 53539 KB/s, 1 seconds passed
... 52%, 66560 KB, 52352 KB/s, 1 seconds passed
... 52%, 66592 KB, 52360 KB/s, 1 seconds passed
... 52%, 66624 KB, 52368 KB/s, 1 seconds passed
... 52%, 66656 KB, 52375 KB/s, 1 seconds passed
... 52%, 66688 KB, 52384 KB/s, 1 seconds passed
... 52%, 66720 KB, 52394 KB/s, 1 seconds passed
... 52%, 66752 KB, 52404 KB/s, 1 seconds passed
... 53%, 66784 KB, 52415 KB/s, 1 seconds passed
... 53%, 66816 KB, 52428 KB/s, 1 seconds passed
... 53%, 66848 KB, 52441 KB/s, 1 seconds passed
... 53%, 66880 KB, 52424 KB/s, 1 seconds passed
... 53%, 66912 KB, 52431 KB/s, 1 seconds passed
... 53%, 66944 KB, 52441 KB/s, 1 seconds passed
... 53%, 66976 KB, 52451 KB/s, 1 seconds passed
... 53%, 67008 KB, 52461 KB/s, 1 seconds passed
... 53%, 67040 KB, 52470 KB/s, 1 seconds passed
... 53%, 67072 KB, 52480 KB/s, 1 seconds passed
... 53%, 67104 KB, 52490 KB/s, 1 seconds passed
... 53%, 67136 KB, 52498 KB/s, 1 seconds passed
... 53%, 67168 KB, 52508 KB/s, 1 seconds passed
... 53%, 67200 KB, 52518 KB/s, 1 seconds passed
... 53%, 67232 KB, 52528 KB/s, 1 seconds passed
... 53%, 67264 KB, 52537 KB/s, 1 seconds passed
... 53%, 67296 KB, 52547 KB/s, 1 seconds passed
... 53%, 67328 KB, 52556 KB/s, 1 seconds passed
... 53%, 67360 KB, 52565 KB/s, 1 seconds passed
... 53%, 67392 KB, 52574 KB/s, 1 seconds passed

.. parsed-literal::

    ... 53%, 67424 KB, 52584 KB/s, 1 seconds passed
... 53%, 67456 KB, 52594 KB/s, 1 seconds passed
... 53%, 67488 KB, 52603 KB/s, 1 seconds passed
... 53%, 67520 KB, 52612 KB/s, 1 seconds passed
... 53%, 67552 KB, 52623 KB/s, 1 seconds passed
... 53%, 67584 KB, 52631 KB/s, 1 seconds passed
... 53%, 67616 KB, 52641 KB/s, 1 seconds passed
... 53%, 67648 KB, 52651 KB/s, 1 seconds passed
... 53%, 67680 KB, 52661 KB/s, 1 seconds passed
... 53%, 67712 KB, 52671 KB/s, 1 seconds passed
... 53%, 67744 KB, 52680 KB/s, 1 seconds passed
... 53%, 67776 KB, 52690 KB/s, 1 seconds passed
... 53%, 67808 KB, 52700 KB/s, 1 seconds passed
... 53%, 67840 KB, 52709 KB/s, 1 seconds passed
... 53%, 67872 KB, 52719 KB/s, 1 seconds passed
... 53%, 67904 KB, 52728 KB/s, 1 seconds passed
... 53%, 67936 KB, 52737 KB/s, 1 seconds passed
... 53%, 67968 KB, 52746 KB/s, 1 seconds passed
... 53%, 68000 KB, 52756 KB/s, 1 seconds passed
... 54%, 68032 KB, 52766 KB/s, 1 seconds passed
... 54%, 68064 KB, 52781 KB/s, 1 seconds passed
... 54%, 68096 KB, 52795 KB/s, 1 seconds passed
... 54%, 68128 KB, 52810 KB/s, 1 seconds passed
... 54%, 68160 KB, 52825 KB/s, 1 seconds passed
... 54%, 68192 KB, 52839 KB/s, 1 seconds passed
... 54%, 68224 KB, 52854 KB/s, 1 seconds passed
... 54%, 68256 KB, 52868 KB/s, 1 seconds passed
... 54%, 68288 KB, 52883 KB/s, 1 seconds passed
... 54%, 68320 KB, 52898 KB/s, 1 seconds passed
... 54%, 68352 KB, 52913 KB/s, 1 seconds passed
... 54%, 68384 KB, 52927 KB/s, 1 seconds passed
... 54%, 68416 KB, 52941 KB/s, 1 seconds passed
... 54%, 68448 KB, 52956 KB/s, 1 seconds passed
... 54%, 68480 KB, 52971 KB/s, 1 seconds passed
... 54%, 68512 KB, 52985 KB/s, 1 seconds passed
... 54%, 68544 KB, 52999 KB/s, 1 seconds passed
... 54%, 68576 KB, 53014 KB/s, 1 seconds passed
... 54%, 68608 KB, 53029 KB/s, 1 seconds passed
... 54%, 68640 KB, 53044 KB/s, 1 seconds passed
... 54%, 68672 KB, 53058 KB/s, 1 seconds passed
... 54%, 68704 KB, 53072 KB/s, 1 seconds passed
... 54%, 68736 KB, 53087 KB/s, 1 seconds passed
... 54%, 68768 KB, 53102 KB/s, 1 seconds passed
... 54%, 68800 KB, 53117 KB/s, 1 seconds passed
... 54%, 68832 KB, 53131 KB/s, 1 seconds passed
... 54%, 68864 KB, 53146 KB/s, 1 seconds passed
... 54%, 68896 KB, 53160 KB/s, 1 seconds passed
... 54%, 68928 KB, 53174 KB/s, 1 seconds passed
... 54%, 68960 KB, 53189 KB/s, 1 seconds passed
... 54%, 68992 KB, 53204 KB/s, 1 seconds passed
... 54%, 69024 KB, 53219 KB/s, 1 seconds passed
... 54%, 69056 KB, 53233 KB/s, 1 seconds passed
... 54%, 69088 KB, 53248 KB/s, 1 seconds passed
... 54%, 69120 KB, 53263 KB/s, 1 seconds passed
... 54%, 69152 KB, 53277 KB/s, 1 seconds passed
... 54%, 69184 KB, 53292 KB/s, 1 seconds passed
... 54%, 69216 KB, 53307 KB/s, 1 seconds passed
... 54%, 69248 KB, 53321 KB/s, 1 seconds passed
... 55%, 69280 KB, 53335 KB/s, 1 seconds passed
... 55%, 69312 KB, 53350 KB/s, 1 seconds passed
... 55%, 69344 KB, 53367 KB/s, 1 seconds passed
... 55%, 69376 KB, 53384 KB/s, 1 seconds passed
... 55%, 69408 KB, 53402 KB/s, 1 seconds passed
... 55%, 69440 KB, 53420 KB/s, 1 seconds passed
... 55%, 69472 KB, 53437 KB/s, 1 seconds passed
... 55%, 69504 KB, 53454 KB/s, 1 seconds passed
... 55%, 69536 KB, 53472 KB/s, 1 seconds passed
... 55%, 69568 KB, 53490 KB/s, 1 seconds passed
... 55%, 69600 KB, 53508 KB/s, 1 seconds passed
... 55%, 69632 KB, 53526 KB/s, 1 seconds passed
... 55%, 69664 KB, 53543 KB/s, 1 seconds passed
... 55%, 69696 KB, 53561 KB/s, 1 seconds passed
... 55%, 69728 KB, 53578 KB/s, 1 seconds passed
... 55%, 69760 KB, 53596 KB/s, 1 seconds passed
... 55%, 69792 KB, 53614 KB/s, 1 seconds passed
... 55%, 69824 KB, 53632 KB/s, 1 seconds passed
... 55%, 69856 KB, 53650 KB/s, 1 seconds passed
... 55%, 69888 KB, 53668 KB/s, 1 seconds passed
... 55%, 69920 KB, 53685 KB/s, 1 seconds passed
... 55%, 69952 KB, 53703 KB/s, 1 seconds passed
... 55%, 69984 KB, 53720 KB/s, 1 seconds passed
... 55%, 70016 KB, 53732 KB/s, 1 seconds passed
... 55%, 70048 KB, 53740 KB/s, 1 seconds passed
... 55%, 70080 KB, 53756 KB/s, 1 seconds passed
... 55%, 70112 KB, 53772 KB/s, 1 seconds passed
... 55%, 70144 KB, 53784 KB/s, 1 seconds passed
... 55%, 70176 KB, 53797 KB/s, 1 seconds passed
... 55%, 70208 KB, 53809 KB/s, 1 seconds passed
... 55%, 70240 KB, 53823 KB/s, 1 seconds passed
... 55%, 70272 KB, 53836 KB/s, 1 seconds passed
... 55%, 70304 KB, 53848 KB/s, 1 seconds passed
... 55%, 70336 KB, 53861 KB/s, 1 seconds passed
... 55%, 70368 KB, 53875 KB/s, 1 seconds passed
... 55%, 70400 KB, 53889 KB/s, 1 seconds passed
... 55%, 70432 KB, 53900 KB/s, 1 seconds passed
... 55%, 70464 KB, 53912 KB/s, 1 seconds passed
... 55%, 70496 KB, 53925 KB/s, 1 seconds passed
... 55%, 70528 KB, 53939 KB/s, 1 seconds passed
... 56%, 70560 KB, 53950 KB/s, 1 seconds passed
... 56%, 70592 KB, 53963 KB/s, 1 seconds passed
... 56%, 70624 KB, 53977 KB/s, 1 seconds passed
... 56%, 70656 KB, 53993 KB/s, 1 seconds passed
... 56%, 70688 KB, 54006 KB/s, 1 seconds passed
... 56%, 70720 KB, 54020 KB/s, 1 seconds passed
... 56%, 70752 KB, 54033 KB/s, 1 seconds passed
... 56%, 70784 KB, 53916 KB/s, 1 seconds passed
... 56%, 70816 KB, 53930 KB/s, 1 seconds passed
... 56%, 70848 KB, 53943 KB/s, 1 seconds passed
... 56%, 70880 KB, 53957 KB/s, 1 seconds passed
... 56%, 70912 KB, 53878 KB/s, 1 seconds passed
... 56%, 70944 KB, 53885 KB/s, 1 seconds passed
... 56%, 70976 KB, 53899 KB/s, 1 seconds passed
... 56%, 71008 KB, 53918 KB/s, 1 seconds passed
... 56%, 71040 KB, 53932 KB/s, 1 seconds passed
... 56%, 71072 KB, 53943 KB/s, 1 seconds passed
... 56%, 71104 KB, 53957 KB/s, 1 seconds passed
... 56%, 71136 KB, 53970 KB/s, 1 seconds passed
... 56%, 71168 KB, 53984 KB/s, 1 seconds passed
... 56%, 71200 KB, 53995 KB/s, 1 seconds passed
... 56%, 71232 KB, 54009 KB/s, 1 seconds passed
... 56%, 71264 KB, 54020 KB/s, 1 seconds passed
... 56%, 71296 KB, 54034 KB/s, 1 seconds passed
... 56%, 71328 KB, 54047 KB/s, 1 seconds passed
... 56%, 71360 KB, 54058 KB/s, 1 seconds passed
... 56%, 71392 KB, 54074 KB/s, 1 seconds passed
... 56%, 71424 KB, 54085 KB/s, 1 seconds passed
... 56%, 71456 KB, 54096 KB/s, 1 seconds passed
... 56%, 71488 KB, 54110 KB/s, 1 seconds passed
... 56%, 71520 KB, 54123 KB/s, 1 seconds passed
... 56%, 71552 KB, 54137 KB/s, 1 seconds passed
... 56%, 71584 KB, 54148 KB/s, 1 seconds passed
... 56%, 71616 KB, 54161 KB/s, 1 seconds passed
... 56%, 71648 KB, 54177 KB/s, 1 seconds passed

.. parsed-literal::

    ... 56%, 71680 KB, 51120 KB/s, 1 seconds passed
... 56%, 71712 KB, 51123 KB/s, 1 seconds passed
... 56%, 71744 KB, 51132 KB/s, 1 seconds passed
... 56%, 71776 KB, 51140 KB/s, 1 seconds passed
... 57%, 71808 KB, 51149 KB/s, 1 seconds passed
... 57%, 71840 KB, 51157 KB/s, 1 seconds passed
... 57%, 71872 KB, 51166 KB/s, 1 seconds passed
... 57%, 71904 KB, 51175 KB/s, 1 seconds passed
... 57%, 71936 KB, 51184 KB/s, 1 seconds passed
... 57%, 71968 KB, 51193 KB/s, 1 seconds passed
... 57%, 72000 KB, 51202 KB/s, 1 seconds passed
... 57%, 72032 KB, 51211 KB/s, 1 seconds passed
... 57%, 72064 KB, 51220 KB/s, 1 seconds passed
... 57%, 72096 KB, 51228 KB/s, 1 seconds passed
... 57%, 72128 KB, 51237 KB/s, 1 seconds passed
... 57%, 72160 KB, 51244 KB/s, 1 seconds passed
... 57%, 72192 KB, 51253 KB/s, 1 seconds passed
... 57%, 72224 KB, 51262 KB/s, 1 seconds passed
... 57%, 72256 KB, 51271 KB/s, 1 seconds passed
... 57%, 72288 KB, 51278 KB/s, 1 seconds passed
... 57%, 72320 KB, 51287 KB/s, 1 seconds passed
... 57%, 72352 KB, 51296 KB/s, 1 seconds passed
... 57%, 72384 KB, 51306 KB/s, 1 seconds passed
... 57%, 72416 KB, 51315 KB/s, 1 seconds passed
... 57%, 72448 KB, 51325 KB/s, 1 seconds passed
... 57%, 72480 KB, 51335 KB/s, 1 seconds passed
... 57%, 72512 KB, 51343 KB/s, 1 seconds passed
... 57%, 72544 KB, 51354 KB/s, 1 seconds passed
... 57%, 72576 KB, 51362 KB/s, 1 seconds passed
... 57%, 72608 KB, 51373 KB/s, 1 seconds passed
... 57%, 72640 KB, 51384 KB/s, 1 seconds passed
... 57%, 72672 KB, 51394 KB/s, 1 seconds passed
... 57%, 72704 KB, 51405 KB/s, 1 seconds passed
... 57%, 72736 KB, 51417 KB/s, 1 seconds passed
... 57%, 72768 KB, 51427 KB/s, 1 seconds passed
... 57%, 72800 KB, 51436 KB/s, 1 seconds passed
... 57%, 72832 KB, 51448 KB/s, 1 seconds passed
... 57%, 72864 KB, 51457 KB/s, 1 seconds passed
... 57%, 72896 KB, 51467 KB/s, 1 seconds passed
... 57%, 72928 KB, 51478 KB/s, 1 seconds passed
... 57%, 72960 KB, 51489 KB/s, 1 seconds passed
... 57%, 72992 KB, 51499 KB/s, 1 seconds passed
... 57%, 73024 KB, 51510 KB/s, 1 seconds passed
... 58%, 73056 KB, 51521 KB/s, 1 seconds passed
... 58%, 73088 KB, 51531 KB/s, 1 seconds passed
... 58%, 73120 KB, 51542 KB/s, 1 seconds passed
... 58%, 73152 KB, 51553 KB/s, 1 seconds passed
... 58%, 73184 KB, 51564 KB/s, 1 seconds passed
... 58%, 73216 KB, 51574 KB/s, 1 seconds passed
... 58%, 73248 KB, 51586 KB/s, 1 seconds passed
... 58%, 73280 KB, 51600 KB/s, 1 seconds passed
... 58%, 73312 KB, 51614 KB/s, 1 seconds passed
... 58%, 73344 KB, 51629 KB/s, 1 seconds passed
... 58%, 73376 KB, 51644 KB/s, 1 seconds passed
... 58%, 73408 KB, 51658 KB/s, 1 seconds passed
... 58%, 73440 KB, 51672 KB/s, 1 seconds passed
... 58%, 73472 KB, 51687 KB/s, 1 seconds passed
... 58%, 73504 KB, 51701 KB/s, 1 seconds passed
... 58%, 73536 KB, 51716 KB/s, 1 seconds passed
... 58%, 73568 KB, 51732 KB/s, 1 seconds passed
... 58%, 73600 KB, 51746 KB/s, 1 seconds passed
... 58%, 73632 KB, 51760 KB/s, 1 seconds passed
... 58%, 73664 KB, 51775 KB/s, 1 seconds passed
... 58%, 73696 KB, 51789 KB/s, 1 seconds passed
... 58%, 73728 KB, 51803 KB/s, 1 seconds passed
... 58%, 73760 KB, 51818 KB/s, 1 seconds passed
... 58%, 73792 KB, 51834 KB/s, 1 seconds passed
... 58%, 73824 KB, 51848 KB/s, 1 seconds passed
... 58%, 73856 KB, 51861 KB/s, 1 seconds passed
... 58%, 73888 KB, 51875 KB/s, 1 seconds passed
... 58%, 73920 KB, 51890 KB/s, 1 seconds passed
... 58%, 73952 KB, 51905 KB/s, 1 seconds passed
... 58%, 73984 KB, 51919 KB/s, 1 seconds passed
... 58%, 74016 KB, 51934 KB/s, 1 seconds passed
... 58%, 74048 KB, 51948 KB/s, 1 seconds passed
... 58%, 74080 KB, 51963 KB/s, 1 seconds passed
... 58%, 74112 KB, 51977 KB/s, 1 seconds passed
... 58%, 74144 KB, 51992 KB/s, 1 seconds passed
... 58%, 74176 KB, 52006 KB/s, 1 seconds passed
... 58%, 74208 KB, 52021 KB/s, 1 seconds passed
... 58%, 74240 KB, 52036 KB/s, 1 seconds passed
... 58%, 74272 KB, 52050 KB/s, 1 seconds passed
... 58%, 74304 KB, 52064 KB/s, 1 seconds passed
... 59%, 74336 KB, 52079 KB/s, 1 seconds passed
... 59%, 74368 KB, 52094 KB/s, 1 seconds passed
... 59%, 74400 KB, 52110 KB/s, 1 seconds passed
... 59%, 74432 KB, 52125 KB/s, 1 seconds passed

.. parsed-literal::

    ... 59%, 74464 KB, 51695 KB/s, 1 seconds passed
... 59%, 74496 KB, 51701 KB/s, 1 seconds passed
... 59%, 74528 KB, 51709 KB/s, 1 seconds passed
... 59%, 74560 KB, 51719 KB/s, 1 seconds passed
... 59%, 74592 KB, 51728 KB/s, 1 seconds passed
... 59%, 74624 KB, 51739 KB/s, 1 seconds passed
... 59%, 74656 KB, 51749 KB/s, 1 seconds passed
... 59%, 74688 KB, 51759 KB/s, 1 seconds passed
... 59%, 74720 KB, 51769 KB/s, 1 seconds passed
... 59%, 74752 KB, 51780 KB/s, 1 seconds passed
... 59%, 74784 KB, 51782 KB/s, 1 seconds passed
... 59%, 74816 KB, 51792 KB/s, 1 seconds passed
... 59%, 74848 KB, 51691 KB/s, 1 seconds passed
... 59%, 74880 KB, 51695 KB/s, 1 seconds passed
... 59%, 74912 KB, 51699 KB/s, 1 seconds passed
... 59%, 74944 KB, 51707 KB/s, 1 seconds passed
... 59%, 74976 KB, 51714 KB/s, 1 seconds passed
... 59%, 75008 KB, 51722 KB/s, 1 seconds passed
... 59%, 75040 KB, 51731 KB/s, 1 seconds passed
... 59%, 75072 KB, 51740 KB/s, 1 seconds passed
... 59%, 75104 KB, 51749 KB/s, 1 seconds passed
... 59%, 75136 KB, 51757 KB/s, 1 seconds passed
... 59%, 75168 KB, 51765 KB/s, 1 seconds passed
... 59%, 75200 KB, 51775 KB/s, 1 seconds passed
... 59%, 75232 KB, 51786 KB/s, 1 seconds passed
... 59%, 75264 KB, 51797 KB/s, 1 seconds passed
... 59%, 75296 KB, 51763 KB/s, 1 seconds passed
... 59%, 75328 KB, 51768 KB/s, 1 seconds passed
... 59%, 75360 KB, 51773 KB/s, 1 seconds passed
... 59%, 75392 KB, 51780 KB/s, 1 seconds passed
... 59%, 75424 KB, 51788 KB/s, 1 seconds passed
... 59%, 75456 KB, 51796 KB/s, 1 seconds passed
... 59%, 75488 KB, 51804 KB/s, 1 seconds passed
... 59%, 75520 KB, 51813 KB/s, 1 seconds passed
... 59%, 75552 KB, 51821 KB/s, 1 seconds passed
... 60%, 75584 KB, 51830 KB/s, 1 seconds passed
... 60%, 75616 KB, 51839 KB/s, 1 seconds passed
... 60%, 75648 KB, 51847 KB/s, 1 seconds passed
... 60%, 75680 KB, 51855 KB/s, 1 seconds passed
... 60%, 75712 KB, 51863 KB/s, 1 seconds passed
... 60%, 75744 KB, 51871 KB/s, 1 seconds passed
... 60%, 75776 KB, 51879 KB/s, 1 seconds passed
... 60%, 75808 KB, 51888 KB/s, 1 seconds passed
... 60%, 75840 KB, 51900 KB/s, 1 seconds passed
... 60%, 75872 KB, 51911 KB/s, 1 seconds passed
... 60%, 75904 KB, 51923 KB/s, 1 seconds passed
... 60%, 75936 KB, 51935 KB/s, 1 seconds passed
... 60%, 75968 KB, 51947 KB/s, 1 seconds passed
... 60%, 76000 KB, 51958 KB/s, 1 seconds passed
... 60%, 76032 KB, 51970 KB/s, 1 seconds passed
... 60%, 76064 KB, 51982 KB/s, 1 seconds passed
... 60%, 76096 KB, 51994 KB/s, 1 seconds passed
... 60%, 76128 KB, 52006 KB/s, 1 seconds passed
... 60%, 76160 KB, 52016 KB/s, 1 seconds passed
... 60%, 76192 KB, 52028 KB/s, 1 seconds passed
... 60%, 76224 KB, 52040 KB/s, 1 seconds passed
... 60%, 76256 KB, 52052 KB/s, 1 seconds passed
... 60%, 76288 KB, 52063 KB/s, 1 seconds passed
... 60%, 76320 KB, 52075 KB/s, 1 seconds passed
... 60%, 76352 KB, 52087 KB/s, 1 seconds passed
... 60%, 76384 KB, 52099 KB/s, 1 seconds passed
... 60%, 76416 KB, 52110 KB/s, 1 seconds passed
... 60%, 76448 KB, 52122 KB/s, 1 seconds passed
... 60%, 76480 KB, 52134 KB/s, 1 seconds passed
... 60%, 76512 KB, 52146 KB/s, 1 seconds passed
... 60%, 76544 KB, 52158 KB/s, 1 seconds passed
... 60%, 76576 KB, 52172 KB/s, 1 seconds passed
... 60%, 76608 KB, 52185 KB/s, 1 seconds passed
... 60%, 76640 KB, 52199 KB/s, 1 seconds passed
... 60%, 76672 KB, 52213 KB/s, 1 seconds passed
... 60%, 76704 KB, 52227 KB/s, 1 seconds passed
... 60%, 76736 KB, 52240 KB/s, 1 seconds passed
... 60%, 76768 KB, 52254 KB/s, 1 seconds passed

.. parsed-literal::

    ... 60%, 76800 KB, 51552 KB/s, 1 seconds passed
... 61%, 76832 KB, 51557 KB/s, 1 seconds passed
... 61%, 76864 KB, 51565 KB/s, 1 seconds passed
... 61%, 76896 KB, 51574 KB/s, 1 seconds passed
... 61%, 76928 KB, 51581 KB/s, 1 seconds passed
... 61%, 76960 KB, 51590 KB/s, 1 seconds passed
... 61%, 76992 KB, 51598 KB/s, 1 seconds passed
... 61%, 77024 KB, 51606 KB/s, 1 seconds passed
... 61%, 77056 KB, 51615 KB/s, 1 seconds passed
... 61%, 77088 KB, 51623 KB/s, 1 seconds passed
... 61%, 77120 KB, 51631 KB/s, 1 seconds passed
... 61%, 77152 KB, 51640 KB/s, 1 seconds passed
... 61%, 77184 KB, 51647 KB/s, 1 seconds passed
... 61%, 77216 KB, 51656 KB/s, 1 seconds passed
... 61%, 77248 KB, 51664 KB/s, 1 seconds passed
... 61%, 77280 KB, 51673 KB/s, 1 seconds passed
... 61%, 77312 KB, 51681 KB/s, 1 seconds passed
... 61%, 77344 KB, 51689 KB/s, 1 seconds passed
... 61%, 77376 KB, 51697 KB/s, 1 seconds passed
... 61%, 77408 KB, 51705 KB/s, 1 seconds passed
... 61%, 77440 KB, 51713 KB/s, 1 seconds passed
... 61%, 77472 KB, 51720 KB/s, 1 seconds passed
... 61%, 77504 KB, 51728 KB/s, 1 seconds passed
... 61%, 77536 KB, 51736 KB/s, 1 seconds passed
... 61%, 77568 KB, 51744 KB/s, 1 seconds passed
... 61%, 77600 KB, 51753 KB/s, 1 seconds passed
... 61%, 77632 KB, 51761 KB/s, 1 seconds passed
... 61%, 77664 KB, 51768 KB/s, 1 seconds passed
... 61%, 77696 KB, 51776 KB/s, 1 seconds passed
... 61%, 77728 KB, 51784 KB/s, 1 seconds passed
... 61%, 77760 KB, 51794 KB/s, 1 seconds passed
... 61%, 77792 KB, 51807 KB/s, 1 seconds passed
... 61%, 77824 KB, 51820 KB/s, 1 seconds passed
... 61%, 77856 KB, 51833 KB/s, 1 seconds passed
... 61%, 77888 KB, 51846 KB/s, 1 seconds passed
... 61%, 77920 KB, 51858 KB/s, 1 seconds passed
... 61%, 77952 KB, 51871 KB/s, 1 seconds passed
... 61%, 77984 KB, 51884 KB/s, 1 seconds passed
... 61%, 78016 KB, 51896 KB/s, 1 seconds passed
... 61%, 78048 KB, 51909 KB/s, 1 seconds passed
... 61%, 78080 KB, 51921 KB/s, 1 seconds passed
... 62%, 78112 KB, 51933 KB/s, 1 seconds passed
... 62%, 78144 KB, 51946 KB/s, 1 seconds passed
... 62%, 78176 KB, 51959 KB/s, 1 seconds passed
... 62%, 78208 KB, 51972 KB/s, 1 seconds passed
... 62%, 78240 KB, 51984 KB/s, 1 seconds passed
... 62%, 78272 KB, 51997 KB/s, 1 seconds passed
... 62%, 78304 KB, 52010 KB/s, 1 seconds passed
... 62%, 78336 KB, 52022 KB/s, 1 seconds passed
... 62%, 78368 KB, 52035 KB/s, 1 seconds passed
... 62%, 78400 KB, 52047 KB/s, 1 seconds passed
... 62%, 78432 KB, 52060 KB/s, 1 seconds passed
... 62%, 78464 KB, 52072 KB/s, 1 seconds passed
... 62%, 78496 KB, 52085 KB/s, 1 seconds passed
... 62%, 78528 KB, 52097 KB/s, 1 seconds passed
... 62%, 78560 KB, 52110 KB/s, 1 seconds passed
... 62%, 78592 KB, 52122 KB/s, 1 seconds passed
... 62%, 78624 KB, 52135 KB/s, 1 seconds passed
... 62%, 78656 KB, 52148 KB/s, 1 seconds passed
... 62%, 78688 KB, 52161 KB/s, 1 seconds passed
... 62%, 78720 KB, 52173 KB/s, 1 seconds passed
... 62%, 78752 KB, 52186 KB/s, 1 seconds passed
... 62%, 78784 KB, 52198 KB/s, 1 seconds passed
... 62%, 78816 KB, 52211 KB/s, 1 seconds passed
... 62%, 78848 KB, 52224 KB/s, 1 seconds passed
... 62%, 78880 KB, 52237 KB/s, 1 seconds passed
... 62%, 78912 KB, 52250 KB/s, 1 seconds passed
... 62%, 78944 KB, 52263 KB/s, 1 seconds passed
... 62%, 78976 KB, 52276 KB/s, 1 seconds passed
... 62%, 79008 KB, 52288 KB/s, 1 seconds passed
... 62%, 79040 KB, 52302 KB/s, 1 seconds passed
... 62%, 79072 KB, 52318 KB/s, 1 seconds passed
... 62%, 79104 KB, 52333 KB/s, 1 seconds passed
... 62%, 79136 KB, 52349 KB/s, 1 seconds passed
... 62%, 79168 KB, 52364 KB/s, 1 seconds passed
... 62%, 79200 KB, 52379 KB/s, 1 seconds passed
... 62%, 79232 KB, 52394 KB/s, 1 seconds passed
... 62%, 79264 KB, 52410 KB/s, 1 seconds passed
... 62%, 79296 KB, 52425 KB/s, 1 seconds passed
... 62%, 79328 KB, 52441 KB/s, 1 seconds passed
... 63%, 79360 KB, 52456 KB/s, 1 seconds passed
... 63%, 79392 KB, 52472 KB/s, 1 seconds passed
... 63%, 79424 KB, 52487 KB/s, 1 seconds passed
... 63%, 79456 KB, 52503 KB/s, 1 seconds passed
... 63%, 79488 KB, 52517 KB/s, 1 seconds passed
... 63%, 79520 KB, 52529 KB/s, 1 seconds passed
... 63%, 79552 KB, 52538 KB/s, 1 seconds passed
... 63%, 79584 KB, 52548 KB/s, 1 seconds passed
... 63%, 79616 KB, 52563 KB/s, 1 seconds passed
... 63%, 79648 KB, 52575 KB/s, 1 seconds passed
... 63%, 79680 KB, 52586 KB/s, 1 seconds passed
... 63%, 79712 KB, 52596 KB/s, 1 seconds passed
... 63%, 79744 KB, 52609 KB/s, 1 seconds passed
... 63%, 79776 KB, 52619 KB/s, 1 seconds passed
... 63%, 79808 KB, 52630 KB/s, 1 seconds passed
... 63%, 79840 KB, 52641 KB/s, 1 seconds passed
... 63%, 79872 KB, 52654 KB/s, 1 seconds passed
... 63%, 79904 KB, 52663 KB/s, 1 seconds passed
... 63%, 79936 KB, 52674 KB/s, 1 seconds passed
... 63%, 79968 KB, 52686 KB/s, 1 seconds passed
... 63%, 80000 KB, 52697 KB/s, 1 seconds passed
... 63%, 80032 KB, 52708 KB/s, 1 seconds passed
... 63%, 80064 KB, 52723 KB/s, 1 seconds passed
... 63%, 80096 KB, 52733 KB/s, 1 seconds passed
... 63%, 80128 KB, 52743 KB/s, 1 seconds passed
... 63%, 80160 KB, 52755 KB/s, 1 seconds passed
... 63%, 80192 KB, 52767 KB/s, 1 seconds passed
... 63%, 80224 KB, 52779 KB/s, 1 seconds passed
... 63%, 80256 KB, 52791 KB/s, 1 seconds passed
... 63%, 80288 KB, 52799 KB/s, 1 seconds passed
... 63%, 80320 KB, 52813 KB/s, 1 seconds passed
... 63%, 80352 KB, 52823 KB/s, 1 seconds passed
... 63%, 80384 KB, 52837 KB/s, 1 seconds passed
... 63%, 80416 KB, 52843 KB/s, 1 seconds passed
... 63%, 80448 KB, 52858 KB/s, 1 seconds passed
... 63%, 80480 KB, 52871 KB/s, 1 seconds passed
... 63%, 80512 KB, 52880 KB/s, 1 seconds passed
... 63%, 80544 KB, 52887 KB/s, 1 seconds passed
... 63%, 80576 KB, 52894 KB/s, 1 seconds passed
... 63%, 80608 KB, 52903 KB/s, 1 seconds passed
... 64%, 80640 KB, 52580 KB/s, 1 seconds passed
... 64%, 80672 KB, 52585 KB/s, 1 seconds passed
... 64%, 80704 KB, 52591 KB/s, 1 seconds passed
... 64%, 80736 KB, 52598 KB/s, 1 seconds passed
... 64%, 80768 KB, 52606 KB/s, 1 seconds passed
... 64%, 80800 KB, 52615 KB/s, 1 seconds passed
... 64%, 80832 KB, 52623 KB/s, 1 seconds passed
... 64%, 80864 KB, 52630 KB/s, 1 seconds passed
... 64%, 80896 KB, 52638 KB/s, 1 seconds passed
... 64%, 80928 KB, 52646 KB/s, 1 seconds passed
... 64%, 80960 KB, 52654 KB/s, 1 seconds passed
... 64%, 80992 KB, 52663 KB/s, 1 seconds passed

.. parsed-literal::

    ... 64%, 81024 KB, 52671 KB/s, 1 seconds passed
... 64%, 81056 KB, 52679 KB/s, 1 seconds passed
... 64%, 81088 KB, 52687 KB/s, 1 seconds passed
... 64%, 81120 KB, 52696 KB/s, 1 seconds passed
... 64%, 81152 KB, 52704 KB/s, 1 seconds passed
... 64%, 81184 KB, 52711 KB/s, 1 seconds passed
... 64%, 81216 KB, 52718 KB/s, 1 seconds passed
... 64%, 81248 KB, 52726 KB/s, 1 seconds passed
... 64%, 81280 KB, 52735 KB/s, 1 seconds passed
... 64%, 81312 KB, 52743 KB/s, 1 seconds passed
... 64%, 81344 KB, 52751 KB/s, 1 seconds passed
... 64%, 81376 KB, 52759 KB/s, 1 seconds passed
... 64%, 81408 KB, 52769 KB/s, 1 seconds passed
... 64%, 81440 KB, 52779 KB/s, 1 seconds passed
... 64%, 81472 KB, 52789 KB/s, 1 seconds passed
... 64%, 81504 KB, 52799 KB/s, 1 seconds passed
... 64%, 81536 KB, 52808 KB/s, 1 seconds passed
... 64%, 81568 KB, 52817 KB/s, 1 seconds passed
... 64%, 81600 KB, 52827 KB/s, 1 seconds passed
... 64%, 81632 KB, 52836 KB/s, 1 seconds passed
... 64%, 81664 KB, 52846 KB/s, 1 seconds passed
... 64%, 81696 KB, 52857 KB/s, 1 seconds passed
... 64%, 81728 KB, 52869 KB/s, 1 seconds passed
... 64%, 81760 KB, 52881 KB/s, 1 seconds passed
... 64%, 81792 KB, 52894 KB/s, 1 seconds passed
... 64%, 81824 KB, 52906 KB/s, 1 seconds passed
... 64%, 81856 KB, 52918 KB/s, 1 seconds passed
... 65%, 81888 KB, 52931 KB/s, 1 seconds passed
... 65%, 81920 KB, 52197 KB/s, 1 seconds passed
... 65%, 81952 KB, 52200 KB/s, 1 seconds passed
... 65%, 81984 KB, 52207 KB/s, 1 seconds passed
... 65%, 82016 KB, 52214 KB/s, 1 seconds passed
... 65%, 82048 KB, 52221 KB/s, 1 seconds passed
... 65%, 82080 KB, 52228 KB/s, 1 seconds passed
... 65%, 82112 KB, 52236 KB/s, 1 seconds passed
... 65%, 82144 KB, 52242 KB/s, 1 seconds passed
... 65%, 82176 KB, 52249 KB/s, 1 seconds passed
... 65%, 82208 KB, 52257 KB/s, 1 seconds passed
... 65%, 82240 KB, 52266 KB/s, 1 seconds passed
... 65%, 82272 KB, 52273 KB/s, 1 seconds passed
... 65%, 82304 KB, 52281 KB/s, 1 seconds passed
... 65%, 82336 KB, 52289 KB/s, 1 seconds passed
... 65%, 82368 KB, 52296 KB/s, 1 seconds passed
... 65%, 82400 KB, 52304 KB/s, 1 seconds passed
... 65%, 82432 KB, 52311 KB/s, 1 seconds passed
... 65%, 82464 KB, 52318 KB/s, 1 seconds passed
... 65%, 82496 KB, 52325 KB/s, 1 seconds passed
... 65%, 82528 KB, 52332 KB/s, 1 seconds passed
... 65%, 82560 KB, 52340 KB/s, 1 seconds passed
... 65%, 82592 KB, 52348 KB/s, 1 seconds passed
... 65%, 82624 KB, 52355 KB/s, 1 seconds passed
... 65%, 82656 KB, 52363 KB/s, 1 seconds passed
... 65%, 82688 KB, 52369 KB/s, 1 seconds passed
... 65%, 82720 KB, 52377 KB/s, 1 seconds passed
... 65%, 82752 KB, 52385 KB/s, 1 seconds passed
... 65%, 82784 KB, 52392 KB/s, 1 seconds passed
... 65%, 82816 KB, 52400 KB/s, 1 seconds passed
... 65%, 82848 KB, 52407 KB/s, 1 seconds passed
... 65%, 82880 KB, 52415 KB/s, 1 seconds passed
... 65%, 82912 KB, 52423 KB/s, 1 seconds passed
... 65%, 82944 KB, 52431 KB/s, 1 seconds passed
... 65%, 82976 KB, 52441 KB/s, 1 seconds passed
... 65%, 83008 KB, 52453 KB/s, 1 seconds passed
... 65%, 83040 KB, 52465 KB/s, 1 seconds passed
... 65%, 83072 KB, 52477 KB/s, 1 seconds passed
... 65%, 83104 KB, 52489 KB/s, 1 seconds passed
... 66%, 83136 KB, 52501 KB/s, 1 seconds passed
... 66%, 83168 KB, 52514 KB/s, 1 seconds passed
... 66%, 83200 KB, 52525 KB/s, 1 seconds passed
... 66%, 83232 KB, 52537 KB/s, 1 seconds passed
... 66%, 83264 KB, 52550 KB/s, 1 seconds passed
... 66%, 83296 KB, 52562 KB/s, 1 seconds passed
... 66%, 83328 KB, 52574 KB/s, 1 seconds passed
... 66%, 83360 KB, 52587 KB/s, 1 seconds passed
... 66%, 83392 KB, 52599 KB/s, 1 seconds passed
... 66%, 83424 KB, 52611 KB/s, 1 seconds passed
... 66%, 83456 KB, 52624 KB/s, 1 seconds passed
... 66%, 83488 KB, 52636 KB/s, 1 seconds passed
... 66%, 83520 KB, 52649 KB/s, 1 seconds passed
... 66%, 83552 KB, 52661 KB/s, 1 seconds passed
... 66%, 83584 KB, 52673 KB/s, 1 seconds passed
... 66%, 83616 KB, 52685 KB/s, 1 seconds passed
... 66%, 83648 KB, 52698 KB/s, 1 seconds passed
... 66%, 83680 KB, 52709 KB/s, 1 seconds passed
... 66%, 83712 KB, 52721 KB/s, 1 seconds passed
... 66%, 83744 KB, 52733 KB/s, 1 seconds passed
... 66%, 83776 KB, 52745 KB/s, 1 seconds passed
... 66%, 83808 KB, 52757 KB/s, 1 seconds passed
... 66%, 83840 KB, 52770 KB/s, 1 seconds passed
... 66%, 83872 KB, 52781 KB/s, 1 seconds passed
... 66%, 83904 KB, 52794 KB/s, 1 seconds passed
... 66%, 83936 KB, 52806 KB/s, 1 seconds passed

.. parsed-literal::

    ... 66%, 83968 KB, 52818 KB/s, 1 seconds passed
... 66%, 84000 KB, 52831 KB/s, 1 seconds passed
... 66%, 84032 KB, 52842 KB/s, 1 seconds passed
... 66%, 84064 KB, 52855 KB/s, 1 seconds passed
... 66%, 84096 KB, 52867 KB/s, 1 seconds passed
... 66%, 84128 KB, 52879 KB/s, 1 seconds passed
... 66%, 84160 KB, 52892 KB/s, 1 seconds passed
... 66%, 84192 KB, 52904 KB/s, 1 seconds passed
... 66%, 84224 KB, 52916 KB/s, 1 seconds passed
... 66%, 84256 KB, 52928 KB/s, 1 seconds passed
... 66%, 84288 KB, 52939 KB/s, 1 seconds passed
... 66%, 84320 KB, 52953 KB/s, 1 seconds passed
... 66%, 84352 KB, 52967 KB/s, 1 seconds passed
... 66%, 84384 KB, 52982 KB/s, 1 seconds passed
... 67%, 84416 KB, 52997 KB/s, 1 seconds passed
... 67%, 84448 KB, 53011 KB/s, 1 seconds passed
... 67%, 84480 KB, 53026 KB/s, 1 seconds passed
... 67%, 84512 KB, 53039 KB/s, 1 seconds passed
... 67%, 84544 KB, 53054 KB/s, 1 seconds passed
... 67%, 84576 KB, 53069 KB/s, 1 seconds passed
... 67%, 84608 KB, 53083 KB/s, 1 seconds passed
... 67%, 84640 KB, 53098 KB/s, 1 seconds passed
... 67%, 84672 KB, 53110 KB/s, 1 seconds passed
... 67%, 84704 KB, 53121 KB/s, 1 seconds passed
... 67%, 84736 KB, 53129 KB/s, 1 seconds passed
... 67%, 84768 KB, 53140 KB/s, 1 seconds passed
... 67%, 84800 KB, 53151 KB/s, 1 seconds passed
... 67%, 84832 KB, 53163 KB/s, 1 seconds passed
... 67%, 84864 KB, 53174 KB/s, 1 seconds passed
... 67%, 84896 KB, 53185 KB/s, 1 seconds passed
... 67%, 84928 KB, 53197 KB/s, 1 seconds passed
... 67%, 84960 KB, 53208 KB/s, 1 seconds passed
... 67%, 84992 KB, 53219 KB/s, 1 seconds passed
... 67%, 85024 KB, 53227 KB/s, 1 seconds passed
... 67%, 85056 KB, 53238 KB/s, 1 seconds passed
... 67%, 85088 KB, 53251 KB/s, 1 seconds passed
... 67%, 85120 KB, 53262 KB/s, 1 seconds passed
... 67%, 85152 KB, 53274 KB/s, 1 seconds passed
... 67%, 85184 KB, 53285 KB/s, 1 seconds passed
... 67%, 85216 KB, 53294 KB/s, 1 seconds passed
... 67%, 85248 KB, 53305 KB/s, 1 seconds passed
... 67%, 85280 KB, 53315 KB/s, 1 seconds passed
... 67%, 85312 KB, 53324 KB/s, 1 seconds passed
... 67%, 85344 KB, 53334 KB/s, 1 seconds passed
... 67%, 85376 KB, 53345 KB/s, 1 seconds passed
... 67%, 85408 KB, 53355 KB/s, 1 seconds passed
... 67%, 85440 KB, 52771 KB/s, 1 seconds passed
... 67%, 85472 KB, 52778 KB/s, 1 seconds passed
... 67%, 85504 KB, 52785 KB/s, 1 seconds passed
... 67%, 85536 KB, 52792 KB/s, 1 seconds passed
... 67%, 85568 KB, 52800 KB/s, 1 seconds passed
... 67%, 85600 KB, 52808 KB/s, 1 seconds passed
... 67%, 85632 KB, 52816 KB/s, 1 seconds passed
... 68%, 85664 KB, 52823 KB/s, 1 seconds passed
... 68%, 85696 KB, 52831 KB/s, 1 seconds passed
... 68%, 85728 KB, 52838 KB/s, 1 seconds passed
... 68%, 85760 KB, 52845 KB/s, 1 seconds passed
... 68%, 85792 KB, 52852 KB/s, 1 seconds passed
... 68%, 85824 KB, 52860 KB/s, 1 seconds passed
... 68%, 85856 KB, 52867 KB/s, 1 seconds passed
... 68%, 85888 KB, 52874 KB/s, 1 seconds passed
... 68%, 85920 KB, 52881 KB/s, 1 seconds passed
... 68%, 85952 KB, 52889 KB/s, 1 seconds passed
... 68%, 85984 KB, 52896 KB/s, 1 seconds passed
... 68%, 86016 KB, 52904 KB/s, 1 seconds passed
... 68%, 86048 KB, 52911 KB/s, 1 seconds passed
... 68%, 86080 KB, 52918 KB/s, 1 seconds passed
... 68%, 86112 KB, 52925 KB/s, 1 seconds passed
... 68%, 86144 KB, 52933 KB/s, 1 seconds passed
... 68%, 86176 KB, 52941 KB/s, 1 seconds passed
... 68%, 86208 KB, 52947 KB/s, 1 seconds passed
... 68%, 86240 KB, 52955 KB/s, 1 seconds passed
... 68%, 86272 KB, 52962 KB/s, 1 seconds passed
... 68%, 86304 KB, 52970 KB/s, 1 seconds passed
... 68%, 86336 KB, 52977 KB/s, 1 seconds passed
... 68%, 86368 KB, 52984 KB/s, 1 seconds passed
... 68%, 86400 KB, 52992 KB/s, 1 seconds passed
... 68%, 86432 KB, 53000 KB/s, 1 seconds passed
... 68%, 86464 KB, 53007 KB/s, 1 seconds passed
... 68%, 86496 KB, 53015 KB/s, 1 seconds passed
... 68%, 86528 KB, 53022 KB/s, 1 seconds passed
... 68%, 86560 KB, 53029 KB/s, 1 seconds passed
... 68%, 86592 KB, 53038 KB/s, 1 seconds passed
... 68%, 86624 KB, 53050 KB/s, 1 seconds passed
... 68%, 86656 KB, 53062 KB/s, 1 seconds passed
... 68%, 86688 KB, 53076 KB/s, 1 seconds passed
... 68%, 86720 KB, 53089 KB/s, 1 seconds passed
... 68%, 86752 KB, 53102 KB/s, 1 seconds passed
... 68%, 86784 KB, 53116 KB/s, 1 seconds passed
... 68%, 86816 KB, 53129 KB/s, 1 seconds passed
... 68%, 86848 KB, 53141 KB/s, 1 seconds passed
... 68%, 86880 KB, 53155 KB/s, 1 seconds passed
... 69%, 86912 KB, 53168 KB/s, 1 seconds passed
... 69%, 86944 KB, 53182 KB/s, 1 seconds passed
... 69%, 86976 KB, 53195 KB/s, 1 seconds passed
... 69%, 87008 KB, 53208 KB/s, 1 seconds passed
... 69%, 87040 KB, 53090 KB/s, 1 seconds passed
... 69%, 87072 KB, 53088 KB/s, 1 seconds passed
... 69%, 87104 KB, 53100 KB/s, 1 seconds passed

.. parsed-literal::

    ... 69%, 87136 KB, 53089 KB/s, 1 seconds passed
... 69%, 87168 KB, 53094 KB/s, 1 seconds passed
... 69%, 87200 KB, 53105 KB/s, 1 seconds passed
... 69%, 87232 KB, 53117 KB/s, 1 seconds passed
... 69%, 87264 KB, 53128 KB/s, 1 seconds passed
... 69%, 87296 KB, 53125 KB/s, 1 seconds passed
... 69%, 87328 KB, 53135 KB/s, 1 seconds passed
... 69%, 87360 KB, 53145 KB/s, 1 seconds passed
... 69%, 87392 KB, 53156 KB/s, 1 seconds passed
... 69%, 87424 KB, 53166 KB/s, 1 seconds passed
... 69%, 87456 KB, 53176 KB/s, 1 seconds passed
... 69%, 87488 KB, 53187 KB/s, 1 seconds passed
... 69%, 87520 KB, 53197 KB/s, 1 seconds passed
... 69%, 87552 KB, 53208 KB/s, 1 seconds passed
... 69%, 87584 KB, 53219 KB/s, 1 seconds passed
... 69%, 87616 KB, 53229 KB/s, 1 seconds passed
... 69%, 87648 KB, 53240 KB/s, 1 seconds passed
... 69%, 87680 KB, 53250 KB/s, 1 seconds passed
... 69%, 87712 KB, 53260 KB/s, 1 seconds passed
... 69%, 87744 KB, 53271 KB/s, 1 seconds passed
... 69%, 87776 KB, 53281 KB/s, 1 seconds passed
... 69%, 87808 KB, 53291 KB/s, 1 seconds passed
... 69%, 87840 KB, 53301 KB/s, 1 seconds passed
... 69%, 87872 KB, 53312 KB/s, 1 seconds passed
... 69%, 87904 KB, 53322 KB/s, 1 seconds passed
... 69%, 87936 KB, 53333 KB/s, 1 seconds passed
... 69%, 87968 KB, 53344 KB/s, 1 seconds passed
... 69%, 88000 KB, 53354 KB/s, 1 seconds passed
... 69%, 88032 KB, 53365 KB/s, 1 seconds passed
... 69%, 88064 KB, 53376 KB/s, 1 seconds passed
... 69%, 88096 KB, 53385 KB/s, 1 seconds passed
... 69%, 88128 KB, 53396 KB/s, 1 seconds passed
... 69%, 88160 KB, 53405 KB/s, 1 seconds passed
... 70%, 88192 KB, 53416 KB/s, 1 seconds passed
... 70%, 88224 KB, 53427 KB/s, 1 seconds passed
... 70%, 88256 KB, 53437 KB/s, 1 seconds passed
... 70%, 88288 KB, 53448 KB/s, 1 seconds passed
... 70%, 88320 KB, 53458 KB/s, 1 seconds passed
... 70%, 88352 KB, 53468 KB/s, 1 seconds passed
... 70%, 88384 KB, 53479 KB/s, 1 seconds passed
... 70%, 88416 KB, 53490 KB/s, 1 seconds passed
... 70%, 88448 KB, 53499 KB/s, 1 seconds passed
... 70%, 88480 KB, 53510 KB/s, 1 seconds passed
... 70%, 88512 KB, 53521 KB/s, 1 seconds passed
... 70%, 88544 KB, 53530 KB/s, 1 seconds passed
... 70%, 88576 KB, 53541 KB/s, 1 seconds passed
... 70%, 88608 KB, 53551 KB/s, 1 seconds passed
... 70%, 88640 KB, 53562 KB/s, 1 seconds passed
... 70%, 88672 KB, 53571 KB/s, 1 seconds passed
... 70%, 88704 KB, 53585 KB/s, 1 seconds passed
... 70%, 88736 KB, 53596 KB/s, 1 seconds passed
... 70%, 88768 KB, 53605 KB/s, 1 seconds passed
... 70%, 88800 KB, 53616 KB/s, 1 seconds passed
... 70%, 88832 KB, 53627 KB/s, 1 seconds passed
... 70%, 88864 KB, 53638 KB/s, 1 seconds passed
... 70%, 88896 KB, 53647 KB/s, 1 seconds passed
... 70%, 88928 KB, 53658 KB/s, 1 seconds passed
... 70%, 88960 KB, 53663 KB/s, 1 seconds passed
... 70%, 88992 KB, 53674 KB/s, 1 seconds passed
... 70%, 89024 KB, 53683 KB/s, 1 seconds passed
... 70%, 89056 KB, 53694 KB/s, 1 seconds passed
... 70%, 89088 KB, 53704 KB/s, 1 seconds passed
... 70%, 89120 KB, 53715 KB/s, 1 seconds passed
... 70%, 89152 KB, 53724 KB/s, 1 seconds passed
... 70%, 89184 KB, 53735 KB/s, 1 seconds passed
... 70%, 89216 KB, 53746 KB/s, 1 seconds passed
... 70%, 89248 KB, 53755 KB/s, 1 seconds passed
... 70%, 89280 KB, 53766 KB/s, 1 seconds passed
... 70%, 89312 KB, 53778 KB/s, 1 seconds passed
... 70%, 89344 KB, 53789 KB/s, 1 seconds passed
... 70%, 89376 KB, 53799 KB/s, 1 seconds passed
... 70%, 89408 KB, 53810 KB/s, 1 seconds passed
... 71%, 89440 KB, 53819 KB/s, 1 seconds passed
... 71%, 89472 KB, 53830 KB/s, 1 seconds passed
... 71%, 89504 KB, 53841 KB/s, 1 seconds passed
... 71%, 89536 KB, 53850 KB/s, 1 seconds passed
... 71%, 89568 KB, 53862 KB/s, 1 seconds passed
... 71%, 89600 KB, 53871 KB/s, 1 seconds passed
... 71%, 89632 KB, 53882 KB/s, 1 seconds passed
... 71%, 89664 KB, 53889 KB/s, 1 seconds passed
... 71%, 89696 KB, 53900 KB/s, 1 seconds passed
... 71%, 89728 KB, 53911 KB/s, 1 seconds passed
... 71%, 89760 KB, 53921 KB/s, 1 seconds passed
... 71%, 89792 KB, 53930 KB/s, 1 seconds passed
... 71%, 89824 KB, 53940 KB/s, 1 seconds passed
... 71%, 89856 KB, 53952 KB/s, 1 seconds passed
... 71%, 89888 KB, 53962 KB/s, 1 seconds passed
... 71%, 89920 KB, 53973 KB/s, 1 seconds passed
... 71%, 89952 KB, 53982 KB/s, 1 seconds passed
... 71%, 89984 KB, 53993 KB/s, 1 seconds passed
... 71%, 90016 KB, 54004 KB/s, 1 seconds passed
... 71%, 90048 KB, 54014 KB/s, 1 seconds passed
... 71%, 90080 KB, 54025 KB/s, 1 seconds passed
... 71%, 90112 KB, 54035 KB/s, 1 seconds passed
... 71%, 90144 KB, 54044 KB/s, 1 seconds passed
... 71%, 90176 KB, 54055 KB/s, 1 seconds passed
... 71%, 90208 KB, 54066 KB/s, 1 seconds passed
... 71%, 90240 KB, 54075 KB/s, 1 seconds passed
... 71%, 90272 KB, 54085 KB/s, 1 seconds passed
... 71%, 90304 KB, 54093 KB/s, 1 seconds passed
... 71%, 90336 KB, 54102 KB/s, 1 seconds passed
... 71%, 90368 KB, 54112 KB/s, 1 seconds passed
... 71%, 90400 KB, 54126 KB/s, 1 seconds passed
... 71%, 90432 KB, 54137 KB/s, 1 seconds passed
... 71%, 90464 KB, 54146 KB/s, 1 seconds passed
... 71%, 90496 KB, 54157 KB/s, 1 seconds passed
... 71%, 90528 KB, 54167 KB/s, 1 seconds passed
... 71%, 90560 KB, 54176 KB/s, 1 seconds passed
... 71%, 90592 KB, 54187 KB/s, 1 seconds passed
... 71%, 90624 KB, 54198 KB/s, 1 seconds passed
... 71%, 90656 KB, 54203 KB/s, 1 seconds passed
... 72%, 90688 KB, 54213 KB/s, 1 seconds passed
... 72%, 90720 KB, 54219 KB/s, 1 seconds passed
... 72%, 90752 KB, 54233 KB/s, 1 seconds passed
... 72%, 90784 KB, 54245 KB/s, 1 seconds passed
... 72%, 90816 KB, 54254 KB/s, 1 seconds passed
... 72%, 90848 KB, 54264 KB/s, 1 seconds passed
... 72%, 90880 KB, 54275 KB/s, 1 seconds passed
... 72%, 90912 KB, 54284 KB/s, 1 seconds passed
... 72%, 90944 KB, 54296 KB/s, 1 seconds passed
... 72%, 90976 KB, 54307 KB/s, 1 seconds passed
... 72%, 91008 KB, 54316 KB/s, 1 seconds passed
... 72%, 91040 KB, 54326 KB/s, 1 seconds passed
... 72%, 91072 KB, 54337 KB/s, 1 seconds passed
... 72%, 91104 KB, 54348 KB/s, 1 seconds passed
... 72%, 91136 KB, 54357 KB/s, 1 seconds passed
... 72%, 91168 KB, 54367 KB/s, 1 seconds passed
... 72%, 91200 KB, 54378 KB/s, 1 seconds passed
... 72%, 91232 KB, 54388 KB/s, 1 seconds passed
... 72%, 91264 KB, 54397 KB/s, 1 seconds passed
... 72%, 91296 KB, 54408 KB/s, 1 seconds passed
... 72%, 91328 KB, 54417 KB/s, 1 seconds passed
... 72%, 91360 KB, 54427 KB/s, 1 seconds passed
... 72%, 91392 KB, 54438 KB/s, 1 seconds passed
... 72%, 91424 KB, 54447 KB/s, 1 seconds passed
... 72%, 91456 KB, 54457 KB/s, 1 seconds passed
... 72%, 91488 KB, 54468 KB/s, 1 seconds passed
... 72%, 91520 KB, 54452 KB/s, 1 seconds passed
... 72%, 91552 KB, 54462 KB/s, 1 seconds passed
... 72%, 91584 KB, 54473 KB/s, 1 seconds passed
... 72%, 91616 KB, 54484 KB/s, 1 seconds passed
... 72%, 91648 KB, 54492 KB/s, 1 seconds passed
... 72%, 91680 KB, 54503 KB/s, 1 seconds passed
... 72%, 91712 KB, 54513 KB/s, 1 seconds passed
... 72%, 91744 KB, 54524 KB/s, 1 seconds passed
... 72%, 91776 KB, 54533 KB/s, 1 seconds passed
... 72%, 91808 KB, 54543 KB/s, 1 seconds passed
... 72%, 91840 KB, 54551 KB/s, 1 seconds passed
... 72%, 91872 KB, 54556 KB/s, 1 seconds passed
... 72%, 91904 KB, 54565 KB/s, 1 seconds passed
... 72%, 91936 KB, 54580 KB/s, 1 seconds passed
... 73%, 91968 KB, 54592 KB/s, 1 seconds passed
... 73%, 92000 KB, 54599 KB/s, 1 seconds passed
... 73%, 92032 KB, 54610 KB/s, 1 seconds passed
... 73%, 92064 KB, 54620 KB/s, 1 seconds passed
... 73%, 92096 KB, 54630 KB/s, 1 seconds passed
... 73%, 92128 KB, 54643 KB/s, 1 seconds passed

.. parsed-literal::

    ... 73%, 92160 KB, 53349 KB/s, 1 seconds passed
... 73%, 92192 KB, 53354 KB/s, 1 seconds passed
... 73%, 92224 KB, 53347 KB/s, 1 seconds passed
... 73%, 92256 KB, 53354 KB/s, 1 seconds passed
... 73%, 92288 KB, 53358 KB/s, 1 seconds passed
... 73%, 92320 KB, 53365 KB/s, 1 seconds passed
... 73%, 92352 KB, 53372 KB/s, 1 seconds passed
... 73%, 92384 KB, 53379 KB/s, 1 seconds passed
... 73%, 92416 KB, 53386 KB/s, 1 seconds passed
... 73%, 92448 KB, 53394 KB/s, 1 seconds passed
... 73%, 92480 KB, 53402 KB/s, 1 seconds passed
... 73%, 92512 KB, 53407 KB/s, 1 seconds passed
... 73%, 92544 KB, 53414 KB/s, 1 seconds passed
... 73%, 92576 KB, 53420 KB/s, 1 seconds passed
... 73%, 92608 KB, 53427 KB/s, 1 seconds passed
... 73%, 92640 KB, 53433 KB/s, 1 seconds passed
... 73%, 92672 KB, 53440 KB/s, 1 seconds passed
... 73%, 92704 KB, 53447 KB/s, 1 seconds passed
... 73%, 92736 KB, 53454 KB/s, 1 seconds passed
... 73%, 92768 KB, 53461 KB/s, 1 seconds passed
... 73%, 92800 KB, 53468 KB/s, 1 seconds passed
... 73%, 92832 KB, 53474 KB/s, 1 seconds passed
... 73%, 92864 KB, 53480 KB/s, 1 seconds passed
... 73%, 92896 KB, 53487 KB/s, 1 seconds passed
... 73%, 92928 KB, 53493 KB/s, 1 seconds passed
... 73%, 92960 KB, 53500 KB/s, 1 seconds passed
... 73%, 92992 KB, 53508 KB/s, 1 seconds passed
... 73%, 93024 KB, 53514 KB/s, 1 seconds passed
... 73%, 93056 KB, 53521 KB/s, 1 seconds passed
... 73%, 93088 KB, 53527 KB/s, 1 seconds passed
... 73%, 93120 KB, 53533 KB/s, 1 seconds passed
... 73%, 93152 KB, 53540 KB/s, 1 seconds passed
... 73%, 93184 KB, 53547 KB/s, 1 seconds passed
... 74%, 93216 KB, 53553 KB/s, 1 seconds passed
... 74%, 93248 KB, 53560 KB/s, 1 seconds passed
... 74%, 93280 KB, 53566 KB/s, 1 seconds passed
... 74%, 93312 KB, 53573 KB/s, 1 seconds passed
... 74%, 93344 KB, 53580 KB/s, 1 seconds passed
... 74%, 93376 KB, 53587 KB/s, 1 seconds passed
... 74%, 93408 KB, 53593 KB/s, 1 seconds passed
... 74%, 93440 KB, 53601 KB/s, 1 seconds passed

.. parsed-literal::

    ... 74%, 93472 KB, 53608 KB/s, 1 seconds passed
... 74%, 93504 KB, 53618 KB/s, 1 seconds passed
... 74%, 93536 KB, 53628 KB/s, 1 seconds passed
... 74%, 93568 KB, 53639 KB/s, 1 seconds passed
... 74%, 93600 KB, 53650 KB/s, 1 seconds passed
... 74%, 93632 KB, 53660 KB/s, 1 seconds passed
... 74%, 93664 KB, 53671 KB/s, 1 seconds passed
... 74%, 93696 KB, 53682 KB/s, 1 seconds passed
... 74%, 93728 KB, 53693 KB/s, 1 seconds passed
... 74%, 93760 KB, 53703 KB/s, 1 seconds passed
... 74%, 93792 KB, 53714 KB/s, 1 seconds passed
... 74%, 93824 KB, 53725 KB/s, 1 seconds passed
... 74%, 93856 KB, 53735 KB/s, 1 seconds passed
... 74%, 93888 KB, 53746 KB/s, 1 seconds passed
... 74%, 93920 KB, 53757 KB/s, 1 seconds passed
... 74%, 93952 KB, 53768 KB/s, 1 seconds passed
... 74%, 93984 KB, 53779 KB/s, 1 seconds passed
... 74%, 94016 KB, 53789 KB/s, 1 seconds passed
... 74%, 94048 KB, 53799 KB/s, 1 seconds passed
... 74%, 94080 KB, 53810 KB/s, 1 seconds passed
... 74%, 94112 KB, 53820 KB/s, 1 seconds passed
... 74%, 94144 KB, 53831 KB/s, 1 seconds passed
... 74%, 94176 KB, 53842 KB/s, 1 seconds passed
... 74%, 94208 KB, 53852 KB/s, 1 seconds passed
... 74%, 94240 KB, 53863 KB/s, 1 seconds passed
... 74%, 94272 KB, 53874 KB/s, 1 seconds passed
... 74%, 94304 KB, 53884 KB/s, 1 seconds passed
... 74%, 94336 KB, 53895 KB/s, 1 seconds passed
... 74%, 94368 KB, 53906 KB/s, 1 seconds passed
... 74%, 94400 KB, 53916 KB/s, 1 seconds passed
... 74%, 94432 KB, 53927 KB/s, 1 seconds passed
... 74%, 94464 KB, 53938 KB/s, 1 seconds passed
... 75%, 94496 KB, 53948 KB/s, 1 seconds passed
... 75%, 94528 KB, 53959 KB/s, 1 seconds passed
... 75%, 94560 KB, 53969 KB/s, 1 seconds passed
... 75%, 94592 KB, 53979 KB/s, 1 seconds passed
... 75%, 94624 KB, 53989 KB/s, 1 seconds passed
... 75%, 94656 KB, 54000 KB/s, 1 seconds passed
... 75%, 94688 KB, 54011 KB/s, 1 seconds passed
... 75%, 94720 KB, 54022 KB/s, 1 seconds passed
... 75%, 94752 KB, 54032 KB/s, 1 seconds passed
... 75%, 94784 KB, 54044 KB/s, 1 seconds passed
... 75%, 94816 KB, 54057 KB/s, 1 seconds passed
... 75%, 94848 KB, 54070 KB/s, 1 seconds passed
... 75%, 94880 KB, 54083 KB/s, 1 seconds passed
... 75%, 94912 KB, 54096 KB/s, 1 seconds passed
... 75%, 94944 KB, 54109 KB/s, 1 seconds passed
... 75%, 94976 KB, 54122 KB/s, 1 seconds passed
... 75%, 95008 KB, 54135 KB/s, 1 seconds passed
... 75%, 95040 KB, 54148 KB/s, 1 seconds passed
... 75%, 95072 KB, 54161 KB/s, 1 seconds passed
... 75%, 95104 KB, 54174 KB/s, 1 seconds passed
... 75%, 95136 KB, 54187 KB/s, 1 seconds passed
... 75%, 95168 KB, 54200 KB/s, 1 seconds passed
... 75%, 95200 KB, 54213 KB/s, 1 seconds passed
... 75%, 95232 KB, 54226 KB/s, 1 seconds passed
... 75%, 95264 KB, 54239 KB/s, 1 seconds passed
... 75%, 95296 KB, 54252 KB/s, 1 seconds passed
... 75%, 95328 KB, 54265 KB/s, 1 seconds passed
... 75%, 95360 KB, 54278 KB/s, 1 seconds passed
... 75%, 95392 KB, 54291 KB/s, 1 seconds passed
... 75%, 95424 KB, 54302 KB/s, 1 seconds passed
... 75%, 95456 KB, 54311 KB/s, 1 seconds passed
... 75%, 95488 KB, 54321 KB/s, 1 seconds passed
... 75%, 95520 KB, 54331 KB/s, 1 seconds passed
... 75%, 95552 KB, 54340 KB/s, 1 seconds passed
... 75%, 95584 KB, 54345 KB/s, 1 seconds passed
... 75%, 95616 KB, 54350 KB/s, 1 seconds passed
... 75%, 95648 KB, 54360 KB/s, 1 seconds passed
... 75%, 95680 KB, 54373 KB/s, 1 seconds passed
... 75%, 95712 KB, 54386 KB/s, 1 seconds passed
... 76%, 95744 KB, 54398 KB/s, 1 seconds passed
... 76%, 95776 KB, 54408 KB/s, 1 seconds passed
... 76%, 95808 KB, 54413 KB/s, 1 seconds passed
... 76%, 95840 KB, 54418 KB/s, 1 seconds passed
... 76%, 95872 KB, 54431 KB/s, 1 seconds passed
... 76%, 95904 KB, 54444 KB/s, 1 seconds passed
... 76%, 95936 KB, 54456 KB/s, 1 seconds passed
... 76%, 95968 KB, 54466 KB/s, 1 seconds passed
... 76%, 96000 KB, 54476 KB/s, 1 seconds passed
... 76%, 96032 KB, 54484 KB/s, 1 seconds passed
... 76%, 96064 KB, 54403 KB/s, 1 seconds passed
... 76%, 96096 KB, 54408 KB/s, 1 seconds passed
... 76%, 96128 KB, 54413 KB/s, 1 seconds passed
... 76%, 96160 KB, 54419 KB/s, 1 seconds passed
... 76%, 96192 KB, 54425 KB/s, 1 seconds passed
... 76%, 96224 KB, 54432 KB/s, 1 seconds passed
... 76%, 96256 KB, 54440 KB/s, 1 seconds passed
... 76%, 96288 KB, 54449 KB/s, 1 seconds passed
... 76%, 96320 KB, 54458 KB/s, 1 seconds passed
... 76%, 96352 KB, 54467 KB/s, 1 seconds passed
... 76%, 96384 KB, 54476 KB/s, 1 seconds passed
... 76%, 96416 KB, 54486 KB/s, 1 seconds passed
... 76%, 96448 KB, 54495 KB/s, 1 seconds passed
... 76%, 96480 KB, 54504 KB/s, 1 seconds passed
... 76%, 96512 KB, 54513 KB/s, 1 seconds passed
... 76%, 96544 KB, 54522 KB/s, 1 seconds passed
... 76%, 96576 KB, 54530 KB/s, 1 seconds passed
... 76%, 96608 KB, 54540 KB/s, 1 seconds passed
... 76%, 96640 KB, 54548 KB/s, 1 seconds passed
... 76%, 96672 KB, 54557 KB/s, 1 seconds passed
... 76%, 96704 KB, 54566 KB/s, 1 seconds passed
... 76%, 96736 KB, 54574 KB/s, 1 seconds passed
... 76%, 96768 KB, 54584 KB/s, 1 seconds passed
... 76%, 96800 KB, 54592 KB/s, 1 seconds passed
... 76%, 96832 KB, 54601 KB/s, 1 seconds passed
... 76%, 96864 KB, 54610 KB/s, 1 seconds passed
... 76%, 96896 KB, 54619 KB/s, 1 seconds passed
... 76%, 96928 KB, 54628 KB/s, 1 seconds passed
... 76%, 96960 KB, 54637 KB/s, 1 seconds passed
... 77%, 96992 KB, 54647 KB/s, 1 seconds passed
... 77%, 97024 KB, 54658 KB/s, 1 seconds passed
... 77%, 97056 KB, 54669 KB/s, 1 seconds passed
... 77%, 97088 KB, 54680 KB/s, 1 seconds passed
... 77%, 97120 KB, 54690 KB/s, 1 seconds passed
... 77%, 97152 KB, 54701 KB/s, 1 seconds passed
... 77%, 97184 KB, 54711 KB/s, 1 seconds passed
... 77%, 97216 KB, 54722 KB/s, 1 seconds passed
... 77%, 97248 KB, 54734 KB/s, 1 seconds passed

.. parsed-literal::

    ... 77%, 97280 KB, 53683 KB/s, 1 seconds passed
... 77%, 97312 KB, 53684 KB/s, 1 seconds passed
... 77%, 97344 KB, 53689 KB/s, 1 seconds passed
... 77%, 97376 KB, 53695 KB/s, 1 seconds passed
... 77%, 97408 KB, 53701 KB/s, 1 seconds passed
... 77%, 97440 KB, 53707 KB/s, 1 seconds passed
... 77%, 97472 KB, 53714 KB/s, 1 seconds passed
... 77%, 97504 KB, 53723 KB/s, 1 seconds passed
... 77%, 97536 KB, 53731 KB/s, 1 seconds passed
... 77%, 97568 KB, 53737 KB/s, 1 seconds passed
... 77%, 97600 KB, 53743 KB/s, 1 seconds passed
... 77%, 97632 KB, 53749 KB/s, 1 seconds passed
... 77%, 97664 KB, 53754 KB/s, 1 seconds passed
... 77%, 97696 KB, 53761 KB/s, 1 seconds passed
... 77%, 97728 KB, 53767 KB/s, 1 seconds passed
... 77%, 97760 KB, 53774 KB/s, 1 seconds passed
... 77%, 97792 KB, 53779 KB/s, 1 seconds passed
... 77%, 97824 KB, 53786 KB/s, 1 seconds passed
... 77%, 97856 KB, 53792 KB/s, 1 seconds passed
... 77%, 97888 KB, 53798 KB/s, 1 seconds passed
... 77%, 97920 KB, 53804 KB/s, 1 seconds passed
... 77%, 97952 KB, 53810 KB/s, 1 seconds passed
... 77%, 97984 KB, 53816 KB/s, 1 seconds passed
... 77%, 98016 KB, 53823 KB/s, 1 seconds passed
... 77%, 98048 KB, 53829 KB/s, 1 seconds passed
... 77%, 98080 KB, 53836 KB/s, 1 seconds passed
... 77%, 98112 KB, 53842 KB/s, 1 seconds passed
... 77%, 98144 KB, 53849 KB/s, 1 seconds passed
... 77%, 98176 KB, 53855 KB/s, 1 seconds passed
... 77%, 98208 KB, 53861 KB/s, 1 seconds passed
... 77%, 98240 KB, 53868 KB/s, 1 seconds passed
... 78%, 98272 KB, 53873 KB/s, 1 seconds passed
... 78%, 98304 KB, 53878 KB/s, 1 seconds passed
... 78%, 98336 KB, 53888 KB/s, 1 seconds passed
... 78%, 98368 KB, 53897 KB/s, 1 seconds passed
... 78%, 98400 KB, 53907 KB/s, 1 seconds passed
... 78%, 98432 KB, 53917 KB/s, 1 seconds passed
... 78%, 98464 KB, 53927 KB/s, 1 seconds passed
... 78%, 98496 KB, 53937 KB/s, 1 seconds passed
... 78%, 98528 KB, 53947 KB/s, 1 seconds passed
... 78%, 98560 KB, 53957 KB/s, 1 seconds passed
... 78%, 98592 KB, 53967 KB/s, 1 seconds passed
... 78%, 98624 KB, 53978 KB/s, 1 seconds passed
... 78%, 98656 KB, 53988 KB/s, 1 seconds passed
... 78%, 98688 KB, 53999 KB/s, 1 seconds passed
... 78%, 98720 KB, 54009 KB/s, 1 seconds passed
... 78%, 98752 KB, 54019 KB/s, 1 seconds passed
... 78%, 98784 KB, 54029 KB/s, 1 seconds passed
... 78%, 98816 KB, 54040 KB/s, 1 seconds passed
... 78%, 98848 KB, 54050 KB/s, 1 seconds passed
... 78%, 98880 KB, 54060 KB/s, 1 seconds passed
... 78%, 98912 KB, 54070 KB/s, 1 seconds passed
... 78%, 98944 KB, 54080 KB/s, 1 seconds passed
... 78%, 98976 KB, 54090 KB/s, 1 seconds passed
... 78%, 99008 KB, 54101 KB/s, 1 seconds passed
... 78%, 99040 KB, 54111 KB/s, 1 seconds passed
... 78%, 99072 KB, 54122 KB/s, 1 seconds passed
... 78%, 99104 KB, 54131 KB/s, 1 seconds passed
... 78%, 99136 KB, 54141 KB/s, 1 seconds passed
... 78%, 99168 KB, 54151 KB/s, 1 seconds passed
... 78%, 99200 KB, 54162 KB/s, 1 seconds passed
... 78%, 99232 KB, 54172 KB/s, 1 seconds passed
... 78%, 99264 KB, 54182 KB/s, 1 seconds passed
... 78%, 99296 KB, 54192 KB/s, 1 seconds passed
... 78%, 99328 KB, 54202 KB/s, 1 seconds passed
... 78%, 99360 KB, 54213 KB/s, 1 seconds passed
... 78%, 99392 KB, 54223 KB/s, 1 seconds passed
... 78%, 99424 KB, 54233 KB/s, 1 seconds passed
... 78%, 99456 KB, 54243 KB/s, 1 seconds passed
... 78%, 99488 KB, 54254 KB/s, 1 seconds passed
... 79%, 99520 KB, 54264 KB/s, 1 seconds passed
... 79%, 99552 KB, 54275 KB/s, 1 seconds passed
... 79%, 99584 KB, 54285 KB/s, 1 seconds passed
... 79%, 99616 KB, 54295 KB/s, 1 seconds passed
... 79%, 99648 KB, 54308 KB/s, 1 seconds passed
... 79%, 99680 KB, 54320 KB/s, 1 seconds passed
... 79%, 99712 KB, 54333 KB/s, 1 seconds passed
... 79%, 99744 KB, 54345 KB/s, 1 seconds passed
... 79%, 99776 KB, 54358 KB/s, 1 seconds passed
... 79%, 99808 KB, 54370 KB/s, 1 seconds passed
... 79%, 99840 KB, 54383 KB/s, 1 seconds passed
... 79%, 99872 KB, 54395 KB/s, 1 seconds passed
... 79%, 99904 KB, 54407 KB/s, 1 seconds passed
... 79%, 99936 KB, 54419 KB/s, 1 seconds passed
... 79%, 99968 KB, 54432 KB/s, 1 seconds passed
... 79%, 100000 KB, 54444 KB/s, 1 seconds passed
... 79%, 100032 KB, 54456 KB/s, 1 seconds passed
... 79%, 100064 KB, 54466 KB/s, 1 seconds passed
... 79%, 100096 KB, 54474 KB/s, 1 seconds passed
... 79%, 100128 KB, 54483 KB/s, 1 seconds passed
... 79%, 100160 KB, 54493 KB/s, 1 seconds passed
... 79%, 100192 KB, 54501 KB/s, 1 seconds passed
... 79%, 100224 KB, 54511 KB/s, 1 seconds passed
... 79%, 100256 KB, 54520 KB/s, 1 seconds passed
... 79%, 100288 KB, 54528 KB/s, 1 seconds passed
... 79%, 100320 KB, 54538 KB/s, 1 seconds passed
... 79%, 100352 KB, 54547 KB/s, 1 seconds passed
... 79%, 100384 KB, 54555 KB/s, 1 seconds passed
... 79%, 100416 KB, 54566 KB/s, 1 seconds passed
... 79%, 100448 KB, 54575 KB/s, 1 seconds passed
... 79%, 100480 KB, 54582 KB/s, 1 seconds passed
... 79%, 100512 KB, 54594 KB/s, 1 seconds passed
... 79%, 100544 KB, 54602 KB/s, 1 seconds passed
... 79%, 100576 KB, 54610 KB/s, 1 seconds passed
... 79%, 100608 KB, 54618 KB/s, 1 seconds passed
... 79%, 100640 KB, 54625 KB/s, 1 seconds passed
... 79%, 100672 KB, 54630 KB/s, 1 seconds passed
... 79%, 100704 KB, 54635 KB/s, 1 seconds passed
... 79%, 100736 KB, 54647 KB/s, 1 seconds passed
... 80%, 100768 KB, 54659 KB/s, 1 seconds passed
... 80%, 100800 KB, 54672 KB/s, 1 seconds passed
... 80%, 100832 KB, 54683 KB/s, 1 seconds passed
... 80%, 100864 KB, 54690 KB/s, 1 seconds passed
... 80%, 100896 KB, 54701 KB/s, 1 seconds passed
... 80%, 100928 KB, 54709 KB/s, 1 seconds passed
... 80%, 100960 KB, 54719 KB/s, 1 seconds passed
... 80%, 100992 KB, 54728 KB/s, 1 seconds passed
... 80%, 101024 KB, 54738 KB/s, 1 seconds passed
... 80%, 101056 KB, 54746 KB/s, 1 seconds passed
... 80%, 101088 KB, 54757 KB/s, 1 seconds passed

.. parsed-literal::

    ... 80%, 101120 KB, 54767 KB/s, 1 seconds passed
... 80%, 101152 KB, 54623 KB/s, 1 seconds passed
... 80%, 101184 KB, 54633 KB/s, 1 seconds passed
... 80%, 101216 KB, 54641 KB/s, 1 seconds passed
... 80%, 101248 KB, 54651 KB/s, 1 seconds passed
... 80%, 101280 KB, 54659 KB/s, 1 seconds passed
... 80%, 101312 KB, 54669 KB/s, 1 seconds passed
... 80%, 101344 KB, 54678 KB/s, 1 seconds passed
... 80%, 101376 KB, 54688 KB/s, 1 seconds passed
... 80%, 101408 KB, 54696 KB/s, 1 seconds passed
... 80%, 101440 KB, 54705 KB/s, 1 seconds passed
... 80%, 101472 KB, 54715 KB/s, 1 seconds passed
... 80%, 101504 KB, 54724 KB/s, 1 seconds passed
... 80%, 101536 KB, 54732 KB/s, 1 seconds passed
... 80%, 101568 KB, 53812 KB/s, 1 seconds passed
... 80%, 101600 KB, 53814 KB/s, 1 seconds passed
... 80%, 101632 KB, 53819 KB/s, 1 seconds passed
... 80%, 101664 KB, 53826 KB/s, 1 seconds passed
... 80%, 101696 KB, 53831 KB/s, 1 seconds passed
... 80%, 101728 KB, 53837 KB/s, 1 seconds passed
... 80%, 101760 KB, 53843 KB/s, 1 seconds passed
... 80%, 101792 KB, 53851 KB/s, 1 seconds passed
... 80%, 101824 KB, 53857 KB/s, 1 seconds passed
... 80%, 101856 KB, 53863 KB/s, 1 seconds passed
... 80%, 101888 KB, 53868 KB/s, 1 seconds passed
... 80%, 101920 KB, 53874 KB/s, 1 seconds passed
... 80%, 101952 KB, 53879 KB/s, 1 seconds passed
... 80%, 101984 KB, 53886 KB/s, 1 seconds passed
... 80%, 102016 KB, 53892 KB/s, 1 seconds passed
... 81%, 102048 KB, 53898 KB/s, 1 seconds passed
... 81%, 102080 KB, 53904 KB/s, 1 seconds passed
... 81%, 102112 KB, 53910 KB/s, 1 seconds passed
... 81%, 102144 KB, 53916 KB/s, 1 seconds passed
... 81%, 102176 KB, 53921 KB/s, 1 seconds passed
... 81%, 102208 KB, 53926 KB/s, 1 seconds passed
... 81%, 102240 KB, 53931 KB/s, 1 seconds passed
... 81%, 102272 KB, 53938 KB/s, 1 seconds passed
... 81%, 102304 KB, 53946 KB/s, 1 seconds passed
... 81%, 102336 KB, 53955 KB/s, 1 seconds passed
... 81%, 102368 KB, 53964 KB/s, 1 seconds passed

.. parsed-literal::

    ... 81%, 102400 KB, 52799 KB/s, 1 seconds passed
... 81%, 102432 KB, 52800 KB/s, 1 seconds passed
... 81%, 102464 KB, 52805 KB/s, 1 seconds passed
... 81%, 102496 KB, 52812 KB/s, 1 seconds passed
... 81%, 102528 KB, 52819 KB/s, 1 seconds passed
... 81%, 102560 KB, 52826 KB/s, 1 seconds passed
... 81%, 102592 KB, 52832 KB/s, 1 seconds passed
... 81%, 102624 KB, 52838 KB/s, 1 seconds passed
... 81%, 102656 KB, 52844 KB/s, 1 seconds passed
... 81%, 102688 KB, 52850 KB/s, 1 seconds passed
... 81%, 102720 KB, 52855 KB/s, 1 seconds passed
... 81%, 102752 KB, 52861 KB/s, 1 seconds passed
... 81%, 102784 KB, 52865 KB/s, 1 seconds passed
... 81%, 102816 KB, 52871 KB/s, 1 seconds passed
... 81%, 102848 KB, 52877 KB/s, 1 seconds passed
... 81%, 102880 KB, 52883 KB/s, 1 seconds passed
... 81%, 102912 KB, 52888 KB/s, 1 seconds passed
... 81%, 102944 KB, 52894 KB/s, 1 seconds passed
... 81%, 102976 KB, 52900 KB/s, 1 seconds passed
... 81%, 103008 KB, 52906 KB/s, 1 seconds passed
... 81%, 103040 KB, 52911 KB/s, 1 seconds passed
... 81%, 103072 KB, 52917 KB/s, 1 seconds passed
... 81%, 103104 KB, 52923 KB/s, 1 seconds passed

.. parsed-literal::

    ... 81%, 103136 KB, 52928 KB/s, 1 seconds passed
... 81%, 103168 KB, 52934 KB/s, 1 seconds passed
... 81%, 103200 KB, 52940 KB/s, 1 seconds passed
... 81%, 103232 KB, 52945 KB/s, 1 seconds passed
... 81%, 103264 KB, 52951 KB/s, 1 seconds passed
... 82%, 103296 KB, 52957 KB/s, 1 seconds passed
... 82%, 103328 KB, 52963 KB/s, 1 seconds passed
... 82%, 103360 KB, 52969 KB/s, 1 seconds passed
... 82%, 103392 KB, 52975 KB/s, 1 seconds passed
... 82%, 103424 KB, 52980 KB/s, 1 seconds passed
... 82%, 103456 KB, 52986 KB/s, 1 seconds passed
... 82%, 103488 KB, 52991 KB/s, 1 seconds passed
... 82%, 103520 KB, 52997 KB/s, 1 seconds passed
... 82%, 103552 KB, 53003 KB/s, 1 seconds passed
... 82%, 103584 KB, 53009 KB/s, 1 seconds passed
... 82%, 103616 KB, 53016 KB/s, 1 seconds passed
... 82%, 103648 KB, 53022 KB/s, 1 seconds passed
... 82%, 103680 KB, 53028 KB/s, 1 seconds passed
... 82%, 103712 KB, 53034 KB/s, 1 seconds passed
... 82%, 103744 KB, 53040 KB/s, 1 seconds passed
... 82%, 103776 KB, 53048 KB/s, 1 seconds passed
... 82%, 103808 KB, 53058 KB/s, 1 seconds passed
... 82%, 103840 KB, 53067 KB/s, 1 seconds passed
... 82%, 103872 KB, 53077 KB/s, 1 seconds passed
... 82%, 103904 KB, 53087 KB/s, 1 seconds passed
... 82%, 103936 KB, 53097 KB/s, 1 seconds passed
... 82%, 103968 KB, 53106 KB/s, 1 seconds passed
... 82%, 104000 KB, 53116 KB/s, 1 seconds passed
... 82%, 104032 KB, 53126 KB/s, 1 seconds passed
... 82%, 104064 KB, 53135 KB/s, 1 seconds passed
... 82%, 104096 KB, 53144 KB/s, 1 seconds passed
... 82%, 104128 KB, 53154 KB/s, 1 seconds passed
... 82%, 104160 KB, 53163 KB/s, 1 seconds passed
... 82%, 104192 KB, 53173 KB/s, 1 seconds passed
... 82%, 104224 KB, 53182 KB/s, 1 seconds passed
... 82%, 104256 KB, 53192 KB/s, 1 seconds passed
... 82%, 104288 KB, 53201 KB/s, 1 seconds passed
... 82%, 104320 KB, 53211 KB/s, 1 seconds passed
... 82%, 104352 KB, 53220 KB/s, 1 seconds passed
... 82%, 104384 KB, 53230 KB/s, 1 seconds passed
... 82%, 104416 KB, 53239 KB/s, 1 seconds passed
... 82%, 104448 KB, 53249 KB/s, 1 seconds passed
... 82%, 104480 KB, 53259 KB/s, 1 seconds passed
... 82%, 104512 KB, 53268 KB/s, 1 seconds passed
... 83%, 104544 KB, 53278 KB/s, 1 seconds passed
... 83%, 104576 KB, 53288 KB/s, 1 seconds passed
... 83%, 104608 KB, 53297 KB/s, 1 seconds passed
... 83%, 104640 KB, 53307 KB/s, 1 seconds passed
... 83%, 104672 KB, 53317 KB/s, 1 seconds passed
... 83%, 104704 KB, 53327 KB/s, 1 seconds passed
... 83%, 104736 KB, 53337 KB/s, 1 seconds passed
... 83%, 104768 KB, 53346 KB/s, 1 seconds passed
... 83%, 104800 KB, 53356 KB/s, 1 seconds passed
... 83%, 104832 KB, 53365 KB/s, 1 seconds passed
... 83%, 104864 KB, 53375 KB/s, 1 seconds passed
... 83%, 104896 KB, 53384 KB/s, 1 seconds passed
... 83%, 104928 KB, 53394 KB/s, 1 seconds passed
... 83%, 104960 KB, 53404 KB/s, 1 seconds passed
... 83%, 104992 KB, 53413 KB/s, 1 seconds passed
... 83%, 105024 KB, 53423 KB/s, 1 seconds passed
... 83%, 105056 KB, 53434 KB/s, 1 seconds passed
... 83%, 105088 KB, 53445 KB/s, 1 seconds passed
... 83%, 105120 KB, 53457 KB/s, 1 seconds passed
... 83%, 105152 KB, 53469 KB/s, 1 seconds passed
... 83%, 105184 KB, 53480 KB/s, 1 seconds passed
... 83%, 105216 KB, 53492 KB/s, 1 seconds passed
... 83%, 105248 KB, 53504 KB/s, 1 seconds passed
... 83%, 105280 KB, 53515 KB/s, 1 seconds passed
... 83%, 105312 KB, 53527 KB/s, 1 seconds passed
... 83%, 105344 KB, 53538 KB/s, 1 seconds passed
... 83%, 105376 KB, 53550 KB/s, 1 seconds passed
... 83%, 105408 KB, 53561 KB/s, 1 seconds passed
... 83%, 105440 KB, 53572 KB/s, 1 seconds passed
... 83%, 105472 KB, 53583 KB/s, 1 seconds passed
... 83%, 105504 KB, 53595 KB/s, 1 seconds passed
... 83%, 105536 KB, 53605 KB/s, 1 seconds passed
... 83%, 105568 KB, 53617 KB/s, 1 seconds passed
... 83%, 105600 KB, 53629 KB/s, 1 seconds passed
... 83%, 105632 KB, 53640 KB/s, 1 seconds passed
... 83%, 105664 KB, 53652 KB/s, 1 seconds passed
... 83%, 105696 KB, 53663 KB/s, 1 seconds passed
... 83%, 105728 KB, 53675 KB/s, 1 seconds passed
... 83%, 105760 KB, 53687 KB/s, 1 seconds passed
... 83%, 105792 KB, 53699 KB/s, 1 seconds passed
... 84%, 105824 KB, 53711 KB/s, 1 seconds passed
... 84%, 105856 KB, 53722 KB/s, 1 seconds passed
... 84%, 105888 KB, 53734 KB/s, 1 seconds passed
... 84%, 105920 KB, 53745 KB/s, 1 seconds passed
... 84%, 105952 KB, 53754 KB/s, 1 seconds passed
... 84%, 105984 KB, 53763 KB/s, 1 seconds passed
... 84%, 106016 KB, 53773 KB/s, 1 seconds passed
... 84%, 106048 KB, 53782 KB/s, 1 seconds passed
... 84%, 106080 KB, 53788 KB/s, 1 seconds passed
... 84%, 106112 KB, 53797 KB/s, 1 seconds passed
... 84%, 106144 KB, 53804 KB/s, 1 seconds passed
... 84%, 106176 KB, 53813 KB/s, 1 seconds passed
... 84%, 106208 KB, 53822 KB/s, 1 seconds passed
... 84%, 106240 KB, 53833 KB/s, 1 seconds passed
... 84%, 106272 KB, 53721 KB/s, 1 seconds passed
... 84%, 106304 KB, 53706 KB/s, 1 seconds passed
... 84%, 106336 KB, 53713 KB/s, 1 seconds passed
... 84%, 106368 KB, 53720 KB/s, 1 seconds passed
... 84%, 106400 KB, 53728 KB/s, 1 seconds passed
... 84%, 106432 KB, 53735 KB/s, 1 seconds passed
... 84%, 106464 KB, 53743 KB/s, 1 seconds passed
... 84%, 106496 KB, 53750 KB/s, 1 seconds passed
... 84%, 106528 KB, 53758 KB/s, 1 seconds passed
... 84%, 106560 KB, 53765 KB/s, 1 seconds passed
... 84%, 106592 KB, 53773 KB/s, 1 seconds passed
... 84%, 106624 KB, 53781 KB/s, 1 seconds passed
... 84%, 106656 KB, 53788 KB/s, 1 seconds passed
... 84%, 106688 KB, 53795 KB/s, 1 seconds passed
... 84%, 106720 KB, 53803 KB/s, 1 seconds passed
... 84%, 106752 KB, 53810 KB/s, 1 seconds passed
... 84%, 106784 KB, 53817 KB/s, 1 seconds passed
... 84%, 106816 KB, 53825 KB/s, 1 seconds passed
... 84%, 106848 KB, 53833 KB/s, 1 seconds passed
... 84%, 106880 KB, 53840 KB/s, 1 seconds passed
... 84%, 106912 KB, 53847 KB/s, 1 seconds passed
... 84%, 106944 KB, 53855 KB/s, 1 seconds passed
... 84%, 106976 KB, 53862 KB/s, 1 seconds passed
... 84%, 107008 KB, 53870 KB/s, 1 seconds passed
... 84%, 107040 KB, 53877 KB/s, 1 seconds passed
... 85%, 107072 KB, 53885 KB/s, 1 seconds passed
... 85%, 107104 KB, 53892 KB/s, 1 seconds passed
... 85%, 107136 KB, 53899 KB/s, 1 seconds passed
... 85%, 107168 KB, 53906 KB/s, 1 seconds passed
... 85%, 107200 KB, 53913 KB/s, 1 seconds passed
... 85%, 107232 KB, 53920 KB/s, 1 seconds passed
... 85%, 107264 KB, 53927 KB/s, 1 seconds passed
... 85%, 107296 KB, 53935 KB/s, 1 seconds passed
... 85%, 107328 KB, 53942 KB/s, 1 seconds passed
... 85%, 107360 KB, 53950 KB/s, 1 seconds passed
... 85%, 107392 KB, 53959 KB/s, 1 seconds passed
... 85%, 107424 KB, 53968 KB/s, 1 seconds passed
... 85%, 107456 KB, 53977 KB/s, 1 seconds passed
... 85%, 107488 KB, 53987 KB/s, 1 seconds passed

.. parsed-literal::

    ... 85%, 107520 KB, 52640 KB/s, 2 seconds passed
... 85%, 107552 KB, 52643 KB/s, 2 seconds passed
... 85%, 107584 KB, 52648 KB/s, 2 seconds passed
... 85%, 107616 KB, 52653 KB/s, 2 seconds passed
... 85%, 107648 KB, 52656 KB/s, 2 seconds passed
... 85%, 107680 KB, 52661 KB/s, 2 seconds passed
... 85%, 107712 KB, 52666 KB/s, 2 seconds passed
... 85%, 107744 KB, 52671 KB/s, 2 seconds passed
... 85%, 107776 KB, 52676 KB/s, 2 seconds passed
... 85%, 107808 KB, 52682 KB/s, 2 seconds passed
... 85%, 107840 KB, 52688 KB/s, 2 seconds passed
... 85%, 107872 KB, 52693 KB/s, 2 seconds passed
... 85%, 107904 KB, 52698 KB/s, 2 seconds passed
... 85%, 107936 KB, 52704 KB/s, 2 seconds passed
... 85%, 107968 KB, 52709 KB/s, 2 seconds passed
... 85%, 108000 KB, 52715 KB/s, 2 seconds passed
... 85%, 108032 KB, 52720 KB/s, 2 seconds passed
... 85%, 108064 KB, 52726 KB/s, 2 seconds passed
... 85%, 108096 KB, 52732 KB/s, 2 seconds passed
... 85%, 108128 KB, 52738 KB/s, 2 seconds passed
... 85%, 108160 KB, 52743 KB/s, 2 seconds passed
... 85%, 108192 KB, 52749 KB/s, 2 seconds passed

.. parsed-literal::

    ... 85%, 108224 KB, 52754 KB/s, 2 seconds passed
... 85%, 108256 KB, 52761 KB/s, 2 seconds passed
... 85%, 108288 KB, 52765 KB/s, 2 seconds passed
... 86%, 108320 KB, 52771 KB/s, 2 seconds passed
... 86%, 108352 KB, 52777 KB/s, 2 seconds passed
... 86%, 108384 KB, 52783 KB/s, 2 seconds passed
... 86%, 108416 KB, 52788 KB/s, 2 seconds passed
... 86%, 108448 KB, 52795 KB/s, 2 seconds passed
... 86%, 108480 KB, 52802 KB/s, 2 seconds passed
... 86%, 108512 KB, 52810 KB/s, 2 seconds passed
... 86%, 108544 KB, 52818 KB/s, 2 seconds passed
... 86%, 108576 KB, 52825 KB/s, 2 seconds passed
... 86%, 108608 KB, 52831 KB/s, 2 seconds passed
... 86%, 108640 KB, 52836 KB/s, 2 seconds passed
... 86%, 108672 KB, 52845 KB/s, 2 seconds passed
... 86%, 108704 KB, 52854 KB/s, 2 seconds passed
... 86%, 108736 KB, 52863 KB/s, 2 seconds passed
... 86%, 108768 KB, 52873 KB/s, 2 seconds passed
... 86%, 108800 KB, 52883 KB/s, 2 seconds passed
... 86%, 108832 KB, 52894 KB/s, 2 seconds passed
... 86%, 108864 KB, 52905 KB/s, 2 seconds passed
... 86%, 108896 KB, 52915 KB/s, 2 seconds passed
... 86%, 108928 KB, 52926 KB/s, 2 seconds passed
... 86%, 108960 KB, 52936 KB/s, 2 seconds passed
... 86%, 108992 KB, 52947 KB/s, 2 seconds passed
... 86%, 109024 KB, 52958 KB/s, 2 seconds passed
... 86%, 109056 KB, 52876 KB/s, 2 seconds passed
... 86%, 109088 KB, 52880 KB/s, 2 seconds passed
... 86%, 109120 KB, 52889 KB/s, 2 seconds passed
... 86%, 109152 KB, 52898 KB/s, 2 seconds passed
... 86%, 109184 KB, 52907 KB/s, 2 seconds passed
... 86%, 109216 KB, 52916 KB/s, 2 seconds passed
... 86%, 109248 KB, 52925 KB/s, 2 seconds passed
... 86%, 109280 KB, 52934 KB/s, 2 seconds passed
... 86%, 109312 KB, 52943 KB/s, 2 seconds passed
... 86%, 109344 KB, 52951 KB/s, 2 seconds passed
... 86%, 109376 KB, 52960 KB/s, 2 seconds passed
... 86%, 109408 KB, 52969 KB/s, 2 seconds passed
... 86%, 109440 KB, 52976 KB/s, 2 seconds passed
... 86%, 109472 KB, 52985 KB/s, 2 seconds passed
... 86%, 109504 KB, 52993 KB/s, 2 seconds passed
... 86%, 109536 KB, 53002 KB/s, 2 seconds passed
... 86%, 109568 KB, 53010 KB/s, 2 seconds passed
... 87%, 109600 KB, 53018 KB/s, 2 seconds passed
... 87%, 109632 KB, 53027 KB/s, 2 seconds passed
... 87%, 109664 KB, 53036 KB/s, 2 seconds passed
... 87%, 109696 KB, 53043 KB/s, 2 seconds passed
... 87%, 109728 KB, 53052 KB/s, 2 seconds passed
... 87%, 109760 KB, 53060 KB/s, 2 seconds passed
... 87%, 109792 KB, 53069 KB/s, 2 seconds passed
... 87%, 109824 KB, 53078 KB/s, 2 seconds passed
... 87%, 109856 KB, 53085 KB/s, 2 seconds passed
... 87%, 109888 KB, 53093 KB/s, 2 seconds passed
... 87%, 109920 KB, 53102 KB/s, 2 seconds passed
... 87%, 109952 KB, 53111 KB/s, 2 seconds passed
... 87%, 109984 KB, 53118 KB/s, 2 seconds passed
... 87%, 110016 KB, 53127 KB/s, 2 seconds passed
... 87%, 110048 KB, 53135 KB/s, 2 seconds passed
... 87%, 110080 KB, 53143 KB/s, 2 seconds passed
... 87%, 110112 KB, 53151 KB/s, 2 seconds passed
... 87%, 110144 KB, 53160 KB/s, 2 seconds passed
... 87%, 110176 KB, 53168 KB/s, 2 seconds passed
... 87%, 110208 KB, 53176 KB/s, 2 seconds passed
... 87%, 110240 KB, 53185 KB/s, 2 seconds passed
... 87%, 110272 KB, 53193 KB/s, 2 seconds passed
... 87%, 110304 KB, 53202 KB/s, 2 seconds passed
... 87%, 110336 KB, 53209 KB/s, 2 seconds passed
... 87%, 110368 KB, 53218 KB/s, 2 seconds passed
... 87%, 110400 KB, 53226 KB/s, 2 seconds passed
... 87%, 110432 KB, 53235 KB/s, 2 seconds passed
... 87%, 110464 KB, 53242 KB/s, 2 seconds passed
... 87%, 110496 KB, 53251 KB/s, 2 seconds passed
... 87%, 110528 KB, 53259 KB/s, 2 seconds passed
... 87%, 110560 KB, 53268 KB/s, 2 seconds passed
... 87%, 110592 KB, 53275 KB/s, 2 seconds passed
... 87%, 110624 KB, 53284 KB/s, 2 seconds passed
... 87%, 110656 KB, 53292 KB/s, 2 seconds passed
... 87%, 110688 KB, 53301 KB/s, 2 seconds passed
... 87%, 110720 KB, 53308 KB/s, 2 seconds passed
... 87%, 110752 KB, 53317 KB/s, 2 seconds passed
... 87%, 110784 KB, 53326 KB/s, 2 seconds passed
... 87%, 110816 KB, 53333 KB/s, 2 seconds passed
... 88%, 110848 KB, 53342 KB/s, 2 seconds passed
... 88%, 110880 KB, 53350 KB/s, 2 seconds passed
... 88%, 110912 KB, 53359 KB/s, 2 seconds passed
... 88%, 110944 KB, 53367 KB/s, 2 seconds passed
... 88%, 110976 KB, 53375 KB/s, 2 seconds passed
... 88%, 111008 KB, 53382 KB/s, 2 seconds passed
... 88%, 111040 KB, 53391 KB/s, 2 seconds passed
... 88%, 111072 KB, 53398 KB/s, 2 seconds passed
... 88%, 111104 KB, 53408 KB/s, 2 seconds passed
... 88%, 111136 KB, 53416 KB/s, 2 seconds passed
... 88%, 111168 KB, 53423 KB/s, 2 seconds passed
... 88%, 111200 KB, 53432 KB/s, 2 seconds passed
... 88%, 111232 KB, 53441 KB/s, 2 seconds passed
... 88%, 111264 KB, 53448 KB/s, 2 seconds passed
... 88%, 111296 KB, 53457 KB/s, 2 seconds passed
... 88%, 111328 KB, 53464 KB/s, 2 seconds passed
... 88%, 111360 KB, 53473 KB/s, 2 seconds passed
... 88%, 111392 KB, 53482 KB/s, 2 seconds passed
... 88%, 111424 KB, 53489 KB/s, 2 seconds passed
... 88%, 111456 KB, 53498 KB/s, 2 seconds passed
... 88%, 111488 KB, 53506 KB/s, 2 seconds passed
... 88%, 111520 KB, 53514 KB/s, 2 seconds passed
... 88%, 111552 KB, 53522 KB/s, 2 seconds passed
... 88%, 111584 KB, 53530 KB/s, 2 seconds passed
... 88%, 111616 KB, 53538 KB/s, 2 seconds passed
... 88%, 111648 KB, 53547 KB/s, 2 seconds passed
... 88%, 111680 KB, 53554 KB/s, 2 seconds passed
... 88%, 111712 KB, 53563 KB/s, 2 seconds passed
... 88%, 111744 KB, 53570 KB/s, 2 seconds passed
... 88%, 111776 KB, 53579 KB/s, 2 seconds passed
... 88%, 111808 KB, 53409 KB/s, 2 seconds passed
... 88%, 111840 KB, 53415 KB/s, 2 seconds passed
... 88%, 111872 KB, 53425 KB/s, 2 seconds passed
... 88%, 111904 KB, 53433 KB/s, 2 seconds passed
... 88%, 111936 KB, 53440 KB/s, 2 seconds passed
... 88%, 111968 KB, 53449 KB/s, 2 seconds passed
... 88%, 112000 KB, 53458 KB/s, 2 seconds passed
... 88%, 112032 KB, 53465 KB/s, 2 seconds passed
... 88%, 112064 KB, 53474 KB/s, 2 seconds passed
... 88%, 112096 KB, 53481 KB/s, 2 seconds passed
... 89%, 112128 KB, 53490 KB/s, 2 seconds passed
... 89%, 112160 KB, 53498 KB/s, 2 seconds passed
... 89%, 112192 KB, 53504 KB/s, 2 seconds passed
... 89%, 112224 KB, 53512 KB/s, 2 seconds passed
... 89%, 112256 KB, 53520 KB/s, 2 seconds passed
... 89%, 112288 KB, 53528 KB/s, 2 seconds passed
... 89%, 112320 KB, 53537 KB/s, 2 seconds passed
... 89%, 112352 KB, 53545 KB/s, 2 seconds passed
... 89%, 112384 KB, 53553 KB/s, 2 seconds passed
... 89%, 112416 KB, 53561 KB/s, 2 seconds passed
... 89%, 112448 KB, 53569 KB/s, 2 seconds passed
... 89%, 112480 KB, 53577 KB/s, 2 seconds passed
... 89%, 112512 KB, 53585 KB/s, 2 seconds passed
... 89%, 112544 KB, 53592 KB/s, 2 seconds passed
... 89%, 112576 KB, 53600 KB/s, 2 seconds passed
... 89%, 112608 KB, 53610 KB/s, 2 seconds passed

.. parsed-literal::

    ... 89%, 112640 KB, 51383 KB/s, 2 seconds passed
... 89%, 112672 KB, 51385 KB/s, 2 seconds passed
... 89%, 112704 KB, 51389 KB/s, 2 seconds passed
... 89%, 112736 KB, 51394 KB/s, 2 seconds passed
... 89%, 112768 KB, 51399 KB/s, 2 seconds passed
... 89%, 112800 KB, 51404 KB/s, 2 seconds passed
... 89%, 112832 KB, 51409 KB/s, 2 seconds passed
... 89%, 112864 KB, 51415 KB/s, 2 seconds passed
... 89%, 112896 KB, 51420 KB/s, 2 seconds passed
... 89%, 112928 KB, 51425 KB/s, 2 seconds passed
... 89%, 112960 KB, 51430 KB/s, 2 seconds passed
... 89%, 112992 KB, 51436 KB/s, 2 seconds passed
... 89%, 113024 KB, 51443 KB/s, 2 seconds passed
... 89%, 113056 KB, 51450 KB/s, 2 seconds passed
... 89%, 113088 KB, 51457 KB/s, 2 seconds passed
... 89%, 113120 KB, 51465 KB/s, 2 seconds passed
... 89%, 113152 KB, 51473 KB/s, 2 seconds passed
... 89%, 113184 KB, 51481 KB/s, 2 seconds passed
... 89%, 113216 KB, 51480 KB/s, 2 seconds passed
... 89%, 113248 KB, 51485 KB/s, 2 seconds passed
... 89%, 113280 KB, 51490 KB/s, 2 seconds passed
... 89%, 113312 KB, 51496 KB/s, 2 seconds passed
... 89%, 113344 KB, 51502 KB/s, 2 seconds passed
... 90%, 113376 KB, 51507 KB/s, 2 seconds passed
... 90%, 113408 KB, 51514 KB/s, 2 seconds passed
... 90%, 113440 KB, 51500 KB/s, 2 seconds passed
... 90%, 113472 KB, 51502 KB/s, 2 seconds passed
... 90%, 113504 KB, 51508 KB/s, 2 seconds passed
... 90%, 113536 KB, 51513 KB/s, 2 seconds passed
... 90%, 113568 KB, 51518 KB/s, 2 seconds passed

.. parsed-literal::

    ... 90%, 113600 KB, 51523 KB/s, 2 seconds passed
... 90%, 113632 KB, 51529 KB/s, 2 seconds passed
... 90%, 113664 KB, 51535 KB/s, 2 seconds passed
... 90%, 113696 KB, 51540 KB/s, 2 seconds passed
... 90%, 113728 KB, 51546 KB/s, 2 seconds passed
... 90%, 113760 KB, 51551 KB/s, 2 seconds passed
... 90%, 113792 KB, 51557 KB/s, 2 seconds passed
... 90%, 113824 KB, 51564 KB/s, 2 seconds passed
... 90%, 113856 KB, 51571 KB/s, 2 seconds passed
... 90%, 113888 KB, 51578 KB/s, 2 seconds passed
... 90%, 113920 KB, 51586 KB/s, 2 seconds passed
... 90%, 113952 KB, 51594 KB/s, 2 seconds passed
... 90%, 113984 KB, 51601 KB/s, 2 seconds passed
... 90%, 114016 KB, 51609 KB/s, 2 seconds passed
... 90%, 114048 KB, 51617 KB/s, 2 seconds passed
... 90%, 114080 KB, 51625 KB/s, 2 seconds passed
... 90%, 114112 KB, 51632 KB/s, 2 seconds passed
... 90%, 114144 KB, 51640 KB/s, 2 seconds passed
... 90%, 114176 KB, 51648 KB/s, 2 seconds passed
... 90%, 114208 KB, 51655 KB/s, 2 seconds passed
... 90%, 114240 KB, 51663 KB/s, 2 seconds passed
... 90%, 114272 KB, 51671 KB/s, 2 seconds passed
... 90%, 114304 KB, 51679 KB/s, 2 seconds passed
... 90%, 114336 KB, 51686 KB/s, 2 seconds passed
... 90%, 114368 KB, 51694 KB/s, 2 seconds passed
... 90%, 114400 KB, 51702 KB/s, 2 seconds passed
... 90%, 114432 KB, 51710 KB/s, 2 seconds passed
... 90%, 114464 KB, 51717 KB/s, 2 seconds passed
... 90%, 114496 KB, 51725 KB/s, 2 seconds passed
... 90%, 114528 KB, 51733 KB/s, 2 seconds passed
... 90%, 114560 KB, 51740 KB/s, 2 seconds passed
... 90%, 114592 KB, 51748 KB/s, 2 seconds passed
... 91%, 114624 KB, 51756 KB/s, 2 seconds passed
... 91%, 114656 KB, 51764 KB/s, 2 seconds passed
... 91%, 114688 KB, 51772 KB/s, 2 seconds passed
... 91%, 114720 KB, 51780 KB/s, 2 seconds passed
... 91%, 114752 KB, 51787 KB/s, 2 seconds passed
... 91%, 114784 KB, 51795 KB/s, 2 seconds passed
... 91%, 114816 KB, 51803 KB/s, 2 seconds passed
... 91%, 114848 KB, 51810 KB/s, 2 seconds passed
... 91%, 114880 KB, 51818 KB/s, 2 seconds passed
... 91%, 114912 KB, 51826 KB/s, 2 seconds passed
... 91%, 114944 KB, 51834 KB/s, 2 seconds passed
... 91%, 114976 KB, 51843 KB/s, 2 seconds passed
... 91%, 115008 KB, 51853 KB/s, 2 seconds passed
... 91%, 115040 KB, 51863 KB/s, 2 seconds passed
... 91%, 115072 KB, 51873 KB/s, 2 seconds passed
... 91%, 115104 KB, 51883 KB/s, 2 seconds passed
... 91%, 115136 KB, 51893 KB/s, 2 seconds passed
... 91%, 115168 KB, 51903 KB/s, 2 seconds passed
... 91%, 115200 KB, 51913 KB/s, 2 seconds passed
... 91%, 115232 KB, 51923 KB/s, 2 seconds passed
... 91%, 115264 KB, 51933 KB/s, 2 seconds passed
... 91%, 115296 KB, 51943 KB/s, 2 seconds passed
... 91%, 115328 KB, 51953 KB/s, 2 seconds passed
... 91%, 115360 KB, 51963 KB/s, 2 seconds passed
... 91%, 115392 KB, 51973 KB/s, 2 seconds passed
... 91%, 115424 KB, 51983 KB/s, 2 seconds passed
... 91%, 115456 KB, 51993 KB/s, 2 seconds passed
... 91%, 115488 KB, 52003 KB/s, 2 seconds passed
... 91%, 115520 KB, 52013 KB/s, 2 seconds passed
... 91%, 115552 KB, 52022 KB/s, 2 seconds passed
... 91%, 115584 KB, 52032 KB/s, 2 seconds passed
... 91%, 115616 KB, 52040 KB/s, 2 seconds passed
... 91%, 115648 KB, 52048 KB/s, 2 seconds passed
... 91%, 115680 KB, 52057 KB/s, 2 seconds passed
... 91%, 115712 KB, 52065 KB/s, 2 seconds passed
... 91%, 115744 KB, 52073 KB/s, 2 seconds passed
... 91%, 115776 KB, 52081 KB/s, 2 seconds passed
... 91%, 115808 KB, 52088 KB/s, 2 seconds passed
... 91%, 115840 KB, 52096 KB/s, 2 seconds passed
... 91%, 115872 KB, 52103 KB/s, 2 seconds passed
... 92%, 115904 KB, 52111 KB/s, 2 seconds passed
... 92%, 115936 KB, 52118 KB/s, 2 seconds passed
... 92%, 115968 KB, 52128 KB/s, 2 seconds passed
... 92%, 116000 KB, 52136 KB/s, 2 seconds passed
... 92%, 116032 KB, 52142 KB/s, 2 seconds passed
... 92%, 116064 KB, 52152 KB/s, 2 seconds passed
... 92%, 116096 KB, 52161 KB/s, 2 seconds passed
... 92%, 116128 KB, 52169 KB/s, 2 seconds passed
... 92%, 116160 KB, 52177 KB/s, 2 seconds passed
... 92%, 116192 KB, 52184 KB/s, 2 seconds passed
... 92%, 116224 KB, 52191 KB/s, 2 seconds passed
... 92%, 116256 KB, 52198 KB/s, 2 seconds passed
... 92%, 116288 KB, 52205 KB/s, 2 seconds passed
... 92%, 116320 KB, 52213 KB/s, 2 seconds passed
... 92%, 116352 KB, 52222 KB/s, 2 seconds passed
... 92%, 116384 KB, 52231 KB/s, 2 seconds passed
... 92%, 116416 KB, 52238 KB/s, 2 seconds passed
... 92%, 116448 KB, 52246 KB/s, 2 seconds passed
... 92%, 116480 KB, 52254 KB/s, 2 seconds passed
... 92%, 116512 KB, 52261 KB/s, 2 seconds passed
... 92%, 116544 KB, 52269 KB/s, 2 seconds passed
... 92%, 116576 KB, 52147 KB/s, 2 seconds passed
... 92%, 116608 KB, 52155 KB/s, 2 seconds passed
... 92%, 116640 KB, 52165 KB/s, 2 seconds passed
... 92%, 116672 KB, 52173 KB/s, 2 seconds passed
... 92%, 116704 KB, 52181 KB/s, 2 seconds passed
... 92%, 116736 KB, 52188 KB/s, 2 seconds passed
... 92%, 116768 KB, 52196 KB/s, 2 seconds passed
... 92%, 116800 KB, 52204 KB/s, 2 seconds passed
... 92%, 116832 KB, 52211 KB/s, 2 seconds passed
... 92%, 116864 KB, 52219 KB/s, 2 seconds passed
... 92%, 116896 KB, 52228 KB/s, 2 seconds passed
... 92%, 116928 KB, 52235 KB/s, 2 seconds passed
... 92%, 116960 KB, 52244 KB/s, 2 seconds passed
... 92%, 116992 KB, 52251 KB/s, 2 seconds passed
... 92%, 117024 KB, 52259 KB/s, 2 seconds passed
... 92%, 117056 KB, 52267 KB/s, 2 seconds passed
... 92%, 117088 KB, 52275 KB/s, 2 seconds passed
... 92%, 117120 KB, 52282 KB/s, 2 seconds passed
... 93%, 117152 KB, 52290 KB/s, 2 seconds passed
... 93%, 117184 KB, 52298 KB/s, 2 seconds passed
... 93%, 117216 KB, 52305 KB/s, 2 seconds passed
... 93%, 117248 KB, 52313 KB/s, 2 seconds passed
... 93%, 117280 KB, 52321 KB/s, 2 seconds passed
... 93%, 117312 KB, 52328 KB/s, 2 seconds passed
... 93%, 117344 KB, 52336 KB/s, 2 seconds passed
... 93%, 117376 KB, 52343 KB/s, 2 seconds passed
... 93%, 117408 KB, 52351 KB/s, 2 seconds passed
... 93%, 117440 KB, 52360 KB/s, 2 seconds passed
... 93%, 117472 KB, 52368 KB/s, 2 seconds passed
... 93%, 117504 KB, 52374 KB/s, 2 seconds passed
... 93%, 117536 KB, 52383 KB/s, 2 seconds passed
... 93%, 117568 KB, 52391 KB/s, 2 seconds passed
... 93%, 117600 KB, 52399 KB/s, 2 seconds passed
... 93%, 117632 KB, 52406 KB/s, 2 seconds passed
... 93%, 117664 KB, 52414 KB/s, 2 seconds passed
... 93%, 117696 KB, 52422 KB/s, 2 seconds passed
... 93%, 117728 KB, 52430 KB/s, 2 seconds passed

.. parsed-literal::

    ... 93%, 117760 KB, 51456 KB/s, 2 seconds passed
... 93%, 117792 KB, 51459 KB/s, 2 seconds passed
... 93%, 117824 KB, 51464 KB/s, 2 seconds passed
... 93%, 117856 KB, 51469 KB/s, 2 seconds passed
... 93%, 117888 KB, 51476 KB/s, 2 seconds passed
... 93%, 117920 KB, 51477 KB/s, 2 seconds passed
... 93%, 117952 KB, 51482 KB/s, 2 seconds passed
... 93%, 117984 KB, 51487 KB/s, 2 seconds passed
... 93%, 118016 KB, 51493 KB/s, 2 seconds passed
... 93%, 118048 KB, 51497 KB/s, 2 seconds passed
... 93%, 118080 KB, 51502 KB/s, 2 seconds passed
... 93%, 118112 KB, 51508 KB/s, 2 seconds passed
... 93%, 118144 KB, 51512 KB/s, 2 seconds passed
... 93%, 118176 KB, 51518 KB/s, 2 seconds passed
... 93%, 118208 KB, 51523 KB/s, 2 seconds passed
... 93%, 118240 KB, 51528 KB/s, 2 seconds passed
... 93%, 118272 KB, 51533 KB/s, 2 seconds passed
... 93%, 118304 KB, 51539 KB/s, 2 seconds passed
... 93%, 118336 KB, 51543 KB/s, 2 seconds passed
... 93%, 118368 KB, 51549 KB/s, 2 seconds passed
... 94%, 118400 KB, 51554 KB/s, 2 seconds passed
... 94%, 118432 KB, 51560 KB/s, 2 seconds passed
... 94%, 118464 KB, 51565 KB/s, 2 seconds passed
... 94%, 118496 KB, 51570 KB/s, 2 seconds passed
... 94%, 118528 KB, 51576 KB/s, 2 seconds passed
... 94%, 118560 KB, 51582 KB/s, 2 seconds passed
... 94%, 118592 KB, 51590 KB/s, 2 seconds passed
... 94%, 118624 KB, 51597 KB/s, 2 seconds passed
... 94%, 118656 KB, 51604 KB/s, 2 seconds passed
... 94%, 118688 KB, 51612 KB/s, 2 seconds passed
... 94%, 118720 KB, 51619 KB/s, 2 seconds passed
... 94%, 118752 KB, 51626 KB/s, 2 seconds passed
... 94%, 118784 KB, 51633 KB/s, 2 seconds passed
... 94%, 118816 KB, 51640 KB/s, 2 seconds passed
... 94%, 118848 KB, 51648 KB/s, 2 seconds passed
... 94%, 118880 KB, 51655 KB/s, 2 seconds passed
... 94%, 118912 KB, 51662 KB/s, 2 seconds passed
... 94%, 118944 KB, 51670 KB/s, 2 seconds passed
... 94%, 118976 KB, 51677 KB/s, 2 seconds passed
... 94%, 119008 KB, 51684 KB/s, 2 seconds passed
... 94%, 119040 KB, 51692 KB/s, 2 seconds passed
... 94%, 119072 KB, 51699 KB/s, 2 seconds passed
... 94%, 119104 KB, 51707 KB/s, 2 seconds passed
... 94%, 119136 KB, 51714 KB/s, 2 seconds passed
... 94%, 119168 KB, 51721 KB/s, 2 seconds passed
... 94%, 119200 KB, 51729 KB/s, 2 seconds passed
... 94%, 119232 KB, 51736 KB/s, 2 seconds passed
... 94%, 119264 KB, 51744 KB/s, 2 seconds passed
... 94%, 119296 KB, 51751 KB/s, 2 seconds passed
... 94%, 119328 KB, 51758 KB/s, 2 seconds passed
... 94%, 119360 KB, 51766 KB/s, 2 seconds passed
... 94%, 119392 KB, 51773 KB/s, 2 seconds passed
... 94%, 119424 KB, 51781 KB/s, 2 seconds passed
... 94%, 119456 KB, 51788 KB/s, 2 seconds passed
... 94%, 119488 KB, 51795 KB/s, 2 seconds passed
... 94%, 119520 KB, 51803 KB/s, 2 seconds passed

.. parsed-literal::

    ... 94%, 119552 KB, 51810 KB/s, 2 seconds passed
... 94%, 119584 KB, 51818 KB/s, 2 seconds passed
... 94%, 119616 KB, 51824 KB/s, 2 seconds passed
... 94%, 119648 KB, 51832 KB/s, 2 seconds passed
... 95%, 119680 KB, 51840 KB/s, 2 seconds passed
... 95%, 119712 KB, 51849 KB/s, 2 seconds passed
... 95%, 119744 KB, 51859 KB/s, 2 seconds passed
... 95%, 119776 KB, 51869 KB/s, 2 seconds passed
... 95%, 119808 KB, 51878 KB/s, 2 seconds passed
... 95%, 119840 KB, 51888 KB/s, 2 seconds passed
... 95%, 119872 KB, 51897 KB/s, 2 seconds passed
... 95%, 119904 KB, 51907 KB/s, 2 seconds passed
... 95%, 119936 KB, 51916 KB/s, 2 seconds passed
... 95%, 119968 KB, 51926 KB/s, 2 seconds passed
... 95%, 120000 KB, 51935 KB/s, 2 seconds passed
... 95%, 120032 KB, 51945 KB/s, 2 seconds passed
... 95%, 120064 KB, 51955 KB/s, 2 seconds passed
... 95%, 120096 KB, 51964 KB/s, 2 seconds passed
... 95%, 120128 KB, 51974 KB/s, 2 seconds passed
... 95%, 120160 KB, 51983 KB/s, 2 seconds passed
... 95%, 120192 KB, 51993 KB/s, 2 seconds passed
... 95%, 120224 KB, 52003 KB/s, 2 seconds passed
... 95%, 120256 KB, 52012 KB/s, 2 seconds passed
... 95%, 120288 KB, 52021 KB/s, 2 seconds passed
... 95%, 120320 KB, 52031 KB/s, 2 seconds passed
... 95%, 120352 KB, 52041 KB/s, 2 seconds passed
... 95%, 120384 KB, 52051 KB/s, 2 seconds passed
... 95%, 120416 KB, 52060 KB/s, 2 seconds passed
... 95%, 120448 KB, 52069 KB/s, 2 seconds passed
... 95%, 120480 KB, 52079 KB/s, 2 seconds passed
... 95%, 120512 KB, 52088 KB/s, 2 seconds passed
... 95%, 120544 KB, 52098 KB/s, 2 seconds passed
... 95%, 120576 KB, 52108 KB/s, 2 seconds passed
... 95%, 120608 KB, 52117 KB/s, 2 seconds passed
... 95%, 120640 KB, 52125 KB/s, 2 seconds passed
... 95%, 120672 KB, 52132 KB/s, 2 seconds passed
... 95%, 120704 KB, 52141 KB/s, 2 seconds passed
... 95%, 120736 KB, 52149 KB/s, 2 seconds passed
... 95%, 120768 KB, 52157 KB/s, 2 seconds passed
... 95%, 120800 KB, 52164 KB/s, 2 seconds passed
... 95%, 120832 KB, 52172 KB/s, 2 seconds passed
... 95%, 120864 KB, 52180 KB/s, 2 seconds passed
... 95%, 120896 KB, 52186 KB/s, 2 seconds passed
... 96%, 120928 KB, 52194 KB/s, 2 seconds passed
... 96%, 120960 KB, 52203 KB/s, 2 seconds passed
... 96%, 120992 KB, 52210 KB/s, 2 seconds passed
... 96%, 121024 KB, 52217 KB/s, 2 seconds passed
... 96%, 121056 KB, 52224 KB/s, 2 seconds passed
... 96%, 121088 KB, 52232 KB/s, 2 seconds passed
... 96%, 121120 KB, 52240 KB/s, 2 seconds passed
... 96%, 121152 KB, 52246 KB/s, 2 seconds passed
... 96%, 121184 KB, 52254 KB/s, 2 seconds passed
... 96%, 121216 KB, 52261 KB/s, 2 seconds passed
... 96%, 121248 KB, 52269 KB/s, 2 seconds passed
... 96%, 121280 KB, 52277 KB/s, 2 seconds passed
... 96%, 121312 KB, 52284 KB/s, 2 seconds passed
... 96%, 121344 KB, 52293 KB/s, 2 seconds passed
... 96%, 121376 KB, 52301 KB/s, 2 seconds passed
... 96%, 121408 KB, 52305 KB/s, 2 seconds passed
... 96%, 121440 KB, 52309 KB/s, 2 seconds passed
... 96%, 121472 KB, 52318 KB/s, 2 seconds passed
... 96%, 121504 KB, 52329 KB/s, 2 seconds passed
... 96%, 121536 KB, 52336 KB/s, 2 seconds passed
... 96%, 121568 KB, 52343 KB/s, 2 seconds passed
... 96%, 121600 KB, 52352 KB/s, 2 seconds passed
... 96%, 121632 KB, 52275 KB/s, 2 seconds passed
... 96%, 121664 KB, 52282 KB/s, 2 seconds passed
... 96%, 121696 KB, 52290 KB/s, 2 seconds passed
... 96%, 121728 KB, 52298 KB/s, 2 seconds passed
... 96%, 121760 KB, 52304 KB/s, 2 seconds passed
... 96%, 121792 KB, 52308 KB/s, 2 seconds passed
... 96%, 121824 KB, 52312 KB/s, 2 seconds passed
... 96%, 121856 KB, 52317 KB/s, 2 seconds passed
... 96%, 121888 KB, 52323 KB/s, 2 seconds passed
... 96%, 121920 KB, 52328 KB/s, 2 seconds passed
... 96%, 121952 KB, 52334 KB/s, 2 seconds passed
... 96%, 121984 KB, 52339 KB/s, 2 seconds passed
... 96%, 122016 KB, 52344 KB/s, 2 seconds passed
... 96%, 122048 KB, 52350 KB/s, 2 seconds passed
... 96%, 122080 KB, 52355 KB/s, 2 seconds passed
... 96%, 122112 KB, 52361 KB/s, 2 seconds passed
... 96%, 122144 KB, 52366 KB/s, 2 seconds passed
... 97%, 122176 KB, 52372 KB/s, 2 seconds passed
... 97%, 122208 KB, 52377 KB/s, 2 seconds passed
... 97%, 122240 KB, 52382 KB/s, 2 seconds passed
... 97%, 122272 KB, 52387 KB/s, 2 seconds passed
... 97%, 122304 KB, 52392 KB/s, 2 seconds passed
... 97%, 122336 KB, 52398 KB/s, 2 seconds passed
... 97%, 122368 KB, 52403 KB/s, 2 seconds passed
... 97%, 122400 KB, 52409 KB/s, 2 seconds passed
... 97%, 122432 KB, 52414 KB/s, 2 seconds passed
... 97%, 122464 KB, 52419 KB/s, 2 seconds passed
... 97%, 122496 KB, 52425 KB/s, 2 seconds passed
... 97%, 122528 KB, 52430 KB/s, 2 seconds passed
... 97%, 122560 KB, 52436 KB/s, 2 seconds passed
... 97%, 122592 KB, 52441 KB/s, 2 seconds passed
... 97%, 122624 KB, 52448 KB/s, 2 seconds passed
... 97%, 122656 KB, 52455 KB/s, 2 seconds passed
... 97%, 122688 KB, 52462 KB/s, 2 seconds passed
... 97%, 122720 KB, 52470 KB/s, 2 seconds passed
... 97%, 122752 KB, 52477 KB/s, 2 seconds passed
... 97%, 122784 KB, 52486 KB/s, 2 seconds passed
... 97%, 122816 KB, 52495 KB/s, 2 seconds passed
... 97%, 122848 KB, 52505 KB/s, 2 seconds passed

.. parsed-literal::

    ... 97%, 122880 KB, 50623 KB/s, 2 seconds passed
... 97%, 122912 KB, 50625 KB/s, 2 seconds passed
... 97%, 122944 KB, 50629 KB/s, 2 seconds passed
... 97%, 122976 KB, 50634 KB/s, 2 seconds passed
... 97%, 123008 KB, 50639 KB/s, 2 seconds passed
... 97%, 123040 KB, 50644 KB/s, 2 seconds passed
... 97%, 123072 KB, 50648 KB/s, 2 seconds passed
... 97%, 123104 KB, 50653 KB/s, 2 seconds passed
... 97%, 123136 KB, 50658 KB/s, 2 seconds passed
... 97%, 123168 KB, 50663 KB/s, 2 seconds passed
... 97%, 123200 KB, 50669 KB/s, 2 seconds passed
... 97%, 123232 KB, 50674 KB/s, 2 seconds passed
... 97%, 123264 KB, 50678 KB/s, 2 seconds passed
... 97%, 123296 KB, 50684 KB/s, 2 seconds passed
... 97%, 123328 KB, 50689 KB/s, 2 seconds passed
... 97%, 123360 KB, 50694 KB/s, 2 seconds passed
... 97%, 123392 KB, 50699 KB/s, 2 seconds passed
... 97%, 123424 KB, 50703 KB/s, 2 seconds passed
... 98%, 123456 KB, 50710 KB/s, 2 seconds passed
... 98%, 123488 KB, 50717 KB/s, 2 seconds passed
... 98%, 123520 KB, 50724 KB/s, 2 seconds passed
... 98%, 123552 KB, 50731 KB/s, 2 seconds passed
... 98%, 123584 KB, 50738 KB/s, 2 seconds passed
... 98%, 123616 KB, 50745 KB/s, 2 seconds passed
... 98%, 123648 KB, 50752 KB/s, 2 seconds passed
... 98%, 123680 KB, 50759 KB/s, 2 seconds passed
... 98%, 123712 KB, 50731 KB/s, 2 seconds passed
... 98%, 123744 KB, 50734 KB/s, 2 seconds passed
... 98%, 123776 KB, 50738 KB/s, 2 seconds passed
... 98%, 123808 KB, 50743 KB/s, 2 seconds passed
... 98%, 123840 KB, 50749 KB/s, 2 seconds passed
... 98%, 123872 KB, 50755 KB/s, 2 seconds passed
... 98%, 123904 KB, 50762 KB/s, 2 seconds passed
... 98%, 123936 KB, 50769 KB/s, 2 seconds passed
... 98%, 123968 KB, 50774 KB/s, 2 seconds passed
... 98%, 124000 KB, 50781 KB/s, 2 seconds passed
... 98%, 124032 KB, 50787 KB/s, 2 seconds passed
... 98%, 124064 KB, 50794 KB/s, 2 seconds passed
... 98%, 124096 KB, 50800 KB/s, 2 seconds passed
... 98%, 124128 KB, 50807 KB/s, 2 seconds passed
... 98%, 124160 KB, 50814 KB/s, 2 seconds passed
... 98%, 124192 KB, 50821 KB/s, 2 seconds passed
... 98%, 124224 KB, 50827 KB/s, 2 seconds passed
... 98%, 124256 KB, 50834 KB/s, 2 seconds passed
... 98%, 124288 KB, 50840 KB/s, 2 seconds passed
... 98%, 124320 KB, 50847 KB/s, 2 seconds passed
... 98%, 124352 KB, 50854 KB/s, 2 seconds passed
... 98%, 124384 KB, 50861 KB/s, 2 seconds passed
... 98%, 124416 KB, 50867 KB/s, 2 seconds passed
... 98%, 124448 KB, 50874 KB/s, 2 seconds passed
... 98%, 124480 KB, 50881 KB/s, 2 seconds passed
... 98%, 124512 KB, 50888 KB/s, 2 seconds passed
... 98%, 124544 KB, 50895 KB/s, 2 seconds passed
... 98%, 124576 KB, 50902 KB/s, 2 seconds passed
... 98%, 124608 KB, 50908 KB/s, 2 seconds passed
... 98%, 124640 KB, 50915 KB/s, 2 seconds passed
... 98%, 124672 KB, 50922 KB/s, 2 seconds passed
... 99%, 124704 KB, 50928 KB/s, 2 seconds passed
... 99%, 124736 KB, 50935 KB/s, 2 seconds passed
... 99%, 124768 KB, 50942 KB/s, 2 seconds passed
... 99%, 124800 KB, 50949 KB/s, 2 seconds passed
... 99%, 124832 KB, 50956 KB/s, 2 seconds passed
... 99%, 124864 KB, 50964 KB/s, 2 seconds passed
... 99%, 124896 KB, 50973 KB/s, 2 seconds passed
... 99%, 124928 KB, 50982 KB/s, 2 seconds passed
... 99%, 124960 KB, 50991 KB/s, 2 seconds passed
... 99%, 124992 KB, 50999 KB/s, 2 seconds passed
... 99%, 125024 KB, 51008 KB/s, 2 seconds passed
... 99%, 125056 KB, 51017 KB/s, 2 seconds passed
... 99%, 125088 KB, 51025 KB/s, 2 seconds passed
... 99%, 125120 KB, 51034 KB/s, 2 seconds passed
... 99%, 125152 KB, 51043 KB/s, 2 seconds passed
... 99%, 125184 KB, 51051 KB/s, 2 seconds passed
... 99%, 125216 KB, 51060 KB/s, 2 seconds passed
... 99%, 125248 KB, 51068 KB/s, 2 seconds passed
... 99%, 125280 KB, 51077 KB/s, 2 seconds passed
... 99%, 125312 KB, 51086 KB/s, 2 seconds passed
... 99%, 125344 KB, 51094 KB/s, 2 seconds passed
... 99%, 125376 KB, 51103 KB/s, 2 seconds passed
... 99%, 125408 KB, 51112 KB/s, 2 seconds passed
... 99%, 125440 KB, 51121 KB/s, 2 seconds passed
... 99%, 125472 KB, 51130 KB/s, 2 seconds passed
... 99%, 125504 KB, 51138 KB/s, 2 seconds passed
... 99%, 125536 KB, 51147 KB/s, 2 seconds passed
... 99%, 125568 KB, 51154 KB/s, 2 seconds passed
... 99%, 125600 KB, 51160 KB/s, 2 seconds passed
... 99%, 125632 KB, 51169 KB/s, 2 seconds passed
... 99%, 125664 KB, 51175 KB/s, 2 seconds passed
... 99%, 125696 KB, 51180 KB/s, 2 seconds passed
... 99%, 125728 KB, 51188 KB/s, 2 seconds passed
... 99%, 125760 KB, 51197 KB/s, 2 seconds passed
... 99%, 125792 KB, 51204 KB/s, 2 seconds passed
... 99%, 125824 KB, 51212 KB/s, 2 seconds passed
... 99%, 125856 KB, 51219 KB/s, 2 seconds passed
... 99%, 125888 KB, 51224 KB/s, 2 seconds passed
... 99%, 125920 KB, 51232 KB/s, 2 seconds passed
... 99%, 125952 KB, 51241 KB/s, 2 seconds passed
... 100%, 125953 KB, 51237 KB/s, 2 seconds passed



.. parsed-literal::


    ========== Downloading models/public/colorization-v2/model/__init__.py


.. parsed-literal::

    ... 100%, 0 KB, 288 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/base_color.py


.. parsed-literal::

    ... 100%, 0 KB, 1806 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/eccv16.py


.. parsed-literal::

    ... 100%, 4 KB, 17288 KB/s, 0 seconds passed


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
        convert_command = f"omz_converter " f"--name {MODEL_NAME} " f"--download_dir {MODEL_DIR} " f"--precisions {PRECISION}"
        ! $convert_command


.. parsed-literal::

    ========== Converting colorization-v2 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=models/public/colorization-v2 --model-name=ECCVGenerator --weights=models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth --import-module=model --input-shape=1,1,256,256 --output-file=models/public/colorization-v2/colorization-v2-eccv16.onnx --input-names=data_l --output-names=color_ab



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting colorization-v2 to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=models/public/colorization-v2/FP16 --model_name=colorization-v2 --input=data_l --output=color_ab --input_model=models/public/colorization-v2/colorization-v2-eccv16.onnx '--layout=data_l(NCHW)' '--input_shape=[1, 1, 256, 256]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/notebooks/vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-661/.workspace/scm/ov-notebook/notebooks/vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.bin





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
            image = cv2.cvtColor(cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2RGB)
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
        img_url_0,
        filename="test_0.jpg",
        directory="data",
        show_progress=False,
        silent=True,
        timeout=30,
    )
    assert Path(image_file_0).exists()

    image_file_1 = utils.download_file(
        img_url_1,
        filename="test_1.jpg",
        directory="data",
        show_progress=False,
        silent=True,
        timeout=30,
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
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv2.cvtColor(img_lab_out, cv2.COLOR_Lab2RGB), 0, 1)
        colorized_image = (cv2.resize(img_bgr_out, (w_in, h_in)) * 255).astype(np.uint8)
        return colorized_image

.. code:: ipython3

    color_img_0 = colorize(test_img_0)
    color_img_1 = colorize(test_img_1)

Display Colorized Image
-----------------------



.. code:: ipython3

    plot_output(test_img_0, color_img_0)



.. image:: vision-image-colorization-with-output_files/vision-image-colorization-with-output_21_0.png


.. code:: ipython3

    plot_output(test_img_1, color_img_1)



.. image:: vision-image-colorization-with-output_files/vision-image-colorization-with-output_22_0.png

