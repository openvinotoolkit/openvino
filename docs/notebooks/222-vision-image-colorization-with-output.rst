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

    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)
    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.8.8)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)


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

    ... 0%, 32 KB, 932 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 932 KB/s, 0 seconds passed
... 0%, 96 KB, 1338 KB/s, 0 seconds passed
... 0%, 128 KB, 1717 KB/s, 0 seconds passed
... 0%, 160 KB, 1557 KB/s, 0 seconds passed
... 0%, 192 KB, 1827 KB/s, 0 seconds passed
... 0%, 224 KB, 2106 KB/s, 0 seconds passed
... 0%, 256 KB, 2370 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 288 KB, 2615 KB/s, 0 seconds passed
... 0%, 320 KB, 2345 KB/s, 0 seconds passed
... 0%, 352 KB, 2560 KB/s, 0 seconds passed
... 0%, 384 KB, 2784 KB/s, 0 seconds passed
... 0%, 416 KB, 3009 KB/s, 0 seconds passed
... 0%, 448 KB, 3229 KB/s, 0 seconds passed
... 0%, 480 KB, 3443 KB/s, 0 seconds passed
... 0%, 512 KB, 3654 KB/s, 0 seconds passed
... 0%, 544 KB, 3873 KB/s, 0 seconds passed
... 0%, 576 KB, 4086 KB/s, 0 seconds passed
... 0%, 608 KB, 4216 KB/s, 0 seconds passed
... 0%, 640 KB, 4428 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 672 KB, 3916 KB/s, 0 seconds passed
... 0%, 704 KB, 4051 KB/s, 0 seconds passed
... 0%, 736 KB, 4227 KB/s, 0 seconds passed
... 0%, 768 KB, 4402 KB/s, 0 seconds passed
... 0%, 800 KB, 4578 KB/s, 0 seconds passed
... 0%, 832 KB, 4753 KB/s, 0 seconds passed
... 0%, 864 KB, 4927 KB/s, 0 seconds passed
... 0%, 896 KB, 5100 KB/s, 0 seconds passed
... 0%, 928 KB, 5273 KB/s, 0 seconds passed
... 0%, 960 KB, 5446 KB/s, 0 seconds passed
... 0%, 992 KB, 5617 KB/s, 0 seconds passed
... 0%, 1024 KB, 5788 KB/s, 0 seconds passed
... 0%, 1056 KB, 5958 KB/s, 0 seconds passed
... 0%, 1088 KB, 6128 KB/s, 0 seconds passed
... 0%, 1120 KB, 6297 KB/s, 0 seconds passed
... 0%, 1152 KB, 6466 KB/s, 0 seconds passed
... 0%, 1184 KB, 6634 KB/s, 0 seconds passed
... 0%, 1216 KB, 6802 KB/s, 0 seconds passed
... 0%, 1248 KB, 6969 KB/s, 0 seconds passed
... 1%, 1280 KB, 7137 KB/s, 0 seconds passed
... 1%, 1312 KB, 6344 KB/s, 0 seconds passed
... 1%, 1344 KB, 6485 KB/s, 0 seconds passed
... 1%, 1376 KB, 6628 KB/s, 0 seconds passed
... 1%, 1408 KB, 6772 KB/s, 0 seconds passed
... 1%, 1440 KB, 6915 KB/s, 0 seconds passed
... 1%, 1472 KB, 7058 KB/s, 0 seconds passed
... 1%, 1504 KB, 7200 KB/s, 0 seconds passed
... 1%, 1536 KB, 7343 KB/s, 0 seconds passed
... 1%, 1568 KB, 7486 KB/s, 0 seconds passed
... 1%, 1600 KB, 7627 KB/s, 0 seconds passed
... 1%, 1632 KB, 7767 KB/s, 0 seconds passed
... 1%, 1664 KB, 7907 KB/s, 0 seconds passed
... 1%, 1696 KB, 8044 KB/s, 0 seconds passed
... 1%, 1728 KB, 8184 KB/s, 0 seconds passed
... 1%, 1760 KB, 8321 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 1792 KB, 8457 KB/s, 0 seconds passed
... 1%, 1824 KB, 8595 KB/s, 0 seconds passed
... 1%, 1856 KB, 8733 KB/s, 0 seconds passed
... 1%, 1888 KB, 8870 KB/s, 0 seconds passed
... 1%, 1920 KB, 9007 KB/s, 0 seconds passed
... 1%, 1952 KB, 9145 KB/s, 0 seconds passed
... 1%, 1984 KB, 9281 KB/s, 0 seconds passed
... 1%, 2016 KB, 9417 KB/s, 0 seconds passed
... 1%, 2048 KB, 9553 KB/s, 0 seconds passed
... 1%, 2080 KB, 9688 KB/s, 0 seconds passed
... 1%, 2112 KB, 9823 KB/s, 0 seconds passed
... 1%, 2144 KB, 9958 KB/s, 0 seconds passed
... 1%, 2176 KB, 10092 KB/s, 0 seconds passed
... 1%, 2208 KB, 10226 KB/s, 0 seconds passed
... 1%, 2240 KB, 10359 KB/s, 0 seconds passed
... 1%, 2272 KB, 10492 KB/s, 0 seconds passed
... 1%, 2304 KB, 10625 KB/s, 0 seconds passed
... 1%, 2336 KB, 10757 KB/s, 0 seconds passed
... 1%, 2368 KB, 10889 KB/s, 0 seconds passed
... 1%, 2400 KB, 11020 KB/s, 0 seconds passed
... 1%, 2432 KB, 11151 KB/s, 0 seconds passed
... 1%, 2464 KB, 11282 KB/s, 0 seconds passed
... 1%, 2496 KB, 11413 KB/s, 0 seconds passed
... 2%, 2528 KB, 11543 KB/s, 0 seconds passed
... 2%, 2560 KB, 11672 KB/s, 0 seconds passed
... 2%, 2592 KB, 11801 KB/s, 0 seconds passed
... 2%, 2624 KB, 11931 KB/s, 0 seconds passed
... 2%, 2656 KB, 11098 KB/s, 0 seconds passed
... 2%, 2688 KB, 11204 KB/s, 0 seconds passed
... 2%, 2720 KB, 11249 KB/s, 0 seconds passed
... 2%, 2752 KB, 11365 KB/s, 0 seconds passed
... 2%, 2784 KB, 11483 KB/s, 0 seconds passed
... 2%, 2816 KB, 11600 KB/s, 0 seconds passed
... 2%, 2848 KB, 11717 KB/s, 0 seconds passed
... 2%, 2880 KB, 11833 KB/s, 0 seconds passed
... 2%, 2912 KB, 11948 KB/s, 0 seconds passed
... 2%, 2944 KB, 12064 KB/s, 0 seconds passed
... 2%, 2976 KB, 12181 KB/s, 0 seconds passed
... 2%, 3008 KB, 12296 KB/s, 0 seconds passed
... 2%, 3040 KB, 12411 KB/s, 0 seconds passed
... 2%, 3072 KB, 12525 KB/s, 0 seconds passed
... 2%, 3104 KB, 12640 KB/s, 0 seconds passed
... 2%, 3136 KB, 12755 KB/s, 0 seconds passed
... 2%, 3168 KB, 12867 KB/s, 0 seconds passed
... 2%, 3200 KB, 12981 KB/s, 0 seconds passed
... 2%, 3232 KB, 13095 KB/s, 0 seconds passed
... 2%, 3264 KB, 13209 KB/s, 0 seconds passed
... 2%, 3296 KB, 13322 KB/s, 0 seconds passed
... 2%, 3328 KB, 13435 KB/s, 0 seconds passed
... 2%, 3360 KB, 13547 KB/s, 0 seconds passed
... 2%, 3392 KB, 13659 KB/s, 0 seconds passed
... 2%, 3424 KB, 13771 KB/s, 0 seconds passed
... 2%, 3456 KB, 13883 KB/s, 0 seconds passed
... 2%, 3488 KB, 13994 KB/s, 0 seconds passed
... 2%, 3520 KB, 14105 KB/s, 0 seconds passed
... 2%, 3552 KB, 14216 KB/s, 0 seconds passed
... 2%, 3584 KB, 14326 KB/s, 0 seconds passed
... 2%, 3616 KB, 14438 KB/s, 0 seconds passed
... 2%, 3648 KB, 14550 KB/s, 0 seconds passed
... 2%, 3680 KB, 14663 KB/s, 0 seconds passed
... 2%, 3712 KB, 14776 KB/s, 0 seconds passed
... 2%, 3744 KB, 14888 KB/s, 0 seconds passed
... 2%, 3776 KB, 15000 KB/s, 0 seconds passed
... 3%, 3808 KB, 15113 KB/s, 0 seconds passed
... 3%, 3840 KB, 15225 KB/s, 0 seconds passed
... 3%, 3872 KB, 15336 KB/s, 0 seconds passed
... 3%, 3904 KB, 15446 KB/s, 0 seconds passed
... 3%, 3936 KB, 15558 KB/s, 0 seconds passed
... 3%, 3968 KB, 15669 KB/s, 0 seconds passed
... 3%, 4000 KB, 15777 KB/s, 0 seconds passed
... 3%, 4032 KB, 15887 KB/s, 0 seconds passed
... 3%, 4064 KB, 15998 KB/s, 0 seconds passed
... 3%, 4096 KB, 16107 KB/s, 0 seconds passed
... 3%, 4128 KB, 16217 KB/s, 0 seconds passed
... 3%, 4160 KB, 16327 KB/s, 0 seconds passed
... 3%, 4192 KB, 16437 KB/s, 0 seconds passed
... 3%, 4224 KB, 16546 KB/s, 0 seconds passed
... 3%, 4256 KB, 16655 KB/s, 0 seconds passed
... 3%, 4288 KB, 16764 KB/s, 0 seconds passed
... 3%, 4320 KB, 16873 KB/s, 0 seconds passed
... 3%, 4352 KB, 16982 KB/s, 0 seconds passed
... 3%, 4384 KB, 16715 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 4416 KB, 16533 KB/s, 0 seconds passed
... 3%, 4448 KB, 16627 KB/s, 0 seconds passed
... 3%, 4480 KB, 16725 KB/s, 0 seconds passed
... 3%, 4512 KB, 16824 KB/s, 0 seconds passed
... 3%, 4544 KB, 16924 KB/s, 0 seconds passed
... 3%, 4576 KB, 17023 KB/s, 0 seconds passed
... 3%, 4608 KB, 17122 KB/s, 0 seconds passed
... 3%, 4640 KB, 17222 KB/s, 0 seconds passed
... 3%, 4672 KB, 17321 KB/s, 0 seconds passed
... 3%, 4704 KB, 17421 KB/s, 0 seconds passed
... 3%, 4736 KB, 17519 KB/s, 0 seconds passed
... 3%, 4768 KB, 17616 KB/s, 0 seconds passed
... 3%, 4800 KB, 17715 KB/s, 0 seconds passed
... 3%, 4832 KB, 17813 KB/s, 0 seconds passed
... 3%, 4864 KB, 17911 KB/s, 0 seconds passed
... 3%, 4896 KB, 18009 KB/s, 0 seconds passed
... 3%, 4928 KB, 18106 KB/s, 0 seconds passed
... 3%, 4960 KB, 18203 KB/s, 0 seconds passed
... 3%, 4992 KB, 18299 KB/s, 0 seconds passed
... 3%, 5024 KB, 18395 KB/s, 0 seconds passed
... 4%, 5056 KB, 18492 KB/s, 0 seconds passed
... 4%, 5088 KB, 18588 KB/s, 0 seconds passed
... 4%, 5120 KB, 18684 KB/s, 0 seconds passed
... 4%, 5152 KB, 18780 KB/s, 0 seconds passed
... 4%, 5184 KB, 18876 KB/s, 0 seconds passed
... 4%, 5216 KB, 18969 KB/s, 0 seconds passed
... 4%, 5248 KB, 19064 KB/s, 0 seconds passed
... 4%, 5280 KB, 19159 KB/s, 0 seconds passed
... 4%, 5312 KB, 19255 KB/s, 0 seconds passed
... 4%, 5344 KB, 19350 KB/s, 0 seconds passed
... 4%, 5376 KB, 19445 KB/s, 0 seconds passed
... 4%, 5408 KB, 19538 KB/s, 0 seconds passed
... 4%, 5440 KB, 19632 KB/s, 0 seconds passed
... 4%, 5472 KB, 19725 KB/s, 0 seconds passed
... 4%, 5504 KB, 19819 KB/s, 0 seconds passed
... 4%, 5536 KB, 19912 KB/s, 0 seconds passed
... 4%, 5568 KB, 20005 KB/s, 0 seconds passed
... 4%, 5600 KB, 20097 KB/s, 0 seconds passed
... 4%, 5632 KB, 20190 KB/s, 0 seconds passed
... 4%, 5664 KB, 20282 KB/s, 0 seconds passed
... 4%, 5696 KB, 20375 KB/s, 0 seconds passed
... 4%, 5728 KB, 20468 KB/s, 0 seconds passed
... 4%, 5760 KB, 20558 KB/s, 0 seconds passed
... 4%, 5792 KB, 20650 KB/s, 0 seconds passed
... 4%, 5824 KB, 20744 KB/s, 0 seconds passed
... 4%, 5856 KB, 20843 KB/s, 0 seconds passed
... 4%, 5888 KB, 20941 KB/s, 0 seconds passed
... 4%, 5920 KB, 21039 KB/s, 0 seconds passed
... 4%, 5952 KB, 21138 KB/s, 0 seconds passed
... 4%, 5984 KB, 21236 KB/s, 0 seconds passed
... 4%, 6016 KB, 21332 KB/s, 0 seconds passed
... 4%, 6048 KB, 21430 KB/s, 0 seconds passed
... 4%, 6080 KB, 21527 KB/s, 0 seconds passed
... 4%, 6112 KB, 21625 KB/s, 0 seconds passed
... 4%, 6144 KB, 21723 KB/s, 0 seconds passed
... 4%, 6176 KB, 21820 KB/s, 0 seconds passed
... 4%, 6208 KB, 21918 KB/s, 0 seconds passed
... 4%, 6240 KB, 22015 KB/s, 0 seconds passed
... 4%, 6272 KB, 22113 KB/s, 0 seconds passed
... 5%, 6304 KB, 22209 KB/s, 0 seconds passed
... 5%, 6336 KB, 22306 KB/s, 0 seconds passed
... 5%, 6368 KB, 22403 KB/s, 0 seconds passed
... 5%, 6400 KB, 22498 KB/s, 0 seconds passed
... 5%, 6432 KB, 22593 KB/s, 0 seconds passed
... 5%, 6464 KB, 22679 KB/s, 0 seconds passed
... 5%, 6496 KB, 22771 KB/s, 0 seconds passed
... 5%, 6528 KB, 22861 KB/s, 0 seconds passed
... 5%, 6560 KB, 22952 KB/s, 0 seconds passed
... 5%, 6592 KB, 23042 KB/s, 0 seconds passed
... 5%, 6624 KB, 22746 KB/s, 0 seconds passed
... 5%, 6656 KB, 22836 KB/s, 0 seconds passed
... 5%, 6688 KB, 22924 KB/s, 0 seconds passed
... 5%, 6720 KB, 23014 KB/s, 0 seconds passed
... 5%, 6752 KB, 23102 KB/s, 0 seconds passed
... 5%, 6784 KB, 23189 KB/s, 0 seconds passed
... 5%, 6816 KB, 23278 KB/s, 0 seconds passed
... 5%, 6848 KB, 23366 KB/s, 0 seconds passed
... 5%, 6880 KB, 23450 KB/s, 0 seconds passed
... 5%, 6912 KB, 23537 KB/s, 0 seconds passed
... 5%, 6944 KB, 23625 KB/s, 0 seconds passed
... 5%, 6976 KB, 23711 KB/s, 0 seconds passed
... 5%, 7008 KB, 23799 KB/s, 0 seconds passed
... 5%, 7040 KB, 23877 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 7072 KB, 13840 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 7104 KB, 13662 KB/s, 0 seconds passed
... 5%, 7136 KB, 13577 KB/s, 0 seconds passed
... 5%, 7168 KB, 13627 KB/s, 0 seconds passed
... 5%, 7200 KB, 13545 KB/s, 0 seconds passed
... 5%, 7232 KB, 13594 KB/s, 0 seconds passed
... 5%, 7264 KB, 13644 KB/s, 0 seconds passed
... 5%, 7296 KB, 13695 KB/s, 0 seconds passed
... 5%, 7328 KB, 13748 KB/s, 0 seconds passed
... 5%, 7360 KB, 13670 KB/s, 0 seconds passed
... 5%, 7392 KB, 13717 KB/s, 0 seconds passed
... 5%, 7424 KB, 13766 KB/s, 0 seconds passed
... 5%, 7456 KB, 13818 KB/s, 0 seconds passed
... 5%, 7488 KB, 13869 KB/s, 0 seconds passed
... 5%, 7520 KB, 13920 KB/s, 0 seconds passed
... 5%, 7552 KB, 13972 KB/s, 0 seconds passed
... 6%, 7584 KB, 14023 KB/s, 0 seconds passed
... 6%, 7616 KB, 14074 KB/s, 0 seconds passed
... 6%, 7648 KB, 14125 KB/s, 0 seconds passed
... 6%, 7680 KB, 14176 KB/s, 0 seconds passed
... 6%, 7712 KB, 14227 KB/s, 0 seconds passed
... 6%, 7744 KB, 14277 KB/s, 0 seconds passed
... 6%, 7776 KB, 14328 KB/s, 0 seconds passed
... 6%, 7808 KB, 14379 KB/s, 0 seconds passed
... 6%, 7840 KB, 14430 KB/s, 0 seconds passed
... 6%, 7872 KB, 14481 KB/s, 0 seconds passed
... 6%, 7904 KB, 14532 KB/s, 0 seconds passed
... 6%, 7936 KB, 14582 KB/s, 0 seconds passed
... 6%, 7968 KB, 14633 KB/s, 0 seconds passed
... 6%, 8000 KB, 14683 KB/s, 0 seconds passed
... 6%, 8032 KB, 14734 KB/s, 0 seconds passed
... 6%, 8064 KB, 14784 KB/s, 0 seconds passed
... 6%, 8096 KB, 14835 KB/s, 0 seconds passed
... 6%, 8128 KB, 14885 KB/s, 0 seconds passed
... 6%, 8160 KB, 14935 KB/s, 0 seconds passed
... 6%, 8192 KB, 14985 KB/s, 0 seconds passed
... 6%, 8224 KB, 15035 KB/s, 0 seconds passed
... 6%, 8256 KB, 15085 KB/s, 0 seconds passed
... 6%, 8288 KB, 15135 KB/s, 0 seconds passed
... 6%, 8320 KB, 15185 KB/s, 0 seconds passed
... 6%, 8352 KB, 15235 KB/s, 0 seconds passed
... 6%, 8384 KB, 15285 KB/s, 0 seconds passed
... 6%, 8416 KB, 15335 KB/s, 0 seconds passed
... 6%, 8448 KB, 15384 KB/s, 0 seconds passed
... 6%, 8480 KB, 15434 KB/s, 0 seconds passed
... 6%, 8512 KB, 15483 KB/s, 0 seconds passed
... 6%, 8544 KB, 15532 KB/s, 0 seconds passed
... 6%, 8576 KB, 15582 KB/s, 0 seconds passed
... 6%, 8608 KB, 15631 KB/s, 0 seconds passed
... 6%, 8640 KB, 15681 KB/s, 0 seconds passed
... 6%, 8672 KB, 15731 KB/s, 0 seconds passed
... 6%, 8704 KB, 15780 KB/s, 0 seconds passed
... 6%, 8736 KB, 15829 KB/s, 0 seconds passed
... 6%, 8768 KB, 15879 KB/s, 0 seconds passed
... 6%, 8800 KB, 15928 KB/s, 0 seconds passed
... 7%, 8832 KB, 15977 KB/s, 0 seconds passed
... 7%, 8864 KB, 16026 KB/s, 0 seconds passed
... 7%, 8896 KB, 16075 KB/s, 0 seconds passed
... 7%, 8928 KB, 16127 KB/s, 0 seconds passed
... 7%, 8960 KB, 16178 KB/s, 0 seconds passed
... 7%, 8992 KB, 16230 KB/s, 0 seconds passed
... 7%, 9024 KB, 16282 KB/s, 0 seconds passed
... 7%, 9056 KB, 16334 KB/s, 0 seconds passed
... 7%, 9088 KB, 16386 KB/s, 0 seconds passed
... 7%, 9120 KB, 16437 KB/s, 0 seconds passed
... 7%, 9152 KB, 16489 KB/s, 0 seconds passed
... 7%, 9184 KB, 16540 KB/s, 0 seconds passed
... 7%, 9216 KB, 16592 KB/s, 0 seconds passed
... 7%, 9248 KB, 16643 KB/s, 0 seconds passed
... 7%, 9280 KB, 16694 KB/s, 0 seconds passed
... 7%, 9312 KB, 16746 KB/s, 0 seconds passed
... 7%, 9344 KB, 16797 KB/s, 0 seconds passed
... 7%, 9376 KB, 16849 KB/s, 0 seconds passed
... 7%, 9408 KB, 16900 KB/s, 0 seconds passed
... 7%, 9440 KB, 16950 KB/s, 0 seconds passed
... 7%, 9472 KB, 17002 KB/s, 0 seconds passed
... 7%, 9504 KB, 17053 KB/s, 0 seconds passed
... 7%, 9536 KB, 17104 KB/s, 0 seconds passed
... 7%, 9568 KB, 17155 KB/s, 0 seconds passed
... 7%, 9600 KB, 17204 KB/s, 0 seconds passed
... 7%, 9632 KB, 17253 KB/s, 0 seconds passed
... 7%, 9664 KB, 17302 KB/s, 0 seconds passed
... 7%, 9696 KB, 17349 KB/s, 0 seconds passed
... 7%, 9728 KB, 17398 KB/s, 0 seconds passed
... 7%, 9760 KB, 17447 KB/s, 0 seconds passed
... 7%, 9792 KB, 17496 KB/s, 0 seconds passed
... 7%, 9824 KB, 17545 KB/s, 0 seconds passed
... 7%, 9856 KB, 17592 KB/s, 0 seconds passed
... 7%, 9888 KB, 17641 KB/s, 0 seconds passed
... 7%, 9920 KB, 17689 KB/s, 0 seconds passed
... 7%, 9952 KB, 17738 KB/s, 0 seconds passed
... 7%, 9984 KB, 17785 KB/s, 0 seconds passed
... 7%, 10016 KB, 17834 KB/s, 0 seconds passed
... 7%, 10048 KB, 17882 KB/s, 0 seconds passed
... 8%, 10080 KB, 17930 KB/s, 0 seconds passed
... 8%, 10112 KB, 17979 KB/s, 0 seconds passed
... 8%, 10144 KB, 18026 KB/s, 0 seconds passed
... 8%, 10176 KB, 18074 KB/s, 0 seconds passed
... 8%, 10208 KB, 18122 KB/s, 0 seconds passed
... 8%, 10240 KB, 18166 KB/s, 0 seconds passed
... 8%, 10272 KB, 18215 KB/s, 0 seconds passed
... 8%, 10304 KB, 18263 KB/s, 0 seconds passed
... 8%, 10336 KB, 18311 KB/s, 0 seconds passed
... 8%, 10368 KB, 18359 KB/s, 0 seconds passed
... 8%, 10400 KB, 18405 KB/s, 0 seconds passed
... 8%, 10432 KB, 18453 KB/s, 0 seconds passed
... 8%, 10464 KB, 18501 KB/s, 0 seconds passed
... 8%, 10496 KB, 18549 KB/s, 0 seconds passed
... 8%, 10528 KB, 18597 KB/s, 0 seconds passed
... 8%, 10560 KB, 18644 KB/s, 0 seconds passed
... 8%, 10592 KB, 18692 KB/s, 0 seconds passed
... 8%, 10624 KB, 18738 KB/s, 0 seconds passed
... 8%, 10656 KB, 18786 KB/s, 0 seconds passed
... 8%, 10688 KB, 18833 KB/s, 0 seconds passed
... 8%, 10720 KB, 18881 KB/s, 0 seconds passed
... 8%, 10752 KB, 18928 KB/s, 0 seconds passed
... 8%, 10784 KB, 18976 KB/s, 0 seconds passed
... 8%, 10816 KB, 19017 KB/s, 0 seconds passed
... 8%, 10848 KB, 19053 KB/s, 0 seconds passed
... 8%, 10880 KB, 19093 KB/s, 0 seconds passed
... 8%, 10912 KB, 19135 KB/s, 0 seconds passed
... 8%, 10944 KB, 19178 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 10976 KB, 19219 KB/s, 0 seconds passed
... 8%, 11008 KB, 19262 KB/s, 0 seconds passed
... 8%, 11040 KB, 19304 KB/s, 0 seconds passed
... 8%, 11072 KB, 19346 KB/s, 0 seconds passed
... 8%, 11104 KB, 19376 KB/s, 0 seconds passed
... 8%, 11136 KB, 19401 KB/s, 0 seconds passed
... 8%, 11168 KB, 19427 KB/s, 0 seconds passed
... 8%, 11200 KB, 19454 KB/s, 0 seconds passed
... 8%, 11232 KB, 19494 KB/s, 0 seconds passed
... 8%, 11264 KB, 19533 KB/s, 0 seconds passed
... 8%, 11296 KB, 19574 KB/s, 0 seconds passed
... 8%, 11328 KB, 19616 KB/s, 0 seconds passed
... 9%, 11360 KB, 19658 KB/s, 0 seconds passed
... 9%, 11392 KB, 19700 KB/s, 0 seconds passed
... 9%, 11424 KB, 19743 KB/s, 0 seconds passed
... 9%, 11456 KB, 19785 KB/s, 0 seconds passed
... 9%, 11488 KB, 19827 KB/s, 0 seconds passed
... 9%, 11520 KB, 19870 KB/s, 0 seconds passed
... 9%, 11552 KB, 19912 KB/s, 0 seconds passed
... 9%, 11584 KB, 19953 KB/s, 0 seconds passed
... 9%, 11616 KB, 19994 KB/s, 0 seconds passed
... 9%, 11648 KB, 20036 KB/s, 0 seconds passed
... 9%, 11680 KB, 20078 KB/s, 0 seconds passed
... 9%, 11712 KB, 20119 KB/s, 0 seconds passed
... 9%, 11744 KB, 20161 KB/s, 0 seconds passed
... 9%, 11776 KB, 20203 KB/s, 0 seconds passed
... 9%, 11808 KB, 20244 KB/s, 0 seconds passed
... 9%, 11840 KB, 20286 KB/s, 0 seconds passed
... 9%, 11872 KB, 20329 KB/s, 0 seconds passed
... 9%, 11904 KB, 20376 KB/s, 0 seconds passed
... 9%, 11936 KB, 20422 KB/s, 0 seconds passed
... 9%, 11968 KB, 20467 KB/s, 0 seconds passed
... 9%, 12000 KB, 20513 KB/s, 0 seconds passed
... 9%, 12032 KB, 20558 KB/s, 0 seconds passed
... 9%, 12064 KB, 20605 KB/s, 0 seconds passed
... 9%, 12096 KB, 20651 KB/s, 0 seconds passed
... 9%, 12128 KB, 20697 KB/s, 0 seconds passed
... 9%, 12160 KB, 20743 KB/s, 0 seconds passed
... 9%, 12192 KB, 20789 KB/s, 0 seconds passed
... 9%, 12224 KB, 20835 KB/s, 0 seconds passed
... 9%, 12256 KB, 20881 KB/s, 0 seconds passed
... 9%, 12288 KB, 20927 KB/s, 0 seconds passed
... 9%, 12320 KB, 20972 KB/s, 0 seconds passed
... 9%, 12352 KB, 21017 KB/s, 0 seconds passed
... 9%, 12384 KB, 21063 KB/s, 0 seconds passed
... 9%, 12416 KB, 21108 KB/s, 0 seconds passed
... 9%, 12448 KB, 21154 KB/s, 0 seconds passed
... 9%, 12480 KB, 21199 KB/s, 0 seconds passed
... 9%, 12512 KB, 21245 KB/s, 0 seconds passed
... 9%, 12544 KB, 21290 KB/s, 0 seconds passed
... 9%, 12576 KB, 21336 KB/s, 0 seconds passed
... 10%, 12608 KB, 21381 KB/s, 0 seconds passed
... 10%, 12640 KB, 21426 KB/s, 0 seconds passed
... 10%, 12672 KB, 21471 KB/s, 0 seconds passed
... 10%, 12704 KB, 21517 KB/s, 0 seconds passed
... 10%, 12736 KB, 21562 KB/s, 0 seconds passed
... 10%, 12768 KB, 21607 KB/s, 0 seconds passed
... 10%, 12800 KB, 21652 KB/s, 0 seconds passed
... 10%, 12832 KB, 21697 KB/s, 0 seconds passed
... 10%, 12864 KB, 21742 KB/s, 0 seconds passed
... 10%, 12896 KB, 21787 KB/s, 0 seconds passed
... 10%, 12928 KB, 21832 KB/s, 0 seconds passed
... 10%, 12960 KB, 21877 KB/s, 0 seconds passed
... 10%, 12992 KB, 21922 KB/s, 0 seconds passed
... 10%, 13024 KB, 21966 KB/s, 0 seconds passed
... 10%, 13056 KB, 22011 KB/s, 0 seconds passed
... 10%, 13088 KB, 22056 KB/s, 0 seconds passed
... 10%, 13120 KB, 22101 KB/s, 0 seconds passed
... 10%, 13152 KB, 22146 KB/s, 0 seconds passed
... 10%, 13184 KB, 22193 KB/s, 0 seconds passed
... 10%, 13216 KB, 22240 KB/s, 0 seconds passed
... 10%, 13248 KB, 22288 KB/s, 0 seconds passed
... 10%, 13280 KB, 22336 KB/s, 0 seconds passed
... 10%, 13312 KB, 22383 KB/s, 0 seconds passed
... 10%, 13344 KB, 22431 KB/s, 0 seconds passed
... 10%, 13376 KB, 22478 KB/s, 0 seconds passed
... 10%, 13408 KB, 22525 KB/s, 0 seconds passed
... 10%, 13440 KB, 22573 KB/s, 0 seconds passed
... 10%, 13472 KB, 22620 KB/s, 0 seconds passed
... 10%, 13504 KB, 22667 KB/s, 0 seconds passed
... 10%, 13536 KB, 22714 KB/s, 0 seconds passed
... 10%, 13568 KB, 22762 KB/s, 0 seconds passed
... 10%, 13600 KB, 22809 KB/s, 0 seconds passed
... 10%, 13632 KB, 22856 KB/s, 0 seconds passed
... 10%, 13664 KB, 22903 KB/s, 0 seconds passed
... 10%, 13696 KB, 22950 KB/s, 0 seconds passed
... 10%, 13728 KB, 22997 KB/s, 0 seconds passed
... 10%, 13760 KB, 23043 KB/s, 0 seconds passed
... 10%, 13792 KB, 23091 KB/s, 0 seconds passed
... 10%, 13824 KB, 23138 KB/s, 0 seconds passed
... 11%, 13856 KB, 23185 KB/s, 0 seconds passed
... 11%, 13888 KB, 23232 KB/s, 0 seconds passed
... 11%, 13920 KB, 23279 KB/s, 0 seconds passed
... 11%, 13952 KB, 23326 KB/s, 0 seconds passed
... 11%, 13984 KB, 23373 KB/s, 0 seconds passed
... 11%, 14016 KB, 23420 KB/s, 0 seconds passed
... 11%, 14048 KB, 23467 KB/s, 0 seconds passed
... 11%, 14080 KB, 23514 KB/s, 0 seconds passed
... 11%, 14112 KB, 23561 KB/s, 0 seconds passed
... 11%, 14144 KB, 23607 KB/s, 0 seconds passed
... 11%, 14176 KB, 23654 KB/s, 0 seconds passed
... 11%, 14208 KB, 23701 KB/s, 0 seconds passed
... 11%, 14240 KB, 23748 KB/s, 0 seconds passed
... 11%, 14272 KB, 23795 KB/s, 0 seconds passed
... 11%, 14304 KB, 23841 KB/s, 0 seconds passed
... 11%, 14336 KB, 23888 KB/s, 0 seconds passed
... 11%, 14368 KB, 23935 KB/s, 0 seconds passed
... 11%, 14400 KB, 23982 KB/s, 0 seconds passed
... 11%, 14432 KB, 24028 KB/s, 0 seconds passed
... 11%, 14464 KB, 24075 KB/s, 0 seconds passed
... 11%, 14496 KB, 24122 KB/s, 0 seconds passed
... 11%, 14528 KB, 24169 KB/s, 0 seconds passed
... 11%, 14560 KB, 24215 KB/s, 0 seconds passed
... 11%, 14592 KB, 24257 KB/s, 0 seconds passed
... 11%, 14624 KB, 24299 KB/s, 0 seconds passed
... 11%, 14656 KB, 24342 KB/s, 0 seconds passed
... 11%, 14688 KB, 24382 KB/s, 0 seconds passed
... 11%, 14720 KB, 24425 KB/s, 0 seconds passed
... 11%, 14752 KB, 24463 KB/s, 0 seconds passed
... 11%, 14784 KB, 24505 KB/s, 0 seconds passed
... 11%, 14816 KB, 24548 KB/s, 0 seconds passed
... 11%, 14848 KB, 24590 KB/s, 0 seconds passed
... 11%, 14880 KB, 24630 KB/s, 0 seconds passed
... 11%, 14912 KB, 24675 KB/s, 0 seconds passed
... 11%, 14944 KB, 24715 KB/s, 0 seconds passed
... 11%, 14976 KB, 24757 KB/s, 0 seconds passed
... 11%, 15008 KB, 24799 KB/s, 0 seconds passed
... 11%, 15040 KB, 24839 KB/s, 0 seconds passed
... 11%, 15072 KB, 24881 KB/s, 0 seconds passed
... 11%, 15104 KB, 24921 KB/s, 0 seconds passed
... 12%, 15136 KB, 24963 KB/s, 0 seconds passed
... 12%, 15168 KB, 25005 KB/s, 0 seconds passed
... 12%, 15200 KB, 25045 KB/s, 0 seconds passed
... 12%, 15232 KB, 25086 KB/s, 0 seconds passed
... 12%, 15264 KB, 25128 KB/s, 0 seconds passed
... 12%, 15296 KB, 25168 KB/s, 0 seconds passed
... 12%, 15328 KB, 25210 KB/s, 0 seconds passed
... 12%, 15360 KB, 25256 KB/s, 0 seconds passed
... 12%, 15392 KB, 25298 KB/s, 0 seconds passed
... 12%, 15424 KB, 25337 KB/s, 0 seconds passed
... 12%, 15456 KB, 25379 KB/s, 0 seconds passed
... 12%, 15488 KB, 25420 KB/s, 0 seconds passed
... 12%, 15520 KB, 25460 KB/s, 0 seconds passed
... 12%, 15552 KB, 25501 KB/s, 0 seconds passed
... 12%, 15584 KB, 25543 KB/s, 0 seconds passed
... 12%, 15616 KB, 25582 KB/s, 0 seconds passed
... 12%, 15648 KB, 25623 KB/s, 0 seconds passed
... 12%, 15680 KB, 25665 KB/s, 0 seconds passed
... 12%, 15712 KB, 25704 KB/s, 0 seconds passed
... 12%, 15744 KB, 25745 KB/s, 0 seconds passed
... 12%, 15776 KB, 25787 KB/s, 0 seconds passed
... 12%, 15808 KB, 25826 KB/s, 0 seconds passed
... 12%, 15840 KB, 25867 KB/s, 0 seconds passed
... 12%, 15872 KB, 25908 KB/s, 0 seconds passed
... 12%, 15904 KB, 25947 KB/s, 0 seconds passed
... 12%, 15936 KB, 25988 KB/s, 0 seconds passed
... 12%, 15968 KB, 26029 KB/s, 0 seconds passed
... 12%, 16000 KB, 26068 KB/s, 0 seconds passed
... 12%, 16032 KB, 26109 KB/s, 0 seconds passed
... 12%, 16064 KB, 26150 KB/s, 0 seconds passed
... 12%, 16096 KB, 26188 KB/s, 0 seconds passed
... 12%, 16128 KB, 26223 KB/s, 0 seconds passed
... 12%, 16160 KB, 26257 KB/s, 0 seconds passed
... 12%, 16192 KB, 26292 KB/s, 0 seconds passed
... 12%, 16224 KB, 26327 KB/s, 0 seconds passed
... 12%, 16256 KB, 26373 KB/s, 0 seconds passed
... 12%, 16288 KB, 26419 KB/s, 0 seconds passed
... 12%, 16320 KB, 26464 KB/s, 0 seconds passed
... 12%, 16352 KB, 26502 KB/s, 0 seconds passed
... 13%, 16384 KB, 26536 KB/s, 0 seconds passed
... 13%, 16416 KB, 26583 KB/s, 0 seconds passed
... 13%, 16448 KB, 26629 KB/s, 0 seconds passed
... 13%, 16480 KB, 26670 KB/s, 0 seconds passed
... 13%, 16512 KB, 26704 KB/s, 0 seconds passed
... 13%, 16544 KB, 26741 KB/s, 0 seconds passed
... 13%, 16576 KB, 26776 KB/s, 0 seconds passed
... 13%, 16608 KB, 26814 KB/s, 0 seconds passed
... 13%, 16640 KB, 26861 KB/s, 0 seconds passed
... 13%, 16672 KB, 26904 KB/s, 0 seconds passed
... 13%, 16704 KB, 26942 KB/s, 0 seconds passed
... 13%, 16736 KB, 26976 KB/s, 0 seconds passed
... 13%, 16768 KB, 27009 KB/s, 0 seconds passed
... 13%, 16800 KB, 27043 KB/s, 0 seconds passed
... 13%, 16832 KB, 27085 KB/s, 0 seconds passed
... 13%, 16864 KB, 27131 KB/s, 0 seconds passed
... 13%, 16896 KB, 27177 KB/s, 0 seconds passed
... 13%, 16928 KB, 27221 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 16960 KB, 27261 KB/s, 0 seconds passed
... 13%, 16992 KB, 27301 KB/s, 0 seconds passed
... 13%, 17024 KB, 27338 KB/s, 0 seconds passed
... 13%, 17056 KB, 27378 KB/s, 0 seconds passed
... 13%, 17088 KB, 27418 KB/s, 0 seconds passed
... 13%, 17120 KB, 27458 KB/s, 0 seconds passed
... 13%, 17152 KB, 27494 KB/s, 0 seconds passed
... 13%, 17184 KB, 27533 KB/s, 0 seconds passed
... 13%, 17216 KB, 27572 KB/s, 0 seconds passed
... 13%, 17248 KB, 27611 KB/s, 0 seconds passed
... 13%, 17280 KB, 27646 KB/s, 0 seconds passed
... 13%, 17312 KB, 27687 KB/s, 0 seconds passed
... 13%, 17344 KB, 27727 KB/s, 0 seconds passed
... 13%, 17376 KB, 27767 KB/s, 0 seconds passed
... 13%, 17408 KB, 27806 KB/s, 0 seconds passed
... 13%, 17440 KB, 27846 KB/s, 0 seconds passed
... 13%, 17472 KB, 27885 KB/s, 0 seconds passed
... 13%, 17504 KB, 27923 KB/s, 0 seconds passed
... 13%, 17536 KB, 27962 KB/s, 0 seconds passed
... 13%, 17568 KB, 28001 KB/s, 0 seconds passed
... 13%, 17600 KB, 28038 KB/s, 0 seconds passed
... 13%, 17632 KB, 28073 KB/s, 0 seconds passed
... 14%, 17664 KB, 28110 KB/s, 0 seconds passed
... 14%, 17696 KB, 28146 KB/s, 0 seconds passed
... 14%, 17728 KB, 28176 KB/s, 0 seconds passed
... 14%, 17760 KB, 28219 KB/s, 0 seconds passed
... 14%, 17792 KB, 28263 KB/s, 0 seconds passed
... 14%, 17824 KB, 28305 KB/s, 0 seconds passed
... 14%, 17856 KB, 28342 KB/s, 0 seconds passed
... 14%, 17888 KB, 28381 KB/s, 0 seconds passed
... 14%, 17920 KB, 28418 KB/s, 0 seconds passed
... 14%, 17952 KB, 28454 KB/s, 0 seconds passed
... 14%, 17984 KB, 28493 KB/s, 0 seconds passed
... 14%, 18016 KB, 28529 KB/s, 0 seconds passed
... 14%, 18048 KB, 28560 KB/s, 0 seconds passed
... 14%, 18080 KB, 28592 KB/s, 0 seconds passed
... 14%, 18112 KB, 28637 KB/s, 0 seconds passed
... 14%, 18144 KB, 28681 KB/s, 0 seconds passed
... 14%, 18176 KB, 28721 KB/s, 0 seconds passed
... 14%, 18208 KB, 28762 KB/s, 0 seconds passed
... 14%, 18240 KB, 28801 KB/s, 0 seconds passed
... 14%, 18272 KB, 28837 KB/s, 0 seconds passed
... 14%, 18304 KB, 28869 KB/s, 0 seconds passed
... 14%, 18336 KB, 28902 KB/s, 0 seconds passed
... 14%, 18368 KB, 28934 KB/s, 0 seconds passed
... 14%, 18400 KB, 28968 KB/s, 0 seconds passed
... 14%, 18432 KB, 29002 KB/s, 0 seconds passed
... 14%, 18464 KB, 29035 KB/s, 0 seconds passed
... 14%, 18496 KB, 29069 KB/s, 0 seconds passed
... 14%, 18528 KB, 29100 KB/s, 0 seconds passed
... 14%, 18560 KB, 29133 KB/s, 0 seconds passed
... 14%, 18592 KB, 29166 KB/s, 0 seconds passed
... 14%, 18624 KB, 29200 KB/s, 0 seconds passed
... 14%, 18656 KB, 29234 KB/s, 0 seconds passed
... 14%, 18688 KB, 29266 KB/s, 0 seconds passed
... 14%, 18720 KB, 29299 KB/s, 0 seconds passed
... 14%, 18752 KB, 29333 KB/s, 0 seconds passed
... 14%, 18784 KB, 29367 KB/s, 0 seconds passed
... 14%, 18816 KB, 29401 KB/s, 0 seconds passed
... 14%, 18848 KB, 29434 KB/s, 0 seconds passed
... 14%, 18880 KB, 29466 KB/s, 0 seconds passed
... 15%, 18912 KB, 29500 KB/s, 0 seconds passed
... 15%, 18944 KB, 29532 KB/s, 0 seconds passed
... 15%, 18976 KB, 29566 KB/s, 0 seconds passed
... 15%, 19008 KB, 29599 KB/s, 0 seconds passed
... 15%, 19040 KB, 29631 KB/s, 0 seconds passed
... 15%, 19072 KB, 29664 KB/s, 0 seconds passed
... 15%, 19104 KB, 29697 KB/s, 0 seconds passed
... 15%, 19136 KB, 29730 KB/s, 0 seconds passed
... 15%, 19168 KB, 29763 KB/s, 0 seconds passed
... 15%, 19200 KB, 29796 KB/s, 0 seconds passed
... 15%, 19232 KB, 29835 KB/s, 0 seconds passed
... 15%, 19264 KB, 29872 KB/s, 0 seconds passed
... 15%, 19296 KB, 29911 KB/s, 0 seconds passed
... 15%, 19328 KB, 29950 KB/s, 0 seconds passed
... 15%, 19360 KB, 29989 KB/s, 0 seconds passed
... 15%, 19392 KB, 30028 KB/s, 0 seconds passed
... 15%, 19424 KB, 30067 KB/s, 0 seconds passed
... 15%, 19456 KB, 30105 KB/s, 0 seconds passed
... 15%, 19488 KB, 30144 KB/s, 0 seconds passed
... 15%, 19520 KB, 30182 KB/s, 0 seconds passed
... 15%, 19552 KB, 30221 KB/s, 0 seconds passed
... 15%, 19584 KB, 30260 KB/s, 0 seconds passed
... 15%, 19616 KB, 30297 KB/s, 0 seconds passed
... 15%, 19648 KB, 30335 KB/s, 0 seconds passed
... 15%, 19680 KB, 30373 KB/s, 0 seconds passed
... 15%, 19712 KB, 30412 KB/s, 0 seconds passed
... 15%, 19744 KB, 30451 KB/s, 0 seconds passed
... 15%, 19776 KB, 30488 KB/s, 0 seconds passed
... 15%, 19808 KB, 30526 KB/s, 0 seconds passed
... 15%, 19840 KB, 30565 KB/s, 0 seconds passed
... 15%, 19872 KB, 30603 KB/s, 0 seconds passed
... 15%, 19904 KB, 30641 KB/s, 0 seconds passed
... 15%, 19936 KB, 30680 KB/s, 0 seconds passed
... 15%, 19968 KB, 30718 KB/s, 0 seconds passed
... 15%, 20000 KB, 30756 KB/s, 0 seconds passed
... 15%, 20032 KB, 30794 KB/s, 0 seconds passed
... 15%, 20064 KB, 30832 KB/s, 0 seconds passed
... 15%, 20096 KB, 30870 KB/s, 0 seconds passed
... 15%, 20128 KB, 30908 KB/s, 0 seconds passed
... 16%, 20160 KB, 30946 KB/s, 0 seconds passed
... 16%, 20192 KB, 30985 KB/s, 0 seconds passed
... 16%, 20224 KB, 31023 KB/s, 0 seconds passed
... 16%, 20256 KB, 31061 KB/s, 0 seconds passed
... 16%, 20288 KB, 31098 KB/s, 0 seconds passed
... 16%, 20320 KB, 31136 KB/s, 0 seconds passed
... 16%, 20352 KB, 31173 KB/s, 0 seconds passed
... 16%, 20384 KB, 31212 KB/s, 0 seconds passed
... 16%, 20416 KB, 31250 KB/s, 0 seconds passed
... 16%, 20448 KB, 31288 KB/s, 0 seconds passed
... 16%, 20480 KB, 31325 KB/s, 0 seconds passed
... 16%, 20512 KB, 31362 KB/s, 0 seconds passed
... 16%, 20544 KB, 31400 KB/s, 0 seconds passed
... 16%, 20576 KB, 31437 KB/s, 0 seconds passed
... 16%, 20608 KB, 31476 KB/s, 0 seconds passed
... 16%, 20640 KB, 31515 KB/s, 0 seconds passed
... 16%, 20672 KB, 31556 KB/s, 0 seconds passed
... 16%, 20704 KB, 31597 KB/s, 0 seconds passed
... 16%, 20736 KB, 31639 KB/s, 0 seconds passed
... 16%, 20768 KB, 31679 KB/s, 0 seconds passed
... 16%, 20800 KB, 31719 KB/s, 0 seconds passed
... 16%, 20832 KB, 31755 KB/s, 0 seconds passed
... 16%, 20864 KB, 31792 KB/s, 0 seconds passed
... 16%, 20896 KB, 31827 KB/s, 0 seconds passed
... 16%, 20928 KB, 31864 KB/s, 0 seconds passed
... 16%, 20960 KB, 31899 KB/s, 0 seconds passed
... 16%, 20992 KB, 31933 KB/s, 0 seconds passed
... 16%, 21024 KB, 31966 KB/s, 0 seconds passed
... 16%, 21056 KB, 32002 KB/s, 0 seconds passed
... 16%, 21088 KB, 32038 KB/s, 0 seconds passed
... 16%, 21120 KB, 32074 KB/s, 0 seconds passed
... 16%, 21152 KB, 32110 KB/s, 0 seconds passed
... 16%, 21184 KB, 32145 KB/s, 0 seconds passed
... 16%, 21216 KB, 32176 KB/s, 0 seconds passed
... 16%, 21248 KB, 32212 KB/s, 0 seconds passed
... 16%, 21280 KB, 32247 KB/s, 0 seconds passed
... 16%, 21312 KB, 32283 KB/s, 0 seconds passed
... 16%, 21344 KB, 32319 KB/s, 0 seconds passed
... 16%, 21376 KB, 32354 KB/s, 0 seconds passed
... 16%, 21408 KB, 32387 KB/s, 0 seconds passed
... 17%, 21440 KB, 32423 KB/s, 0 seconds passed
... 17%, 21472 KB, 32458 KB/s, 0 seconds passed
... 17%, 21504 KB, 32494 KB/s, 0 seconds passed
... 17%, 21536 KB, 32529 KB/s, 0 seconds passed
... 17%, 21568 KB, 32562 KB/s, 0 seconds passed
... 17%, 21600 KB, 32595 KB/s, 0 seconds passed
... 17%, 21632 KB, 32631 KB/s, 0 seconds passed
... 17%, 21664 KB, 32666 KB/s, 0 seconds passed
... 17%, 21696 KB, 32701 KB/s, 0 seconds passed
... 17%, 21728 KB, 32737 KB/s, 0 seconds passed
... 17%, 21760 KB, 32772 KB/s, 0 seconds passed
... 17%, 21792 KB, 32804 KB/s, 0 seconds passed
... 17%, 21824 KB, 32835 KB/s, 0 seconds passed
... 17%, 21856 KB, 32870 KB/s, 0 seconds passed
... 17%, 21888 KB, 32902 KB/s, 0 seconds passed
... 17%, 21920 KB, 32938 KB/s, 0 seconds passed
... 17%, 21952 KB, 32973 KB/s, 0 seconds passed
... 17%, 21984 KB, 33008 KB/s, 0 seconds passed
... 17%, 22016 KB, 33043 KB/s, 0 seconds passed
... 17%, 22048 KB, 33075 KB/s, 0 seconds passed
... 17%, 22080 KB, 33108 KB/s, 0 seconds passed
... 17%, 22112 KB, 33143 KB/s, 0 seconds passed
... 17%, 22144 KB, 33178 KB/s, 0 seconds passed
... 17%, 22176 KB, 33210 KB/s, 0 seconds passed
... 17%, 22208 KB, 33242 KB/s, 0 seconds passed
... 17%, 22240 KB, 33270 KB/s, 0 seconds passed
... 17%, 22272 KB, 33296 KB/s, 0 seconds passed
... 17%, 22304 KB, 33324 KB/s, 0 seconds passed
... 17%, 22336 KB, 33365 KB/s, 0 seconds passed
... 17%, 22368 KB, 33406 KB/s, 0 seconds passed
... 17%, 22400 KB, 33448 KB/s, 0 seconds passed
... 17%, 22432 KB, 33484 KB/s, 0 seconds passed
... 17%, 22464 KB, 33516 KB/s, 0 seconds passed
... 17%, 22496 KB, 33556 KB/s, 0 seconds passed
... 17%, 22528 KB, 33591 KB/s, 0 seconds passed
... 17%, 22560 KB, 33620 KB/s, 0 seconds passed
... 17%, 22592 KB, 33655 KB/s, 0 seconds passed
... 17%, 22624 KB, 33687 KB/s, 0 seconds passed
... 17%, 22656 KB, 33721 KB/s, 0 seconds passed
... 18%, 22688 KB, 33748 KB/s, 0 seconds passed
... 18%, 22720 KB, 33786 KB/s, 0 seconds passed
... 18%, 22752 KB, 33824 KB/s, 0 seconds passed
... 18%, 22784 KB, 33853 KB/s, 0 seconds passed
... 18%, 22816 KB, 33880 KB/s, 0 seconds passed

.. parsed-literal::

    ... 18%, 22848 KB, 33920 KB/s, 0 seconds passed
... 18%, 22880 KB, 33958 KB/s, 0 seconds passed
... 18%, 22912 KB, 33992 KB/s, 0 seconds passed
... 18%, 22944 KB, 34022 KB/s, 0 seconds passed
... 18%, 22976 KB, 34056 KB/s, 0 seconds passed
... 18%, 23008 KB, 34090 KB/s, 0 seconds passed
... 18%, 23040 KB, 34080 KB/s, 0 seconds passed
... 18%, 23072 KB, 34119 KB/s, 0 seconds passed
... 18%, 23104 KB, 34159 KB/s, 0 seconds passed
... 18%, 23136 KB, 34200 KB/s, 0 seconds passed
... 18%, 23168 KB, 34240 KB/s, 0 seconds passed
... 18%, 23200 KB, 34281 KB/s, 0 seconds passed
... 18%, 23232 KB, 34321 KB/s, 0 seconds passed
... 18%, 23264 KB, 34355 KB/s, 0 seconds passed
... 18%, 23296 KB, 34392 KB/s, 0 seconds passed
... 18%, 23328 KB, 34420 KB/s, 0 seconds passed
... 18%, 23360 KB, 34454 KB/s, 0 seconds passed
... 18%, 23392 KB, 34485 KB/s, 0 seconds passed
... 18%, 23424 KB, 34519 KB/s, 0 seconds passed
... 18%, 23456 KB, 34551 KB/s, 0 seconds passed
... 18%, 23488 KB, 34584 KB/s, 0 seconds passed
... 18%, 23520 KB, 34621 KB/s, 0 seconds passed
... 18%, 23552 KB, 34652 KB/s, 0 seconds passed
... 18%, 23584 KB, 34686 KB/s, 0 seconds passed
... 18%, 23616 KB, 34720 KB/s, 0 seconds passed
... 18%, 23648 KB, 34753 KB/s, 0 seconds passed
... 18%, 23680 KB, 34778 KB/s, 0 seconds passed
... 18%, 23712 KB, 34804 KB/s, 0 seconds passed
... 18%, 23744 KB, 34841 KB/s, 0 seconds passed
... 18%, 23776 KB, 34879 KB/s, 0 seconds passed
... 18%, 23808 KB, 34910 KB/s, 0 seconds passed
... 18%, 23840 KB, 34948 KB/s, 0 seconds passed
... 18%, 23872 KB, 34979 KB/s, 0 seconds passed
... 18%, 23904 KB, 35005 KB/s, 0 seconds passed
... 19%, 23936 KB, 35030 KB/s, 0 seconds passed
... 19%, 23968 KB, 35070 KB/s, 0 seconds passed
... 19%, 24000 KB, 35106 KB/s, 0 seconds passed
... 19%, 24032 KB, 35140 KB/s, 0 seconds passed
... 19%, 24064 KB, 35173 KB/s, 0 seconds passed
... 19%, 24096 KB, 35206 KB/s, 0 seconds passed
... 19%, 24128 KB, 35240 KB/s, 0 seconds passed
... 19%, 24160 KB, 35273 KB/s, 0 seconds passed
... 19%, 24192 KB, 35304 KB/s, 0 seconds passed
... 19%, 24224 KB, 35337 KB/s, 0 seconds passed
... 19%, 24256 KB, 35370 KB/s, 0 seconds passed
... 19%, 24288 KB, 35403 KB/s, 0 seconds passed
... 19%, 24320 KB, 35428 KB/s, 0 seconds passed
... 19%, 24352 KB, 35461 KB/s, 0 seconds passed
... 19%, 24384 KB, 35495 KB/s, 0 seconds passed
... 19%, 24416 KB, 35527 KB/s, 0 seconds passed
... 19%, 24448 KB, 35557 KB/s, 0 seconds passed
... 19%, 24480 KB, 35590 KB/s, 0 seconds passed
... 19%, 24512 KB, 35623 KB/s, 0 seconds passed
... 19%, 24544 KB, 35653 KB/s, 0 seconds passed
... 19%, 24576 KB, 35676 KB/s, 0 seconds passed
... 19%, 24608 KB, 35701 KB/s, 0 seconds passed
... 19%, 24640 KB, 35740 KB/s, 0 seconds passed
... 19%, 24672 KB, 35781 KB/s, 0 seconds passed
... 19%, 24704 KB, 35815 KB/s, 0 seconds passed
... 19%, 24736 KB, 35848 KB/s, 0 seconds passed
... 19%, 24768 KB, 35878 KB/s, 0 seconds passed
... 19%, 24800 KB, 35913 KB/s, 0 seconds passed
... 19%, 24832 KB, 35943 KB/s, 0 seconds passed
... 19%, 24864 KB, 35973 KB/s, 0 seconds passed
... 19%, 24896 KB, 36006 KB/s, 0 seconds passed
... 19%, 24928 KB, 36036 KB/s, 0 seconds passed
... 19%, 24960 KB, 36069 KB/s, 0 seconds passed
... 19%, 24992 KB, 36102 KB/s, 0 seconds passed
... 19%, 25024 KB, 36131 KB/s, 0 seconds passed
... 19%, 25056 KB, 36167 KB/s, 0 seconds passed
... 19%, 25088 KB, 36197 KB/s, 0 seconds passed
... 19%, 25120 KB, 36230 KB/s, 0 seconds passed
... 19%, 25152 KB, 36262 KB/s, 0 seconds passed
... 19%, 25184 KB, 36292 KB/s, 0 seconds passed
... 20%, 25216 KB, 36324 KB/s, 0 seconds passed
... 20%, 25248 KB, 36354 KB/s, 0 seconds passed
... 20%, 25280 KB, 36384 KB/s, 0 seconds passed
... 20%, 25312 KB, 36416 KB/s, 0 seconds passed
... 20%, 25344 KB, 36446 KB/s, 0 seconds passed
... 20%, 25376 KB, 36481 KB/s, 0 seconds passed
... 20%, 25408 KB, 36510 KB/s, 0 seconds passed
... 20%, 25440 KB, 36543 KB/s, 0 seconds passed
... 20%, 25472 KB, 36575 KB/s, 0 seconds passed
... 20%, 25504 KB, 36607 KB/s, 0 seconds passed
... 20%, 25536 KB, 36637 KB/s, 0 seconds passed
... 20%, 25568 KB, 36668 KB/s, 0 seconds passed
... 20%, 25600 KB, 36703 KB/s, 0 seconds passed
... 20%, 25632 KB, 36733 KB/s, 0 seconds passed
... 20%, 25664 KB, 36765 KB/s, 0 seconds passed
... 20%, 25696 KB, 36797 KB/s, 0 seconds passed
... 20%, 25728 KB, 36821 KB/s, 0 seconds passed
... 20%, 25760 KB, 36853 KB/s, 0 seconds passed
... 20%, 25792 KB, 36879 KB/s, 0 seconds passed
... 20%, 25824 KB, 36903 KB/s, 0 seconds passed
... 20%, 25856 KB, 36926 KB/s, 0 seconds passed
... 20%, 25888 KB, 36956 KB/s, 0 seconds passed
... 20%, 25920 KB, 36994 KB/s, 0 seconds passed
... 20%, 25952 KB, 37033 KB/s, 0 seconds passed
... 20%, 25984 KB, 37069 KB/s, 0 seconds passed
... 20%, 26016 KB, 37102 KB/s, 0 seconds passed
... 20%, 26048 KB, 37133 KB/s, 0 seconds passed
... 20%, 26080 KB, 37162 KB/s, 0 seconds passed
... 20%, 26112 KB, 37194 KB/s, 0 seconds passed
... 20%, 26144 KB, 37223 KB/s, 0 seconds passed
... 20%, 26176 KB, 37255 KB/s, 0 seconds passed
... 20%, 26208 KB, 37287 KB/s, 0 seconds passed
... 20%, 26240 KB, 37318 KB/s, 0 seconds passed
... 20%, 26272 KB, 37347 KB/s, 0 seconds passed
... 20%, 26304 KB, 37379 KB/s, 0 seconds passed
... 20%, 26336 KB, 37411 KB/s, 0 seconds passed
... 20%, 26368 KB, 37439 KB/s, 0 seconds passed
... 20%, 26400 KB, 37468 KB/s, 0 seconds passed
... 20%, 26432 KB, 37500 KB/s, 0 seconds passed
... 21%, 26464 KB, 37528 KB/s, 0 seconds passed
... 21%, 26496 KB, 37560 KB/s, 0 seconds passed
... 21%, 26528 KB, 37589 KB/s, 0 seconds passed
... 21%, 26560 KB, 37620 KB/s, 0 seconds passed
... 21%, 26592 KB, 37652 KB/s, 0 seconds passed
... 21%, 26624 KB, 37683 KB/s, 0 seconds passed
... 21%, 26656 KB, 37712 KB/s, 0 seconds passed
... 21%, 26688 KB, 37742 KB/s, 0 seconds passed
... 21%, 26720 KB, 37774 KB/s, 0 seconds passed
... 21%, 26752 KB, 37802 KB/s, 0 seconds passed
... 21%, 26784 KB, 37833 KB/s, 0 seconds passed
... 21%, 26816 KB, 37862 KB/s, 0 seconds passed
... 21%, 26848 KB, 37893 KB/s, 0 seconds passed
... 21%, 26880 KB, 37925 KB/s, 0 seconds passed
... 21%, 26912 KB, 37956 KB/s, 0 seconds passed
... 21%, 26944 KB, 37984 KB/s, 0 seconds passed
... 21%, 26976 KB, 38015 KB/s, 0 seconds passed
... 21%, 27008 KB, 38046 KB/s, 0 seconds passed
... 21%, 27040 KB, 38075 KB/s, 0 seconds passed
... 21%, 27072 KB, 38106 KB/s, 0 seconds passed
... 21%, 27104 KB, 38134 KB/s, 0 seconds passed
... 21%, 27136 KB, 38165 KB/s, 0 seconds passed
... 21%, 27168 KB, 38196 KB/s, 0 seconds passed
... 21%, 27200 KB, 38225 KB/s, 0 seconds passed
... 21%, 27232 KB, 38256 KB/s, 0 seconds passed
... 21%, 27264 KB, 38284 KB/s, 0 seconds passed
... 21%, 27296 KB, 38312 KB/s, 0 seconds passed
... 21%, 27328 KB, 38343 KB/s, 0 seconds passed
... 21%, 27360 KB, 38374 KB/s, 0 seconds passed
... 21%, 27392 KB, 38402 KB/s, 0 seconds passed
... 21%, 27424 KB, 38433 KB/s, 0 seconds passed
... 21%, 27456 KB, 38463 KB/s, 0 seconds passed
... 21%, 27488 KB, 38494 KB/s, 0 seconds passed
... 21%, 27520 KB, 38522 KB/s, 0 seconds passed
... 21%, 27552 KB, 38553 KB/s, 0 seconds passed
... 21%, 27584 KB, 38584 KB/s, 0 seconds passed
... 21%, 27616 KB, 38615 KB/s, 0 seconds passed
... 21%, 27648 KB, 38642 KB/s, 0 seconds passed
... 21%, 27680 KB, 38673 KB/s, 0 seconds passed
... 22%, 27712 KB, 38703 KB/s, 0 seconds passed
... 22%, 27744 KB, 38722 KB/s, 0 seconds passed
... 22%, 27776 KB, 38743 KB/s, 0 seconds passed
... 22%, 27808 KB, 38767 KB/s, 0 seconds passed
... 22%, 27840 KB, 38802 KB/s, 0 seconds passed
... 22%, 27872 KB, 38839 KB/s, 0 seconds passed
... 22%, 27904 KB, 38876 KB/s, 0 seconds passed
... 22%, 27936 KB, 38907 KB/s, 0 seconds passed
... 22%, 27968 KB, 38938 KB/s, 0 seconds passed
... 22%, 28000 KB, 38968 KB/s, 0 seconds passed
... 22%, 28032 KB, 38996 KB/s, 0 seconds passed
... 22%, 28064 KB, 39026 KB/s, 0 seconds passed
... 22%, 28096 KB, 39056 KB/s, 0 seconds passed
... 22%, 28128 KB, 39084 KB/s, 0 seconds passed
... 22%, 28160 KB, 39114 KB/s, 0 seconds passed
... 22%, 28192 KB, 39141 KB/s, 0 seconds passed
... 22%, 28224 KB, 39171 KB/s, 0 seconds passed
... 22%, 28256 KB, 39202 KB/s, 0 seconds passed
... 22%, 28288 KB, 39232 KB/s, 0 seconds passed
... 22%, 28320 KB, 39262 KB/s, 0 seconds passed
... 22%, 28352 KB, 39290 KB/s, 0 seconds passed
... 22%, 28384 KB, 39320 KB/s, 0 seconds passed
... 22%, 28416 KB, 39350 KB/s, 0 seconds passed
... 22%, 28448 KB, 39377 KB/s, 0 seconds passed
... 22%, 28480 KB, 39402 KB/s, 0 seconds passed
... 22%, 28512 KB, 39434 KB/s, 0 seconds passed
... 22%, 28544 KB, 39464 KB/s, 0 seconds passed
... 22%, 28576 KB, 39494 KB/s, 0 seconds passed
... 22%, 28608 KB, 39524 KB/s, 0 seconds passed
... 22%, 28640 KB, 39551 KB/s, 0 seconds passed
... 22%, 28672 KB, 39581 KB/s, 0 seconds passed
... 22%, 28704 KB, 39611 KB/s, 0 seconds passed

.. parsed-literal::

    ... 22%, 28736 KB, 39638 KB/s, 0 seconds passed
... 22%, 28768 KB, 39668 KB/s, 0 seconds passed
... 22%, 28800 KB, 39695 KB/s, 0 seconds passed
... 22%, 28832 KB, 39725 KB/s, 0 seconds passed
... 22%, 28864 KB, 39754 KB/s, 0 seconds passed
... 22%, 28896 KB, 39783 KB/s, 0 seconds passed
... 22%, 28928 KB, 39813 KB/s, 0 seconds passed
... 22%, 28960 KB, 39837 KB/s, 0 seconds passed
... 23%, 28992 KB, 39867 KB/s, 0 seconds passed
... 23%, 29024 KB, 39897 KB/s, 0 seconds passed
... 23%, 29056 KB, 39926 KB/s, 0 seconds passed
... 23%, 29088 KB, 39953 KB/s, 0 seconds passed
... 23%, 29120 KB, 39981 KB/s, 0 seconds passed
... 23%, 29152 KB, 40010 KB/s, 0 seconds passed
... 23%, 29184 KB, 40040 KB/s, 0 seconds passed
... 23%, 29216 KB, 40070 KB/s, 0 seconds passed
... 23%, 29248 KB, 40096 KB/s, 0 seconds passed
... 23%, 29280 KB, 40126 KB/s, 0 seconds passed
... 23%, 29312 KB, 40155 KB/s, 0 seconds passed
... 23%, 29344 KB, 40185 KB/s, 0 seconds passed
... 23%, 29376 KB, 40205 KB/s, 0 seconds passed
... 23%, 29408 KB, 40235 KB/s, 0 seconds passed
... 23%, 29440 KB, 40264 KB/s, 0 seconds passed
... 23%, 29472 KB, 40291 KB/s, 0 seconds passed
... 23%, 29504 KB, 40320 KB/s, 0 seconds passed
... 23%, 29536 KB, 40346 KB/s, 0 seconds passed
... 23%, 29568 KB, 40376 KB/s, 0 seconds passed
... 23%, 29600 KB, 40402 KB/s, 0 seconds passed
... 23%, 29632 KB, 40432 KB/s, 0 seconds passed
... 23%, 29664 KB, 40461 KB/s, 0 seconds passed
... 23%, 29696 KB, 40490 KB/s, 0 seconds passed
... 23%, 29728 KB, 40516 KB/s, 0 seconds passed
... 23%, 29760 KB, 40545 KB/s, 0 seconds passed
... 23%, 29792 KB, 40572 KB/s, 0 seconds passed
... 23%, 29824 KB, 40601 KB/s, 0 seconds passed
... 23%, 29856 KB, 40630 KB/s, 0 seconds passed
... 23%, 29888 KB, 40659 KB/s, 0 seconds passed
... 23%, 29920 KB, 40685 KB/s, 0 seconds passed
... 23%, 29952 KB, 40714 KB/s, 0 seconds passed
... 23%, 29984 KB, 40741 KB/s, 0 seconds passed
... 23%, 30016 KB, 40770 KB/s, 0 seconds passed
... 23%, 30048 KB, 40799 KB/s, 0 seconds passed
... 23%, 30080 KB, 40828 KB/s, 0 seconds passed
... 23%, 30112 KB, 40857 KB/s, 0 seconds passed
... 23%, 30144 KB, 40883 KB/s, 0 seconds passed
... 23%, 30176 KB, 40911 KB/s, 0 seconds passed
... 23%, 30208 KB, 40937 KB/s, 0 seconds passed
... 24%, 30240 KB, 40966 KB/s, 0 seconds passed
... 24%, 30272 KB, 40995 KB/s, 0 seconds passed
... 24%, 30304 KB, 41021 KB/s, 0 seconds passed
... 24%, 30336 KB, 41050 KB/s, 0 seconds passed
... 24%, 30368 KB, 41078 KB/s, 0 seconds passed
... 24%, 30400 KB, 41104 KB/s, 0 seconds passed
... 24%, 30432 KB, 41133 KB/s, 0 seconds passed
... 24%, 30464 KB, 41159 KB/s, 0 seconds passed
... 24%, 30496 KB, 41188 KB/s, 0 seconds passed
... 24%, 30528 KB, 41217 KB/s, 0 seconds passed
... 24%, 30560 KB, 41245 KB/s, 0 seconds passed
... 24%, 30592 KB, 41271 KB/s, 0 seconds passed
... 24%, 30624 KB, 41300 KB/s, 0 seconds passed
... 24%, 30656 KB, 41328 KB/s, 0 seconds passed
... 24%, 30688 KB, 41357 KB/s, 0 seconds passed
... 24%, 30720 KB, 40991 KB/s, 0 seconds passed
... 24%, 30752 KB, 41015 KB/s, 0 seconds passed
... 24%, 30784 KB, 41004 KB/s, 0 seconds passed
... 24%, 30816 KB, 41026 KB/s, 0 seconds passed
... 24%, 30848 KB, 41049 KB/s, 0 seconds passed
... 24%, 30880 KB, 41072 KB/s, 0 seconds passed
... 24%, 30912 KB, 41095 KB/s, 0 seconds passed
... 24%, 30944 KB, 41117 KB/s, 0 seconds passed
... 24%, 30976 KB, 41139 KB/s, 0 seconds passed
... 24%, 31008 KB, 41160 KB/s, 0 seconds passed
... 24%, 31040 KB, 41183 KB/s, 0 seconds passed
... 24%, 31072 KB, 41205 KB/s, 0 seconds passed
... 24%, 31104 KB, 41227 KB/s, 0 seconds passed
... 24%, 31136 KB, 41249 KB/s, 0 seconds passed
... 24%, 31168 KB, 41272 KB/s, 0 seconds passed
... 24%, 31200 KB, 41294 KB/s, 0 seconds passed
... 24%, 31232 KB, 41317 KB/s, 0 seconds passed
... 24%, 31264 KB, 41339 KB/s, 0 seconds passed
... 24%, 31296 KB, 41360 KB/s, 0 seconds passed
... 24%, 31328 KB, 41382 KB/s, 0 seconds passed
... 24%, 31360 KB, 41404 KB/s, 0 seconds passed
... 24%, 31392 KB, 41426 KB/s, 0 seconds passed
... 24%, 31424 KB, 41448 KB/s, 0 seconds passed
... 24%, 31456 KB, 41470 KB/s, 0 seconds passed
... 24%, 31488 KB, 41493 KB/s, 0 seconds passed
... 25%, 31520 KB, 41514 KB/s, 0 seconds passed
... 25%, 31552 KB, 41536 KB/s, 0 seconds passed
... 25%, 31584 KB, 41559 KB/s, 0 seconds passed
... 25%, 31616 KB, 41581 KB/s, 0 seconds passed
... 25%, 31648 KB, 41604 KB/s, 0 seconds passed
... 25%, 31680 KB, 41630 KB/s, 0 seconds passed
... 25%, 31712 KB, 41653 KB/s, 0 seconds passed
... 25%, 31744 KB, 41676 KB/s, 0 seconds passed
... 25%, 31776 KB, 41699 KB/s, 0 seconds passed
... 25%, 31808 KB, 41721 KB/s, 0 seconds passed
... 25%, 31840 KB, 41743 KB/s, 0 seconds passed
... 25%, 31872 KB, 41765 KB/s, 0 seconds passed
... 25%, 31904 KB, 41786 KB/s, 0 seconds passed
... 25%, 31936 KB, 41809 KB/s, 0 seconds passed
... 25%, 31968 KB, 41830 KB/s, 0 seconds passed
... 25%, 32000 KB, 41851 KB/s, 0 seconds passed
... 25%, 32032 KB, 41872 KB/s, 0 seconds passed
... 25%, 32064 KB, 41894 KB/s, 0 seconds passed
... 25%, 32096 KB, 41916 KB/s, 0 seconds passed
... 25%, 32128 KB, 41938 KB/s, 0 seconds passed
... 25%, 32160 KB, 41962 KB/s, 0 seconds passed
... 25%, 32192 KB, 41992 KB/s, 0 seconds passed
... 25%, 32224 KB, 42022 KB/s, 0 seconds passed
... 25%, 32256 KB, 42051 KB/s, 0 seconds passed
... 25%, 32288 KB, 42080 KB/s, 0 seconds passed
... 25%, 32320 KB, 42110 KB/s, 0 seconds passed
... 25%, 32352 KB, 42141 KB/s, 0 seconds passed
... 25%, 32384 KB, 42173 KB/s, 0 seconds passed
... 25%, 32416 KB, 42203 KB/s, 0 seconds passed
... 25%, 32448 KB, 42233 KB/s, 0 seconds passed
... 25%, 32480 KB, 42265 KB/s, 0 seconds passed
... 25%, 32512 KB, 42294 KB/s, 0 seconds passed
... 25%, 32544 KB, 42326 KB/s, 0 seconds passed

.. parsed-literal::

    ... 25%, 32576 KB, 41265 KB/s, 0 seconds passed
... 25%, 32608 KB, 41278 KB/s, 0 seconds passed
... 25%, 32640 KB, 41290 KB/s, 0 seconds passed
... 25%, 32672 KB, 41293 KB/s, 0 seconds passed
... 25%, 32704 KB, 41297 KB/s, 0 seconds passed
... 25%, 32736 KB, 41315 KB/s, 0 seconds passed
... 26%, 32768 KB, 40984 KB/s, 0 seconds passed
... 26%, 32800 KB, 41003 KB/s, 0 seconds passed
... 26%, 32832 KB, 41020 KB/s, 0 seconds passed
... 26%, 32864 KB, 41038 KB/s, 0 seconds passed
... 26%, 32896 KB, 41059 KB/s, 0 seconds passed
... 26%, 32928 KB, 41080 KB/s, 0 seconds passed
... 26%, 32960 KB, 41102 KB/s, 0 seconds passed
... 26%, 32992 KB, 41124 KB/s, 0 seconds passed
... 26%, 33024 KB, 41144 KB/s, 0 seconds passed
... 26%, 33056 KB, 41167 KB/s, 0 seconds passed
... 26%, 33088 KB, 41185 KB/s, 0 seconds passed
... 26%, 33120 KB, 41206 KB/s, 0 seconds passed
... 26%, 33152 KB, 41228 KB/s, 0 seconds passed
... 26%, 33184 KB, 41249 KB/s, 0 seconds passed
... 26%, 33216 KB, 41268 KB/s, 0 seconds passed
... 26%, 33248 KB, 41290 KB/s, 0 seconds passed

.. parsed-literal::

    ... 26%, 33280 KB, 40133 KB/s, 0 seconds passed
... 26%, 33312 KB, 40139 KB/s, 0 seconds passed
... 26%, 33344 KB, 40147 KB/s, 0 seconds passed
... 26%, 33376 KB, 40157 KB/s, 0 seconds passed
... 26%, 33408 KB, 40167 KB/s, 0 seconds passed
... 26%, 33440 KB, 40176 KB/s, 0 seconds passed
... 26%, 33472 KB, 40184 KB/s, 0 seconds passed
... 26%, 33504 KB, 40196 KB/s, 0 seconds passed
... 26%, 33536 KB, 40214 KB/s, 0 seconds passed
... 26%, 33568 KB, 40234 KB/s, 0 seconds passed
... 26%, 33600 KB, 40253 KB/s, 0 seconds passed
... 26%, 33632 KB, 40272 KB/s, 0 seconds passed
... 26%, 33664 KB, 40292 KB/s, 0 seconds passed
... 26%, 33696 KB, 40312 KB/s, 0 seconds passed
... 26%, 33728 KB, 40332 KB/s, 0 seconds passed
... 26%, 33760 KB, 40350 KB/s, 0 seconds passed
... 26%, 33792 KB, 40370 KB/s, 0 seconds passed
... 26%, 33824 KB, 40390 KB/s, 0 seconds passed
... 26%, 33856 KB, 40409 KB/s, 0 seconds passed
... 26%, 33888 KB, 40429 KB/s, 0 seconds passed
... 26%, 33920 KB, 40449 KB/s, 0 seconds passed
... 26%, 33952 KB, 40469 KB/s, 0 seconds passed
... 26%, 33984 KB, 40488 KB/s, 0 seconds passed
... 27%, 34016 KB, 40508 KB/s, 0 seconds passed
... 27%, 34048 KB, 40528 KB/s, 0 seconds passed
... 27%, 34080 KB, 40548 KB/s, 0 seconds passed
... 27%, 34112 KB, 40567 KB/s, 0 seconds passed
... 27%, 34144 KB, 40586 KB/s, 0 seconds passed
... 27%, 34176 KB, 40606 KB/s, 0 seconds passed
... 27%, 34208 KB, 40625 KB/s, 0 seconds passed
... 27%, 34240 KB, 40645 KB/s, 0 seconds passed
... 27%, 34272 KB, 40665 KB/s, 0 seconds passed
... 27%, 34304 KB, 40684 KB/s, 0 seconds passed
... 27%, 34336 KB, 40704 KB/s, 0 seconds passed
... 27%, 34368 KB, 40724 KB/s, 0 seconds passed
... 27%, 34400 KB, 40743 KB/s, 0 seconds passed
... 27%, 34432 KB, 40762 KB/s, 0 seconds passed
... 27%, 34464 KB, 40781 KB/s, 0 seconds passed
... 27%, 34496 KB, 40801 KB/s, 0 seconds passed
... 27%, 34528 KB, 40820 KB/s, 0 seconds passed
... 27%, 34560 KB, 40840 KB/s, 0 seconds passed
... 27%, 34592 KB, 40860 KB/s, 0 seconds passed
... 27%, 34624 KB, 40885 KB/s, 0 seconds passed
... 27%, 34656 KB, 40911 KB/s, 0 seconds passed
... 27%, 34688 KB, 40938 KB/s, 0 seconds passed
... 27%, 34720 KB, 40964 KB/s, 0 seconds passed
... 27%, 34752 KB, 40990 KB/s, 0 seconds passed
... 27%, 34784 KB, 41016 KB/s, 0 seconds passed
... 27%, 34816 KB, 41042 KB/s, 0 seconds passed
... 27%, 34848 KB, 41068 KB/s, 0 seconds passed
... 27%, 34880 KB, 41095 KB/s, 0 seconds passed
... 27%, 34912 KB, 41123 KB/s, 0 seconds passed
... 27%, 34944 KB, 41151 KB/s, 0 seconds passed
... 27%, 34976 KB, 41180 KB/s, 0 seconds passed
... 27%, 35008 KB, 41208 KB/s, 0 seconds passed
... 27%, 35040 KB, 41237 KB/s, 0 seconds passed
... 27%, 35072 KB, 41265 KB/s, 0 seconds passed
... 27%, 35104 KB, 41294 KB/s, 0 seconds passed
... 27%, 35136 KB, 41322 KB/s, 0 seconds passed
... 27%, 35168 KB, 41350 KB/s, 0 seconds passed
... 27%, 35200 KB, 41378 KB/s, 0 seconds passed
... 27%, 35232 KB, 41407 KB/s, 0 seconds passed
... 27%, 35264 KB, 41435 KB/s, 0 seconds passed
... 28%, 35296 KB, 41463 KB/s, 0 seconds passed
... 28%, 35328 KB, 41491 KB/s, 0 seconds passed
... 28%, 35360 KB, 41519 KB/s, 0 seconds passed
... 28%, 35392 KB, 41547 KB/s, 0 seconds passed
... 28%, 35424 KB, 41576 KB/s, 0 seconds passed
... 28%, 35456 KB, 41051 KB/s, 0 seconds passed
... 28%, 35488 KB, 41072 KB/s, 0 seconds passed
... 28%, 35520 KB, 41094 KB/s, 0 seconds passed
... 28%, 35552 KB, 41115 KB/s, 0 seconds passed
... 28%, 35584 KB, 41137 KB/s, 0 seconds passed
... 28%, 35616 KB, 41159 KB/s, 0 seconds passed
... 28%, 35648 KB, 41131 KB/s, 0 seconds passed
... 28%, 35680 KB, 41148 KB/s, 0 seconds passed
... 28%, 35712 KB, 41165 KB/s, 0 seconds passed
... 28%, 35744 KB, 41185 KB/s, 0 seconds passed
... 28%, 35776 KB, 41206 KB/s, 0 seconds passed
... 28%, 35808 KB, 41010 KB/s, 0 seconds passed
... 28%, 35840 KB, 41025 KB/s, 0 seconds passed
... 28%, 35872 KB, 41044 KB/s, 0 seconds passed
... 28%, 35904 KB, 41063 KB/s, 0 seconds passed
... 28%, 35936 KB, 41082 KB/s, 0 seconds passed
... 28%, 35968 KB, 41102 KB/s, 0 seconds passed
... 28%, 36000 KB, 41120 KB/s, 0 seconds passed
... 28%, 36032 KB, 41140 KB/s, 0 seconds passed
... 28%, 36064 KB, 41158 KB/s, 0 seconds passed
... 28%, 36096 KB, 41177 KB/s, 0 seconds passed
... 28%, 36128 KB, 41195 KB/s, 0 seconds passed
... 28%, 36160 KB, 41214 KB/s, 0 seconds passed
... 28%, 36192 KB, 41232 KB/s, 0 seconds passed
... 28%, 36224 KB, 41251 KB/s, 0 seconds passed

.. parsed-literal::

    ... 28%, 36256 KB, 41269 KB/s, 0 seconds passed
... 28%, 36288 KB, 41288 KB/s, 0 seconds passed
... 28%, 36320 KB, 41306 KB/s, 0 seconds passed
... 28%, 36352 KB, 41324 KB/s, 0 seconds passed
... 28%, 36384 KB, 41343 KB/s, 0 seconds passed
... 28%, 36416 KB, 41360 KB/s, 0 seconds passed
... 28%, 36448 KB, 41372 KB/s, 0 seconds passed
... 28%, 36480 KB, 41390 KB/s, 0 seconds passed
... 28%, 36512 KB, 41408 KB/s, 0 seconds passed
... 29%, 36544 KB, 41426 KB/s, 0 seconds passed
... 29%, 36576 KB, 41445 KB/s, 0 seconds passed
... 29%, 36608 KB, 41464 KB/s, 0 seconds passed
... 29%, 36640 KB, 41482 KB/s, 0 seconds passed
... 29%, 36672 KB, 41500 KB/s, 0 seconds passed
... 29%, 36704 KB, 41519 KB/s, 0 seconds passed
... 29%, 36736 KB, 41537 KB/s, 0 seconds passed
... 29%, 36768 KB, 41554 KB/s, 0 seconds passed
... 29%, 36800 KB, 41572 KB/s, 0 seconds passed
... 29%, 36832 KB, 41590 KB/s, 0 seconds passed
... 29%, 36864 KB, 41609 KB/s, 0 seconds passed
... 29%, 36896 KB, 41627 KB/s, 0 seconds passed
... 29%, 36928 KB, 41645 KB/s, 0 seconds passed
... 29%, 36960 KB, 41664 KB/s, 0 seconds passed
... 29%, 36992 KB, 41686 KB/s, 0 seconds passed
... 29%, 37024 KB, 41713 KB/s, 0 seconds passed
... 29%, 37056 KB, 41740 KB/s, 0 seconds passed
... 29%, 37088 KB, 41767 KB/s, 0 seconds passed
... 29%, 37120 KB, 41794 KB/s, 0 seconds passed
... 29%, 37152 KB, 41820 KB/s, 0 seconds passed
... 29%, 37184 KB, 41847 KB/s, 0 seconds passed
... 29%, 37216 KB, 41874 KB/s, 0 seconds passed
... 29%, 37248 KB, 41900 KB/s, 0 seconds passed
... 29%, 37280 KB, 41926 KB/s, 0 seconds passed
... 29%, 37312 KB, 41951 KB/s, 0 seconds passed
... 29%, 37344 KB, 41976 KB/s, 0 seconds passed
... 29%, 37376 KB, 42000 KB/s, 0 seconds passed
... 29%, 37408 KB, 42025 KB/s, 0 seconds passed
... 29%, 37440 KB, 42049 KB/s, 0 seconds passed
... 29%, 37472 KB, 42074 KB/s, 0 seconds passed
... 29%, 37504 KB, 42098 KB/s, 0 seconds passed
... 29%, 37536 KB, 42123 KB/s, 0 seconds passed
... 29%, 37568 KB, 42147 KB/s, 0 seconds passed
... 29%, 37600 KB, 42172 KB/s, 0 seconds passed
... 29%, 37632 KB, 42196 KB/s, 0 seconds passed
... 29%, 37664 KB, 42221 KB/s, 0 seconds passed
... 29%, 37696 KB, 42245 KB/s, 0 seconds passed
... 29%, 37728 KB, 42269 KB/s, 0 seconds passed
... 29%, 37760 KB, 42293 KB/s, 0 seconds passed
... 30%, 37792 KB, 42317 KB/s, 0 seconds passed
... 30%, 37824 KB, 42342 KB/s, 0 seconds passed
... 30%, 37856 KB, 42366 KB/s, 0 seconds passed
... 30%, 37888 KB, 42391 KB/s, 0 seconds passed
... 30%, 37920 KB, 42415 KB/s, 0 seconds passed
... 30%, 37952 KB, 42439 KB/s, 0 seconds passed
... 30%, 37984 KB, 42464 KB/s, 0 seconds passed
... 30%, 38016 KB, 42488 KB/s, 0 seconds passed
... 30%, 38048 KB, 42512 KB/s, 0 seconds passed
... 30%, 38080 KB, 42536 KB/s, 0 seconds passed
... 30%, 38112 KB, 42561 KB/s, 0 seconds passed
... 30%, 38144 KB, 42581 KB/s, 0 seconds passed
... 30%, 38176 KB, 42604 KB/s, 0 seconds passed
... 30%, 38208 KB, 42627 KB/s, 0 seconds passed
... 30%, 38240 KB, 42647 KB/s, 0 seconds passed
... 30%, 38272 KB, 42671 KB/s, 0 seconds passed
... 30%, 38304 KB, 42695 KB/s, 0 seconds passed
... 30%, 38336 KB, 42719 KB/s, 0 seconds passed
... 30%, 38368 KB, 42744 KB/s, 0 seconds passed
... 30%, 38400 KB, 42766 KB/s, 0 seconds passed
... 30%, 38432 KB, 42789 KB/s, 0 seconds passed
... 30%, 38464 KB, 42812 KB/s, 0 seconds passed
... 30%, 38496 KB, 42825 KB/s, 0 seconds passed
... 30%, 38528 KB, 42841 KB/s, 0 seconds passed
... 30%, 38560 KB, 42867 KB/s, 0 seconds passed
... 30%, 38592 KB, 42896 KB/s, 0 seconds passed
... 30%, 38624 KB, 42924 KB/s, 0 seconds passed
... 30%, 38656 KB, 42944 KB/s, 0 seconds passed
... 30%, 38688 KB, 42959 KB/s, 0 seconds passed
... 30%, 38720 KB, 42984 KB/s, 0 seconds passed
... 30%, 38752 KB, 43007 KB/s, 0 seconds passed
... 30%, 38784 KB, 43028 KB/s, 0 seconds passed
... 30%, 38816 KB, 43044 KB/s, 0 seconds passed
... 30%, 38848 KB, 43060 KB/s, 0 seconds passed
... 30%, 38880 KB, 43079 KB/s, 0 seconds passed
... 30%, 38912 KB, 43106 KB/s, 0 seconds passed
... 30%, 38944 KB, 43134 KB/s, 0 seconds passed
... 30%, 38976 KB, 43162 KB/s, 0 seconds passed
... 30%, 39008 KB, 43188 KB/s, 0 seconds passed
... 30%, 39040 KB, 43202 KB/s, 0 seconds passed
... 31%, 39072 KB, 43217 KB/s, 0 seconds passed
... 31%, 39104 KB, 43240 KB/s, 0 seconds passed
... 31%, 39136 KB, 43267 KB/s, 0 seconds passed
... 31%, 39168 KB, 43293 KB/s, 0 seconds passed
... 31%, 39200 KB, 43321 KB/s, 0 seconds passed
... 31%, 39232 KB, 43344 KB/s, 0 seconds passed
... 31%, 39264 KB, 43363 KB/s, 0 seconds passed
... 31%, 39296 KB, 43377 KB/s, 0 seconds passed
... 31%, 39328 KB, 43392 KB/s, 0 seconds passed
... 31%, 39360 KB, 43420 KB/s, 0 seconds passed
... 31%, 39392 KB, 43448 KB/s, 0 seconds passed
... 31%, 39424 KB, 43474 KB/s, 0 seconds passed
... 31%, 39456 KB, 43497 KB/s, 0 seconds passed
... 31%, 39488 KB, 43520 KB/s, 0 seconds passed
... 31%, 39520 KB, 43540 KB/s, 0 seconds passed
... 31%, 39552 KB, 43563 KB/s, 0 seconds passed
... 31%, 39584 KB, 43585 KB/s, 0 seconds passed
... 31%, 39616 KB, 43606 KB/s, 0 seconds passed
... 31%, 39648 KB, 43628 KB/s, 0 seconds passed
... 31%, 39680 KB, 43651 KB/s, 0 seconds passed
... 31%, 39712 KB, 43673 KB/s, 0 seconds passed
... 31%, 39744 KB, 43693 KB/s, 0 seconds passed
... 31%, 39776 KB, 43716 KB/s, 0 seconds passed
... 31%, 39808 KB, 43739 KB/s, 0 seconds passed
... 31%, 39840 KB, 43759 KB/s, 0 seconds passed
... 31%, 39872 KB, 43776 KB/s, 0 seconds passed
... 31%, 39904 KB, 43791 KB/s, 0 seconds passed
... 31%, 39936 KB, 43816 KB/s, 0 seconds passed
... 31%, 39968 KB, 43845 KB/s, 0 seconds passed
... 31%, 40000 KB, 43860 KB/s, 0 seconds passed
... 31%, 40032 KB, 43875 KB/s, 0 seconds passed
... 31%, 40064 KB, 43901 KB/s, 0 seconds passed
... 31%, 40096 KB, 43927 KB/s, 0 seconds passed
... 31%, 40128 KB, 43954 KB/s, 0 seconds passed
... 31%, 40160 KB, 43974 KB/s, 0 seconds passed
... 31%, 40192 KB, 43997 KB/s, 0 seconds passed
... 31%, 40224 KB, 44016 KB/s, 0 seconds passed
... 31%, 40256 KB, 44031 KB/s, 0 seconds passed
... 31%, 40288 KB, 44056 KB/s, 0 seconds passed
... 32%, 40320 KB, 44078 KB/s, 0 seconds passed
... 32%, 40352 KB, 44100 KB/s, 0 seconds passed
... 32%, 40384 KB, 44125 KB/s, 0 seconds passed
... 32%, 40416 KB, 44147 KB/s, 0 seconds passed
... 32%, 40448 KB, 44170 KB/s, 0 seconds passed
... 32%, 40480 KB, 44192 KB/s, 0 seconds passed
... 32%, 40512 KB, 44204 KB/s, 0 seconds passed
... 32%, 40544 KB, 44217 KB/s, 0 seconds passed
... 32%, 40576 KB, 44232 KB/s, 0 seconds passed
... 32%, 40608 KB, 44250 KB/s, 0 seconds passed
... 32%, 40640 KB, 44268 KB/s, 0 seconds passed
... 32%, 40672 KB, 44286 KB/s, 0 seconds passed
... 32%, 40704 KB, 44305 KB/s, 0 seconds passed
... 32%, 40736 KB, 44324 KB/s, 0 seconds passed
... 32%, 40768 KB, 44342 KB/s, 0 seconds passed
... 32%, 40800 KB, 44360 KB/s, 0 seconds passed
... 32%, 40832 KB, 44379 KB/s, 0 seconds passed
... 32%, 40864 KB, 44400 KB/s, 0 seconds passed
... 32%, 40896 KB, 44421 KB/s, 0 seconds passed
... 32%, 40928 KB, 44443 KB/s, 0 seconds passed

.. parsed-literal::

    ... 32%, 40960 KB, 43955 KB/s, 0 seconds passed
... 32%, 40992 KB, 43971 KB/s, 0 seconds passed
... 32%, 41024 KB, 43987 KB/s, 0 seconds passed
... 32%, 41056 KB, 44004 KB/s, 0 seconds passed
... 32%, 41088 KB, 44021 KB/s, 0 seconds passed
... 32%, 41120 KB, 44038 KB/s, 0 seconds passed
... 32%, 41152 KB, 44053 KB/s, 0 seconds passed
... 32%, 41184 KB, 44069 KB/s, 0 seconds passed
... 32%, 41216 KB, 44086 KB/s, 0 seconds passed
... 32%, 41248 KB, 44103 KB/s, 0 seconds passed
... 32%, 41280 KB, 44120 KB/s, 0 seconds passed
... 32%, 41312 KB, 44137 KB/s, 0 seconds passed
... 32%, 41344 KB, 44154 KB/s, 0 seconds passed
... 32%, 41376 KB, 44170 KB/s, 0 seconds passed
... 32%, 41408 KB, 44186 KB/s, 0 seconds passed
... 32%, 41440 KB, 44203 KB/s, 0 seconds passed
... 32%, 41472 KB, 44218 KB/s, 0 seconds passed
... 32%, 41504 KB, 44235 KB/s, 0 seconds passed
... 32%, 41536 KB, 44252 KB/s, 0 seconds passed
... 33%, 41568 KB, 44269 KB/s, 0 seconds passed
... 33%, 41600 KB, 44284 KB/s, 0 seconds passed
... 33%, 41632 KB, 44301 KB/s, 0 seconds passed
... 33%, 41664 KB, 44317 KB/s, 0 seconds passed
... 33%, 41696 KB, 44337 KB/s, 0 seconds passed
... 33%, 41728 KB, 44357 KB/s, 0 seconds passed
... 33%, 41760 KB, 44377 KB/s, 0 seconds passed
... 33%, 41792 KB, 44397 KB/s, 0 seconds passed
... 33%, 41824 KB, 44418 KB/s, 0 seconds passed
... 33%, 41856 KB, 44438 KB/s, 0 seconds passed
... 33%, 41888 KB, 44459 KB/s, 0 seconds passed
... 33%, 41920 KB, 44449 KB/s, 0 seconds passed
... 33%, 41952 KB, 44464 KB/s, 0 seconds passed
... 33%, 41984 KB, 44480 KB/s, 0 seconds passed
... 33%, 42016 KB, 44496 KB/s, 0 seconds passed
... 33%, 42048 KB, 44512 KB/s, 0 seconds passed
... 33%, 42080 KB, 44528 KB/s, 0 seconds passed
... 33%, 42112 KB, 44545 KB/s, 0 seconds passed
... 33%, 42144 KB, 44561 KB/s, 0 seconds passed
... 33%, 42176 KB, 44576 KB/s, 0 seconds passed
... 33%, 42208 KB, 44592 KB/s, 0 seconds passed
... 33%, 42240 KB, 44608 KB/s, 0 seconds passed
... 33%, 42272 KB, 44625 KB/s, 0 seconds passed
... 33%, 42304 KB, 44646 KB/s, 0 seconds passed
... 33%, 42336 KB, 44667 KB/s, 0 seconds passed
... 33%, 42368 KB, 44690 KB/s, 0 seconds passed
... 33%, 42400 KB, 44711 KB/s, 0 seconds passed
... 33%, 42432 KB, 44732 KB/s, 0 seconds passed
... 33%, 42464 KB, 44754 KB/s, 0 seconds passed
... 33%, 42496 KB, 44775 KB/s, 0 seconds passed
... 33%, 42528 KB, 44797 KB/s, 0 seconds passed
... 33%, 42560 KB, 44818 KB/s, 0 seconds passed
... 33%, 42592 KB, 44840 KB/s, 0 seconds passed
... 33%, 42624 KB, 44862 KB/s, 0 seconds passed
... 33%, 42656 KB, 44884 KB/s, 0 seconds passed
... 33%, 42688 KB, 44905 KB/s, 0 seconds passed
... 33%, 42720 KB, 44927 KB/s, 0 seconds passed
... 33%, 42752 KB, 44948 KB/s, 0 seconds passed
... 33%, 42784 KB, 44969 KB/s, 0 seconds passed
... 33%, 42816 KB, 44991 KB/s, 0 seconds passed
... 34%, 42848 KB, 45012 KB/s, 0 seconds passed
... 34%, 42880 KB, 45034 KB/s, 0 seconds passed
... 34%, 42912 KB, 45056 KB/s, 0 seconds passed
... 34%, 42944 KB, 45076 KB/s, 0 seconds passed
... 34%, 42976 KB, 45098 KB/s, 0 seconds passed
... 34%, 43008 KB, 45119 KB/s, 0 seconds passed
... 34%, 43040 KB, 45140 KB/s, 0 seconds passed
... 34%, 43072 KB, 45162 KB/s, 0 seconds passed
... 34%, 43104 KB, 45183 KB/s, 0 seconds passed
... 34%, 43136 KB, 45205 KB/s, 0 seconds passed
... 34%, 43168 KB, 45227 KB/s, 0 seconds passed
... 34%, 43200 KB, 45248 KB/s, 0 seconds passed
... 34%, 43232 KB, 45270 KB/s, 0 seconds passed
... 34%, 43264 KB, 45291 KB/s, 0 seconds passed
... 34%, 43296 KB, 45312 KB/s, 0 seconds passed
... 34%, 43328 KB, 45334 KB/s, 0 seconds passed
... 34%, 43360 KB, 45355 KB/s, 0 seconds passed
... 34%, 43392 KB, 45376 KB/s, 0 seconds passed
... 34%, 43424 KB, 45397 KB/s, 0 seconds passed
... 34%, 43456 KB, 45418 KB/s, 0 seconds passed
... 34%, 43488 KB, 45440 KB/s, 0 seconds passed
... 34%, 43520 KB, 45461 KB/s, 0 seconds passed
... 34%, 43552 KB, 45482 KB/s, 0 seconds passed
... 34%, 43584 KB, 45504 KB/s, 0 seconds passed
... 34%, 43616 KB, 45525 KB/s, 0 seconds passed
... 34%, 43648 KB, 45546 KB/s, 0 seconds passed
... 34%, 43680 KB, 45569 KB/s, 0 seconds passed
... 34%, 43712 KB, 45592 KB/s, 0 seconds passed
... 34%, 43744 KB, 45611 KB/s, 0 seconds passed
... 34%, 43776 KB, 45636 KB/s, 0 seconds passed
... 34%, 43808 KB, 45658 KB/s, 0 seconds passed
... 34%, 43840 KB, 45673 KB/s, 0 seconds passed
... 34%, 43872 KB, 45687 KB/s, 0 seconds passed
... 34%, 43904 KB, 45704 KB/s, 0 seconds passed
... 34%, 43936 KB, 45730 KB/s, 0 seconds passed
... 34%, 43968 KB, 45755 KB/s, 0 seconds passed
... 34%, 44000 KB, 45778 KB/s, 0 seconds passed
... 34%, 44032 KB, 45797 KB/s, 0 seconds passed
... 34%, 44064 KB, 45815 KB/s, 0 seconds passed
... 35%, 44096 KB, 45829 KB/s, 0 seconds passed
... 35%, 44128 KB, 45851 KB/s, 0 seconds passed
... 35%, 44160 KB, 45876 KB/s, 0 seconds passed
... 35%, 44192 KB, 45897 KB/s, 0 seconds passed
... 35%, 44224 KB, 45918 KB/s, 0 seconds passed
... 35%, 44256 KB, 45938 KB/s, 0 seconds passed
... 35%, 44288 KB, 45956 KB/s, 0 seconds passed
... 35%, 44320 KB, 45977 KB/s, 0 seconds passed
... 35%, 44352 KB, 45998 KB/s, 0 seconds passed
... 35%, 44384 KB, 46016 KB/s, 0 seconds passed
... 35%, 44416 KB, 46039 KB/s, 0 seconds passed
... 35%, 44448 KB, 46057 KB/s, 0 seconds passed
... 35%, 44480 KB, 46078 KB/s, 0 seconds passed
... 35%, 44512 KB, 46098 KB/s, 0 seconds passed
... 35%, 44544 KB, 46116 KB/s, 0 seconds passed
... 35%, 44576 KB, 46134 KB/s, 0 seconds passed
... 35%, 44608 KB, 46157 KB/s, 0 seconds passed
... 35%, 44640 KB, 46175 KB/s, 0 seconds passed
... 35%, 44672 KB, 46196 KB/s, 0 seconds passed
... 35%, 44704 KB, 46216 KB/s, 0 seconds passed
... 35%, 44736 KB, 46227 KB/s, 0 seconds passed
... 35%, 44768 KB, 46240 KB/s, 0 seconds passed
... 35%, 44800 KB, 46264 KB/s, 0 seconds passed
... 35%, 44832 KB, 46290 KB/s, 0 seconds passed
... 35%, 44864 KB, 46305 KB/s, 0 seconds passed
... 35%, 44896 KB, 46318 KB/s, 0 seconds passed
... 35%, 44928 KB, 46342 KB/s, 0 seconds passed
... 35%, 44960 KB, 46368 KB/s, 0 seconds passed
... 35%, 44992 KB, 46391 KB/s, 0 seconds passed
... 35%, 45024 KB, 46406 KB/s, 0 seconds passed
... 35%, 45056 KB, 46427 KB/s, 0 seconds passed
... 35%, 45088 KB, 46447 KB/s, 0 seconds passed
... 35%, 45120 KB, 46465 KB/s, 0 seconds passed
... 35%, 45152 KB, 46485 KB/s, 0 seconds passed
... 35%, 45184 KB, 46414 KB/s, 0 seconds passed
... 35%, 45216 KB, 46432 KB/s, 0 seconds passed
... 35%, 45248 KB, 46453 KB/s, 0 seconds passed
... 35%, 45280 KB, 46469 KB/s, 0 seconds passed
... 35%, 45312 KB, 46482 KB/s, 0 seconds passed
... 36%, 45344 KB, 46505 KB/s, 0 seconds passed
... 36%, 45376 KB, 46530 KB/s, 0 seconds passed
... 36%, 45408 KB, 46551 KB/s, 0 seconds passed
... 36%, 45440 KB, 46569 KB/s, 0 seconds passed
... 36%, 45472 KB, 46589 KB/s, 0 seconds passed
... 36%, 45504 KB, 46609 KB/s, 0 seconds passed
... 36%, 45536 KB, 46627 KB/s, 0 seconds passed
... 36%, 45568 KB, 46647 KB/s, 0 seconds passed
... 36%, 45600 KB, 46665 KB/s, 0 seconds passed
... 36%, 45632 KB, 46685 KB/s, 0 seconds passed
... 36%, 45664 KB, 46705 KB/s, 0 seconds passed
... 36%, 45696 KB, 46726 KB/s, 0 seconds passed
... 36%, 45728 KB, 46746 KB/s, 0 seconds passed
... 36%, 45760 KB, 46763 KB/s, 0 seconds passed
... 36%, 45792 KB, 46781 KB/s, 0 seconds passed
... 36%, 45824 KB, 46801 KB/s, 0 seconds passed
... 36%, 45856 KB, 46821 KB/s, 0 seconds passed
... 36%, 45888 KB, 46841 KB/s, 0 seconds passed
... 36%, 45920 KB, 46859 KB/s, 0 seconds passed
... 36%, 45952 KB, 46882 KB/s, 0 seconds passed
... 36%, 45984 KB, 46899 KB/s, 0 seconds passed
... 36%, 46016 KB, 46919 KB/s, 0 seconds passed
... 36%, 46048 KB, 46939 KB/s, 0 seconds passed

.. parsed-literal::

    ... 36%, 46080 KB, 43839 KB/s, 1 seconds passed
... 36%, 46112 KB, 43848 KB/s, 1 seconds passed
... 36%, 46144 KB, 43861 KB/s, 1 seconds passed
... 36%, 46176 KB, 43875 KB/s, 1 seconds passed
... 36%, 46208 KB, 43887 KB/s, 1 seconds passed
... 36%, 46240 KB, 43902 KB/s, 1 seconds passed
... 36%, 46272 KB, 43916 KB/s, 1 seconds passed
... 36%, 46304 KB, 43931 KB/s, 1 seconds passed
... 36%, 46336 KB, 43945 KB/s, 1 seconds passed
... 36%, 46368 KB, 43959 KB/s, 1 seconds passed
... 36%, 46400 KB, 43974 KB/s, 1 seconds passed
... 36%, 46432 KB, 43988 KB/s, 1 seconds passed
... 36%, 46464 KB, 44002 KB/s, 1 seconds passed
... 36%, 46496 KB, 44017 KB/s, 1 seconds passed
... 36%, 46528 KB, 44031 KB/s, 1 seconds passed
... 36%, 46560 KB, 44044 KB/s, 1 seconds passed
... 36%, 46592 KB, 44059 KB/s, 1 seconds passed
... 37%, 46624 KB, 44073 KB/s, 1 seconds passed
... 37%, 46656 KB, 44088 KB/s, 1 seconds passed
... 37%, 46688 KB, 44102 KB/s, 1 seconds passed
... 37%, 46720 KB, 44116 KB/s, 1 seconds passed
... 37%, 46752 KB, 44132 KB/s, 1 seconds passed
... 37%, 46784 KB, 44147 KB/s, 1 seconds passed
... 37%, 46816 KB, 44162 KB/s, 1 seconds passed
... 37%, 46848 KB, 44178 KB/s, 1 seconds passed
... 37%, 46880 KB, 44192 KB/s, 1 seconds passed
... 37%, 46912 KB, 44208 KB/s, 1 seconds passed
... 37%, 46944 KB, 44223 KB/s, 1 seconds passed
... 37%, 46976 KB, 44237 KB/s, 1 seconds passed
... 37%, 47008 KB, 44252 KB/s, 1 seconds passed
... 37%, 47040 KB, 44268 KB/s, 1 seconds passed
... 37%, 47072 KB, 44283 KB/s, 1 seconds passed
... 37%, 47104 KB, 44299 KB/s, 1 seconds passed
... 37%, 47136 KB, 44314 KB/s, 1 seconds passed
... 37%, 47168 KB, 44330 KB/s, 1 seconds passed
... 37%, 47200 KB, 44345 KB/s, 1 seconds passed
... 37%, 47232 KB, 44358 KB/s, 1 seconds passed
... 37%, 47264 KB, 44374 KB/s, 1 seconds passed
... 37%, 47296 KB, 44389 KB/s, 1 seconds passed
... 37%, 47328 KB, 44405 KB/s, 1 seconds passed
... 37%, 47360 KB, 44421 KB/s, 1 seconds passed
... 37%, 47392 KB, 44436 KB/s, 1 seconds passed
... 37%, 47424 KB, 44452 KB/s, 1 seconds passed
... 37%, 47456 KB, 44467 KB/s, 1 seconds passed
... 37%, 47488 KB, 44483 KB/s, 1 seconds passed
... 37%, 47520 KB, 44498 KB/s, 1 seconds passed
... 37%, 47552 KB, 44513 KB/s, 1 seconds passed
... 37%, 47584 KB, 44528 KB/s, 1 seconds passed
... 37%, 47616 KB, 44543 KB/s, 1 seconds passed
... 37%, 47648 KB, 44559 KB/s, 1 seconds passed
... 37%, 47680 KB, 44577 KB/s, 1 seconds passed
... 37%, 47712 KB, 44597 KB/s, 1 seconds passed
... 37%, 47744 KB, 44617 KB/s, 1 seconds passed
... 37%, 47776 KB, 44637 KB/s, 1 seconds passed
... 37%, 47808 KB, 44658 KB/s, 1 seconds passed
... 37%, 47840 KB, 44678 KB/s, 1 seconds passed
... 38%, 47872 KB, 44697 KB/s, 1 seconds passed
... 38%, 47904 KB, 44718 KB/s, 1 seconds passed
... 38%, 47936 KB, 44738 KB/s, 1 seconds passed
... 38%, 47968 KB, 44758 KB/s, 1 seconds passed
... 38%, 48000 KB, 44778 KB/s, 1 seconds passed
... 38%, 48032 KB, 44799 KB/s, 1 seconds passed
... 38%, 48064 KB, 44819 KB/s, 1 seconds passed
... 38%, 48096 KB, 44838 KB/s, 1 seconds passed
... 38%, 48128 KB, 44858 KB/s, 1 seconds passed
... 38%, 48160 KB, 44878 KB/s, 1 seconds passed
... 38%, 48192 KB, 44898 KB/s, 1 seconds passed
... 38%, 48224 KB, 44918 KB/s, 1 seconds passed
... 38%, 48256 KB, 44938 KB/s, 1 seconds passed
... 38%, 48288 KB, 44958 KB/s, 1 seconds passed
... 38%, 48320 KB, 44978 KB/s, 1 seconds passed
... 38%, 48352 KB, 44998 KB/s, 1 seconds passed
... 38%, 48384 KB, 45018 KB/s, 1 seconds passed
... 38%, 48416 KB, 45038 KB/s, 1 seconds passed
... 38%, 48448 KB, 45058 KB/s, 1 seconds passed
... 38%, 48480 KB, 45077 KB/s, 1 seconds passed
... 38%, 48512 KB, 45097 KB/s, 1 seconds passed
... 38%, 48544 KB, 45118 KB/s, 1 seconds passed
... 38%, 48576 KB, 45138 KB/s, 1 seconds passed
... 38%, 48608 KB, 45158 KB/s, 1 seconds passed
... 38%, 48640 KB, 45176 KB/s, 1 seconds passed
... 38%, 48672 KB, 45196 KB/s, 1 seconds passed
... 38%, 48704 KB, 45215 KB/s, 1 seconds passed
... 38%, 48736 KB, 45235 KB/s, 1 seconds passed
... 38%, 48768 KB, 45255 KB/s, 1 seconds passed
... 38%, 48800 KB, 45275 KB/s, 1 seconds passed
... 38%, 48832 KB, 45295 KB/s, 1 seconds passed
... 38%, 48864 KB, 45315 KB/s, 1 seconds passed
... 38%, 48896 KB, 45335 KB/s, 1 seconds passed
... 38%, 48928 KB, 45355 KB/s, 1 seconds passed
... 38%, 48960 KB, 45374 KB/s, 1 seconds passed
... 38%, 48992 KB, 45394 KB/s, 1 seconds passed
... 38%, 49024 KB, 45414 KB/s, 1 seconds passed
... 38%, 49056 KB, 45437 KB/s, 1 seconds passed
... 38%, 49088 KB, 45460 KB/s, 1 seconds passed
... 38%, 49120 KB, 45482 KB/s, 1 seconds passed
... 39%, 49152 KB, 45505 KB/s, 1 seconds passed
... 39%, 49184 KB, 45528 KB/s, 1 seconds passed
... 39%, 49216 KB, 45551 KB/s, 1 seconds passed
... 39%, 49248 KB, 45572 KB/s, 1 seconds passed
... 39%, 49280 KB, 45595 KB/s, 1 seconds passed
... 39%, 49312 KB, 45618 KB/s, 1 seconds passed
... 39%, 49344 KB, 45641 KB/s, 1 seconds passed
... 39%, 49376 KB, 45663 KB/s, 1 seconds passed
... 39%, 49408 KB, 45686 KB/s, 1 seconds passed
... 39%, 49440 KB, 45709 KB/s, 1 seconds passed
... 39%, 49472 KB, 45732 KB/s, 1 seconds passed
... 39%, 49504 KB, 45754 KB/s, 1 seconds passed
... 39%, 49536 KB, 45777 KB/s, 1 seconds passed
... 39%, 49568 KB, 45800 KB/s, 1 seconds passed
... 39%, 49600 KB, 45823 KB/s, 1 seconds passed
... 39%, 49632 KB, 45846 KB/s, 1 seconds passed
... 39%, 49664 KB, 45868 KB/s, 1 seconds passed
... 39%, 49696 KB, 45891 KB/s, 1 seconds passed
... 39%, 49728 KB, 45914 KB/s, 1 seconds passed
... 39%, 49760 KB, 45937 KB/s, 1 seconds passed

.. parsed-literal::

    ... 39%, 49792 KB, 45958 KB/s, 1 seconds passed
... 39%, 49824 KB, 45977 KB/s, 1 seconds passed
... 39%, 49856 KB, 45865 KB/s, 1 seconds passed
... 39%, 49888 KB, 45855 KB/s, 1 seconds passed
... 39%, 49920 KB, 45874 KB/s, 1 seconds passed
... 39%, 49952 KB, 45893 KB/s, 1 seconds passed
... 39%, 49984 KB, 45912 KB/s, 1 seconds passed
... 39%, 50016 KB, 45930 KB/s, 1 seconds passed
... 39%, 50048 KB, 45946 KB/s, 1 seconds passed
... 39%, 50080 KB, 45964 KB/s, 1 seconds passed
... 39%, 50112 KB, 45983 KB/s, 1 seconds passed
... 39%, 50144 KB, 46001 KB/s, 1 seconds passed
... 39%, 50176 KB, 46017 KB/s, 1 seconds passed
... 39%, 50208 KB, 46035 KB/s, 1 seconds passed
... 39%, 50240 KB, 46051 KB/s, 1 seconds passed
... 39%, 50272 KB, 46070 KB/s, 1 seconds passed
... 39%, 50304 KB, 46088 KB/s, 1 seconds passed
... 39%, 50336 KB, 46104 KB/s, 1 seconds passed
... 39%, 50368 KB, 46120 KB/s, 1 seconds passed
... 40%, 50400 KB, 46131 KB/s, 1 seconds passed
... 40%, 50432 KB, 46143 KB/s, 1 seconds passed
... 40%, 50464 KB, 46155 KB/s, 1 seconds passed
... 40%, 50496 KB, 46178 KB/s, 1 seconds passed
... 40%, 50528 KB, 46200 KB/s, 1 seconds passed
... 40%, 50560 KB, 46223 KB/s, 1 seconds passed
... 40%, 50592 KB, 46246 KB/s, 1 seconds passed
... 40%, 50624 KB, 46238 KB/s, 1 seconds passed
... 40%, 50656 KB, 46256 KB/s, 1 seconds passed
... 40%, 50688 KB, 46278 KB/s, 1 seconds passed
... 40%, 50720 KB, 46296 KB/s, 1 seconds passed
... 40%, 50752 KB, 46312 KB/s, 1 seconds passed
... 40%, 50784 KB, 46328 KB/s, 1 seconds passed
... 40%, 50816 KB, 46338 KB/s, 1 seconds passed
... 40%, 50848 KB, 46350 KB/s, 1 seconds passed
... 40%, 50880 KB, 46366 KB/s, 1 seconds passed
... 40%, 50912 KB, 46389 KB/s, 1 seconds passed
... 40%, 50944 KB, 46411 KB/s, 1 seconds passed
... 40%, 50976 KB, 46430 KB/s, 1 seconds passed
... 40%, 51008 KB, 46448 KB/s, 1 seconds passed
... 40%, 51040 KB, 46466 KB/s, 1 seconds passed
... 40%, 51072 KB, 46482 KB/s, 1 seconds passed
... 40%, 51104 KB, 46500 KB/s, 1 seconds passed
... 40%, 51136 KB, 46518 KB/s, 1 seconds passed
... 40%, 51168 KB, 46536 KB/s, 1 seconds passed

.. parsed-literal::

    ... 40%, 51200 KB, 45035 KB/s, 1 seconds passed
... 40%, 51232 KB, 45042 KB/s, 1 seconds passed
... 40%, 51264 KB, 45052 KB/s, 1 seconds passed
... 40%, 51296 KB, 45068 KB/s, 1 seconds passed
... 40%, 51328 KB, 45016 KB/s, 1 seconds passed
... 40%, 51360 KB, 45019 KB/s, 1 seconds passed
... 40%, 51392 KB, 45030 KB/s, 1 seconds passed
... 40%, 51424 KB, 45042 KB/s, 1 seconds passed
... 40%, 51456 KB, 45055 KB/s, 1 seconds passed
... 40%, 51488 KB, 45066 KB/s, 1 seconds passed
... 40%, 51520 KB, 45079 KB/s, 1 seconds passed
... 40%, 51552 KB, 45092 KB/s, 1 seconds passed
... 40%, 51584 KB, 45105 KB/s, 1 seconds passed
... 40%, 51616 KB, 45119 KB/s, 1 seconds passed
... 41%, 51648 KB, 45132 KB/s, 1 seconds passed
... 41%, 51680 KB, 45143 KB/s, 1 seconds passed
... 41%, 51712 KB, 45156 KB/s, 1 seconds passed
... 41%, 51744 KB, 45169 KB/s, 1 seconds passed
... 41%, 51776 KB, 45181 KB/s, 1 seconds passed
... 41%, 51808 KB, 45194 KB/s, 1 seconds passed
... 41%, 51840 KB, 45208 KB/s, 1 seconds passed
... 41%, 51872 KB, 45221 KB/s, 1 seconds passed
... 41%, 51904 KB, 45234 KB/s, 1 seconds passed
... 41%, 51936 KB, 45246 KB/s, 1 seconds passed
... 41%, 51968 KB, 45259 KB/s, 1 seconds passed
... 41%, 52000 KB, 45271 KB/s, 1 seconds passed
... 41%, 52032 KB, 45284 KB/s, 1 seconds passed
... 41%, 52064 KB, 45297 KB/s, 1 seconds passed
... 41%, 52096 KB, 45310 KB/s, 1 seconds passed
... 41%, 52128 KB, 45326 KB/s, 1 seconds passed
... 41%, 52160 KB, 45342 KB/s, 1 seconds passed
... 41%, 52192 KB, 45357 KB/s, 1 seconds passed
... 41%, 52224 KB, 45372 KB/s, 1 seconds passed
... 41%, 52256 KB, 45388 KB/s, 1 seconds passed
... 41%, 52288 KB, 45404 KB/s, 1 seconds passed
... 41%, 52320 KB, 45420 KB/s, 1 seconds passed
... 41%, 52352 KB, 45436 KB/s, 1 seconds passed
... 41%, 52384 KB, 45452 KB/s, 1 seconds passed
... 41%, 52416 KB, 45466 KB/s, 1 seconds passed
... 41%, 52448 KB, 45482 KB/s, 1 seconds passed
... 41%, 52480 KB, 45498 KB/s, 1 seconds passed
... 41%, 52512 KB, 45513 KB/s, 1 seconds passed
... 41%, 52544 KB, 45529 KB/s, 1 seconds passed
... 41%, 52576 KB, 45545 KB/s, 1 seconds passed
... 41%, 52608 KB, 45560 KB/s, 1 seconds passed
... 41%, 52640 KB, 45576 KB/s, 1 seconds passed
... 41%, 52672 KB, 45592 KB/s, 1 seconds passed
... 41%, 52704 KB, 45608 KB/s, 1 seconds passed
... 41%, 52736 KB, 45624 KB/s, 1 seconds passed
... 41%, 52768 KB, 45640 KB/s, 1 seconds passed
... 41%, 52800 KB, 45655 KB/s, 1 seconds passed
... 41%, 52832 KB, 45670 KB/s, 1 seconds passed
... 41%, 52864 KB, 45686 KB/s, 1 seconds passed
... 41%, 52896 KB, 45702 KB/s, 1 seconds passed
... 42%, 52928 KB, 45718 KB/s, 1 seconds passed
... 42%, 52960 KB, 45733 KB/s, 1 seconds passed
... 42%, 52992 KB, 45749 KB/s, 1 seconds passed
... 42%, 53024 KB, 45765 KB/s, 1 seconds passed
... 42%, 53056 KB, 45780 KB/s, 1 seconds passed
... 42%, 53088 KB, 45796 KB/s, 1 seconds passed
... 42%, 53120 KB, 45812 KB/s, 1 seconds passed
... 42%, 53152 KB, 45827 KB/s, 1 seconds passed
... 42%, 53184 KB, 45846 KB/s, 1 seconds passed
... 42%, 53216 KB, 45865 KB/s, 1 seconds passed
... 42%, 53248 KB, 45885 KB/s, 1 seconds passed
... 42%, 53280 KB, 45903 KB/s, 1 seconds passed
... 42%, 53312 KB, 45922 KB/s, 1 seconds passed
... 42%, 53344 KB, 45941 KB/s, 1 seconds passed
... 42%, 53376 KB, 45961 KB/s, 1 seconds passed
... 42%, 53408 KB, 45980 KB/s, 1 seconds passed
... 42%, 53440 KB, 46000 KB/s, 1 seconds passed
... 42%, 53472 KB, 46019 KB/s, 1 seconds passed
... 42%, 53504 KB, 46038 KB/s, 1 seconds passed
... 42%, 53536 KB, 46057 KB/s, 1 seconds passed
... 42%, 53568 KB, 46077 KB/s, 1 seconds passed
... 42%, 53600 KB, 46097 KB/s, 1 seconds passed
... 42%, 53632 KB, 46116 KB/s, 1 seconds passed
... 42%, 53664 KB, 46136 KB/s, 1 seconds passed
... 42%, 53696 KB, 46155 KB/s, 1 seconds passed
... 42%, 53728 KB, 46174 KB/s, 1 seconds passed
... 42%, 53760 KB, 46194 KB/s, 1 seconds passed
... 42%, 53792 KB, 46214 KB/s, 1 seconds passed
... 42%, 53824 KB, 46233 KB/s, 1 seconds passed
... 42%, 53856 KB, 46253 KB/s, 1 seconds passed
... 42%, 53888 KB, 46272 KB/s, 1 seconds passed
... 42%, 53920 KB, 46290 KB/s, 1 seconds passed
... 42%, 53952 KB, 46309 KB/s, 1 seconds passed
... 42%, 53984 KB, 46328 KB/s, 1 seconds passed
... 42%, 54016 KB, 46348 KB/s, 1 seconds passed
... 42%, 54048 KB, 46367 KB/s, 1 seconds passed
... 42%, 54080 KB, 46386 KB/s, 1 seconds passed
... 42%, 54112 KB, 46405 KB/s, 1 seconds passed
... 42%, 54144 KB, 46424 KB/s, 1 seconds passed
... 43%, 54176 KB, 46443 KB/s, 1 seconds passed
... 43%, 54208 KB, 46463 KB/s, 1 seconds passed
... 43%, 54240 KB, 46483 KB/s, 1 seconds passed
... 43%, 54272 KB, 46502 KB/s, 1 seconds passed
... 43%, 54304 KB, 46521 KB/s, 1 seconds passed
... 43%, 54336 KB, 46540 KB/s, 1 seconds passed
... 43%, 54368 KB, 46559 KB/s, 1 seconds passed
... 43%, 54400 KB, 46579 KB/s, 1 seconds passed
... 43%, 54432 KB, 46598 KB/s, 1 seconds passed
... 43%, 54464 KB, 46617 KB/s, 1 seconds passed
... 43%, 54496 KB, 46636 KB/s, 1 seconds passed
... 43%, 54528 KB, 46655 KB/s, 1 seconds passed
... 43%, 54560 KB, 46674 KB/s, 1 seconds passed
... 43%, 54592 KB, 46693 KB/s, 1 seconds passed
... 43%, 54624 KB, 46712 KB/s, 1 seconds passed
... 43%, 54656 KB, 46734 KB/s, 1 seconds passed
... 43%, 54688 KB, 46755 KB/s, 1 seconds passed
... 43%, 54720 KB, 46776 KB/s, 1 seconds passed
... 43%, 54752 KB, 46793 KB/s, 1 seconds passed
... 43%, 54784 KB, 46809 KB/s, 1 seconds passed
... 43%, 54816 KB, 46824 KB/s, 1 seconds passed
... 43%, 54848 KB, 46841 KB/s, 1 seconds passed
... 43%, 54880 KB, 46856 KB/s, 1 seconds passed
... 43%, 54912 KB, 46872 KB/s, 1 seconds passed
... 43%, 54944 KB, 46889 KB/s, 1 seconds passed
... 43%, 54976 KB, 46904 KB/s, 1 seconds passed
... 43%, 55008 KB, 46887 KB/s, 1 seconds passed
... 43%, 55040 KB, 46904 KB/s, 1 seconds passed
... 43%, 55072 KB, 46920 KB/s, 1 seconds passed
... 43%, 55104 KB, 46692 KB/s, 1 seconds passed
... 43%, 55136 KB, 46709 KB/s, 1 seconds passed
... 43%, 55168 KB, 46724 KB/s, 1 seconds passed
... 43%, 55200 KB, 46740 KB/s, 1 seconds passed
... 43%, 55232 KB, 46757 KB/s, 1 seconds passed
... 43%, 55264 KB, 46774 KB/s, 1 seconds passed
... 43%, 55296 KB, 46790 KB/s, 1 seconds passed
... 43%, 55328 KB, 46807 KB/s, 1 seconds passed
... 43%, 55360 KB, 46824 KB/s, 1 seconds passed
... 43%, 55392 KB, 46838 KB/s, 1 seconds passed
... 44%, 55424 KB, 46854 KB/s, 1 seconds passed
... 44%, 55456 KB, 46867 KB/s, 1 seconds passed
... 44%, 55488 KB, 46884 KB/s, 1 seconds passed
... 44%, 55520 KB, 46900 KB/s, 1 seconds passed
... 44%, 55552 KB, 46915 KB/s, 1 seconds passed
... 44%, 55584 KB, 46929 KB/s, 1 seconds passed
... 44%, 55616 KB, 46943 KB/s, 1 seconds passed
... 44%, 55648 KB, 46956 KB/s, 1 seconds passed
... 44%, 55680 KB, 46971 KB/s, 1 seconds passed
... 44%, 55712 KB, 46985 KB/s, 1 seconds passed

.. parsed-literal::

    ... 44%, 55744 KB, 47000 KB/s, 1 seconds passed
... 44%, 55776 KB, 47015 KB/s, 1 seconds passed
... 44%, 55808 KB, 47030 KB/s, 1 seconds passed
... 44%, 55840 KB, 47044 KB/s, 1 seconds passed
... 44%, 55872 KB, 47058 KB/s, 1 seconds passed
... 44%, 55904 KB, 47073 KB/s, 1 seconds passed
... 44%, 55936 KB, 47087 KB/s, 1 seconds passed
... 44%, 55968 KB, 47102 KB/s, 1 seconds passed
... 44%, 56000 KB, 47115 KB/s, 1 seconds passed
... 44%, 56032 KB, 47129 KB/s, 1 seconds passed
... 44%, 56064 KB, 47144 KB/s, 1 seconds passed
... 44%, 56096 KB, 47158 KB/s, 1 seconds passed
... 44%, 56128 KB, 47172 KB/s, 1 seconds passed
... 44%, 56160 KB, 47189 KB/s, 1 seconds passed
... 44%, 56192 KB, 47206 KB/s, 1 seconds passed
... 44%, 56224 KB, 47225 KB/s, 1 seconds passed
... 44%, 56256 KB, 47247 KB/s, 1 seconds passed
... 44%, 56288 KB, 47269 KB/s, 1 seconds passed
... 44%, 56320 KB, 46698 KB/s, 1 seconds passed
... 44%, 56352 KB, 46691 KB/s, 1 seconds passed
... 44%, 56384 KB, 46701 KB/s, 1 seconds passed
... 44%, 56416 KB, 46712 KB/s, 1 seconds passed
... 44%, 56448 KB, 46724 KB/s, 1 seconds passed
... 44%, 56480 KB, 46678 KB/s, 1 seconds passed
... 44%, 56512 KB, 46686 KB/s, 1 seconds passed
... 44%, 56544 KB, 46697 KB/s, 1 seconds passed
... 44%, 56576 KB, 46708 KB/s, 1 seconds passed
... 44%, 56608 KB, 46719 KB/s, 1 seconds passed
... 44%, 56640 KB, 46730 KB/s, 1 seconds passed
... 44%, 56672 KB, 46742 KB/s, 1 seconds passed
... 45%, 56704 KB, 46750 KB/s, 1 seconds passed
... 45%, 56736 KB, 46761 KB/s, 1 seconds passed
... 45%, 56768 KB, 46772 KB/s, 1 seconds passed
... 45%, 56800 KB, 46783 KB/s, 1 seconds passed
... 45%, 56832 KB, 46796 KB/s, 1 seconds passed
... 45%, 56864 KB, 46808 KB/s, 1 seconds passed
... 45%, 56896 KB, 46820 KB/s, 1 seconds passed
... 45%, 56928 KB, 46832 KB/s, 1 seconds passed
... 45%, 56960 KB, 46843 KB/s, 1 seconds passed
... 45%, 56992 KB, 46856 KB/s, 1 seconds passed
... 45%, 57024 KB, 46865 KB/s, 1 seconds passed
... 45%, 57056 KB, 46876 KB/s, 1 seconds passed
... 45%, 57088 KB, 46888 KB/s, 1 seconds passed
... 45%, 57120 KB, 46900 KB/s, 1 seconds passed
... 45%, 57152 KB, 46913 KB/s, 1 seconds passed
... 45%, 57184 KB, 46924 KB/s, 1 seconds passed
... 45%, 57216 KB, 46936 KB/s, 1 seconds passed
... 45%, 57248 KB, 46948 KB/s, 1 seconds passed
... 45%, 57280 KB, 46961 KB/s, 1 seconds passed
... 45%, 57312 KB, 46973 KB/s, 1 seconds passed
... 45%, 57344 KB, 46985 KB/s, 1 seconds passed
... 45%, 57376 KB, 46996 KB/s, 1 seconds passed
... 45%, 57408 KB, 47013 KB/s, 1 seconds passed
... 45%, 57440 KB, 47030 KB/s, 1 seconds passed
... 45%, 57472 KB, 47047 KB/s, 1 seconds passed
... 45%, 57504 KB, 47065 KB/s, 1 seconds passed
... 45%, 57536 KB, 47082 KB/s, 1 seconds passed
... 45%, 57568 KB, 47099 KB/s, 1 seconds passed
... 45%, 57600 KB, 47116 KB/s, 1 seconds passed
... 45%, 57632 KB, 47133 KB/s, 1 seconds passed
... 45%, 57664 KB, 47150 KB/s, 1 seconds passed
... 45%, 57696 KB, 47166 KB/s, 1 seconds passed
... 45%, 57728 KB, 47183 KB/s, 1 seconds passed
... 45%, 57760 KB, 47200 KB/s, 1 seconds passed
... 45%, 57792 KB, 47217 KB/s, 1 seconds passed
... 45%, 57824 KB, 47234 KB/s, 1 seconds passed
... 45%, 57856 KB, 47251 KB/s, 1 seconds passed
... 45%, 57888 KB, 47267 KB/s, 1 seconds passed
... 45%, 57920 KB, 47284 KB/s, 1 seconds passed
... 46%, 57952 KB, 47300 KB/s, 1 seconds passed
... 46%, 57984 KB, 47317 KB/s, 1 seconds passed
... 46%, 58016 KB, 47334 KB/s, 1 seconds passed
... 46%, 58048 KB, 47351 KB/s, 1 seconds passed
... 46%, 58080 KB, 47368 KB/s, 1 seconds passed
... 46%, 58112 KB, 47385 KB/s, 1 seconds passed
... 46%, 58144 KB, 47402 KB/s, 1 seconds passed
... 46%, 58176 KB, 47419 KB/s, 1 seconds passed
... 46%, 58208 KB, 47436 KB/s, 1 seconds passed
... 46%, 58240 KB, 47452 KB/s, 1 seconds passed
... 46%, 58272 KB, 47469 KB/s, 1 seconds passed
... 46%, 58304 KB, 47486 KB/s, 1 seconds passed
... 46%, 58336 KB, 47503 KB/s, 1 seconds passed
... 46%, 58368 KB, 47520 KB/s, 1 seconds passed
... 46%, 58400 KB, 47536 KB/s, 1 seconds passed
... 46%, 58432 KB, 47552 KB/s, 1 seconds passed
... 46%, 58464 KB, 47569 KB/s, 1 seconds passed
... 46%, 58496 KB, 47586 KB/s, 1 seconds passed
... 46%, 58528 KB, 47603 KB/s, 1 seconds passed
... 46%, 58560 KB, 47620 KB/s, 1 seconds passed
... 46%, 58592 KB, 47637 KB/s, 1 seconds passed
... 46%, 58624 KB, 47653 KB/s, 1 seconds passed
... 46%, 58656 KB, 47669 KB/s, 1 seconds passed
... 46%, 58688 KB, 47686 KB/s, 1 seconds passed
... 46%, 58720 KB, 47704 KB/s, 1 seconds passed
... 46%, 58752 KB, 47724 KB/s, 1 seconds passed
... 46%, 58784 KB, 47744 KB/s, 1 seconds passed
... 46%, 58816 KB, 47764 KB/s, 1 seconds passed
... 46%, 58848 KB, 47783 KB/s, 1 seconds passed
... 46%, 58880 KB, 47803 KB/s, 1 seconds passed
... 46%, 58912 KB, 47822 KB/s, 1 seconds passed
... 46%, 58944 KB, 47842 KB/s, 1 seconds passed
... 46%, 58976 KB, 47861 KB/s, 1 seconds passed
... 46%, 59008 KB, 47881 KB/s, 1 seconds passed
... 46%, 59040 KB, 47900 KB/s, 1 seconds passed
... 46%, 59072 KB, 47920 KB/s, 1 seconds passed
... 46%, 59104 KB, 47939 KB/s, 1 seconds passed
... 46%, 59136 KB, 47959 KB/s, 1 seconds passed
... 46%, 59168 KB, 47973 KB/s, 1 seconds passed
... 47%, 59200 KB, 47989 KB/s, 1 seconds passed
... 47%, 59232 KB, 48005 KB/s, 1 seconds passed
... 47%, 59264 KB, 48018 KB/s, 1 seconds passed
... 47%, 59296 KB, 48034 KB/s, 1 seconds passed
... 47%, 59328 KB, 48050 KB/s, 1 seconds passed
... 47%, 59360 KB, 48059 KB/s, 1 seconds passed
... 47%, 59392 KB, 48073 KB/s, 1 seconds passed
... 47%, 59424 KB, 48082 KB/s, 1 seconds passed
... 47%, 59456 KB, 48101 KB/s, 1 seconds passed
... 47%, 59488 KB, 48121 KB/s, 1 seconds passed
... 47%, 59520 KB, 48139 KB/s, 1 seconds passed
... 47%, 59552 KB, 48153 KB/s, 1 seconds passed
... 47%, 59584 KB, 48168 KB/s, 1 seconds passed
... 47%, 59616 KB, 48184 KB/s, 1 seconds passed

.. parsed-literal::

    ... 47%, 59648 KB, 48197 KB/s, 1 seconds passed
... 47%, 59680 KB, 48211 KB/s, 1 seconds passed
... 47%, 59712 KB, 48226 KB/s, 1 seconds passed
... 47%, 59744 KB, 48244 KB/s, 1 seconds passed
... 47%, 59776 KB, 48258 KB/s, 1 seconds passed
... 47%, 59808 KB, 48273 KB/s, 1 seconds passed
... 47%, 59840 KB, 48289 KB/s, 1 seconds passed
... 47%, 59872 KB, 48298 KB/s, 1 seconds passed
... 47%, 59904 KB, 48314 KB/s, 1 seconds passed
... 47%, 59936 KB, 48330 KB/s, 1 seconds passed
... 47%, 59968 KB, 48343 KB/s, 1 seconds passed
... 47%, 60000 KB, 48362 KB/s, 1 seconds passed
... 47%, 60032 KB, 48378 KB/s, 1 seconds passed
... 47%, 60064 KB, 48394 KB/s, 1 seconds passed
... 47%, 60096 KB, 48405 KB/s, 1 seconds passed
... 47%, 60128 KB, 48415 KB/s, 1 seconds passed
... 47%, 60160 KB, 48425 KB/s, 1 seconds passed
... 47%, 60192 KB, 48171 KB/s, 1 seconds passed
... 47%, 60224 KB, 48161 KB/s, 1 seconds passed
... 47%, 60256 KB, 48180 KB/s, 1 seconds passed
... 47%, 60288 KB, 48194 KB/s, 1 seconds passed
... 47%, 60320 KB, 48206 KB/s, 1 seconds passed
... 47%, 60352 KB, 48216 KB/s, 1 seconds passed
... 47%, 60384 KB, 48228 KB/s, 1 seconds passed
... 47%, 60416 KB, 48239 KB/s, 1 seconds passed
... 47%, 60448 KB, 48249 KB/s, 1 seconds passed
... 48%, 60480 KB, 48261 KB/s, 1 seconds passed
... 48%, 60512 KB, 48273 KB/s, 1 seconds passed
... 48%, 60544 KB, 48283 KB/s, 1 seconds passed
... 48%, 60576 KB, 48295 KB/s, 1 seconds passed
... 48%, 60608 KB, 48306 KB/s, 1 seconds passed
... 48%, 60640 KB, 48318 KB/s, 1 seconds passed
... 48%, 60672 KB, 48329 KB/s, 1 seconds passed
... 48%, 60704 KB, 48341 KB/s, 1 seconds passed
... 48%, 60736 KB, 48352 KB/s, 1 seconds passed
... 48%, 60768 KB, 48363 KB/s, 1 seconds passed
... 48%, 60800 KB, 48374 KB/s, 1 seconds passed
... 48%, 60832 KB, 48387 KB/s, 1 seconds passed
... 48%, 60864 KB, 48401 KB/s, 1 seconds passed
... 48%, 60896 KB, 48415 KB/s, 1 seconds passed
... 48%, 60928 KB, 48430 KB/s, 1 seconds passed
... 48%, 60960 KB, 48444 KB/s, 1 seconds passed
... 48%, 60992 KB, 47888 KB/s, 1 seconds passed
... 48%, 61024 KB, 47895 KB/s, 1 seconds passed
... 48%, 61056 KB, 47904 KB/s, 1 seconds passed
... 48%, 61088 KB, 47914 KB/s, 1 seconds passed
... 48%, 61120 KB, 47924 KB/s, 1 seconds passed
... 48%, 61152 KB, 47935 KB/s, 1 seconds passed
... 48%, 61184 KB, 47947 KB/s, 1 seconds passed
... 48%, 61216 KB, 47958 KB/s, 1 seconds passed
... 48%, 61248 KB, 47967 KB/s, 1 seconds passed
... 48%, 61280 KB, 47978 KB/s, 1 seconds passed
... 48%, 61312 KB, 47991 KB/s, 1 seconds passed
... 48%, 61344 KB, 48006 KB/s, 1 seconds passed
... 48%, 61376 KB, 48020 KB/s, 1 seconds passed
... 48%, 61408 KB, 48035 KB/s, 1 seconds passed

.. parsed-literal::

    ... 48%, 61440 KB, 47180 KB/s, 1 seconds passed
... 48%, 61472 KB, 47187 KB/s, 1 seconds passed
... 48%, 61504 KB, 47196 KB/s, 1 seconds passed
... 48%, 61536 KB, 47209 KB/s, 1 seconds passed
... 48%, 61568 KB, 46974 KB/s, 1 seconds passed
... 48%, 61600 KB, 46980 KB/s, 1 seconds passed
... 48%, 61632 KB, 46989 KB/s, 1 seconds passed
... 48%, 61664 KB, 46999 KB/s, 1 seconds passed
... 48%, 61696 KB, 47012 KB/s, 1 seconds passed
... 49%, 61728 KB, 47016 KB/s, 1 seconds passed
... 49%, 61760 KB, 47026 KB/s, 1 seconds passed
... 49%, 61792 KB, 47037 KB/s, 1 seconds passed
... 49%, 61824 KB, 47047 KB/s, 1 seconds passed
... 49%, 61856 KB, 47057 KB/s, 1 seconds passed
... 49%, 61888 KB, 47069 KB/s, 1 seconds passed
... 49%, 61920 KB, 47079 KB/s, 1 seconds passed
... 49%, 61952 KB, 47089 KB/s, 1 seconds passed
... 49%, 61984 KB, 47100 KB/s, 1 seconds passed
... 49%, 62016 KB, 47110 KB/s, 1 seconds passed
... 49%, 62048 KB, 47119 KB/s, 1 seconds passed
... 49%, 62080 KB, 47129 KB/s, 1 seconds passed
... 49%, 62112 KB, 47139 KB/s, 1 seconds passed
... 49%, 62144 KB, 47150 KB/s, 1 seconds passed
... 49%, 62176 KB, 47160 KB/s, 1 seconds passed
... 49%, 62208 KB, 47170 KB/s, 1 seconds passed
... 49%, 62240 KB, 47181 KB/s, 1 seconds passed
... 49%, 62272 KB, 47191 KB/s, 1 seconds passed
... 49%, 62304 KB, 47202 KB/s, 1 seconds passed
... 49%, 62336 KB, 47212 KB/s, 1 seconds passed
... 49%, 62368 KB, 47222 KB/s, 1 seconds passed
... 49%, 62400 KB, 47232 KB/s, 1 seconds passed
... 49%, 62432 KB, 47242 KB/s, 1 seconds passed
... 49%, 62464 KB, 47254 KB/s, 1 seconds passed
... 49%, 62496 KB, 47269 KB/s, 1 seconds passed
... 49%, 62528 KB, 47284 KB/s, 1 seconds passed
... 49%, 62560 KB, 47299 KB/s, 1 seconds passed
... 49%, 62592 KB, 47314 KB/s, 1 seconds passed
... 49%, 62624 KB, 47329 KB/s, 1 seconds passed
... 49%, 62656 KB, 47345 KB/s, 1 seconds passed
... 49%, 62688 KB, 47360 KB/s, 1 seconds passed
... 49%, 62720 KB, 47375 KB/s, 1 seconds passed
... 49%, 62752 KB, 47390 KB/s, 1 seconds passed
... 49%, 62784 KB, 47405 KB/s, 1 seconds passed
... 49%, 62816 KB, 47420 KB/s, 1 seconds passed
... 49%, 62848 KB, 47434 KB/s, 1 seconds passed
... 49%, 62880 KB, 47450 KB/s, 1 seconds passed
... 49%, 62912 KB, 47465 KB/s, 1 seconds passed
... 49%, 62944 KB, 47481 KB/s, 1 seconds passed
... 49%, 62976 KB, 47496 KB/s, 1 seconds passed
... 50%, 63008 KB, 47511 KB/s, 1 seconds passed
... 50%, 63040 KB, 47526 KB/s, 1 seconds passed
... 50%, 63072 KB, 47541 KB/s, 1 seconds passed
... 50%, 63104 KB, 47556 KB/s, 1 seconds passed
... 50%, 63136 KB, 47572 KB/s, 1 seconds passed
... 50%, 63168 KB, 47587 KB/s, 1 seconds passed
... 50%, 63200 KB, 47602 KB/s, 1 seconds passed
... 50%, 63232 KB, 47618 KB/s, 1 seconds passed
... 50%, 63264 KB, 47633 KB/s, 1 seconds passed
... 50%, 63296 KB, 47647 KB/s, 1 seconds passed
... 50%, 63328 KB, 47662 KB/s, 1 seconds passed
... 50%, 63360 KB, 47677 KB/s, 1 seconds passed
... 50%, 63392 KB, 47693 KB/s, 1 seconds passed
... 50%, 63424 KB, 47708 KB/s, 1 seconds passed
... 50%, 63456 KB, 47722 KB/s, 1 seconds passed
... 50%, 63488 KB, 47738 KB/s, 1 seconds passed
... 50%, 63520 KB, 47753 KB/s, 1 seconds passed
... 50%, 63552 KB, 47768 KB/s, 1 seconds passed
... 50%, 63584 KB, 47784 KB/s, 1 seconds passed
... 50%, 63616 KB, 47799 KB/s, 1 seconds passed
... 50%, 63648 KB, 47814 KB/s, 1 seconds passed
... 50%, 63680 KB, 47829 KB/s, 1 seconds passed
... 50%, 63712 KB, 47845 KB/s, 1 seconds passed
... 50%, 63744 KB, 47860 KB/s, 1 seconds passed
... 50%, 63776 KB, 47879 KB/s, 1 seconds passed
... 50%, 63808 KB, 47897 KB/s, 1 seconds passed
... 50%, 63840 KB, 47915 KB/s, 1 seconds passed
... 50%, 63872 KB, 47933 KB/s, 1 seconds passed
... 50%, 63904 KB, 47951 KB/s, 1 seconds passed
... 50%, 63936 KB, 47968 KB/s, 1 seconds passed
... 50%, 63968 KB, 47986 KB/s, 1 seconds passed
... 50%, 64000 KB, 48004 KB/s, 1 seconds passed

.. parsed-literal::

    ... 50%, 64032 KB, 47575 KB/s, 1 seconds passed
... 50%, 64064 KB, 47581 KB/s, 1 seconds passed
... 50%, 64096 KB, 47595 KB/s, 1 seconds passed
... 50%, 64128 KB, 47611 KB/s, 1 seconds passed
... 50%, 64160 KB, 47627 KB/s, 1 seconds passed
... 50%, 64192 KB, 47532 KB/s, 1 seconds passed
... 50%, 64224 KB, 47545 KB/s, 1 seconds passed
... 51%, 64256 KB, 47560 KB/s, 1 seconds passed
... 51%, 64288 KB, 47576 KB/s, 1 seconds passed
... 51%, 64320 KB, 47590 KB/s, 1 seconds passed
... 51%, 64352 KB, 47574 KB/s, 1 seconds passed
... 51%, 64384 KB, 47586 KB/s, 1 seconds passed
... 51%, 64416 KB, 47598 KB/s, 1 seconds passed
... 51%, 64448 KB, 47612 KB/s, 1 seconds passed
... 51%, 64480 KB, 47625 KB/s, 1 seconds passed
... 51%, 64512 KB, 47637 KB/s, 1 seconds passed
... 51%, 64544 KB, 47650 KB/s, 1 seconds passed
... 51%, 64576 KB, 47663 KB/s, 1 seconds passed
... 51%, 64608 KB, 47677 KB/s, 1 seconds passed
... 51%, 64640 KB, 47690 KB/s, 1 seconds passed
... 51%, 64672 KB, 47703 KB/s, 1 seconds passed
... 51%, 64704 KB, 47716 KB/s, 1 seconds passed
... 51%, 64736 KB, 47730 KB/s, 1 seconds passed
... 51%, 64768 KB, 47694 KB/s, 1 seconds passed
... 51%, 64800 KB, 47703 KB/s, 1 seconds passed
... 51%, 64832 KB, 47716 KB/s, 1 seconds passed
... 51%, 64864 KB, 47728 KB/s, 1 seconds passed
... 51%, 64896 KB, 47741 KB/s, 1 seconds passed
... 51%, 64928 KB, 47754 KB/s, 1 seconds passed
... 51%, 64960 KB, 47767 KB/s, 1 seconds passed
... 51%, 64992 KB, 47780 KB/s, 1 seconds passed
... 51%, 65024 KB, 47791 KB/s, 1 seconds passed
... 51%, 65056 KB, 47804 KB/s, 1 seconds passed
... 51%, 65088 KB, 47817 KB/s, 1 seconds passed
... 51%, 65120 KB, 47830 KB/s, 1 seconds passed
... 51%, 65152 KB, 47843 KB/s, 1 seconds passed
... 51%, 65184 KB, 47857 KB/s, 1 seconds passed
... 51%, 65216 KB, 47872 KB/s, 1 seconds passed
... 51%, 65248 KB, 47886 KB/s, 1 seconds passed
... 51%, 65280 KB, 47901 KB/s, 1 seconds passed
... 51%, 65312 KB, 47915 KB/s, 1 seconds passed
... 51%, 65344 KB, 47930 KB/s, 1 seconds passed
... 51%, 65376 KB, 47944 KB/s, 1 seconds passed
... 51%, 65408 KB, 47958 KB/s, 1 seconds passed
... 51%, 65440 KB, 47972 KB/s, 1 seconds passed
... 51%, 65472 KB, 47986 KB/s, 1 seconds passed
... 52%, 65504 KB, 48000 KB/s, 1 seconds passed
... 52%, 65536 KB, 48014 KB/s, 1 seconds passed
... 52%, 65568 KB, 48029 KB/s, 1 seconds passed
... 52%, 65600 KB, 48043 KB/s, 1 seconds passed
... 52%, 65632 KB, 48058 KB/s, 1 seconds passed
... 52%, 65664 KB, 48072 KB/s, 1 seconds passed
... 52%, 65696 KB, 48087 KB/s, 1 seconds passed
... 52%, 65728 KB, 48101 KB/s, 1 seconds passed
... 52%, 65760 KB, 48116 KB/s, 1 seconds passed
... 52%, 65792 KB, 48130 KB/s, 1 seconds passed
... 52%, 65824 KB, 48145 KB/s, 1 seconds passed
... 52%, 65856 KB, 48159 KB/s, 1 seconds passed
... 52%, 65888 KB, 48173 KB/s, 1 seconds passed
... 52%, 65920 KB, 48188 KB/s, 1 seconds passed
... 52%, 65952 KB, 48199 KB/s, 1 seconds passed
... 52%, 65984 KB, 48213 KB/s, 1 seconds passed
... 52%, 66016 KB, 48227 KB/s, 1 seconds passed
... 52%, 66048 KB, 48240 KB/s, 1 seconds passed
... 52%, 66080 KB, 48254 KB/s, 1 seconds passed
... 52%, 66112 KB, 48268 KB/s, 1 seconds passed
... 52%, 66144 KB, 48282 KB/s, 1 seconds passed
... 52%, 66176 KB, 48296 KB/s, 1 seconds passed
... 52%, 66208 KB, 48310 KB/s, 1 seconds passed
... 52%, 66240 KB, 48322 KB/s, 1 seconds passed
... 52%, 66272 KB, 48336 KB/s, 1 seconds passed
... 52%, 66304 KB, 48348 KB/s, 1 seconds passed
... 52%, 66336 KB, 48362 KB/s, 1 seconds passed
... 52%, 66368 KB, 48376 KB/s, 1 seconds passed
... 52%, 66400 KB, 48390 KB/s, 1 seconds passed
... 52%, 66432 KB, 48404 KB/s, 1 seconds passed
... 52%, 66464 KB, 48416 KB/s, 1 seconds passed
... 52%, 66496 KB, 48432 KB/s, 1 seconds passed
... 52%, 66528 KB, 48444 KB/s, 1 seconds passed

.. parsed-literal::

    ... 52%, 66560 KB, 46842 KB/s, 1 seconds passed
... 52%, 66592 KB, 46809 KB/s, 1 seconds passed
... 52%, 66624 KB, 46816 KB/s, 1 seconds passed
... 52%, 66656 KB, 46827 KB/s, 1 seconds passed
... 52%, 66688 KB, 46527 KB/s, 1 seconds passed
... 52%, 66720 KB, 46534 KB/s, 1 seconds passed
... 52%, 66752 KB, 46542 KB/s, 1 seconds passed
... 53%, 66784 KB, 46551 KB/s, 1 seconds passed
... 53%, 66816 KB, 46560 KB/s, 1 seconds passed
... 53%, 66848 KB, 46570 KB/s, 1 seconds passed
... 53%, 66880 KB, 46580 KB/s, 1 seconds passed
... 53%, 66912 KB, 46589 KB/s, 1 seconds passed
... 53%, 66944 KB, 46598 KB/s, 1 seconds passed
... 53%, 66976 KB, 46607 KB/s, 1 seconds passed
... 53%, 67008 KB, 46617 KB/s, 1 seconds passed
... 53%, 67040 KB, 46626 KB/s, 1 seconds passed
... 53%, 67072 KB, 46636 KB/s, 1 seconds passed
... 53%, 67104 KB, 46646 KB/s, 1 seconds passed
... 53%, 67136 KB, 46656 KB/s, 1 seconds passed
... 53%, 67168 KB, 46666 KB/s, 1 seconds passed
... 53%, 67200 KB, 46676 KB/s, 1 seconds passed
... 53%, 67232 KB, 46686 KB/s, 1 seconds passed
... 53%, 67264 KB, 46696 KB/s, 1 seconds passed
... 53%, 67296 KB, 46704 KB/s, 1 seconds passed
... 53%, 67328 KB, 46714 KB/s, 1 seconds passed
... 53%, 67360 KB, 46724 KB/s, 1 seconds passed
... 53%, 67392 KB, 46734 KB/s, 1 seconds passed
... 53%, 67424 KB, 46744 KB/s, 1 seconds passed

.. parsed-literal::

    ... 53%, 67456 KB, 46754 KB/s, 1 seconds passed
... 53%, 67488 KB, 46764 KB/s, 1 seconds passed
... 53%, 67520 KB, 46776 KB/s, 1 seconds passed
... 53%, 67552 KB, 46788 KB/s, 1 seconds passed
... 53%, 67584 KB, 46801 KB/s, 1 seconds passed
... 53%, 67616 KB, 46814 KB/s, 1 seconds passed
... 53%, 67648 KB, 46825 KB/s, 1 seconds passed
... 53%, 67680 KB, 46838 KB/s, 1 seconds passed
... 53%, 67712 KB, 46850 KB/s, 1 seconds passed
... 53%, 67744 KB, 46863 KB/s, 1 seconds passed
... 53%, 67776 KB, 46876 KB/s, 1 seconds passed
... 53%, 67808 KB, 46888 KB/s, 1 seconds passed
... 53%, 67840 KB, 46900 KB/s, 1 seconds passed
... 53%, 67872 KB, 46913 KB/s, 1 seconds passed
... 53%, 67904 KB, 46926 KB/s, 1 seconds passed
... 53%, 67936 KB, 46939 KB/s, 1 seconds passed
... 53%, 67968 KB, 46952 KB/s, 1 seconds passed
... 53%, 68000 KB, 46964 KB/s, 1 seconds passed
... 54%, 68032 KB, 46977 KB/s, 1 seconds passed
... 54%, 68064 KB, 46990 KB/s, 1 seconds passed
... 54%, 68096 KB, 47002 KB/s, 1 seconds passed
... 54%, 68128 KB, 47015 KB/s, 1 seconds passed
... 54%, 68160 KB, 47028 KB/s, 1 seconds passed
... 54%, 68192 KB, 47040 KB/s, 1 seconds passed
... 54%, 68224 KB, 47053 KB/s, 1 seconds passed
... 54%, 68256 KB, 47065 KB/s, 1 seconds passed
... 54%, 68288 KB, 47078 KB/s, 1 seconds passed
... 54%, 68320 KB, 47091 KB/s, 1 seconds passed
... 54%, 68352 KB, 47103 KB/s, 1 seconds passed
... 54%, 68384 KB, 47116 KB/s, 1 seconds passed
... 54%, 68416 KB, 47129 KB/s, 1 seconds passed
... 54%, 68448 KB, 47142 KB/s, 1 seconds passed
... 54%, 68480 KB, 47155 KB/s, 1 seconds passed
... 54%, 68512 KB, 47167 KB/s, 1 seconds passed
... 54%, 68544 KB, 47179 KB/s, 1 seconds passed
... 54%, 68576 KB, 47192 KB/s, 1 seconds passed
... 54%, 68608 KB, 47205 KB/s, 1 seconds passed
... 54%, 68640 KB, 47221 KB/s, 1 seconds passed
... 54%, 68672 KB, 47237 KB/s, 1 seconds passed
... 54%, 68704 KB, 47252 KB/s, 1 seconds passed
... 54%, 68736 KB, 47267 KB/s, 1 seconds passed
... 54%, 68768 KB, 47283 KB/s, 1 seconds passed
... 54%, 68800 KB, 47299 KB/s, 1 seconds passed
... 54%, 68832 KB, 47314 KB/s, 1 seconds passed
... 54%, 68864 KB, 47329 KB/s, 1 seconds passed
... 54%, 68896 KB, 47345 KB/s, 1 seconds passed
... 54%, 68928 KB, 47361 KB/s, 1 seconds passed
... 54%, 68960 KB, 47376 KB/s, 1 seconds passed
... 54%, 68992 KB, 47392 KB/s, 1 seconds passed
... 54%, 69024 KB, 47407 KB/s, 1 seconds passed
... 54%, 69056 KB, 47423 KB/s, 1 seconds passed
... 54%, 69088 KB, 47438 KB/s, 1 seconds passed
... 54%, 69120 KB, 47453 KB/s, 1 seconds passed
... 54%, 69152 KB, 47468 KB/s, 1 seconds passed
... 54%, 69184 KB, 47484 KB/s, 1 seconds passed
... 54%, 69216 KB, 47499 KB/s, 1 seconds passed
... 54%, 69248 KB, 47513 KB/s, 1 seconds passed
... 55%, 69280 KB, 47528 KB/s, 1 seconds passed
... 55%, 69312 KB, 47541 KB/s, 1 seconds passed
... 55%, 69344 KB, 47547 KB/s, 1 seconds passed
... 55%, 69376 KB, 47057 KB/s, 1 seconds passed
... 55%, 69408 KB, 47063 KB/s, 1 seconds passed
... 55%, 69440 KB, 47072 KB/s, 1 seconds passed
... 55%, 69472 KB, 47081 KB/s, 1 seconds passed
... 55%, 69504 KB, 47091 KB/s, 1 seconds passed
... 55%, 69536 KB, 47101 KB/s, 1 seconds passed
... 55%, 69568 KB, 47109 KB/s, 1 seconds passed
... 55%, 69600 KB, 47118 KB/s, 1 seconds passed
... 55%, 69632 KB, 47128 KB/s, 1 seconds passed
... 55%, 69664 KB, 47140 KB/s, 1 seconds passed
... 55%, 69696 KB, 47152 KB/s, 1 seconds passed
... 55%, 69728 KB, 47164 KB/s, 1 seconds passed
... 55%, 69760 KB, 47175 KB/s, 1 seconds passed
... 55%, 69792 KB, 47185 KB/s, 1 seconds passed
... 55%, 69824 KB, 47194 KB/s, 1 seconds passed
... 55%, 69856 KB, 47204 KB/s, 1 seconds passed
... 55%, 69888 KB, 47214 KB/s, 1 seconds passed
... 55%, 69920 KB, 47222 KB/s, 1 seconds passed
... 55%, 69952 KB, 47232 KB/s, 1 seconds passed
... 55%, 69984 KB, 47241 KB/s, 1 seconds passed
... 55%, 70016 KB, 47251 KB/s, 1 seconds passed
... 55%, 70048 KB, 47260 KB/s, 1 seconds passed
... 55%, 70080 KB, 47269 KB/s, 1 seconds passed
... 55%, 70112 KB, 47279 KB/s, 1 seconds passed
... 55%, 70144 KB, 47288 KB/s, 1 seconds passed
... 55%, 70176 KB, 47299 KB/s, 1 seconds passed
... 55%, 70208 KB, 47311 KB/s, 1 seconds passed
... 55%, 70240 KB, 47322 KB/s, 1 seconds passed
... 55%, 70272 KB, 47332 KB/s, 1 seconds passed
... 55%, 70304 KB, 47343 KB/s, 1 seconds passed
... 55%, 70336 KB, 47354 KB/s, 1 seconds passed
... 55%, 70368 KB, 47366 KB/s, 1 seconds passed
... 55%, 70400 KB, 47377 KB/s, 1 seconds passed
... 55%, 70432 KB, 47388 KB/s, 1 seconds passed
... 55%, 70464 KB, 47399 KB/s, 1 seconds passed
... 55%, 70496 KB, 47410 KB/s, 1 seconds passed
... 55%, 70528 KB, 47421 KB/s, 1 seconds passed
... 56%, 70560 KB, 47432 KB/s, 1 seconds passed
... 56%, 70592 KB, 47443 KB/s, 1 seconds passed
... 56%, 70624 KB, 47455 KB/s, 1 seconds passed
... 56%, 70656 KB, 47465 KB/s, 1 seconds passed
... 56%, 70688 KB, 47476 KB/s, 1 seconds passed
... 56%, 70720 KB, 47487 KB/s, 1 seconds passed
... 56%, 70752 KB, 47499 KB/s, 1 seconds passed
... 56%, 70784 KB, 47510 KB/s, 1 seconds passed
... 56%, 70816 KB, 47520 KB/s, 1 seconds passed
... 56%, 70848 KB, 47531 KB/s, 1 seconds passed
... 56%, 70880 KB, 47543 KB/s, 1 seconds passed
... 56%, 70912 KB, 47554 KB/s, 1 seconds passed
... 56%, 70944 KB, 47565 KB/s, 1 seconds passed
... 56%, 70976 KB, 47576 KB/s, 1 seconds passed
... 56%, 71008 KB, 47587 KB/s, 1 seconds passed
... 56%, 71040 KB, 47598 KB/s, 1 seconds passed
... 56%, 71072 KB, 47608 KB/s, 1 seconds passed
... 56%, 71104 KB, 47620 KB/s, 1 seconds passed
... 56%, 71136 KB, 47632 KB/s, 1 seconds passed
... 56%, 71168 KB, 47646 KB/s, 1 seconds passed

.. parsed-literal::

    ... 56%, 71200 KB, 47661 KB/s, 1 seconds passed
... 56%, 71232 KB, 47675 KB/s, 1 seconds passed
... 56%, 71264 KB, 47690 KB/s, 1 seconds passed
... 56%, 71296 KB, 47705 KB/s, 1 seconds passed
... 56%, 71328 KB, 47719 KB/s, 1 seconds passed
... 56%, 71360 KB, 47734 KB/s, 1 seconds passed
... 56%, 71392 KB, 47749 KB/s, 1 seconds passed
... 56%, 71424 KB, 47763 KB/s, 1 seconds passed
... 56%, 71456 KB, 47778 KB/s, 1 seconds passed
... 56%, 71488 KB, 47793 KB/s, 1 seconds passed
... 56%, 71520 KB, 47807 KB/s, 1 seconds passed
... 56%, 71552 KB, 47822 KB/s, 1 seconds passed
... 56%, 71584 KB, 47836 KB/s, 1 seconds passed
... 56%, 71616 KB, 47850 KB/s, 1 seconds passed
... 56%, 71648 KB, 47865 KB/s, 1 seconds passed
... 56%, 71680 KB, 47880 KB/s, 1 seconds passed
... 56%, 71712 KB, 47895 KB/s, 1 seconds passed
... 56%, 71744 KB, 47909 KB/s, 1 seconds passed
... 56%, 71776 KB, 47923 KB/s, 1 seconds passed
... 57%, 71808 KB, 47938 KB/s, 1 seconds passed
... 57%, 71840 KB, 47952 KB/s, 1 seconds passed
... 57%, 71872 KB, 47966 KB/s, 1 seconds passed
... 57%, 71904 KB, 47981 KB/s, 1 seconds passed
... 57%, 71936 KB, 47995 KB/s, 1 seconds passed
... 57%, 71968 KB, 48010 KB/s, 1 seconds passed
... 57%, 72000 KB, 48024 KB/s, 1 seconds passed
... 57%, 72032 KB, 48038 KB/s, 1 seconds passed
... 57%, 72064 KB, 48053 KB/s, 1 seconds passed
... 57%, 72096 KB, 48068 KB/s, 1 seconds passed
... 57%, 72128 KB, 48082 KB/s, 1 seconds passed
... 57%, 72160 KB, 48097 KB/s, 1 seconds passed
... 57%, 72192 KB, 48112 KB/s, 1 seconds passed
... 57%, 72224 KB, 48127 KB/s, 1 seconds passed
... 57%, 72256 KB, 48136 KB/s, 1 seconds passed
... 57%, 72288 KB, 48149 KB/s, 1 seconds passed
... 57%, 72320 KB, 48055 KB/s, 1 seconds passed
... 57%, 72352 KB, 48063 KB/s, 1 seconds passed
... 57%, 72384 KB, 48070 KB/s, 1 seconds passed
... 57%, 72416 KB, 48081 KB/s, 1 seconds passed
... 57%, 72448 KB, 48093 KB/s, 1 seconds passed
... 57%, 72480 KB, 48105 KB/s, 1 seconds passed
... 57%, 72512 KB, 48117 KB/s, 1 seconds passed
... 57%, 72544 KB, 48129 KB/s, 1 seconds passed
... 57%, 72576 KB, 48140 KB/s, 1 seconds passed
... 57%, 72608 KB, 48152 KB/s, 1 seconds passed
... 57%, 72640 KB, 48164 KB/s, 1 seconds passed
... 57%, 72672 KB, 48174 KB/s, 1 seconds passed
... 57%, 72704 KB, 48185 KB/s, 1 seconds passed
... 57%, 72736 KB, 48197 KB/s, 1 seconds passed
... 57%, 72768 KB, 48209 KB/s, 1 seconds passed
... 57%, 72800 KB, 48221 KB/s, 1 seconds passed
... 57%, 72832 KB, 48233 KB/s, 1 seconds passed
... 57%, 72864 KB, 48245 KB/s, 1 seconds passed
... 57%, 72896 KB, 48256 KB/s, 1 seconds passed
... 57%, 72928 KB, 48267 KB/s, 1 seconds passed
... 57%, 72960 KB, 48279 KB/s, 1 seconds passed
... 57%, 72992 KB, 48291 KB/s, 1 seconds passed
... 57%, 73024 KB, 48302 KB/s, 1 seconds passed
... 58%, 73056 KB, 48314 KB/s, 1 seconds passed
... 58%, 73088 KB, 48325 KB/s, 1 seconds passed
... 58%, 73120 KB, 48336 KB/s, 1 seconds passed
... 58%, 73152 KB, 48348 KB/s, 1 seconds passed
... 58%, 73184 KB, 48359 KB/s, 1 seconds passed
... 58%, 73216 KB, 48372 KB/s, 1 seconds passed
... 58%, 73248 KB, 48386 KB/s, 1 seconds passed
... 58%, 73280 KB, 48399 KB/s, 1 seconds passed
... 58%, 73312 KB, 48412 KB/s, 1 seconds passed
... 58%, 73344 KB, 48426 KB/s, 1 seconds passed
... 58%, 73376 KB, 48439 KB/s, 1 seconds passed
... 58%, 73408 KB, 48453 KB/s, 1 seconds passed
... 58%, 73440 KB, 48466 KB/s, 1 seconds passed
... 58%, 73472 KB, 48480 KB/s, 1 seconds passed
... 58%, 73504 KB, 48494 KB/s, 1 seconds passed
... 58%, 73536 KB, 48507 KB/s, 1 seconds passed
... 58%, 73568 KB, 48520 KB/s, 1 seconds passed
... 58%, 73600 KB, 48533 KB/s, 1 seconds passed
... 58%, 73632 KB, 48545 KB/s, 1 seconds passed
... 58%, 73664 KB, 48558 KB/s, 1 seconds passed
... 58%, 73696 KB, 48569 KB/s, 1 seconds passed
... 58%, 73728 KB, 48583 KB/s, 1 seconds passed
... 58%, 73760 KB, 48596 KB/s, 1 seconds passed
... 58%, 73792 KB, 48605 KB/s, 1 seconds passed
... 58%, 73824 KB, 48618 KB/s, 1 seconds passed
... 58%, 73856 KB, 48632 KB/s, 1 seconds passed
... 58%, 73888 KB, 48643 KB/s, 1 seconds passed
... 58%, 73920 KB, 48655 KB/s, 1 seconds passed
... 58%, 73952 KB, 48668 KB/s, 1 seconds passed
... 58%, 73984 KB, 48680 KB/s, 1 seconds passed
... 58%, 74016 KB, 48693 KB/s, 1 seconds passed
... 58%, 74048 KB, 48705 KB/s, 1 seconds passed
... 58%, 74080 KB, 48716 KB/s, 1 seconds passed
... 58%, 74112 KB, 48729 KB/s, 1 seconds passed
... 58%, 74144 KB, 48740 KB/s, 1 seconds passed
... 58%, 74176 KB, 48752 KB/s, 1 seconds passed
... 58%, 74208 KB, 48765 KB/s, 1 seconds passed
... 58%, 74240 KB, 48777 KB/s, 1 seconds passed
... 58%, 74272 KB, 48788 KB/s, 1 seconds passed
... 58%, 74304 KB, 48801 KB/s, 1 seconds passed
... 59%, 74336 KB, 48813 KB/s, 1 seconds passed
... 59%, 74368 KB, 48826 KB/s, 1 seconds passed
... 59%, 74400 KB, 48836 KB/s, 1 seconds passed
... 59%, 74432 KB, 48850 KB/s, 1 seconds passed
... 59%, 74464 KB, 48862 KB/s, 1 seconds passed
... 59%, 74496 KB, 48873 KB/s, 1 seconds passed
... 59%, 74528 KB, 48885 KB/s, 1 seconds passed
... 59%, 74560 KB, 48898 KB/s, 1 seconds passed
... 59%, 74592 KB, 48909 KB/s, 1 seconds passed
... 59%, 74624 KB, 48922 KB/s, 1 seconds passed
... 59%, 74656 KB, 48934 KB/s, 1 seconds passed
... 59%, 74688 KB, 48945 KB/s, 1 seconds passed
... 59%, 74720 KB, 48958 KB/s, 1 seconds passed
... 59%, 74752 KB, 48970 KB/s, 1 seconds passed
... 59%, 74784 KB, 48983 KB/s, 1 seconds passed
... 59%, 74816 KB, 48993 KB/s, 1 seconds passed
... 59%, 74848 KB, 49006 KB/s, 1 seconds passed
... 59%, 74880 KB, 49018 KB/s, 1 seconds passed
... 59%, 74912 KB, 49031 KB/s, 1 seconds passed
... 59%, 74944 KB, 49042 KB/s, 1 seconds passed
... 59%, 74976 KB, 49054 KB/s, 1 seconds passed
... 59%, 75008 KB, 49068 KB/s, 1 seconds passed
... 59%, 75040 KB, 49079 KB/s, 1 seconds passed
... 59%, 75072 KB, 49092 KB/s, 1 seconds passed
... 59%, 75104 KB, 49104 KB/s, 1 seconds passed
... 59%, 75136 KB, 49115 KB/s, 1 seconds passed
... 59%, 75168 KB, 49128 KB/s, 1 seconds passed
... 59%, 75200 KB, 49140 KB/s, 1 seconds passed
... 59%, 75232 KB, 49151 KB/s, 1 seconds passed
... 59%, 75264 KB, 49162 KB/s, 1 seconds passed
... 59%, 75296 KB, 49173 KB/s, 1 seconds passed
... 59%, 75328 KB, 49181 KB/s, 1 seconds passed
... 59%, 75360 KB, 49188 KB/s, 1 seconds passed
... 59%, 75392 KB, 49200 KB/s, 1 seconds passed
... 59%, 75424 KB, 49215 KB/s, 1 seconds passed
... 59%, 75456 KB, 49231 KB/s, 1 seconds passed
... 59%, 75488 KB, 49246 KB/s, 1 seconds passed
... 59%, 75520 KB, 49259 KB/s, 1 seconds passed
... 59%, 75552 KB, 49268 KB/s, 1 seconds passed
... 60%, 75584 KB, 49276 KB/s, 1 seconds passed
... 60%, 75616 KB, 49283 KB/s, 1 seconds passed
... 60%, 75648 KB, 49300 KB/s, 1 seconds passed
... 60%, 75680 KB, 49316 KB/s, 1 seconds passed
... 60%, 75712 KB, 49328 KB/s, 1 seconds passed
... 60%, 75744 KB, 49340 KB/s, 1 seconds passed
... 60%, 75776 KB, 49353 KB/s, 1 seconds passed
... 60%, 75808 KB, 49365 KB/s, 1 seconds passed
... 60%, 75840 KB, 49376 KB/s, 1 seconds passed
... 60%, 75872 KB, 49387 KB/s, 1 seconds passed
... 60%, 75904 KB, 49395 KB/s, 1 seconds passed
... 60%, 75936 KB, 49407 KB/s, 1 seconds passed
... 60%, 75968 KB, 49423 KB/s, 1 seconds passed
... 60%, 76000 KB, 49436 KB/s, 1 seconds passed
... 60%, 76032 KB, 49446 KB/s, 1 seconds passed
... 60%, 76064 KB, 49459 KB/s, 1 seconds passed
... 60%, 76096 KB, 49471 KB/s, 1 seconds passed
... 60%, 76128 KB, 49484 KB/s, 1 seconds passed
... 60%, 76160 KB, 49494 KB/s, 1 seconds passed
... 60%, 76192 KB, 49507 KB/s, 1 seconds passed
... 60%, 76224 KB, 49519 KB/s, 1 seconds passed
... 60%, 76256 KB, 49530 KB/s, 1 seconds passed
... 60%, 76288 KB, 49542 KB/s, 1 seconds passed
... 60%, 76320 KB, 49521 KB/s, 1 seconds passed
... 60%, 76352 KB, 49533 KB/s, 1 seconds passed
... 60%, 76384 KB, 49544 KB/s, 1 seconds passed
... 60%, 76416 KB, 49556 KB/s, 1 seconds passed
... 60%, 76448 KB, 49569 KB/s, 1 seconds passed
... 60%, 76480 KB, 49581 KB/s, 1 seconds passed
... 60%, 76512 KB, 49591 KB/s, 1 seconds passed
... 60%, 76544 KB, 49604 KB/s, 1 seconds passed
... 60%, 76576 KB, 49616 KB/s, 1 seconds passed
... 60%, 76608 KB, 49628 KB/s, 1 seconds passed
... 60%, 76640 KB, 49641 KB/s, 1 seconds passed
... 60%, 76672 KB, 49651 KB/s, 1 seconds passed
... 60%, 76704 KB, 49662 KB/s, 1 seconds passed
... 60%, 76736 KB, 49673 KB/s, 1 seconds passed

.. parsed-literal::

    ... 60%, 76768 KB, 49680 KB/s, 1 seconds passed
... 60%, 76800 KB, 48329 KB/s, 1 seconds passed
... 61%, 76832 KB, 48334 KB/s, 1 seconds passed
... 61%, 76864 KB, 48342 KB/s, 1 seconds passed
... 61%, 76896 KB, 48352 KB/s, 1 seconds passed
... 61%, 76928 KB, 48361 KB/s, 1 seconds passed
... 61%, 76960 KB, 48370 KB/s, 1 seconds passed
... 61%, 76992 KB, 48378 KB/s, 1 seconds passed
... 61%, 77024 KB, 48387 KB/s, 1 seconds passed
... 61%, 77056 KB, 48396 KB/s, 1 seconds passed
... 61%, 77088 KB, 48407 KB/s, 1 seconds passed
... 61%, 77120 KB, 48415 KB/s, 1 seconds passed
... 61%, 77152 KB, 48424 KB/s, 1 seconds passed
... 61%, 77184 KB, 48433 KB/s, 1 seconds passed
... 61%, 77216 KB, 48442 KB/s, 1 seconds passed
... 61%, 77248 KB, 48450 KB/s, 1 seconds passed
... 61%, 77280 KB, 48459 KB/s, 1 seconds passed
... 61%, 77312 KB, 48468 KB/s, 1 seconds passed
... 61%, 77344 KB, 48477 KB/s, 1 seconds passed
... 61%, 77376 KB, 48487 KB/s, 1 seconds passed
... 61%, 77408 KB, 48499 KB/s, 1 seconds passed

.. parsed-literal::

    ... 61%, 77440 KB, 48498 KB/s, 1 seconds passed
... 61%, 77472 KB, 48506 KB/s, 1 seconds passed
... 61%, 77504 KB, 48480 KB/s, 1 seconds passed
... 61%, 77536 KB, 48489 KB/s, 1 seconds passed
... 61%, 77568 KB, 48498 KB/s, 1 seconds passed
... 61%, 77600 KB, 48506 KB/s, 1 seconds passed
... 61%, 77632 KB, 48515 KB/s, 1 seconds passed
... 61%, 77664 KB, 48523 KB/s, 1 seconds passed
... 61%, 77696 KB, 48532 KB/s, 1 seconds passed
... 61%, 77728 KB, 48540 KB/s, 1 seconds passed
... 61%, 77760 KB, 48549 KB/s, 1 seconds passed
... 61%, 77792 KB, 48559 KB/s, 1 seconds passed
... 61%, 77824 KB, 48567 KB/s, 1 seconds passed
... 61%, 77856 KB, 48576 KB/s, 1 seconds passed
... 61%, 77888 KB, 48585 KB/s, 1 seconds passed
... 61%, 77920 KB, 48595 KB/s, 1 seconds passed
... 61%, 77952 KB, 48606 KB/s, 1 seconds passed
... 61%, 77984 KB, 48617 KB/s, 1 seconds passed
... 61%, 78016 KB, 48629 KB/s, 1 seconds passed
... 61%, 78048 KB, 48640 KB/s, 1 seconds passed
... 61%, 78080 KB, 48651 KB/s, 1 seconds passed
... 62%, 78112 KB, 48662 KB/s, 1 seconds passed
... 62%, 78144 KB, 48673 KB/s, 1 seconds passed
... 62%, 78176 KB, 48685 KB/s, 1 seconds passed
... 62%, 78208 KB, 48696 KB/s, 1 seconds passed
... 62%, 78240 KB, 48708 KB/s, 1 seconds passed
... 62%, 78272 KB, 48719 KB/s, 1 seconds passed
... 62%, 78304 KB, 48730 KB/s, 1 seconds passed
... 62%, 78336 KB, 48741 KB/s, 1 seconds passed
... 62%, 78368 KB, 48753 KB/s, 1 seconds passed
... 62%, 78400 KB, 48764 KB/s, 1 seconds passed
... 62%, 78432 KB, 48775 KB/s, 1 seconds passed
... 62%, 78464 KB, 48787 KB/s, 1 seconds passed
... 62%, 78496 KB, 48798 KB/s, 1 seconds passed
... 62%, 78528 KB, 48809 KB/s, 1 seconds passed
... 62%, 78560 KB, 48821 KB/s, 1 seconds passed
... 62%, 78592 KB, 48832 KB/s, 1 seconds passed
... 62%, 78624 KB, 48843 KB/s, 1 seconds passed
... 62%, 78656 KB, 48855 KB/s, 1 seconds passed
... 62%, 78688 KB, 48866 KB/s, 1 seconds passed
... 62%, 78720 KB, 48878 KB/s, 1 seconds passed
... 62%, 78752 KB, 48889 KB/s, 1 seconds passed
... 62%, 78784 KB, 48900 KB/s, 1 seconds passed
... 62%, 78816 KB, 48912 KB/s, 1 seconds passed
... 62%, 78848 KB, 48923 KB/s, 1 seconds passed
... 62%, 78880 KB, 48935 KB/s, 1 seconds passed
... 62%, 78912 KB, 48946 KB/s, 1 seconds passed
... 62%, 78944 KB, 48957 KB/s, 1 seconds passed
... 62%, 78976 KB, 48968 KB/s, 1 seconds passed
... 62%, 79008 KB, 48980 KB/s, 1 seconds passed
... 62%, 79040 KB, 48990 KB/s, 1 seconds passed
... 62%, 79072 KB, 49001 KB/s, 1 seconds passed
... 62%, 79104 KB, 49014 KB/s, 1 seconds passed
... 62%, 79136 KB, 49028 KB/s, 1 seconds passed
... 62%, 79168 KB, 49042 KB/s, 1 seconds passed
... 62%, 79200 KB, 49056 KB/s, 1 seconds passed
... 62%, 79232 KB, 49070 KB/s, 1 seconds passed
... 62%, 79264 KB, 49084 KB/s, 1 seconds passed
... 62%, 79296 KB, 49098 KB/s, 1 seconds passed
... 62%, 79328 KB, 49113 KB/s, 1 seconds passed
... 63%, 79360 KB, 49126 KB/s, 1 seconds passed
... 63%, 79392 KB, 49141 KB/s, 1 seconds passed
... 63%, 79424 KB, 49155 KB/s, 1 seconds passed
... 63%, 79456 KB, 49169 KB/s, 1 seconds passed
... 63%, 79488 KB, 49183 KB/s, 1 seconds passed
... 63%, 79520 KB, 49197 KB/s, 1 seconds passed
... 63%, 79552 KB, 49211 KB/s, 1 seconds passed
... 63%, 79584 KB, 49221 KB/s, 1 seconds passed
... 63%, 79616 KB, 49233 KB/s, 1 seconds passed
... 63%, 79648 KB, 49245 KB/s, 1 seconds passed
... 63%, 79680 KB, 49255 KB/s, 1 seconds passed
... 63%, 79712 KB, 49267 KB/s, 1 seconds passed
... 63%, 79744 KB, 49279 KB/s, 1 seconds passed
... 63%, 79776 KB, 49290 KB/s, 1 seconds passed
... 63%, 79808 KB, 49302 KB/s, 1 seconds passed
... 63%, 79840 KB, 49312 KB/s, 1 seconds passed
... 63%, 79872 KB, 49324 KB/s, 1 seconds passed
... 63%, 79904 KB, 49336 KB/s, 1 seconds passed
... 63%, 79936 KB, 49347 KB/s, 1 seconds passed
... 63%, 79968 KB, 49357 KB/s, 1 seconds passed
... 63%, 80000 KB, 49369 KB/s, 1 seconds passed
... 63%, 80032 KB, 49379 KB/s, 1 seconds passed
... 63%, 80064 KB, 49391 KB/s, 1 seconds passed
... 63%, 80096 KB, 49403 KB/s, 1 seconds passed
... 63%, 80128 KB, 49413 KB/s, 1 seconds passed
... 63%, 80160 KB, 49426 KB/s, 1 seconds passed
... 63%, 80192 KB, 49438 KB/s, 1 seconds passed
... 63%, 80224 KB, 49448 KB/s, 1 seconds passed
... 63%, 80256 KB, 49458 KB/s, 1 seconds passed
... 63%, 80288 KB, 49470 KB/s, 1 seconds passed
... 63%, 80320 KB, 49481 KB/s, 1 seconds passed
... 63%, 80352 KB, 49493 KB/s, 1 seconds passed
... 63%, 80384 KB, 49503 KB/s, 1 seconds passed
... 63%, 80416 KB, 49515 KB/s, 1 seconds passed
... 63%, 80448 KB, 49527 KB/s, 1 seconds passed
... 63%, 80480 KB, 49537 KB/s, 1 seconds passed
... 63%, 80512 KB, 49548 KB/s, 1 seconds passed
... 63%, 80544 KB, 49558 KB/s, 1 seconds passed
... 63%, 80576 KB, 49570 KB/s, 1 seconds passed
... 63%, 80608 KB, 49582 KB/s, 1 seconds passed
... 64%, 80640 KB, 49594 KB/s, 1 seconds passed
... 64%, 80672 KB, 49605 KB/s, 1 seconds passed
... 64%, 80704 KB, 49615 KB/s, 1 seconds passed
... 64%, 80736 KB, 49619 KB/s, 1 seconds passed
... 64%, 80768 KB, 49622 KB/s, 1 seconds passed
... 64%, 80800 KB, 49629 KB/s, 1 seconds passed
... 64%, 80832 KB, 49641 KB/s, 1 seconds passed
... 64%, 80864 KB, 49656 KB/s, 1 seconds passed
... 64%, 80896 KB, 49668 KB/s, 1 seconds passed
... 64%, 80928 KB, 49678 KB/s, 1 seconds passed
... 64%, 80960 KB, 49688 KB/s, 1 seconds passed
... 64%, 80992 KB, 49700 KB/s, 1 seconds passed
... 64%, 81024 KB, 49712 KB/s, 1 seconds passed
... 64%, 81056 KB, 49722 KB/s, 1 seconds passed
... 64%, 81088 KB, 49538 KB/s, 1 seconds passed
... 64%, 81120 KB, 49545 KB/s, 1 seconds passed
... 64%, 81152 KB, 49554 KB/s, 1 seconds passed
... 64%, 81184 KB, 49563 KB/s, 1 seconds passed
... 64%, 81216 KB, 49571 KB/s, 1 seconds passed
... 64%, 81248 KB, 49580 KB/s, 1 seconds passed
... 64%, 81280 KB, 49588 KB/s, 1 seconds passed
... 64%, 81312 KB, 49597 KB/s, 1 seconds passed
... 64%, 81344 KB, 49606 KB/s, 1 seconds passed
... 64%, 81376 KB, 49616 KB/s, 1 seconds passed
... 64%, 81408 KB, 49626 KB/s, 1 seconds passed
... 64%, 81440 KB, 49613 KB/s, 1 seconds passed
... 64%, 81472 KB, 49620 KB/s, 1 seconds passed
... 64%, 81504 KB, 49627 KB/s, 1 seconds passed
... 64%, 81536 KB, 49635 KB/s, 1 seconds passed
... 64%, 81568 KB, 49644 KB/s, 1 seconds passed
... 64%, 81600 KB, 49652 KB/s, 1 seconds passed
... 64%, 81632 KB, 49661 KB/s, 1 seconds passed
... 64%, 81664 KB, 49669 KB/s, 1 seconds passed
... 64%, 81696 KB, 49678 KB/s, 1 seconds passed
... 64%, 81728 KB, 49686 KB/s, 1 seconds passed
... 64%, 81760 KB, 49694 KB/s, 1 seconds passed
... 64%, 81792 KB, 49703 KB/s, 1 seconds passed
... 64%, 81824 KB, 49713 KB/s, 1 seconds passed
... 64%, 81856 KB, 49724 KB/s, 1 seconds passed
... 65%, 81888 KB, 49734 KB/s, 1 seconds passed

.. parsed-literal::

    ... 65%, 81920 KB, 48373 KB/s, 1 seconds passed
... 65%, 81952 KB, 48377 KB/s, 1 seconds passed
... 65%, 81984 KB, 48384 KB/s, 1 seconds passed
... 65%, 82016 KB, 48392 KB/s, 1 seconds passed
... 65%, 82048 KB, 48399 KB/s, 1 seconds passed
... 65%, 82080 KB, 48409 KB/s, 1 seconds passed
... 65%, 82112 KB, 48419 KB/s, 1 seconds passed
... 65%, 82144 KB, 48427 KB/s, 1 seconds passed
... 65%, 82176 KB, 48435 KB/s, 1 seconds passed
... 65%, 82208 KB, 48442 KB/s, 1 seconds passed
... 65%, 82240 KB, 48450 KB/s, 1 seconds passed
... 65%, 82272 KB, 48458 KB/s, 1 seconds passed
... 65%, 82304 KB, 48466 KB/s, 1 seconds passed
... 65%, 82336 KB, 48474 KB/s, 1 seconds passed

.. parsed-literal::

    ... 65%, 82368 KB, 48482 KB/s, 1 seconds passed
... 65%, 82400 KB, 48490 KB/s, 1 seconds passed
... 65%, 82432 KB, 48497 KB/s, 1 seconds passed
... 65%, 82464 KB, 48505 KB/s, 1 seconds passed
... 65%, 82496 KB, 48511 KB/s, 1 seconds passed
... 65%, 82528 KB, 48513 KB/s, 1 seconds passed
... 65%, 82560 KB, 48515 KB/s, 1 seconds passed
... 65%, 82592 KB, 48517 KB/s, 1 seconds passed
... 65%, 82624 KB, 48520 KB/s, 1 seconds passed
... 65%, 82656 KB, 48521 KB/s, 1 seconds passed
... 65%, 82688 KB, 48523 KB/s, 1 seconds passed
... 65%, 82720 KB, 48524 KB/s, 1 seconds passed
... 65%, 82752 KB, 48531 KB/s, 1 seconds passed
... 65%, 82784 KB, 48542 KB/s, 1 seconds passed
... 65%, 82816 KB, 48553 KB/s, 1 seconds passed
... 65%, 82848 KB, 48564 KB/s, 1 seconds passed
... 65%, 82880 KB, 48576 KB/s, 1 seconds passed
... 65%, 82912 KB, 48587 KB/s, 1 seconds passed
... 65%, 82944 KB, 48599 KB/s, 1 seconds passed
... 65%, 82976 KB, 48611 KB/s, 1 seconds passed
... 65%, 83008 KB, 48622 KB/s, 1 seconds passed
... 65%, 83040 KB, 48633 KB/s, 1 seconds passed
... 65%, 83072 KB, 48645 KB/s, 1 seconds passed
... 65%, 83104 KB, 48657 KB/s, 1 seconds passed
... 66%, 83136 KB, 48669 KB/s, 1 seconds passed
... 66%, 83168 KB, 48680 KB/s, 1 seconds passed
... 66%, 83200 KB, 48691 KB/s, 1 seconds passed
... 66%, 83232 KB, 48703 KB/s, 1 seconds passed
... 66%, 83264 KB, 48715 KB/s, 1 seconds passed
... 66%, 83296 KB, 48727 KB/s, 1 seconds passed
... 66%, 83328 KB, 48738 KB/s, 1 seconds passed
... 66%, 83360 KB, 48750 KB/s, 1 seconds passed
... 66%, 83392 KB, 48762 KB/s, 1 seconds passed
... 66%, 83424 KB, 48774 KB/s, 1 seconds passed
... 66%, 83456 KB, 48785 KB/s, 1 seconds passed
... 66%, 83488 KB, 48797 KB/s, 1 seconds passed
... 66%, 83520 KB, 48809 KB/s, 1 seconds passed
... 66%, 83552 KB, 48820 KB/s, 1 seconds passed
... 66%, 83584 KB, 48832 KB/s, 1 seconds passed
... 66%, 83616 KB, 48844 KB/s, 1 seconds passed
... 66%, 83648 KB, 48855 KB/s, 1 seconds passed
... 66%, 83680 KB, 48867 KB/s, 1 seconds passed
... 66%, 83712 KB, 48878 KB/s, 1 seconds passed
... 66%, 83744 KB, 48889 KB/s, 1 seconds passed
... 66%, 83776 KB, 48901 KB/s, 1 seconds passed
... 66%, 83808 KB, 48912 KB/s, 1 seconds passed
... 66%, 83840 KB, 48924 KB/s, 1 seconds passed
... 66%, 83872 KB, 48936 KB/s, 1 seconds passed
... 66%, 83904 KB, 48948 KB/s, 1 seconds passed
... 66%, 83936 KB, 48959 KB/s, 1 seconds passed
... 66%, 83968 KB, 48971 KB/s, 1 seconds passed
... 66%, 84000 KB, 48983 KB/s, 1 seconds passed
... 66%, 84032 KB, 48997 KB/s, 1 seconds passed
... 66%, 84064 KB, 49010 KB/s, 1 seconds passed
... 66%, 84096 KB, 49024 KB/s, 1 seconds passed
... 66%, 84128 KB, 49038 KB/s, 1 seconds passed
... 66%, 84160 KB, 49052 KB/s, 1 seconds passed
... 66%, 84192 KB, 49066 KB/s, 1 seconds passed
... 66%, 84224 KB, 49080 KB/s, 1 seconds passed
... 66%, 84256 KB, 49094 KB/s, 1 seconds passed
... 66%, 84288 KB, 49108 KB/s, 1 seconds passed
... 66%, 84320 KB, 49121 KB/s, 1 seconds passed
... 66%, 84352 KB, 49134 KB/s, 1 seconds passed
... 66%, 84384 KB, 49148 KB/s, 1 seconds passed
... 67%, 84416 KB, 49161 KB/s, 1 seconds passed
... 67%, 84448 KB, 49175 KB/s, 1 seconds passed
... 67%, 84480 KB, 49189 KB/s, 1 seconds passed
... 67%, 84512 KB, 49203 KB/s, 1 seconds passed
... 67%, 84544 KB, 49217 KB/s, 1 seconds passed
... 67%, 84576 KB, 49231 KB/s, 1 seconds passed
... 67%, 84608 KB, 49245 KB/s, 1 seconds passed
... 67%, 84640 KB, 49259 KB/s, 1 seconds passed
... 67%, 84672 KB, 49272 KB/s, 1 seconds passed
... 67%, 84704 KB, 49286 KB/s, 1 seconds passed
... 67%, 84736 KB, 49299 KB/s, 1 seconds passed
... 67%, 84768 KB, 49310 KB/s, 1 seconds passed
... 67%, 84800 KB, 49321 KB/s, 1 seconds passed
... 67%, 84832 KB, 49332 KB/s, 1 seconds passed
... 67%, 84864 KB, 49343 KB/s, 1 seconds passed
... 67%, 84896 KB, 49354 KB/s, 1 seconds passed
... 67%, 84928 KB, 49364 KB/s, 1 seconds passed
... 67%, 84960 KB, 49375 KB/s, 1 seconds passed
... 67%, 84992 KB, 49381 KB/s, 1 seconds passed
... 67%, 85024 KB, 49391 KB/s, 1 seconds passed
... 67%, 85056 KB, 49398 KB/s, 1 seconds passed
... 67%, 85088 KB, 49405 KB/s, 1 seconds passed
... 67%, 85120 KB, 49418 KB/s, 1 seconds passed
... 67%, 85152 KB, 49432 KB/s, 1 seconds passed
... 67%, 85184 KB, 49446 KB/s, 1 seconds passed
... 67%, 85216 KB, 49455 KB/s, 1 seconds passed
... 67%, 85248 KB, 49466 KB/s, 1 seconds passed
... 67%, 85280 KB, 49477 KB/s, 1 seconds passed
... 67%, 85312 KB, 49488 KB/s, 1 seconds passed
... 67%, 85344 KB, 49499 KB/s, 1 seconds passed
... 67%, 85376 KB, 49510 KB/s, 1 seconds passed
... 67%, 85408 KB, 49519 KB/s, 1 seconds passed
... 67%, 85440 KB, 49530 KB/s, 1 seconds passed
... 67%, 85472 KB, 49542 KB/s, 1 seconds passed
... 67%, 85504 KB, 49551 KB/s, 1 seconds passed
... 67%, 85536 KB, 49557 KB/s, 1 seconds passed
... 67%, 85568 KB, 49566 KB/s, 1 seconds passed
... 67%, 85600 KB, 49581 KB/s, 1 seconds passed
... 67%, 85632 KB, 49593 KB/s, 1 seconds passed
... 68%, 85664 KB, 49602 KB/s, 1 seconds passed
... 68%, 85696 KB, 49609 KB/s, 1 seconds passed
... 68%, 85728 KB, 49624 KB/s, 1 seconds passed
... 68%, 85760 KB, 49635 KB/s, 1 seconds passed
... 68%, 85792 KB, 49645 KB/s, 1 seconds passed
... 68%, 85824 KB, 49657 KB/s, 1 seconds passed
... 68%, 85856 KB, 49597 KB/s, 1 seconds passed
... 68%, 85888 KB, 49607 KB/s, 1 seconds passed
... 68%, 85920 KB, 49620 KB/s, 1 seconds passed
... 68%, 85952 KB, 49625 KB/s, 1 seconds passed
... 68%, 85984 KB, 49633 KB/s, 1 seconds passed
... 68%, 86016 KB, 49648 KB/s, 1 seconds passed
... 68%, 86048 KB, 49662 KB/s, 1 seconds passed
... 68%, 86080 KB, 49667 KB/s, 1 seconds passed
... 68%, 86112 KB, 49673 KB/s, 1 seconds passed
... 68%, 86144 KB, 49680 KB/s, 1 seconds passed
... 68%, 86176 KB, 49687 KB/s, 1 seconds passed
... 68%, 86208 KB, 49702 KB/s, 1 seconds passed
... 68%, 86240 KB, 49717 KB/s, 1 seconds passed
... 68%, 86272 KB, 49732 KB/s, 1 seconds passed
... 68%, 86304 KB, 49745 KB/s, 1 seconds passed
... 68%, 86336 KB, 49756 KB/s, 1 seconds passed
... 68%, 86368 KB, 49761 KB/s, 1 seconds passed
... 68%, 86400 KB, 49768 KB/s, 1 seconds passed
... 68%, 86432 KB, 49777 KB/s, 1 seconds passed
... 68%, 86464 KB, 49792 KB/s, 1 seconds passed
... 68%, 86496 KB, 49804 KB/s, 1 seconds passed
... 68%, 86528 KB, 49817 KB/s, 1 seconds passed
... 68%, 86560 KB, 49829 KB/s, 1 seconds passed
... 68%, 86592 KB, 49840 KB/s, 1 seconds passed
... 68%, 86624 KB, 49849 KB/s, 1 seconds passed
... 68%, 86656 KB, 49860 KB/s, 1 seconds passed
... 68%, 86688 KB, 49866 KB/s, 1 seconds passed
... 68%, 86720 KB, 49872 KB/s, 1 seconds passed
... 68%, 86752 KB, 49879 KB/s, 1 seconds passed
... 68%, 86784 KB, 49894 KB/s, 1 seconds passed
... 68%, 86816 KB, 49909 KB/s, 1 seconds passed
... 68%, 86848 KB, 49920 KB/s, 1 seconds passed
... 68%, 86880 KB, 49931 KB/s, 1 seconds passed
... 69%, 86912 KB, 49942 KB/s, 1 seconds passed
... 69%, 86944 KB, 49953 KB/s, 1 seconds passed
... 69%, 86976 KB, 49962 KB/s, 1 seconds passed
... 69%, 87008 KB, 49973 KB/s, 1 seconds passed

.. parsed-literal::

    ... 69%, 87040 KB, 48980 KB/s, 1 seconds passed
... 69%, 87072 KB, 48984 KB/s, 1 seconds passed
... 69%, 87104 KB, 48990 KB/s, 1 seconds passed
... 69%, 87136 KB, 48998 KB/s, 1 seconds passed
... 69%, 87168 KB, 49006 KB/s, 1 seconds passed
... 69%, 87200 KB, 49015 KB/s, 1 seconds passed
... 69%, 87232 KB, 49023 KB/s, 1 seconds passed
... 69%, 87264 KB, 49030 KB/s, 1 seconds passed
... 69%, 87296 KB, 49038 KB/s, 1 seconds passed
... 69%, 87328 KB, 49045 KB/s, 1 seconds passed
... 69%, 87360 KB, 49052 KB/s, 1 seconds passed
... 69%, 87392 KB, 49060 KB/s, 1 seconds passed
... 69%, 87424 KB, 49067 KB/s, 1 seconds passed
... 69%, 87456 KB, 49075 KB/s, 1 seconds passed
... 69%, 87488 KB, 49082 KB/s, 1 seconds passed
... 69%, 87520 KB, 49090 KB/s, 1 seconds passed
... 69%, 87552 KB, 49098 KB/s, 1 seconds passed
... 69%, 87584 KB, 49106 KB/s, 1 seconds passed
... 69%, 87616 KB, 49113 KB/s, 1 seconds passed
... 69%, 87648 KB, 49121 KB/s, 1 seconds passed
... 69%, 87680 KB, 49128 KB/s, 1 seconds passed
... 69%, 87712 KB, 49135 KB/s, 1 seconds passed
... 69%, 87744 KB, 49143 KB/s, 1 seconds passed
... 69%, 87776 KB, 49150 KB/s, 1 seconds passed
... 69%, 87808 KB, 49159 KB/s, 1 seconds passed
... 69%, 87840 KB, 49168 KB/s, 1 seconds passed
... 69%, 87872 KB, 49177 KB/s, 1 seconds passed
... 69%, 87904 KB, 49186 KB/s, 1 seconds passed
... 69%, 87936 KB, 49195 KB/s, 1 seconds passed
... 69%, 87968 KB, 49204 KB/s, 1 seconds passed
... 69%, 88000 KB, 49214 KB/s, 1 seconds passed
... 69%, 88032 KB, 49223 KB/s, 1 seconds passed
... 69%, 88064 KB, 49231 KB/s, 1 seconds passed
... 69%, 88096 KB, 49240 KB/s, 1 seconds passed
... 69%, 88128 KB, 49248 KB/s, 1 seconds passed
... 69%, 88160 KB, 49257 KB/s, 1 seconds passed
... 70%, 88192 KB, 49266 KB/s, 1 seconds passed
... 70%, 88224 KB, 49275 KB/s, 1 seconds passed
... 70%, 88256 KB, 49284 KB/s, 1 seconds passed
... 70%, 88288 KB, 49293 KB/s, 1 seconds passed
... 70%, 88320 KB, 49302 KB/s, 1 seconds passed
... 70%, 88352 KB, 49311 KB/s, 1 seconds passed
... 70%, 88384 KB, 49320 KB/s, 1 seconds passed
... 70%, 88416 KB, 49329 KB/s, 1 seconds passed
... 70%, 88448 KB, 49337 KB/s, 1 seconds passed
... 70%, 88480 KB, 49346 KB/s, 1 seconds passed
... 70%, 88512 KB, 49355 KB/s, 1 seconds passed
... 70%, 88544 KB, 49364 KB/s, 1 seconds passed
... 70%, 88576 KB, 49373 KB/s, 1 seconds passed
... 70%, 88608 KB, 49382 KB/s, 1 seconds passed
... 70%, 88640 KB, 49391 KB/s, 1 seconds passed
... 70%, 88672 KB, 49400 KB/s, 1 seconds passed
... 70%, 88704 KB, 49409 KB/s, 1 seconds passed
... 70%, 88736 KB, 49417 KB/s, 1 seconds passed
... 70%, 88768 KB, 49428 KB/s, 1 seconds passed
... 70%, 88800 KB, 49440 KB/s, 1 seconds passed
... 70%, 88832 KB, 49452 KB/s, 1 seconds passed
... 70%, 88864 KB, 49464 KB/s, 1 seconds passed
... 70%, 88896 KB, 49476 KB/s, 1 seconds passed
... 70%, 88928 KB, 49488 KB/s, 1 seconds passed
... 70%, 88960 KB, 49500 KB/s, 1 seconds passed
... 70%, 88992 KB, 49512 KB/s, 1 seconds passed
... 70%, 89024 KB, 49524 KB/s, 1 seconds passed
... 70%, 89056 KB, 49536 KB/s, 1 seconds passed
... 70%, 89088 KB, 49548 KB/s, 1 seconds passed
... 70%, 89120 KB, 49560 KB/s, 1 seconds passed
... 70%, 89152 KB, 49572 KB/s, 1 seconds passed
... 70%, 89184 KB, 49584 KB/s, 1 seconds passed
... 70%, 89216 KB, 49596 KB/s, 1 seconds passed
... 70%, 89248 KB, 49607 KB/s, 1 seconds passed
... 70%, 89280 KB, 49619 KB/s, 1 seconds passed
... 70%, 89312 KB, 49631 KB/s, 1 seconds passed
... 70%, 89344 KB, 49643 KB/s, 1 seconds passed
... 70%, 89376 KB, 49654 KB/s, 1 seconds passed
... 70%, 89408 KB, 49666 KB/s, 1 seconds passed
... 71%, 89440 KB, 49678 KB/s, 1 seconds passed
... 71%, 89472 KB, 49690 KB/s, 1 seconds passed
... 71%, 89504 KB, 49702 KB/s, 1 seconds passed
... 71%, 89536 KB, 49714 KB/s, 1 seconds passed

.. parsed-literal::

    ... 71%, 89568 KB, 49726 KB/s, 1 seconds passed
... 71%, 89600 KB, 49738 KB/s, 1 seconds passed
... 71%, 89632 KB, 49750 KB/s, 1 seconds passed
... 71%, 89664 KB, 49761 KB/s, 1 seconds passed
... 71%, 89696 KB, 49773 KB/s, 1 seconds passed
... 71%, 89728 KB, 49785 KB/s, 1 seconds passed
... 71%, 89760 KB, 49797 KB/s, 1 seconds passed
... 71%, 89792 KB, 49809 KB/s, 1 seconds passed
... 71%, 89824 KB, 49818 KB/s, 1 seconds passed
... 71%, 89856 KB, 49827 KB/s, 1 seconds passed
... 71%, 89888 KB, 49835 KB/s, 1 seconds passed
... 71%, 89920 KB, 49843 KB/s, 1 seconds passed
... 71%, 89952 KB, 49851 KB/s, 1 seconds passed
... 71%, 89984 KB, 49858 KB/s, 1 seconds passed
... 71%, 90016 KB, 49866 KB/s, 1 seconds passed
... 71%, 90048 KB, 49875 KB/s, 1 seconds passed
... 71%, 90080 KB, 49886 KB/s, 1 seconds passed
... 71%, 90112 KB, 49898 KB/s, 1 seconds passed
... 71%, 90144 KB, 49912 KB/s, 1 seconds passed
... 71%, 90176 KB, 49925 KB/s, 1 seconds passed
... 71%, 90208 KB, 49939 KB/s, 1 seconds passed
... 71%, 90240 KB, 49952 KB/s, 1 seconds passed
... 71%, 90272 KB, 49966 KB/s, 1 seconds passed
... 71%, 90304 KB, 49980 KB/s, 1 seconds passed
... 71%, 90336 KB, 49993 KB/s, 1 seconds passed
... 71%, 90368 KB, 50007 KB/s, 1 seconds passed
... 71%, 90400 KB, 50021 KB/s, 1 seconds passed
... 71%, 90432 KB, 50034 KB/s, 1 seconds passed
... 71%, 90464 KB, 50048 KB/s, 1 seconds passed
... 71%, 90496 KB, 50060 KB/s, 1 seconds passed
... 71%, 90528 KB, 50071 KB/s, 1 seconds passed
... 71%, 90560 KB, 50081 KB/s, 1 seconds passed
... 71%, 90592 KB, 50089 KB/s, 1 seconds passed
... 71%, 90624 KB, 50096 KB/s, 1 seconds passed
... 71%, 90656 KB, 50102 KB/s, 1 seconds passed
... 72%, 90688 KB, 50116 KB/s, 1 seconds passed
... 72%, 90720 KB, 50130 KB/s, 1 seconds passed
... 72%, 90752 KB, 50141 KB/s, 1 seconds passed
... 72%, 90784 KB, 50151 KB/s, 1 seconds passed
... 72%, 90816 KB, 50161 KB/s, 1 seconds passed
... 72%, 90848 KB, 50169 KB/s, 1 seconds passed
... 72%, 90880 KB, 50179 KB/s, 1 seconds passed
... 72%, 90912 KB, 50190 KB/s, 1 seconds passed
... 72%, 90944 KB, 50200 KB/s, 1 seconds passed
... 72%, 90976 KB, 49974 KB/s, 1 seconds passed
... 72%, 91008 KB, 49980 KB/s, 1 seconds passed

.. parsed-literal::

    ... 72%, 91040 KB, 49009 KB/s, 1 seconds passed
... 72%, 91072 KB, 49014 KB/s, 1 seconds passed
... 72%, 91104 KB, 49020 KB/s, 1 seconds passed
... 72%, 91136 KB, 49027 KB/s, 1 seconds passed
... 72%, 91168 KB, 49034 KB/s, 1 seconds passed
... 72%, 91200 KB, 49042 KB/s, 1 seconds passed
... 72%, 91232 KB, 49049 KB/s, 1 seconds passed
... 72%, 91264 KB, 49055 KB/s, 1 seconds passed
... 72%, 91296 KB, 49062 KB/s, 1 seconds passed
... 72%, 91328 KB, 49069 KB/s, 1 seconds passed
... 72%, 91360 KB, 49076 KB/s, 1 seconds passed
... 72%, 91392 KB, 49082 KB/s, 1 seconds passed
... 72%, 91424 KB, 49090 KB/s, 1 seconds passed
... 72%, 91456 KB, 49097 KB/s, 1 seconds passed
... 72%, 91488 KB, 49104 KB/s, 1 seconds passed
... 72%, 91520 KB, 49112 KB/s, 1 seconds passed
... 72%, 91552 KB, 49119 KB/s, 1 seconds passed
... 72%, 91584 KB, 49126 KB/s, 1 seconds passed
... 72%, 91616 KB, 49133 KB/s, 1 seconds passed
... 72%, 91648 KB, 49139 KB/s, 1 seconds passed
... 72%, 91680 KB, 49147 KB/s, 1 seconds passed
... 72%, 91712 KB, 49153 KB/s, 1 seconds passed
... 72%, 91744 KB, 49162 KB/s, 1 seconds passed
... 72%, 91776 KB, 49170 KB/s, 1 seconds passed
... 72%, 91808 KB, 49179 KB/s, 1 seconds passed
... 72%, 91840 KB, 49188 KB/s, 1 seconds passed
... 72%, 91872 KB, 49197 KB/s, 1 seconds passed
... 72%, 91904 KB, 49208 KB/s, 1 seconds passed
... 72%, 91936 KB, 49218 KB/s, 1 seconds passed
... 73%, 91968 KB, 49229 KB/s, 1 seconds passed
... 73%, 92000 KB, 49239 KB/s, 1 seconds passed
... 73%, 92032 KB, 49249 KB/s, 1 seconds passed
... 73%, 92064 KB, 49260 KB/s, 1 seconds passed
... 73%, 92096 KB, 49270 KB/s, 1 seconds passed
... 73%, 92128 KB, 49281 KB/s, 1 seconds passed
... 73%, 92160 KB, 48826 KB/s, 1 seconds passed
... 73%, 92192 KB, 48830 KB/s, 1 seconds passed
... 73%, 92224 KB, 48836 KB/s, 1 seconds passed
... 73%, 92256 KB, 48842 KB/s, 1 seconds passed
... 73%, 92288 KB, 48849 KB/s, 1 seconds passed
... 73%, 92320 KB, 48856 KB/s, 1 seconds passed
... 73%, 92352 KB, 48863 KB/s, 1 seconds passed
... 73%, 92384 KB, 48870 KB/s, 1 seconds passed
... 73%, 92416 KB, 48876 KB/s, 1 seconds passed
... 73%, 92448 KB, 48883 KB/s, 1 seconds passed
... 73%, 92480 KB, 48890 KB/s, 1 seconds passed
... 73%, 92512 KB, 48897 KB/s, 1 seconds passed
... 73%, 92544 KB, 48904 KB/s, 1 seconds passed
... 73%, 92576 KB, 48911 KB/s, 1 seconds passed
... 73%, 92608 KB, 48920 KB/s, 1 seconds passed
... 73%, 92640 KB, 48929 KB/s, 1 seconds passed
... 73%, 92672 KB, 48938 KB/s, 1 seconds passed
... 73%, 92704 KB, 48947 KB/s, 1 seconds passed
... 73%, 92736 KB, 48956 KB/s, 1 seconds passed
... 73%, 92768 KB, 48957 KB/s, 1 seconds passed
... 73%, 92800 KB, 48964 KB/s, 1 seconds passed
... 73%, 92832 KB, 48971 KB/s, 1 seconds passed
... 73%, 92864 KB, 48978 KB/s, 1 seconds passed
... 73%, 92896 KB, 48985 KB/s, 1 seconds passed
... 73%, 92928 KB, 48991 KB/s, 1 seconds passed
... 73%, 92960 KB, 48998 KB/s, 1 seconds passed
... 73%, 92992 KB, 49006 KB/s, 1 seconds passed
... 73%, 93024 KB, 49015 KB/s, 1 seconds passed
... 73%, 93056 KB, 49024 KB/s, 1 seconds passed
... 73%, 93088 KB, 49033 KB/s, 1 seconds passed
... 73%, 93120 KB, 49042 KB/s, 1 seconds passed
... 73%, 93152 KB, 49051 KB/s, 1 seconds passed
... 73%, 93184 KB, 49060 KB/s, 1 seconds passed
... 74%, 93216 KB, 49069 KB/s, 1 seconds passed
... 74%, 93248 KB, 49078 KB/s, 1 seconds passed
... 74%, 93280 KB, 49087 KB/s, 1 seconds passed
... 74%, 93312 KB, 49095 KB/s, 1 seconds passed
... 74%, 93344 KB, 49104 KB/s, 1 seconds passed
... 74%, 93376 KB, 49113 KB/s, 1 seconds passed
... 74%, 93408 KB, 49121 KB/s, 1 seconds passed
... 74%, 93440 KB, 49130 KB/s, 1 seconds passed
... 74%, 93472 KB, 49139 KB/s, 1 seconds passed
... 74%, 93504 KB, 49148 KB/s, 1 seconds passed
... 74%, 93536 KB, 49157 KB/s, 1 seconds passed
... 74%, 93568 KB, 49166 KB/s, 1 seconds passed
... 74%, 93600 KB, 49175 KB/s, 1 seconds passed

.. parsed-literal::

    ... 74%, 93632 KB, 49184 KB/s, 1 seconds passed
... 74%, 93664 KB, 49193 KB/s, 1 seconds passed
... 74%, 93696 KB, 49202 KB/s, 1 seconds passed
... 74%, 93728 KB, 49210 KB/s, 1 seconds passed
... 74%, 93760 KB, 49219 KB/s, 1 seconds passed
... 74%, 93792 KB, 49227 KB/s, 1 seconds passed
... 74%, 93824 KB, 49232 KB/s, 1 seconds passed
... 74%, 93856 KB, 49237 KB/s, 1 seconds passed
... 74%, 93888 KB, 49241 KB/s, 1 seconds passed
... 74%, 93920 KB, 49245 KB/s, 1 seconds passed
... 74%, 93952 KB, 49252 KB/s, 1 seconds passed
... 74%, 93984 KB, 49260 KB/s, 1 seconds passed
... 74%, 94016 KB, 49267 KB/s, 1 seconds passed
... 74%, 94048 KB, 49277 KB/s, 1 seconds passed
... 74%, 94080 KB, 49288 KB/s, 1 seconds passed
... 74%, 94112 KB, 49299 KB/s, 1 seconds passed
... 74%, 94144 KB, 49310 KB/s, 1 seconds passed
... 74%, 94176 KB, 49321 KB/s, 1 seconds passed
... 74%, 94208 KB, 49332 KB/s, 1 seconds passed
... 74%, 94240 KB, 49344 KB/s, 1 seconds passed
... 74%, 94272 KB, 49355 KB/s, 1 seconds passed
... 74%, 94304 KB, 49367 KB/s, 1 seconds passed
... 74%, 94336 KB, 49378 KB/s, 1 seconds passed
... 74%, 94368 KB, 49389 KB/s, 1 seconds passed
... 74%, 94400 KB, 49400 KB/s, 1 seconds passed
... 74%, 94432 KB, 49411 KB/s, 1 seconds passed
... 74%, 94464 KB, 49422 KB/s, 1 seconds passed
... 75%, 94496 KB, 49434 KB/s, 1 seconds passed
... 75%, 94528 KB, 49445 KB/s, 1 seconds passed
... 75%, 94560 KB, 49457 KB/s, 1 seconds passed
... 75%, 94592 KB, 49468 KB/s, 1 seconds passed
... 75%, 94624 KB, 49480 KB/s, 1 seconds passed
... 75%, 94656 KB, 49491 KB/s, 1 seconds passed
... 75%, 94688 KB, 49502 KB/s, 1 seconds passed
... 75%, 94720 KB, 49514 KB/s, 1 seconds passed
... 75%, 94752 KB, 49526 KB/s, 1 seconds passed
... 75%, 94784 KB, 49537 KB/s, 1 seconds passed
... 75%, 94816 KB, 49548 KB/s, 1 seconds passed
... 75%, 94848 KB, 49560 KB/s, 1 seconds passed
... 75%, 94880 KB, 49571 KB/s, 1 seconds passed
... 75%, 94912 KB, 49583 KB/s, 1 seconds passed
... 75%, 94944 KB, 49594 KB/s, 1 seconds passed
... 75%, 94976 KB, 49606 KB/s, 1 seconds passed
... 75%, 95008 KB, 49617 KB/s, 1 seconds passed
... 75%, 95040 KB, 49628 KB/s, 1 seconds passed
... 75%, 95072 KB, 49639 KB/s, 1 seconds passed
... 75%, 95104 KB, 49651 KB/s, 1 seconds passed
... 75%, 95136 KB, 49662 KB/s, 1 seconds passed
... 75%, 95168 KB, 49674 KB/s, 1 seconds passed
... 75%, 95200 KB, 49685 KB/s, 1 seconds passed
... 75%, 95232 KB, 49697 KB/s, 1 seconds passed
... 75%, 95264 KB, 49708 KB/s, 1 seconds passed
... 75%, 95296 KB, 49717 KB/s, 1 seconds passed
... 75%, 95328 KB, 49726 KB/s, 1 seconds passed
... 75%, 95360 KB, 49736 KB/s, 1 seconds passed
... 75%, 95392 KB, 49746 KB/s, 1 seconds passed
... 75%, 95424 KB, 49756 KB/s, 1 seconds passed
... 75%, 95456 KB, 49764 KB/s, 1 seconds passed
... 75%, 95488 KB, 49774 KB/s, 1 seconds passed
... 75%, 95520 KB, 49783 KB/s, 1 seconds passed
... 75%, 95552 KB, 49793 KB/s, 1 seconds passed
... 75%, 95584 KB, 49800 KB/s, 1 seconds passed
... 75%, 95616 KB, 49808 KB/s, 1 seconds passed
... 75%, 95648 KB, 49817 KB/s, 1 seconds passed
... 75%, 95680 KB, 49823 KB/s, 1 seconds passed
... 75%, 95712 KB, 49828 KB/s, 1 seconds passed
... 76%, 95744 KB, 49835 KB/s, 1 seconds passed
... 76%, 95776 KB, 49846 KB/s, 1 seconds passed
... 76%, 95808 KB, 49859 KB/s, 1 seconds passed
... 76%, 95840 KB, 49872 KB/s, 1 seconds passed
... 76%, 95872 KB, 49885 KB/s, 1 seconds passed
... 76%, 95904 KB, 49893 KB/s, 1 seconds passed
... 76%, 95936 KB, 49903 KB/s, 1 seconds passed
... 76%, 95968 KB, 49913 KB/s, 1 seconds passed
... 76%, 96000 KB, 49923 KB/s, 1 seconds passed
... 76%, 96032 KB, 49931 KB/s, 1 seconds passed
... 76%, 96064 KB, 49941 KB/s, 1 seconds passed
... 76%, 96096 KB, 49952 KB/s, 1 seconds passed
... 76%, 96128 KB, 49939 KB/s, 1 seconds passed
... 76%, 96160 KB, 49950 KB/s, 1 seconds passed
... 76%, 96192 KB, 49960 KB/s, 1 seconds passed
... 76%, 96224 KB, 49970 KB/s, 1 seconds passed
... 76%, 96256 KB, 49980 KB/s, 1 seconds passed
... 76%, 96288 KB, 49988 KB/s, 1 seconds passed
... 76%, 96320 KB, 49996 KB/s, 1 seconds passed
... 76%, 96352 KB, 50006 KB/s, 1 seconds passed
... 76%, 96384 KB, 50015 KB/s, 1 seconds passed
... 76%, 96416 KB, 50025 KB/s, 1 seconds passed
... 76%, 96448 KB, 50034 KB/s, 1 seconds passed
... 76%, 96480 KB, 50043 KB/s, 1 seconds passed
... 76%, 96512 KB, 50053 KB/s, 1 seconds passed
... 76%, 96544 KB, 50062 KB/s, 1 seconds passed
... 76%, 96576 KB, 50072 KB/s, 1 seconds passed
... 76%, 96608 KB, 50080 KB/s, 1 seconds passed
... 76%, 96640 KB, 50090 KB/s, 1 seconds passed
... 76%, 96672 KB, 50100 KB/s, 1 seconds passed
... 76%, 96704 KB, 50108 KB/s, 1 seconds passed
... 76%, 96736 KB, 50118 KB/s, 1 seconds passed
... 76%, 96768 KB, 50128 KB/s, 1 seconds passed
... 76%, 96800 KB, 50136 KB/s, 1 seconds passed
... 76%, 96832 KB, 50072 KB/s, 1 seconds passed
... 76%, 96864 KB, 50080 KB/s, 1 seconds passed
... 76%, 96896 KB, 50090 KB/s, 1 seconds passed
... 76%, 96928 KB, 50100 KB/s, 1 seconds passed
... 76%, 96960 KB, 50108 KB/s, 1 seconds passed
... 77%, 96992 KB, 50116 KB/s, 1 seconds passed
... 77%, 97024 KB, 50121 KB/s, 1 seconds passed
... 77%, 97056 KB, 50132 KB/s, 1 seconds passed
... 77%, 97088 KB, 50145 KB/s, 1 seconds passed
... 77%, 97120 KB, 50155 KB/s, 1 seconds passed
... 77%, 97152 KB, 50163 KB/s, 1 seconds passed
... 77%, 97184 KB, 50173 KB/s, 1 seconds passed
... 77%, 97216 KB, 50183 KB/s, 1 seconds passed
... 77%, 97248 KB, 50192 KB/s, 1 seconds passed

.. parsed-literal::

    ... 77%, 97280 KB, 49764 KB/s, 1 seconds passed
... 77%, 97312 KB, 49769 KB/s, 1 seconds passed
... 77%, 97344 KB, 49776 KB/s, 1 seconds passed
... 77%, 97376 KB, 49783 KB/s, 1 seconds passed
... 77%, 97408 KB, 49790 KB/s, 1 seconds passed
... 77%, 97440 KB, 49794 KB/s, 1 seconds passed
... 77%, 97472 KB, 49800 KB/s, 1 seconds passed
... 77%, 97504 KB, 49807 KB/s, 1 seconds passed
... 77%, 97536 KB, 49814 KB/s, 1 seconds passed
... 77%, 97568 KB, 49821 KB/s, 1 seconds passed
... 77%, 97600 KB, 49829 KB/s, 1 seconds passed
... 77%, 97632 KB, 49836 KB/s, 1 seconds passed
... 77%, 97664 KB, 49842 KB/s, 1 seconds passed
... 77%, 97696 KB, 49849 KB/s, 1 seconds passed
... 77%, 97728 KB, 49856 KB/s, 1 seconds passed
... 77%, 97760 KB, 49863 KB/s, 1 seconds passed
... 77%, 97792 KB, 49870 KB/s, 1 seconds passed
... 77%, 97824 KB, 49876 KB/s, 1 seconds passed
... 77%, 97856 KB, 49883 KB/s, 1 seconds passed
... 77%, 97888 KB, 49889 KB/s, 1 seconds passed
... 77%, 97920 KB, 49896 KB/s, 1 seconds passed
... 77%, 97952 KB, 49903 KB/s, 1 seconds passed
... 77%, 97984 KB, 49910 KB/s, 1 seconds passed
... 77%, 98016 KB, 49917 KB/s, 1 seconds passed
... 77%, 98048 KB, 49924 KB/s, 1 seconds passed
... 77%, 98080 KB, 49931 KB/s, 1 seconds passed
... 77%, 98112 KB, 49937 KB/s, 1 seconds passed
... 77%, 98144 KB, 49944 KB/s, 1 seconds passed
... 77%, 98176 KB, 49951 KB/s, 1 seconds passed
... 77%, 98208 KB, 49958 KB/s, 1 seconds passed
... 77%, 98240 KB, 49965 KB/s, 1 seconds passed
... 78%, 98272 KB, 49972 KB/s, 1 seconds passed
... 78%, 98304 KB, 49979 KB/s, 1 seconds passed
... 78%, 98336 KB, 49986 KB/s, 1 seconds passed
... 78%, 98368 KB, 49995 KB/s, 1 seconds passed
... 78%, 98400 KB, 50005 KB/s, 1 seconds passed
... 78%, 98432 KB, 50015 KB/s, 1 seconds passed
... 78%, 98464 KB, 50025 KB/s, 1 seconds passed
... 78%, 98496 KB, 50035 KB/s, 1 seconds passed
... 78%, 98528 KB, 50045 KB/s, 1 seconds passed
... 78%, 98560 KB, 50055 KB/s, 1 seconds passed
... 78%, 98592 KB, 50065 KB/s, 1 seconds passed
... 78%, 98624 KB, 50075 KB/s, 1 seconds passed
... 78%, 98656 KB, 50085 KB/s, 1 seconds passed
... 78%, 98688 KB, 50095 KB/s, 1 seconds passed
... 78%, 98720 KB, 50105 KB/s, 1 seconds passed
... 78%, 98752 KB, 50116 KB/s, 1 seconds passed
... 78%, 98784 KB, 50126 KB/s, 1 seconds passed
... 78%, 98816 KB, 50136 KB/s, 1 seconds passed
... 78%, 98848 KB, 50146 KB/s, 1 seconds passed
... 78%, 98880 KB, 50156 KB/s, 1 seconds passed
... 78%, 98912 KB, 50166 KB/s, 1 seconds passed
... 78%, 98944 KB, 50176 KB/s, 1 seconds passed
... 78%, 98976 KB, 50186 KB/s, 1 seconds passed
... 78%, 99008 KB, 50196 KB/s, 1 seconds passed
... 78%, 99040 KB, 50206 KB/s, 1 seconds passed
... 78%, 99072 KB, 50216 KB/s, 1 seconds passed
... 78%, 99104 KB, 50226 KB/s, 1 seconds passed
... 78%, 99136 KB, 50236 KB/s, 1 seconds passed
... 78%, 99168 KB, 50246 KB/s, 1 seconds passed
... 78%, 99200 KB, 50256 KB/s, 1 seconds passed
... 78%, 99232 KB, 50266 KB/s, 1 seconds passed
... 78%, 99264 KB, 50276 KB/s, 1 seconds passed
... 78%, 99296 KB, 50287 KB/s, 1 seconds passed
... 78%, 99328 KB, 50297 KB/s, 1 seconds passed
... 78%, 99360 KB, 50307 KB/s, 1 seconds passed
... 78%, 99392 KB, 50317 KB/s, 1 seconds passed
... 78%, 99424 KB, 50327 KB/s, 1 seconds passed
... 78%, 99456 KB, 50337 KB/s, 1 seconds passed
... 78%, 99488 KB, 50347 KB/s, 1 seconds passed
... 79%, 99520 KB, 50357 KB/s, 1 seconds passed
... 79%, 99552 KB, 50367 KB/s, 1 seconds passed
... 79%, 99584 KB, 50377 KB/s, 1 seconds passed
... 79%, 99616 KB, 50387 KB/s, 1 seconds passed
... 79%, 99648 KB, 50397 KB/s, 1 seconds passed
... 79%, 99680 KB, 50407 KB/s, 1 seconds passed
... 79%, 99712 KB, 50419 KB/s, 1 seconds passed
... 79%, 99744 KB, 50431 KB/s, 1 seconds passed
... 79%, 99776 KB, 50443 KB/s, 1 seconds passed
... 79%, 99808 KB, 50455 KB/s, 1 seconds passed
... 79%, 99840 KB, 50467 KB/s, 1 seconds passed
... 79%, 99872 KB, 50479 KB/s, 1 seconds passed
... 79%, 99904 KB, 50491 KB/s, 1 seconds passed
... 79%, 99936 KB, 50503 KB/s, 1 seconds passed
... 79%, 99968 KB, 50515 KB/s, 1 seconds passed
... 79%, 100000 KB, 50527 KB/s, 1 seconds passed
... 79%, 100032 KB, 50539 KB/s, 1 seconds passed
... 79%, 100064 KB, 50551 KB/s, 1 seconds passed
... 79%, 100096 KB, 50563 KB/s, 1 seconds passed
... 79%, 100128 KB, 50575 KB/s, 1 seconds passed
... 79%, 100160 KB, 50587 KB/s, 1 seconds passed
... 79%, 100192 KB, 50595 KB/s, 1 seconds passed
... 79%, 100224 KB, 50600 KB/s, 1 seconds passed
... 79%, 100256 KB, 50605 KB/s, 1 seconds passed
... 79%, 100288 KB, 50614 KB/s, 1 seconds passed
... 79%, 100320 KB, 50626 KB/s, 1 seconds passed
... 79%, 100352 KB, 50638 KB/s, 1 seconds passed
... 79%, 100384 KB, 50650 KB/s, 1 seconds passed
... 79%, 100416 KB, 50659 KB/s, 1 seconds passed
... 79%, 100448 KB, 50667 KB/s, 1 seconds passed
... 79%, 100480 KB, 50673 KB/s, 1 seconds passed
... 79%, 100512 KB, 50679 KB/s, 1 seconds passed
... 79%, 100544 KB, 50691 KB/s, 1 seconds passed
... 79%, 100576 KB, 50703 KB/s, 1 seconds passed
... 79%, 100608 KB, 50715 KB/s, 1 seconds passed
... 79%, 100640 KB, 50723 KB/s, 1 seconds passed
... 79%, 100672 KB, 50732 KB/s, 1 seconds passed
... 79%, 100704 KB, 50742 KB/s, 1 seconds passed
... 79%, 100736 KB, 50750 KB/s, 1 seconds passed
... 80%, 100768 KB, 50759 KB/s, 1 seconds passed
... 80%, 100800 KB, 50769 KB/s, 1 seconds passed
... 80%, 100832 KB, 50778 KB/s, 1 seconds passed
... 80%, 100864 KB, 50786 KB/s, 1 seconds passed
... 80%, 100896 KB, 50796 KB/s, 1 seconds passed
... 80%, 100928 KB, 50802 KB/s, 1 seconds passed
... 80%, 100960 KB, 50812 KB/s, 1 seconds passed
... 80%, 100992 KB, 50820 KB/s, 1 seconds passed
... 80%, 101024 KB, 50829 KB/s, 1 seconds passed
... 80%, 101056 KB, 50839 KB/s, 1 seconds passed
... 80%, 101088 KB, 50848 KB/s, 1 seconds passed
... 80%, 101120 KB, 50856 KB/s, 1 seconds passed
... 80%, 101152 KB, 50867 KB/s, 1 seconds passed
... 80%, 101184 KB, 50812 KB/s, 1 seconds passed
... 80%, 101216 KB, 50810 KB/s, 1 seconds passed
... 80%, 101248 KB, 50808 KB/s, 1 seconds passed
... 80%, 101280 KB, 50813 KB/s, 1 seconds passed
... 80%, 101312 KB, 50822 KB/s, 1 seconds passed
... 80%, 101344 KB, 50833 KB/s, 1 seconds passed
... 80%, 101376 KB, 50841 KB/s, 1 seconds passed
... 80%, 101408 KB, 50847 KB/s, 1 seconds passed
... 80%, 101440 KB, 50852 KB/s, 1 seconds passed
... 80%, 101472 KB, 50857 KB/s, 1 seconds passed
... 80%, 101504 KB, 50870 KB/s, 1 seconds passed
... 80%, 101536 KB, 50883 KB/s, 1 seconds passed

.. parsed-literal::

    ... 80%, 101568 KB, 50526 KB/s, 2 seconds passed
... 80%, 101600 KB, 50532 KB/s, 2 seconds passed
... 80%, 101632 KB, 50539 KB/s, 2 seconds passed
... 80%, 101664 KB, 50545 KB/s, 2 seconds passed
... 80%, 101696 KB, 50552 KB/s, 2 seconds passed
... 80%, 101728 KB, 50558 KB/s, 2 seconds passed
... 80%, 101760 KB, 50565 KB/s, 2 seconds passed
... 80%, 101792 KB, 50570 KB/s, 2 seconds passed
... 80%, 101824 KB, 50577 KB/s, 2 seconds passed
... 80%, 101856 KB, 50584 KB/s, 2 seconds passed
... 80%, 101888 KB, 50590 KB/s, 2 seconds passed
... 80%, 101920 KB, 50597 KB/s, 2 seconds passed
... 80%, 101952 KB, 50604 KB/s, 2 seconds passed
... 80%, 101984 KB, 50610 KB/s, 2 seconds passed
... 80%, 102016 KB, 50616 KB/s, 2 seconds passed
... 81%, 102048 KB, 50622 KB/s, 2 seconds passed
... 81%, 102080 KB, 50629 KB/s, 2 seconds passed
... 81%, 102112 KB, 50635 KB/s, 2 seconds passed
... 81%, 102144 KB, 50641 KB/s, 2 seconds passed
... 81%, 102176 KB, 50649 KB/s, 2 seconds passed
... 81%, 102208 KB, 50657 KB/s, 2 seconds passed
... 81%, 102240 KB, 50666 KB/s, 2 seconds passed
... 81%, 102272 KB, 50675 KB/s, 2 seconds passed
... 81%, 102304 KB, 50683 KB/s, 2 seconds passed
... 81%, 102336 KB, 50692 KB/s, 2 seconds passed
... 81%, 102368 KB, 50701 KB/s, 2 seconds passed

.. parsed-literal::

    ... 81%, 102400 KB, 49338 KB/s, 2 seconds passed
... 81%, 102432 KB, 49342 KB/s, 2 seconds passed
... 81%, 102464 KB, 49348 KB/s, 2 seconds passed
... 81%, 102496 KB, 49355 KB/s, 2 seconds passed
... 81%, 102528 KB, 49038 KB/s, 2 seconds passed
... 81%, 102560 KB, 49040 KB/s, 2 seconds passed
... 81%, 102592 KB, 49046 KB/s, 2 seconds passed
... 81%, 102624 KB, 49052 KB/s, 2 seconds passed
... 81%, 102656 KB, 49058 KB/s, 2 seconds passed
... 81%, 102688 KB, 49064 KB/s, 2 seconds passed
... 81%, 102720 KB, 49070 KB/s, 2 seconds passed
... 81%, 102752 KB, 49077 KB/s, 2 seconds passed
... 81%, 102784 KB, 49083 KB/s, 2 seconds passed
... 81%, 102816 KB, 49089 KB/s, 2 seconds passed
... 81%, 102848 KB, 49096 KB/s, 2 seconds passed
... 81%, 102880 KB, 49102 KB/s, 2 seconds passed
... 81%, 102912 KB, 49109 KB/s, 2 seconds passed
... 81%, 102944 KB, 49115 KB/s, 2 seconds passed
... 81%, 102976 KB, 49122 KB/s, 2 seconds passed
... 81%, 103008 KB, 49128 KB/s, 2 seconds passed
... 81%, 103040 KB, 49134 KB/s, 2 seconds passed
... 81%, 103072 KB, 49141 KB/s, 2 seconds passed
... 81%, 103104 KB, 49147 KB/s, 2 seconds passed
... 81%, 103136 KB, 49153 KB/s, 2 seconds passed
... 81%, 103168 KB, 49158 KB/s, 2 seconds passed
... 81%, 103200 KB, 49164 KB/s, 2 seconds passed
... 81%, 103232 KB, 49171 KB/s, 2 seconds passed
... 81%, 103264 KB, 49178 KB/s, 2 seconds passed
... 82%, 103296 KB, 49185 KB/s, 2 seconds passed
... 82%, 103328 KB, 49190 KB/s, 2 seconds passed
... 82%, 103360 KB, 49197 KB/s, 2 seconds passed
... 82%, 103392 KB, 49203 KB/s, 2 seconds passed
... 82%, 103424 KB, 49210 KB/s, 2 seconds passed
... 82%, 103456 KB, 49216 KB/s, 2 seconds passed
... 82%, 103488 KB, 49221 KB/s, 2 seconds passed
... 82%, 103520 KB, 49228 KB/s, 2 seconds passed
... 82%, 103552 KB, 49235 KB/s, 2 seconds passed
... 82%, 103584 KB, 49242 KB/s, 2 seconds passed
... 82%, 103616 KB, 49247 KB/s, 2 seconds passed
... 82%, 103648 KB, 49253 KB/s, 2 seconds passed
... 82%, 103680 KB, 49257 KB/s, 2 seconds passed
... 82%, 103712 KB, 49264 KB/s, 2 seconds passed
... 82%, 103744 KB, 49271 KB/s, 2 seconds passed
... 82%, 103776 KB, 49277 KB/s, 2 seconds passed
... 82%, 103808 KB, 49284 KB/s, 2 seconds passed
... 82%, 103840 KB, 49290 KB/s, 2 seconds passed
... 82%, 103872 KB, 49297 KB/s, 2 seconds passed
... 82%, 103904 KB, 49303 KB/s, 2 seconds passed
... 82%, 103936 KB, 49308 KB/s, 2 seconds passed
... 82%, 103968 KB, 49314 KB/s, 2 seconds passed
... 82%, 104000 KB, 49322 KB/s, 2 seconds passed

.. parsed-literal::

    ... 82%, 104032 KB, 49326 KB/s, 2 seconds passed
... 82%, 104064 KB, 49333 KB/s, 2 seconds passed
... 82%, 104096 KB, 48665 KB/s, 2 seconds passed
... 82%, 104128 KB, 48669 KB/s, 2 seconds passed
... 82%, 104160 KB, 48675 KB/s, 2 seconds passed
... 82%, 104192 KB, 48680 KB/s, 2 seconds passed
... 82%, 104224 KB, 48686 KB/s, 2 seconds passed
... 82%, 104256 KB, 48692 KB/s, 2 seconds passed
... 82%, 104288 KB, 48698 KB/s, 2 seconds passed
... 82%, 104320 KB, 48705 KB/s, 2 seconds passed
... 82%, 104352 KB, 48711 KB/s, 2 seconds passed
... 82%, 104384 KB, 48717 KB/s, 2 seconds passed
... 82%, 104416 KB, 48723 KB/s, 2 seconds passed
... 82%, 104448 KB, 48730 KB/s, 2 seconds passed
... 82%, 104480 KB, 48736 KB/s, 2 seconds passed
... 82%, 104512 KB, 48742 KB/s, 2 seconds passed
... 83%, 104544 KB, 48748 KB/s, 2 seconds passed
... 83%, 104576 KB, 48754 KB/s, 2 seconds passed
... 83%, 104608 KB, 48760 KB/s, 2 seconds passed
... 83%, 104640 KB, 48766 KB/s, 2 seconds passed
... 83%, 104672 KB, 48773 KB/s, 2 seconds passed
... 83%, 104704 KB, 48779 KB/s, 2 seconds passed
... 83%, 104736 KB, 48785 KB/s, 2 seconds passed
... 83%, 104768 KB, 48791 KB/s, 2 seconds passed
... 83%, 104800 KB, 48799 KB/s, 2 seconds passed
... 83%, 104832 KB, 48807 KB/s, 2 seconds passed
... 83%, 104864 KB, 48815 KB/s, 2 seconds passed
... 83%, 104896 KB, 48823 KB/s, 2 seconds passed
... 83%, 104928 KB, 48831 KB/s, 2 seconds passed
... 83%, 104960 KB, 48838 KB/s, 2 seconds passed
... 83%, 104992 KB, 48847 KB/s, 2 seconds passed
... 83%, 105024 KB, 48857 KB/s, 2 seconds passed
... 83%, 105056 KB, 48868 KB/s, 2 seconds passed

.. parsed-literal::

    ... 83%, 105088 KB, 47850 KB/s, 2 seconds passed
... 83%, 105120 KB, 47842 KB/s, 2 seconds passed
... 83%, 105152 KB, 47837 KB/s, 2 seconds passed
... 83%, 105184 KB, 47836 KB/s, 2 seconds passed
... 83%, 105216 KB, 47833 KB/s, 2 seconds passed
... 83%, 105248 KB, 47833 KB/s, 2 seconds passed
... 83%, 105280 KB, 47832 KB/s, 2 seconds passed
... 83%, 105312 KB, 47839 KB/s, 2 seconds passed
... 83%, 105344 KB, 47846 KB/s, 2 seconds passed
... 83%, 105376 KB, 47854 KB/s, 2 seconds passed
... 83%, 105408 KB, 47862 KB/s, 2 seconds passed
... 83%, 105440 KB, 47870 KB/s, 2 seconds passed
... 83%, 105472 KB, 47877 KB/s, 2 seconds passed
... 83%, 105504 KB, 47885 KB/s, 2 seconds passed
... 83%, 105536 KB, 47893 KB/s, 2 seconds passed
... 83%, 105568 KB, 47901 KB/s, 2 seconds passed
... 83%, 105600 KB, 47909 KB/s, 2 seconds passed
... 83%, 105632 KB, 47917 KB/s, 2 seconds passed
... 83%, 105664 KB, 47924 KB/s, 2 seconds passed
... 83%, 105696 KB, 47932 KB/s, 2 seconds passed
... 83%, 105728 KB, 47940 KB/s, 2 seconds passed
... 83%, 105760 KB, 47947 KB/s, 2 seconds passed
... 83%, 105792 KB, 47955 KB/s, 2 seconds passed
... 84%, 105824 KB, 47963 KB/s, 2 seconds passed
... 84%, 105856 KB, 47971 KB/s, 2 seconds passed
... 84%, 105888 KB, 47979 KB/s, 2 seconds passed
... 84%, 105920 KB, 47987 KB/s, 2 seconds passed
... 84%, 105952 KB, 47995 KB/s, 2 seconds passed
... 84%, 105984 KB, 48003 KB/s, 2 seconds passed
... 84%, 106016 KB, 48011 KB/s, 2 seconds passed
... 84%, 106048 KB, 48019 KB/s, 2 seconds passed
... 84%, 106080 KB, 48026 KB/s, 2 seconds passed
... 84%, 106112 KB, 48034 KB/s, 2 seconds passed
... 84%, 106144 KB, 48042 KB/s, 2 seconds passed
... 84%, 106176 KB, 48050 KB/s, 2 seconds passed
... 84%, 106208 KB, 48059 KB/s, 2 seconds passed
... 84%, 106240 KB, 48068 KB/s, 2 seconds passed
... 84%, 106272 KB, 48077 KB/s, 2 seconds passed
... 84%, 106304 KB, 48086 KB/s, 2 seconds passed
... 84%, 106336 KB, 48095 KB/s, 2 seconds passed

.. parsed-literal::

    ... 84%, 106368 KB, 48104 KB/s, 2 seconds passed
... 84%, 106400 KB, 48113 KB/s, 2 seconds passed
... 84%, 106432 KB, 48122 KB/s, 2 seconds passed
... 84%, 106464 KB, 48131 KB/s, 2 seconds passed
... 84%, 106496 KB, 48140 KB/s, 2 seconds passed
... 84%, 106528 KB, 48149 KB/s, 2 seconds passed
... 84%, 106560 KB, 48158 KB/s, 2 seconds passed
... 84%, 106592 KB, 48167 KB/s, 2 seconds passed
... 84%, 106624 KB, 48176 KB/s, 2 seconds passed
... 84%, 106656 KB, 48185 KB/s, 2 seconds passed
... 84%, 106688 KB, 48194 KB/s, 2 seconds passed
... 84%, 106720 KB, 48203 KB/s, 2 seconds passed
... 84%, 106752 KB, 48212 KB/s, 2 seconds passed
... 84%, 106784 KB, 48221 KB/s, 2 seconds passed
... 84%, 106816 KB, 48230 KB/s, 2 seconds passed
... 84%, 106848 KB, 48239 KB/s, 2 seconds passed
... 84%, 106880 KB, 48248 KB/s, 2 seconds passed
... 84%, 106912 KB, 48257 KB/s, 2 seconds passed
... 84%, 106944 KB, 48266 KB/s, 2 seconds passed
... 84%, 106976 KB, 48275 KB/s, 2 seconds passed
... 84%, 107008 KB, 48284 KB/s, 2 seconds passed
... 84%, 107040 KB, 48293 KB/s, 2 seconds passed
... 85%, 107072 KB, 48302 KB/s, 2 seconds passed
... 85%, 107104 KB, 48311 KB/s, 2 seconds passed
... 85%, 107136 KB, 48319 KB/s, 2 seconds passed
... 85%, 107168 KB, 48328 KB/s, 2 seconds passed
... 85%, 107200 KB, 48337 KB/s, 2 seconds passed
... 85%, 107232 KB, 48346 KB/s, 2 seconds passed
... 85%, 107264 KB, 48355 KB/s, 2 seconds passed
... 85%, 107296 KB, 48365 KB/s, 2 seconds passed
... 85%, 107328 KB, 48374 KB/s, 2 seconds passed
... 85%, 107360 KB, 48383 KB/s, 2 seconds passed
... 85%, 107392 KB, 48393 KB/s, 2 seconds passed
... 85%, 107424 KB, 48402 KB/s, 2 seconds passed
... 85%, 107456 KB, 48411 KB/s, 2 seconds passed
... 85%, 107488 KB, 48421 KB/s, 2 seconds passed

.. parsed-literal::

    ... 85%, 107520 KB, 47103 KB/s, 2 seconds passed
... 85%, 107552 KB, 47107 KB/s, 2 seconds passed
... 85%, 107584 KB, 47114 KB/s, 2 seconds passed
... 85%, 107616 KB, 47121 KB/s, 2 seconds passed
... 85%, 107648 KB, 46996 KB/s, 2 seconds passed
... 85%, 107680 KB, 47001 KB/s, 2 seconds passed
... 85%, 107712 KB, 47008 KB/s, 2 seconds passed
... 85%, 107744 KB, 47016 KB/s, 2 seconds passed
... 85%, 107776 KB, 47023 KB/s, 2 seconds passed
... 85%, 107808 KB, 47031 KB/s, 2 seconds passed
... 85%, 107840 KB, 47038 KB/s, 2 seconds passed
... 85%, 107872 KB, 47045 KB/s, 2 seconds passed
... 85%, 107904 KB, 47053 KB/s, 2 seconds passed
... 85%, 107936 KB, 47060 KB/s, 2 seconds passed
... 85%, 107968 KB, 47068 KB/s, 2 seconds passed
... 85%, 108000 KB, 47075 KB/s, 2 seconds passed
... 85%, 108032 KB, 47083 KB/s, 2 seconds passed
... 85%, 108064 KB, 47091 KB/s, 2 seconds passed
... 85%, 108096 KB, 47098 KB/s, 2 seconds passed
... 85%, 108128 KB, 47106 KB/s, 2 seconds passed
... 85%, 108160 KB, 47113 KB/s, 2 seconds passed
... 85%, 108192 KB, 47121 KB/s, 2 seconds passed
... 85%, 108224 KB, 47129 KB/s, 2 seconds passed
... 85%, 108256 KB, 47136 KB/s, 2 seconds passed
... 85%, 108288 KB, 47143 KB/s, 2 seconds passed
... 86%, 108320 KB, 47151 KB/s, 2 seconds passed
... 86%, 108352 KB, 47159 KB/s, 2 seconds passed
... 86%, 108384 KB, 47166 KB/s, 2 seconds passed
... 86%, 108416 KB, 47173 KB/s, 2 seconds passed
... 86%, 108448 KB, 47181 KB/s, 2 seconds passed
... 86%, 108480 KB, 47189 KB/s, 2 seconds passed
... 86%, 108512 KB, 47196 KB/s, 2 seconds passed
... 86%, 108544 KB, 47204 KB/s, 2 seconds passed
... 86%, 108576 KB, 47212 KB/s, 2 seconds passed
... 86%, 108608 KB, 47219 KB/s, 2 seconds passed
... 86%, 108640 KB, 47227 KB/s, 2 seconds passed
... 86%, 108672 KB, 47234 KB/s, 2 seconds passed
... 86%, 108704 KB, 47243 KB/s, 2 seconds passed
... 86%, 108736 KB, 47253 KB/s, 2 seconds passed
... 86%, 108768 KB, 47262 KB/s, 2 seconds passed
... 86%, 108800 KB, 47272 KB/s, 2 seconds passed
... 86%, 108832 KB, 47282 KB/s, 2 seconds passed
... 86%, 108864 KB, 47291 KB/s, 2 seconds passed
... 86%, 108896 KB, 47301 KB/s, 2 seconds passed
... 86%, 108928 KB, 47310 KB/s, 2 seconds passed
... 86%, 108960 KB, 47320 KB/s, 2 seconds passed
... 86%, 108992 KB, 47329 KB/s, 2 seconds passed
... 86%, 109024 KB, 47339 KB/s, 2 seconds passed
... 86%, 109056 KB, 47349 KB/s, 2 seconds passed
... 86%, 109088 KB, 47358 KB/s, 2 seconds passed
... 86%, 109120 KB, 47368 KB/s, 2 seconds passed
... 86%, 109152 KB, 47378 KB/s, 2 seconds passed
... 86%, 109184 KB, 47387 KB/s, 2 seconds passed
... 86%, 109216 KB, 47397 KB/s, 2 seconds passed
... 86%, 109248 KB, 47406 KB/s, 2 seconds passed
... 86%, 109280 KB, 47416 KB/s, 2 seconds passed
... 86%, 109312 KB, 47425 KB/s, 2 seconds passed
... 86%, 109344 KB, 47433 KB/s, 2 seconds passed
... 86%, 109376 KB, 47440 KB/s, 2 seconds passed
... 86%, 109408 KB, 47450 KB/s, 2 seconds passed
... 86%, 109440 KB, 47457 KB/s, 2 seconds passed
... 86%, 109472 KB, 47465 KB/s, 2 seconds passed
... 86%, 109504 KB, 47473 KB/s, 2 seconds passed
... 86%, 109536 KB, 47481 KB/s, 2 seconds passed
... 86%, 109568 KB, 47489 KB/s, 2 seconds passed
... 87%, 109600 KB, 47498 KB/s, 2 seconds passed
... 87%, 109632 KB, 47506 KB/s, 2 seconds passed
... 87%, 109664 KB, 47514 KB/s, 2 seconds passed
... 87%, 109696 KB, 47523 KB/s, 2 seconds passed
... 87%, 109728 KB, 47527 KB/s, 2 seconds passed
... 87%, 109760 KB, 47528 KB/s, 2 seconds passed
... 87%, 109792 KB, 47533 KB/s, 2 seconds passed
... 87%, 109824 KB, 47537 KB/s, 2 seconds passed
... 87%, 109856 KB, 47540 KB/s, 2 seconds passed
... 87%, 109888 KB, 47545 KB/s, 2 seconds passed
... 87%, 109920 KB, 47550 KB/s, 2 seconds passed
... 87%, 109952 KB, 47556 KB/s, 2 seconds passed
... 87%, 109984 KB, 47561 KB/s, 2 seconds passed
... 87%, 110016 KB, 47565 KB/s, 2 seconds passed
... 87%, 110048 KB, 47572 KB/s, 2 seconds passed
... 87%, 110080 KB, 47580 KB/s, 2 seconds passed

.. parsed-literal::

    ... 87%, 110112 KB, 47588 KB/s, 2 seconds passed
... 87%, 110144 KB, 47595 KB/s, 2 seconds passed
... 87%, 110176 KB, 47603 KB/s, 2 seconds passed
... 87%, 110208 KB, 47611 KB/s, 2 seconds passed
... 87%, 110240 KB, 47403 KB/s, 2 seconds passed
... 87%, 110272 KB, 47406 KB/s, 2 seconds passed
... 87%, 110304 KB, 47411 KB/s, 2 seconds passed
... 87%, 110336 KB, 47416 KB/s, 2 seconds passed
... 87%, 110368 KB, 47424 KB/s, 2 seconds passed
... 87%, 110400 KB, 47338 KB/s, 2 seconds passed
... 87%, 110432 KB, 47343 KB/s, 2 seconds passed
... 87%, 110464 KB, 47349 KB/s, 2 seconds passed
... 87%, 110496 KB, 47354 KB/s, 2 seconds passed
... 87%, 110528 KB, 47321 KB/s, 2 seconds passed
... 87%, 110560 KB, 47327 KB/s, 2 seconds passed
... 87%, 110592 KB, 47332 KB/s, 2 seconds passed
... 87%, 110624 KB, 47338 KB/s, 2 seconds passed
... 87%, 110656 KB, 47343 KB/s, 2 seconds passed
... 87%, 110688 KB, 47349 KB/s, 2 seconds passed
... 87%, 110720 KB, 47355 KB/s, 2 seconds passed
... 87%, 110752 KB, 47361 KB/s, 2 seconds passed
... 87%, 110784 KB, 47367 KB/s, 2 seconds passed
... 87%, 110816 KB, 47373 KB/s, 2 seconds passed
... 88%, 110848 KB, 47379 KB/s, 2 seconds passed
... 88%, 110880 KB, 47386 KB/s, 2 seconds passed
... 88%, 110912 KB, 47393 KB/s, 2 seconds passed
... 88%, 110944 KB, 47399 KB/s, 2 seconds passed
... 88%, 110976 KB, 47405 KB/s, 2 seconds passed
... 88%, 111008 KB, 47410 KB/s, 2 seconds passed
... 88%, 111040 KB, 47416 KB/s, 2 seconds passed
... 88%, 111072 KB, 47422 KB/s, 2 seconds passed
... 88%, 111104 KB, 47428 KB/s, 2 seconds passed
... 88%, 111136 KB, 47434 KB/s, 2 seconds passed
... 88%, 111168 KB, 47440 KB/s, 2 seconds passed
... 88%, 111200 KB, 47446 KB/s, 2 seconds passed
... 88%, 111232 KB, 47452 KB/s, 2 seconds passed
... 88%, 111264 KB, 47458 KB/s, 2 seconds passed
... 88%, 111296 KB, 47464 KB/s, 2 seconds passed
... 88%, 111328 KB, 47470 KB/s, 2 seconds passed
... 88%, 111360 KB, 47475 KB/s, 2 seconds passed
... 88%, 111392 KB, 47481 KB/s, 2 seconds passed
... 88%, 111424 KB, 47486 KB/s, 2 seconds passed
... 88%, 111456 KB, 47492 KB/s, 2 seconds passed
... 88%, 111488 KB, 47498 KB/s, 2 seconds passed
... 88%, 111520 KB, 47504 KB/s, 2 seconds passed
... 88%, 111552 KB, 47511 KB/s, 2 seconds passed
... 88%, 111584 KB, 47518 KB/s, 2 seconds passed
... 88%, 111616 KB, 47524 KB/s, 2 seconds passed
... 88%, 111648 KB, 47531 KB/s, 2 seconds passed
... 88%, 111680 KB, 47539 KB/s, 2 seconds passed
... 88%, 111712 KB, 47547 KB/s, 2 seconds passed
... 88%, 111744 KB, 47554 KB/s, 2 seconds passed
... 88%, 111776 KB, 47562 KB/s, 2 seconds passed
... 88%, 111808 KB, 47565 KB/s, 2 seconds passed
... 88%, 111840 KB, 47572 KB/s, 2 seconds passed
... 88%, 111872 KB, 47581 KB/s, 2 seconds passed
... 88%, 111904 KB, 47590 KB/s, 2 seconds passed
... 88%, 111936 KB, 47598 KB/s, 2 seconds passed
... 88%, 111968 KB, 47607 KB/s, 2 seconds passed
... 88%, 112000 KB, 47615 KB/s, 2 seconds passed
... 88%, 112032 KB, 47623 KB/s, 2 seconds passed
... 88%, 112064 KB, 47629 KB/s, 2 seconds passed
... 88%, 112096 KB, 47634 KB/s, 2 seconds passed
... 89%, 112128 KB, 47642 KB/s, 2 seconds passed
... 89%, 112160 KB, 47650 KB/s, 2 seconds passed
... 89%, 112192 KB, 47659 KB/s, 2 seconds passed
... 89%, 112224 KB, 47668 KB/s, 2 seconds passed
... 89%, 112256 KB, 47676 KB/s, 2 seconds passed
... 89%, 112288 KB, 47685 KB/s, 2 seconds passed
... 89%, 112320 KB, 47694 KB/s, 2 seconds passed
... 89%, 112352 KB, 47702 KB/s, 2 seconds passed
... 89%, 112384 KB, 47711 KB/s, 2 seconds passed
... 89%, 112416 KB, 47718 KB/s, 2 seconds passed
... 89%, 112448 KB, 47725 KB/s, 2 seconds passed
... 89%, 112480 KB, 47732 KB/s, 2 seconds passed
... 89%, 112512 KB, 47741 KB/s, 2 seconds passed
... 89%, 112544 KB, 47749 KB/s, 2 seconds passed
... 89%, 112576 KB, 47757 KB/s, 2 seconds passed
... 89%, 112608 KB, 47764 KB/s, 2 seconds passed
... 89%, 112640 KB, 47675 KB/s, 2 seconds passed
... 89%, 112672 KB, 47680 KB/s, 2 seconds passed
... 89%, 112704 KB, 47687 KB/s, 2 seconds passed
... 89%, 112736 KB, 47695 KB/s, 2 seconds passed
... 89%, 112768 KB, 47705 KB/s, 2 seconds passed
... 89%, 112800 KB, 47714 KB/s, 2 seconds passed
... 89%, 112832 KB, 47723 KB/s, 2 seconds passed
... 89%, 112864 KB, 47731 KB/s, 2 seconds passed
... 89%, 112896 KB, 47740 KB/s, 2 seconds passed

.. parsed-literal::

    ... 89%, 112928 KB, 47748 KB/s, 2 seconds passed
... 89%, 112960 KB, 47755 KB/s, 2 seconds passed
... 89%, 112992 KB, 47762 KB/s, 2 seconds passed
... 89%, 113024 KB, 47770 KB/s, 2 seconds passed
... 89%, 113056 KB, 47779 KB/s, 2 seconds passed
... 89%, 113088 KB, 47785 KB/s, 2 seconds passed
... 89%, 113120 KB, 47791 KB/s, 2 seconds passed
... 89%, 113152 KB, 47798 KB/s, 2 seconds passed
... 89%, 113184 KB, 47803 KB/s, 2 seconds passed
... 89%, 113216 KB, 47809 KB/s, 2 seconds passed
... 89%, 113248 KB, 47815 KB/s, 2 seconds passed
... 89%, 113280 KB, 47820 KB/s, 2 seconds passed
... 89%, 113312 KB, 47826 KB/s, 2 seconds passed
... 89%, 113344 KB, 47807 KB/s, 2 seconds passed
... 90%, 113376 KB, 47813 KB/s, 2 seconds passed
... 90%, 113408 KB, 47820 KB/s, 2 seconds passed
... 90%, 113440 KB, 47829 KB/s, 2 seconds passed
... 90%, 113472 KB, 47834 KB/s, 2 seconds passed
... 90%, 113504 KB, 47839 KB/s, 2 seconds passed
... 90%, 113536 KB, 47845 KB/s, 2 seconds passed
... 90%, 113568 KB, 47851 KB/s, 2 seconds passed
... 90%, 113600 KB, 47856 KB/s, 2 seconds passed
... 90%, 113632 KB, 47862 KB/s, 2 seconds passed
... 90%, 113664 KB, 47868 KB/s, 2 seconds passed
... 90%, 113696 KB, 47873 KB/s, 2 seconds passed
... 90%, 113728 KB, 47880 KB/s, 2 seconds passed
... 90%, 113760 KB, 47885 KB/s, 2 seconds passed
... 90%, 113792 KB, 47891 KB/s, 2 seconds passed
... 90%, 113824 KB, 47897 KB/s, 2 seconds passed
... 90%, 113856 KB, 47903 KB/s, 2 seconds passed
... 90%, 113888 KB, 47909 KB/s, 2 seconds passed
... 90%, 113920 KB, 47915 KB/s, 2 seconds passed
... 90%, 113952 KB, 47921 KB/s, 2 seconds passed
... 90%, 113984 KB, 47927 KB/s, 2 seconds passed
... 90%, 114016 KB, 47933 KB/s, 2 seconds passed
... 90%, 114048 KB, 47939 KB/s, 2 seconds passed
... 90%, 114080 KB, 47945 KB/s, 2 seconds passed
... 90%, 114112 KB, 47951 KB/s, 2 seconds passed
... 90%, 114144 KB, 47957 KB/s, 2 seconds passed
... 90%, 114176 KB, 47962 KB/s, 2 seconds passed
... 90%, 114208 KB, 47968 KB/s, 2 seconds passed
... 90%, 114240 KB, 47974 KB/s, 2 seconds passed
... 90%, 114272 KB, 47980 KB/s, 2 seconds passed
... 90%, 114304 KB, 47987 KB/s, 2 seconds passed
... 90%, 114336 KB, 47996 KB/s, 2 seconds passed
... 90%, 114368 KB, 48005 KB/s, 2 seconds passed
... 90%, 114400 KB, 48013 KB/s, 2 seconds passed
... 90%, 114432 KB, 48022 KB/s, 2 seconds passed
... 90%, 114464 KB, 48029 KB/s, 2 seconds passed
... 90%, 114496 KB, 48034 KB/s, 2 seconds passed
... 90%, 114528 KB, 48039 KB/s, 2 seconds passed
... 90%, 114560 KB, 48046 KB/s, 2 seconds passed
... 90%, 114592 KB, 48052 KB/s, 2 seconds passed
... 91%, 114624 KB, 48058 KB/s, 2 seconds passed
... 91%, 114656 KB, 48065 KB/s, 2 seconds passed
... 91%, 114688 KB, 48071 KB/s, 2 seconds passed
... 91%, 114720 KB, 48076 KB/s, 2 seconds passed
... 91%, 114752 KB, 48082 KB/s, 2 seconds passed
... 91%, 114784 KB, 48088 KB/s, 2 seconds passed
... 91%, 114816 KB, 48095 KB/s, 2 seconds passed
... 91%, 114848 KB, 48100 KB/s, 2 seconds passed
... 91%, 114880 KB, 48108 KB/s, 2 seconds passed
... 91%, 114912 KB, 48113 KB/s, 2 seconds passed
... 91%, 114944 KB, 48121 KB/s, 2 seconds passed
... 91%, 114976 KB, 48019 KB/s, 2 seconds passed
... 91%, 115008 KB, 48006 KB/s, 2 seconds passed
... 91%, 115040 KB, 48009 KB/s, 2 seconds passed
... 91%, 115072 KB, 48015 KB/s, 2 seconds passed
... 91%, 115104 KB, 48020 KB/s, 2 seconds passed
... 91%, 115136 KB, 48026 KB/s, 2 seconds passed
... 91%, 115168 KB, 48031 KB/s, 2 seconds passed
... 91%, 115200 KB, 48036 KB/s, 2 seconds passed
... 91%, 115232 KB, 48042 KB/s, 2 seconds passed
... 91%, 115264 KB, 48048 KB/s, 2 seconds passed
... 91%, 115296 KB, 48054 KB/s, 2 seconds passed
... 91%, 115328 KB, 48059 KB/s, 2 seconds passed
... 91%, 115360 KB, 48065 KB/s, 2 seconds passed
... 91%, 115392 KB, 48071 KB/s, 2 seconds passed
... 91%, 115424 KB, 48077 KB/s, 2 seconds passed
... 91%, 115456 KB, 48083 KB/s, 2 seconds passed
... 91%, 115488 KB, 48088 KB/s, 2 seconds passed
... 91%, 115520 KB, 48094 KB/s, 2 seconds passed
... 91%, 115552 KB, 48099 KB/s, 2 seconds passed
... 91%, 115584 KB, 48105 KB/s, 2 seconds passed
... 91%, 115616 KB, 48110 KB/s, 2 seconds passed
... 91%, 115648 KB, 48116 KB/s, 2 seconds passed
... 91%, 115680 KB, 48122 KB/s, 2 seconds passed
... 91%, 115712 KB, 48127 KB/s, 2 seconds passed
... 91%, 115744 KB, 48132 KB/s, 2 seconds passed
... 91%, 115776 KB, 48137 KB/s, 2 seconds passed
... 91%, 115808 KB, 48143 KB/s, 2 seconds passed
... 91%, 115840 KB, 48148 KB/s, 2 seconds passed
... 91%, 115872 KB, 48154 KB/s, 2 seconds passed
... 92%, 115904 KB, 48162 KB/s, 2 seconds passed
... 92%, 115936 KB, 48169 KB/s, 2 seconds passed
... 92%, 115968 KB, 48177 KB/s, 2 seconds passed
... 92%, 116000 KB, 48184 KB/s, 2 seconds passed
... 92%, 116032 KB, 48192 KB/s, 2 seconds passed
... 92%, 116064 KB, 48199 KB/s, 2 seconds passed
... 92%, 116096 KB, 48207 KB/s, 2 seconds passed
... 92%, 116128 KB, 48214 KB/s, 2 seconds passed
... 92%, 116160 KB, 48221 KB/s, 2 seconds passed
... 92%, 116192 KB, 48229 KB/s, 2 seconds passed
... 92%, 116224 KB, 48113 KB/s, 2 seconds passed
... 92%, 116256 KB, 48118 KB/s, 2 seconds passed

.. parsed-literal::

    ... 92%, 116288 KB, 48124 KB/s, 2 seconds passed
... 92%, 116320 KB, 48130 KB/s, 2 seconds passed
... 92%, 116352 KB, 48138 KB/s, 2 seconds passed
... 92%, 116384 KB, 48145 KB/s, 2 seconds passed
... 92%, 116416 KB, 48152 KB/s, 2 seconds passed
... 92%, 116448 KB, 48158 KB/s, 2 seconds passed
... 92%, 116480 KB, 48166 KB/s, 2 seconds passed
... 92%, 116512 KB, 48172 KB/s, 2 seconds passed
... 92%, 116544 KB, 48179 KB/s, 2 seconds passed
... 92%, 116576 KB, 48186 KB/s, 2 seconds passed
... 92%, 116608 KB, 48194 KB/s, 2 seconds passed
... 92%, 116640 KB, 48201 KB/s, 2 seconds passed
... 92%, 116672 KB, 48208 KB/s, 2 seconds passed
... 92%, 116704 KB, 48215 KB/s, 2 seconds passed
... 92%, 116736 KB, 48221 KB/s, 2 seconds passed
... 92%, 116768 KB, 48228 KB/s, 2 seconds passed
... 92%, 116800 KB, 48235 KB/s, 2 seconds passed
... 92%, 116832 KB, 48242 KB/s, 2 seconds passed
... 92%, 116864 KB, 48249 KB/s, 2 seconds passed
... 92%, 116896 KB, 48255 KB/s, 2 seconds passed
... 92%, 116928 KB, 48262 KB/s, 2 seconds passed
... 92%, 116960 KB, 48268 KB/s, 2 seconds passed
... 92%, 116992 KB, 48275 KB/s, 2 seconds passed
... 92%, 117024 KB, 48282 KB/s, 2 seconds passed
... 92%, 117056 KB, 48288 KB/s, 2 seconds passed
... 92%, 117088 KB, 48295 KB/s, 2 seconds passed
... 92%, 117120 KB, 48301 KB/s, 2 seconds passed
... 93%, 117152 KB, 48308 KB/s, 2 seconds passed
... 93%, 117184 KB, 48314 KB/s, 2 seconds passed
... 93%, 117216 KB, 48321 KB/s, 2 seconds passed
... 93%, 117248 KB, 48328 KB/s, 2 seconds passed
... 93%, 117280 KB, 48334 KB/s, 2 seconds passed
... 93%, 117312 KB, 48341 KB/s, 2 seconds passed
... 93%, 117344 KB, 48347 KB/s, 2 seconds passed
... 93%, 117376 KB, 48354 KB/s, 2 seconds passed
... 93%, 117408 KB, 48361 KB/s, 2 seconds passed
... 93%, 117440 KB, 48367 KB/s, 2 seconds passed
... 93%, 117472 KB, 48374 KB/s, 2 seconds passed
... 93%, 117504 KB, 48380 KB/s, 2 seconds passed
... 93%, 117536 KB, 48387 KB/s, 2 seconds passed
... 93%, 117568 KB, 48395 KB/s, 2 seconds passed
... 93%, 117600 KB, 48403 KB/s, 2 seconds passed
... 93%, 117632 KB, 48411 KB/s, 2 seconds passed
... 93%, 117664 KB, 48419 KB/s, 2 seconds passed
... 93%, 117696 KB, 48428 KB/s, 2 seconds passed
... 93%, 117728 KB, 48436 KB/s, 2 seconds passed
... 93%, 117760 KB, 47758 KB/s, 2 seconds passed
... 93%, 117792 KB, 47759 KB/s, 2 seconds passed
... 93%, 117824 KB, 47764 KB/s, 2 seconds passed
... 93%, 117856 KB, 47769 KB/s, 2 seconds passed

.. parsed-literal::

    ... 93%, 117888 KB, 47775 KB/s, 2 seconds passed
... 93%, 117920 KB, 47780 KB/s, 2 seconds passed
... 93%, 117952 KB, 47786 KB/s, 2 seconds passed
... 93%, 117984 KB, 47791 KB/s, 2 seconds passed
... 93%, 118016 KB, 47796 KB/s, 2 seconds passed
... 93%, 118048 KB, 47802 KB/s, 2 seconds passed
... 93%, 118080 KB, 47807 KB/s, 2 seconds passed
... 93%, 118112 KB, 47812 KB/s, 2 seconds passed
... 93%, 118144 KB, 47818 KB/s, 2 seconds passed
... 93%, 118176 KB, 47823 KB/s, 2 seconds passed
... 93%, 118208 KB, 47828 KB/s, 2 seconds passed
... 93%, 118240 KB, 47834 KB/s, 2 seconds passed
... 93%, 118272 KB, 47839 KB/s, 2 seconds passed
... 93%, 118304 KB, 47844 KB/s, 2 seconds passed
... 93%, 118336 KB, 47849 KB/s, 2 seconds passed
... 93%, 118368 KB, 47854 KB/s, 2 seconds passed
... 94%, 118400 KB, 47860 KB/s, 2 seconds passed
... 94%, 118432 KB, 47865 KB/s, 2 seconds passed
... 94%, 118464 KB, 47871 KB/s, 2 seconds passed
... 94%, 118496 KB, 47876 KB/s, 2 seconds passed
... 94%, 118528 KB, 47882 KB/s, 2 seconds passed
... 94%, 118560 KB, 47887 KB/s, 2 seconds passed
... 94%, 118592 KB, 47892 KB/s, 2 seconds passed
... 94%, 118624 KB, 47897 KB/s, 2 seconds passed
... 94%, 118656 KB, 47903 KB/s, 2 seconds passed
... 94%, 118688 KB, 47908 KB/s, 2 seconds passed
... 94%, 118720 KB, 47914 KB/s, 2 seconds passed
... 94%, 118752 KB, 47919 KB/s, 2 seconds passed
... 94%, 118784 KB, 47924 KB/s, 2 seconds passed
... 94%, 118816 KB, 47930 KB/s, 2 seconds passed
... 94%, 118848 KB, 47935 KB/s, 2 seconds passed
... 94%, 118880 KB, 47940 KB/s, 2 seconds passed
... 94%, 118912 KB, 47945 KB/s, 2 seconds passed
... 94%, 118944 KB, 47951 KB/s, 2 seconds passed
... 94%, 118976 KB, 47956 KB/s, 2 seconds passed
... 94%, 119008 KB, 47962 KB/s, 2 seconds passed
... 94%, 119040 KB, 47968 KB/s, 2 seconds passed
... 94%, 119072 KB, 47973 KB/s, 2 seconds passed
... 94%, 119104 KB, 47979 KB/s, 2 seconds passed
... 94%, 119136 KB, 47984 KB/s, 2 seconds passed
... 94%, 119168 KB, 47990 KB/s, 2 seconds passed
... 94%, 119200 KB, 47997 KB/s, 2 seconds passed
... 94%, 119232 KB, 48005 KB/s, 2 seconds passed
... 94%, 119264 KB, 48014 KB/s, 2 seconds passed
... 94%, 119296 KB, 48022 KB/s, 2 seconds passed
... 94%, 119328 KB, 48029 KB/s, 2 seconds passed
... 94%, 119360 KB, 48037 KB/s, 2 seconds passed
... 94%, 119392 KB, 48045 KB/s, 2 seconds passed
... 94%, 119424 KB, 48054 KB/s, 2 seconds passed
... 94%, 119456 KB, 48062 KB/s, 2 seconds passed
... 94%, 119488 KB, 48070 KB/s, 2 seconds passed
... 94%, 119520 KB, 48078 KB/s, 2 seconds passed
... 94%, 119552 KB, 48086 KB/s, 2 seconds passed
... 94%, 119584 KB, 48094 KB/s, 2 seconds passed
... 94%, 119616 KB, 48102 KB/s, 2 seconds passed
... 94%, 119648 KB, 48110 KB/s, 2 seconds passed
... 95%, 119680 KB, 48118 KB/s, 2 seconds passed
... 95%, 119712 KB, 48127 KB/s, 2 seconds passed
... 95%, 119744 KB, 48135 KB/s, 2 seconds passed
... 95%, 119776 KB, 48143 KB/s, 2 seconds passed
... 95%, 119808 KB, 48151 KB/s, 2 seconds passed
... 95%, 119840 KB, 48159 KB/s, 2 seconds passed
... 95%, 119872 KB, 48167 KB/s, 2 seconds passed
... 95%, 119904 KB, 48175 KB/s, 2 seconds passed
... 95%, 119936 KB, 48183 KB/s, 2 seconds passed
... 95%, 119968 KB, 48191 KB/s, 2 seconds passed
... 95%, 120000 KB, 48199 KB/s, 2 seconds passed
... 95%, 120032 KB, 48208 KB/s, 2 seconds passed
... 95%, 120064 KB, 48216 KB/s, 2 seconds passed
... 95%, 120096 KB, 48224 KB/s, 2 seconds passed
... 95%, 120128 KB, 48232 KB/s, 2 seconds passed
... 95%, 120160 KB, 48240 KB/s, 2 seconds passed
... 95%, 120192 KB, 48248 KB/s, 2 seconds passed
... 95%, 120224 KB, 48256 KB/s, 2 seconds passed
... 95%, 120256 KB, 48264 KB/s, 2 seconds passed
... 95%, 120288 KB, 48272 KB/s, 2 seconds passed
... 95%, 120320 KB, 48280 KB/s, 2 seconds passed
... 95%, 120352 KB, 48288 KB/s, 2 seconds passed
... 95%, 120384 KB, 48296 KB/s, 2 seconds passed
... 95%, 120416 KB, 48304 KB/s, 2 seconds passed
... 95%, 120448 KB, 48312 KB/s, 2 seconds passed
... 95%, 120480 KB, 48320 KB/s, 2 seconds passed
... 95%, 120512 KB, 48330 KB/s, 2 seconds passed
... 95%, 120544 KB, 48340 KB/s, 2 seconds passed
... 95%, 120576 KB, 48349 KB/s, 2 seconds passed
... 95%, 120608 KB, 48359 KB/s, 2 seconds passed
... 95%, 120640 KB, 48368 KB/s, 2 seconds passed
... 95%, 120672 KB, 48378 KB/s, 2 seconds passed
... 95%, 120704 KB, 48387 KB/s, 2 seconds passed
... 95%, 120736 KB, 48397 KB/s, 2 seconds passed
... 95%, 120768 KB, 48406 KB/s, 2 seconds passed
... 95%, 120800 KB, 48416 KB/s, 2 seconds passed
... 95%, 120832 KB, 48425 KB/s, 2 seconds passed
... 95%, 120864 KB, 48435 KB/s, 2 seconds passed
... 95%, 120896 KB, 48444 KB/s, 2 seconds passed
... 96%, 120928 KB, 48454 KB/s, 2 seconds passed
... 96%, 120960 KB, 48463 KB/s, 2 seconds passed
... 96%, 120992 KB, 48473 KB/s, 2 seconds passed
... 96%, 121024 KB, 48482 KB/s, 2 seconds passed
... 96%, 121056 KB, 48492 KB/s, 2 seconds passed
... 96%, 121088 KB, 48500 KB/s, 2 seconds passed
... 96%, 121120 KB, 48510 KB/s, 2 seconds passed
... 96%, 121152 KB, 48519 KB/s, 2 seconds passed
... 96%, 121184 KB, 48529 KB/s, 2 seconds passed
... 96%, 121216 KB, 48539 KB/s, 2 seconds passed
... 96%, 121248 KB, 48548 KB/s, 2 seconds passed
... 96%, 121280 KB, 48558 KB/s, 2 seconds passed
... 96%, 121312 KB, 48568 KB/s, 2 seconds passed
... 96%, 121344 KB, 48577 KB/s, 2 seconds passed
... 96%, 121376 KB, 48587 KB/s, 2 seconds passed
... 96%, 121408 KB, 48596 KB/s, 2 seconds passed
... 96%, 121440 KB, 48606 KB/s, 2 seconds passed
... 96%, 121472 KB, 48615 KB/s, 2 seconds passed
... 96%, 121504 KB, 48625 KB/s, 2 seconds passed
... 96%, 121536 KB, 48635 KB/s, 2 seconds passed
... 96%, 121568 KB, 48644 KB/s, 2 seconds passed
... 96%, 121600 KB, 48654 KB/s, 2 seconds passed
... 96%, 121632 KB, 48571 KB/s, 2 seconds passed
... 96%, 121664 KB, 48579 KB/s, 2 seconds passed
... 96%, 121696 KB, 48587 KB/s, 2 seconds passed
... 96%, 121728 KB, 48594 KB/s, 2 seconds passed
... 96%, 121760 KB, 48587 KB/s, 2 seconds passed
... 96%, 121792 KB, 48594 KB/s, 2 seconds passed
... 96%, 121824 KB, 48603 KB/s, 2 seconds passed
... 96%, 121856 KB, 48610 KB/s, 2 seconds passed
... 96%, 121888 KB, 48618 KB/s, 2 seconds passed
... 96%, 121920 KB, 48625 KB/s, 2 seconds passed
... 96%, 121952 KB, 48632 KB/s, 2 seconds passed
... 96%, 121984 KB, 48640 KB/s, 2 seconds passed
... 96%, 122016 KB, 48648 KB/s, 2 seconds passed
... 96%, 122048 KB, 48654 KB/s, 2 seconds passed
... 96%, 122080 KB, 48662 KB/s, 2 seconds passed
... 96%, 122112 KB, 48670 KB/s, 2 seconds passed
... 96%, 122144 KB, 48676 KB/s, 2 seconds passed
... 97%, 122176 KB, 48682 KB/s, 2 seconds passed
... 97%, 122208 KB, 48690 KB/s, 2 seconds passed
... 97%, 122240 KB, 48697 KB/s, 2 seconds passed
... 97%, 122272 KB, 48704 KB/s, 2 seconds passed
... 97%, 122304 KB, 48713 KB/s, 2 seconds passed
... 97%, 122336 KB, 48720 KB/s, 2 seconds passed
... 97%, 122368 KB, 48728 KB/s, 2 seconds passed
... 97%, 122400 KB, 48735 KB/s, 2 seconds passed
... 97%, 122432 KB, 48742 KB/s, 2 seconds passed
... 97%, 122464 KB, 48747 KB/s, 2 seconds passed
... 97%, 122496 KB, 48756 KB/s, 2 seconds passed
... 97%, 122528 KB, 48764 KB/s, 2 seconds passed
... 97%, 122560 KB, 48772 KB/s, 2 seconds passed
... 97%, 122592 KB, 48780 KB/s, 2 seconds passed
... 97%, 122624 KB, 48787 KB/s, 2 seconds passed
... 97%, 122656 KB, 48794 KB/s, 2 seconds passed
... 97%, 122688 KB, 48801 KB/s, 2 seconds passed
... 97%, 122720 KB, 48808 KB/s, 2 seconds passed
... 97%, 122752 KB, 48816 KB/s, 2 seconds passed
... 97%, 122784 KB, 48823 KB/s, 2 seconds passed
... 97%, 122816 KB, 48827 KB/s, 2 seconds passed
... 97%, 122848 KB, 48835 KB/s, 2 seconds passed

.. parsed-literal::

    ... 97%, 122880 KB, 48610 KB/s, 2 seconds passed
... 97%, 122912 KB, 48613 KB/s, 2 seconds passed
... 97%, 122944 KB, 48622 KB/s, 2 seconds passed
... 97%, 122976 KB, 48626 KB/s, 2 seconds passed
... 97%, 123008 KB, 48612 KB/s, 2 seconds passed
... 97%, 123040 KB, 48619 KB/s, 2 seconds passed
... 97%, 123072 KB, 48627 KB/s, 2 seconds passed
... 97%, 123104 KB, 48636 KB/s, 2 seconds passed
... 97%, 123136 KB, 48644 KB/s, 2 seconds passed
... 97%, 123168 KB, 48651 KB/s, 2 seconds passed
... 97%, 123200 KB, 48659 KB/s, 2 seconds passed
... 97%, 123232 KB, 48663 KB/s, 2 seconds passed
... 97%, 123264 KB, 48671 KB/s, 2 seconds passed
... 97%, 123296 KB, 48678 KB/s, 2 seconds passed
... 97%, 123328 KB, 48686 KB/s, 2 seconds passed
... 97%, 123360 KB, 48693 KB/s, 2 seconds passed
... 97%, 123392 KB, 48700 KB/s, 2 seconds passed
... 97%, 123424 KB, 48707 KB/s, 2 seconds passed
... 98%, 123456 KB, 48713 KB/s, 2 seconds passed
... 98%, 123488 KB, 48720 KB/s, 2 seconds passed
... 98%, 123520 KB, 48727 KB/s, 2 seconds passed
... 98%, 123552 KB, 48734 KB/s, 2 seconds passed
... 98%, 123584 KB, 48740 KB/s, 2 seconds passed
... 98%, 123616 KB, 48747 KB/s, 2 seconds passed
... 98%, 123648 KB, 48754 KB/s, 2 seconds passed
... 98%, 123680 KB, 48761 KB/s, 2 seconds passed
... 98%, 123712 KB, 48768 KB/s, 2 seconds passed
... 98%, 123744 KB, 48774 KB/s, 2 seconds passed
... 98%, 123776 KB, 48781 KB/s, 2 seconds passed
... 98%, 123808 KB, 48788 KB/s, 2 seconds passed
... 98%, 123840 KB, 48795 KB/s, 2 seconds passed
... 98%, 123872 KB, 48801 KB/s, 2 seconds passed
... 98%, 123904 KB, 48808 KB/s, 2 seconds passed
... 98%, 123936 KB, 48815 KB/s, 2 seconds passed
... 98%, 123968 KB, 48822 KB/s, 2 seconds passed
... 98%, 124000 KB, 48829 KB/s, 2 seconds passed
... 98%, 124032 KB, 48836 KB/s, 2 seconds passed
... 98%, 124064 KB, 48843 KB/s, 2 seconds passed
... 98%, 124096 KB, 48849 KB/s, 2 seconds passed
... 98%, 124128 KB, 48856 KB/s, 2 seconds passed
... 98%, 124160 KB, 48863 KB/s, 2 seconds passed
... 98%, 124192 KB, 48870 KB/s, 2 seconds passed
... 98%, 124224 KB, 48876 KB/s, 2 seconds passed
... 98%, 124256 KB, 48883 KB/s, 2 seconds passed
... 98%, 124288 KB, 48890 KB/s, 2 seconds passed
... 98%, 124320 KB, 48896 KB/s, 2 seconds passed
... 98%, 124352 KB, 48903 KB/s, 2 seconds passed
... 98%, 124384 KB, 48910 KB/s, 2 seconds passed
... 98%, 124416 KB, 48917 KB/s, 2 seconds passed
... 98%, 124448 KB, 48924 KB/s, 2 seconds passed
... 98%, 124480 KB, 48931 KB/s, 2 seconds passed
... 98%, 124512 KB, 48938 KB/s, 2 seconds passed
... 98%, 124544 KB, 48946 KB/s, 2 seconds passed
... 98%, 124576 KB, 48955 KB/s, 2 seconds passed
... 98%, 124608 KB, 48964 KB/s, 2 seconds passed
... 98%, 124640 KB, 48973 KB/s, 2 seconds passed
... 98%, 124672 KB, 48981 KB/s, 2 seconds passed
... 99%, 124704 KB, 48990 KB/s, 2 seconds passed
... 99%, 124736 KB, 48999 KB/s, 2 seconds passed
... 99%, 124768 KB, 49008 KB/s, 2 seconds passed
... 99%, 124800 KB, 49017 KB/s, 2 seconds passed
... 99%, 124832 KB, 49026 KB/s, 2 seconds passed
... 99%, 124864 KB, 49034 KB/s, 2 seconds passed
... 99%, 124896 KB, 49042 KB/s, 2 seconds passed
... 99%, 124928 KB, 49047 KB/s, 2 seconds passed
... 99%, 124960 KB, 49054 KB/s, 2 seconds passed
... 99%, 124992 KB, 49062 KB/s, 2 seconds passed
... 99%, 125024 KB, 49071 KB/s, 2 seconds passed
... 99%, 125056 KB, 49076 KB/s, 2 seconds passed
... 99%, 125088 KB, 49083 KB/s, 2 seconds passed
... 99%, 125120 KB, 49091 KB/s, 2 seconds passed
... 99%, 125152 KB, 49098 KB/s, 2 seconds passed
... 99%, 125184 KB, 49106 KB/s, 2 seconds passed
... 99%, 125216 KB, 49111 KB/s, 2 seconds passed
... 99%, 125248 KB, 49119 KB/s, 2 seconds passed
... 99%, 125280 KB, 49126 KB/s, 2 seconds passed
... 99%, 125312 KB, 49135 KB/s, 2 seconds passed
... 99%, 125344 KB, 49143 KB/s, 2 seconds passed
... 99%, 125376 KB, 49150 KB/s, 2 seconds passed
... 99%, 125408 KB, 49155 KB/s, 2 seconds passed
... 99%, 125440 KB, 49162 KB/s, 2 seconds passed
... 99%, 125472 KB, 49170 KB/s, 2 seconds passed
... 99%, 125504 KB, 49176 KB/s, 2 seconds passed
... 99%, 125536 KB, 49184 KB/s, 2 seconds passed
... 99%, 125568 KB, 49191 KB/s, 2 seconds passed
... 99%, 125600 KB, 49199 KB/s, 2 seconds passed
... 99%, 125632 KB, 49206 KB/s, 2 seconds passed
... 99%, 125664 KB, 49214 KB/s, 2 seconds passed
... 99%, 125696 KB, 49221 KB/s, 2 seconds passed
... 99%, 125728 KB, 49228 KB/s, 2 seconds passed
... 99%, 125760 KB, 49235 KB/s, 2 seconds passed
... 99%, 125792 KB, 49242 KB/s, 2 seconds passed
... 99%, 125824 KB, 49249 KB/s, 2 seconds passed
... 99%, 125856 KB, 49257 KB/s, 2 seconds passed
... 99%, 125888 KB, 49263 KB/s, 2 seconds passed
... 99%, 125920 KB, 49269 KB/s, 2 seconds passed
... 99%, 125952 KB, 49279 KB/s, 2 seconds passed
... 100%, 125953 KB, 49276 KB/s, 2 seconds passed



.. parsed-literal::


    ========== Downloading models/public/colorization-v2/model/__init__.py


.. parsed-literal::

    ... 100%, 0 KB, 305 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/base_color.py


.. parsed-literal::

    ... 100%, 0 KB, 1751 KB/s, 0 seconds passed


    ========== Downloading models/public/colorization-v2/model/eccv16.py


.. parsed-literal::

    ... 100%, 4 KB, 17511 KB/s, 0 seconds passed


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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=models/public/colorization-v2 --model-name=ECCVGenerator --weights=models/public/colorization-v2/ckpt/colorization-v2-eccv16.pth --import-module=model --input-shape=1,1,256,256 --output-file=models/public/colorization-v2/colorization-v2-eccv16.onnx --input-names=data_l --output-names=color_ab



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting colorization-v2 to IR (FP16)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=models/public/colorization-v2/FP16 --model_name=colorization-v2 --input=data_l --output=color_ab --input_model=models/public/colorization-v2/colorization-v2-eccv16.onnx '--layout=data_l(NCHW)' '--input_shape=[1, 1, 256, 256]' --compress_to_fp16=True



.. parsed-literal::

    [ INFO ] Generated IR will be compressed to FP16. If you get lower accuracy, please consider disabling compression explicitly by adding argument --compress_to_fp16=False.
    Find more information about compression to FP16 at https://docs.openvino.ai/2023.0/openvino_docs_MO_DG_FP16_Compression.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html


.. parsed-literal::

    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/222-vision-image-colorization/models/public/colorization-v2/FP16/colorization-v2.bin



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

