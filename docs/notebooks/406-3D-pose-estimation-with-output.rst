Live 3D Human Pose Estimation with OpenVINO
===========================================

This notebook demonstrates live 3D Human Pose Estimation with OpenVINO
via a webcam. We utilize the model
`human-pose-estimation-3d-0001 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__. At the end
of this notebook, you will see live inference results from your webcam
(if available). Alternatively, you can also upload a video file to test
out the algorithms. **Make sure you have properly installed
the**\ `Jupyter
extension <https://github.com/jupyter-widgets/pythreejs#jupyterlab>`__\ **and
been using JupyterLab to run the demo as suggested in the
``README.md``**

   **NOTE**: *To use a webcam, you must run this Jupyter notebook on a
   computer with a webcam. If you run on a remote server, the webcam
   will not work. However, you can still do inference on a video file in
   the final step. This demo utilizes the Python interface in
   ``Three.js`` integrated with WebGL to process data from the model
   inference. These results are processed and displayed in the
   notebook.*

*To ensure that the results are displayed correctly, run the code in a
recommended browser on one of the following operating systems:* *Ubuntu,
Windows: Chrome* *macOS: Safari*

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Imports <#imports>`__
-  `The model <#the-model>`__

   -  `Download the model <#download-the-model>`__
   -  `Convert Model to OpenVINO IR
      format <#convert-model-to-openvino-ir-format>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Load the model <#load-the-model>`__

-  `Processing <#processing>`__

   -  `Model Inference <#model-inference>`__
   -  `Draw 2D Pose Overlays <#draw-2d-pose-overlays>`__
   -  `Main Processing Function <#main-processing-function>`__

-  `Run <#run>`__

Prerequisites
-------------



**The ``pythreejs`` extension may not display properly when using the
latest Jupyter Notebook release (2.4.1). Therefore, it is recommended to
use Jupyter Lab instead.**

.. code:: ipython3

    %pip install pythreejs "openvino-dev>=2023.1.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)


.. parsed-literal::

    Requirement already satisfied: openvino-dev>=2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2023.3.0)
    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.2)


.. parsed-literal::

    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (1.23.5)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.14.1)


.. parsed-literal::

    Requirement already satisfied: addict>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.4.0)
    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (0.7.1)
    Requirement already satisfied: jstyleson>=0.0.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (0.0.2)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.8.8)
    Requirement already satisfied: opencv-python in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (4.9.0.80)
    Requirement already satisfied: openvino-telemetry>=2022.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2023.2.1)
    Requirement already satisfied: pillow>=8.1.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (10.2.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2.31.0)
    Requirement already satisfied: scipy>=1.8 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (1.10.1)
    Requirement already satisfied: texttable>=1.6.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (1.7.0)
    Requirement already satisfied: tqdm>=4.54.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (4.66.1)
    Requirement already satisfied: openvino==2023.3.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2023.1.0) (2023.3.0)


.. parsed-literal::

    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)


.. parsed-literal::

    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.2.1)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.10)
    Requirement already satisfied: jupyterlab-widgets~=3.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.10)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (3.3.2)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (2.2.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2023.1.0) (2024.2.2)


.. parsed-literal::

    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.43)
    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.17.2)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)


.. parsed-literal::

    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.13)
    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)


.. parsed-literal::

    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Installing collected packages: traittypes, ipydatawidgets, pythreejs


.. parsed-literal::

    Successfully installed ipydatawidgets-4.3.5 pythreejs-2.4.2 traittypes-0.2.1


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Imports
-------



.. code:: ipython3

    import collections
    import sys
    import time
    from pathlib import Path

    import cv2
    import ipywidgets as widgets
    import numpy as np
    from IPython.display import clear_output, display
    import openvino as ov

    sys.path.append("../utils")
    import notebook_utils as utils

    sys.path.append("./engine")
    import engine.engine3js as engine
    from engine.parse_poses import parse_poses

The model
---------



Download the model
~~~~~~~~~~~~~~~~~~



We use ``omz_downloader``, which is a command line tool from the
``openvino-dev`` package. ``omz_downloader`` automatically creates a
directory structure and downloads the selected model.

.. code:: ipython3

    # directory where model will be downloaded
    base_model_dir = "model"

    # model name as named in Open Model Zoo
    model_name = "human-pose-estimation-3d-0001"
    # selected precision (FP32, FP16)
    precision = "FP32"

    BASE_MODEL_NAME = f"{base_model_dir}/public/{model_name}/{model_name}"
    model_path = Path(BASE_MODEL_NAME).with_suffix(".pth")
    onnx_path = Path(BASE_MODEL_NAME).with_suffix(".onnx")

    ir_model_path = f"model/public/{model_name}/{precision}/{model_name}.xml"
    model_weights_path = f"model/public/{model_name}/{precision}/{model_name}.bin"

    if not model_path.exists():
        download_command = (
            f"omz_downloader " f"--name {model_name} " f"--output_dir {base_model_dir}"
        )
        ! $download_command


.. parsed-literal::

    ################|| Downloading human-pose-estimation-3d-0001 ||################

    ========== Downloading model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


.. parsed-literal::

    ... 0%, 32 KB, 1458 KB/s, 0 seconds passed
... 0%, 64 KB, 1181 KB/s, 0 seconds passed
... 0%, 96 KB, 1748 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 128 KB, 1479 KB/s, 0 seconds passed
... 0%, 160 KB, 1828 KB/s, 0 seconds passed

.. parsed-literal::

    ... 1%, 192 KB, 1614 KB/s, 0 seconds passed
... 1%, 224 KB, 1869 KB/s, 0 seconds passed
... 1%, 256 KB, 2127 KB/s, 0 seconds passed
... 1%, 288 KB, 2035 KB/s, 0 seconds passed
... 1%, 320 KB, 2113 KB/s, 0 seconds passed
... 1%, 352 KB, 2312 KB/s, 0 seconds passed
... 2%, 384 KB, 2515 KB/s, 0 seconds passed

.. parsed-literal::

    ... 2%, 416 KB, 2393 KB/s, 0 seconds passed
... 2%, 448 KB, 2436 KB/s, 0 seconds passed
... 2%, 480 KB, 2598 KB/s, 0 seconds passed
... 2%, 512 KB, 2764 KB/s, 0 seconds passed
... 3%, 544 KB, 2635 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 576 KB, 2664 KB/s, 0 seconds passed
... 3%, 608 KB, 2800 KB/s, 0 seconds passed
... 3%, 640 KB, 2942 KB/s, 0 seconds passed
... 3%, 672 KB, 3083 KB/s, 0 seconds passed
... 3%, 704 KB, 2833 KB/s, 0 seconds passed
... 4%, 736 KB, 2952 KB/s, 0 seconds passed
... 4%, 768 KB, 3072 KB/s, 0 seconds passed

.. parsed-literal::

    ... 4%, 800 KB, 2884 KB/s, 0 seconds passed
... 4%, 832 KB, 2962 KB/s, 0 seconds passed
... 4%, 864 KB, 3063 KB/s, 0 seconds passed
... 4%, 896 KB, 3171 KB/s, 0 seconds passed
... 5%, 928 KB, 2993 KB/s, 0 seconds passed
... 5%, 960 KB, 3065 KB/s, 0 seconds passed
... 5%, 992 KB, 3157 KB/s, 0 seconds passed
... 5%, 1024 KB, 3251 KB/s, 0 seconds passed

.. parsed-literal::

    ... 5%, 1056 KB, 3085 KB/s, 0 seconds passed
... 6%, 1088 KB, 3147 KB/s, 0 seconds passed
... 6%, 1120 KB, 3232 KB/s, 0 seconds passed
... 6%, 1152 KB, 3316 KB/s, 0 seconds passed

.. parsed-literal::

    ... 6%, 1184 KB, 3159 KB/s, 0 seconds passed
... 6%, 1216 KB, 3215 KB/s, 0 seconds passed
... 6%, 1248 KB, 3291 KB/s, 0 seconds passed
... 7%, 1280 KB, 3369 KB/s, 0 seconds passed
... 7%, 1312 KB, 3223 KB/s, 0 seconds passed
... 7%, 1344 KB, 3274 KB/s, 0 seconds passed
... 7%, 1376 KB, 3343 KB/s, 0 seconds passed
... 7%, 1408 KB, 3415 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1440 KB, 3275 KB/s, 0 seconds passed
... 8%, 1472 KB, 3323 KB/s, 0 seconds passed
... 8%, 1504 KB, 3388 KB/s, 0 seconds passed
... 8%, 1536 KB, 3454 KB/s, 0 seconds passed

.. parsed-literal::

    ... 8%, 1568 KB, 3322 KB/s, 0 seconds passed
... 8%, 1600 KB, 3367 KB/s, 0 seconds passed
... 9%, 1632 KB, 3426 KB/s, 0 seconds passed
... 9%, 1664 KB, 3488 KB/s, 0 seconds passed
... 9%, 1696 KB, 3363 KB/s, 0 seconds passed
... 9%, 1728 KB, 3402 KB/s, 0 seconds passed
... 9%, 1760 KB, 3457 KB/s, 0 seconds passed
... 9%, 1792 KB, 3516 KB/s, 0 seconds passed

.. parsed-literal::

    ... 10%, 1824 KB, 3399 KB/s, 0 seconds passed
... 10%, 1856 KB, 3436 KB/s, 0 seconds passed
... 10%, 1888 KB, 3487 KB/s, 0 seconds passed
... 10%, 1920 KB, 3540 KB/s, 0 seconds passed
... 10%, 1952 KB, 3428 KB/s, 0 seconds passed

.. parsed-literal::

    ... 11%, 1984 KB, 3462 KB/s, 0 seconds passed
... 11%, 2016 KB, 3514 KB/s, 0 seconds passed
... 11%, 2048 KB, 3563 KB/s, 0 seconds passed
... 11%, 2080 KB, 3459 KB/s, 0 seconds passed
... 11%, 2112 KB, 3493 KB/s, 0 seconds passed
... 11%, 2144 KB, 3537 KB/s, 0 seconds passed
... 12%, 2176 KB, 3584 KB/s, 0 seconds passed

.. parsed-literal::

    ... 12%, 2208 KB, 3483 KB/s, 0 seconds passed
... 12%, 2240 KB, 3516 KB/s, 0 seconds passed
... 12%, 2272 KB, 3557 KB/s, 0 seconds passed
... 12%, 2304 KB, 3600 KB/s, 0 seconds passed
... 12%, 2336 KB, 3502 KB/s, 0 seconds passed
... 13%, 2368 KB, 3538 KB/s, 0 seconds passed
... 13%, 2400 KB, 3576 KB/s, 0 seconds passed
... 13%, 2432 KB, 3615 KB/s, 0 seconds passed

.. parsed-literal::

    ... 13%, 2464 KB, 3523 KB/s, 0 seconds passed
... 13%, 2496 KB, 3555 KB/s, 0 seconds passed
... 14%, 2528 KB, 3591 KB/s, 0 seconds passed
... 14%, 2560 KB, 3628 KB/s, 0 seconds passed

.. parsed-literal::

    ... 14%, 2592 KB, 3540 KB/s, 0 seconds passed
... 14%, 2624 KB, 3568 KB/s, 0 seconds passed
... 14%, 2656 KB, 3605 KB/s, 0 seconds passed
... 14%, 2688 KB, 3641 KB/s, 0 seconds passed
... 15%, 2720 KB, 3557 KB/s, 0 seconds passed
... 15%, 2752 KB, 3584 KB/s, 0 seconds passed
... 15%, 2784 KB, 3622 KB/s, 0 seconds passed
... 15%, 2816 KB, 3655 KB/s, 0 seconds passed

.. parsed-literal::

    ... 15%, 2848 KB, 3573 KB/s, 0 seconds passed
... 16%, 2880 KB, 3600 KB/s, 0 seconds passed
... 16%, 2912 KB, 3635 KB/s, 0 seconds passed
... 16%, 2944 KB, 3664 KB/s, 0 seconds passed

.. parsed-literal::

    ... 16%, 2976 KB, 3583 KB/s, 0 seconds passed
... 16%, 3008 KB, 3611 KB/s, 0 seconds passed
... 16%, 3040 KB, 3645 KB/s, 0 seconds passed
... 17%, 3072 KB, 3676 KB/s, 0 seconds passed
... 17%, 3104 KB, 3602 KB/s, 0 seconds passed
... 17%, 3136 KB, 3625 KB/s, 0 seconds passed
... 17%, 3168 KB, 3657 KB/s, 0 seconds passed
... 17%, 3200 KB, 3684 KB/s, 0 seconds passed

.. parsed-literal::

    ... 17%, 3232 KB, 3611 KB/s, 0 seconds passed
... 18%, 3264 KB, 3634 KB/s, 0 seconds passed
... 18%, 3296 KB, 3666 KB/s, 0 seconds passed
... 18%, 3328 KB, 3694 KB/s, 0 seconds passed
... 18%, 3360 KB, 3626 KB/s, 0 seconds passed
... 18%, 3392 KB, 3646 KB/s, 0 seconds passed

.. parsed-literal::

    ... 19%, 3424 KB, 3676 KB/s, 0 seconds passed
... 19%, 3456 KB, 3701 KB/s, 0 seconds passed
... 19%, 3488 KB, 3630 KB/s, 0 seconds passed
... 19%, 3520 KB, 3654 KB/s, 0 seconds passed
... 19%, 3552 KB, 3685 KB/s, 0 seconds passed
... 19%, 3584 KB, 3710 KB/s, 0 seconds passed

.. parsed-literal::

    ... 20%, 3616 KB, 3645 KB/s, 0 seconds passed
... 20%, 3648 KB, 3666 KB/s, 0 seconds passed
... 20%, 3680 KB, 3693 KB/s, 0 seconds passed
... 20%, 3712 KB, 3717 KB/s, 0 seconds passed
... 20%, 3744 KB, 3651 KB/s, 1 seconds passed
... 20%, 3776 KB, 3678 KB/s, 1 seconds passed
... 21%, 3808 KB, 3701 KB/s, 1 seconds passed
... 21%, 3840 KB, 3728 KB/s, 1 seconds passed

.. parsed-literal::

    ... 21%, 3872 KB, 3659 KB/s, 1 seconds passed
... 21%, 3904 KB, 3680 KB/s, 1 seconds passed
... 21%, 3936 KB, 3708 KB/s, 1 seconds passed
... 22%, 3968 KB, 3730 KB/s, 1 seconds passed

.. parsed-literal::

    ... 22%, 4000 KB, 3668 KB/s, 1 seconds passed
... 22%, 4032 KB, 3691 KB/s, 1 seconds passed
... 22%, 4064 KB, 3715 KB/s, 1 seconds passed
... 22%, 4096 KB, 3738 KB/s, 1 seconds passed
... 22%, 4128 KB, 3674 KB/s, 1 seconds passed
... 23%, 4160 KB, 3698 KB/s, 1 seconds passed
... 23%, 4192 KB, 3722 KB/s, 1 seconds passed
... 23%, 4224 KB, 3742 KB/s, 1 seconds passed

.. parsed-literal::

    ... 23%, 4256 KB, 3681 KB/s, 1 seconds passed
... 23%, 4288 KB, 3704 KB/s, 1 seconds passed
... 24%, 4320 KB, 3727 KB/s, 1 seconds passed
... 24%, 4352 KB, 3748 KB/s, 1 seconds passed

.. parsed-literal::

    ... 24%, 4384 KB, 3689 KB/s, 1 seconds passed
... 24%, 4416 KB, 3711 KB/s, 1 seconds passed
... 24%, 4448 KB, 3732 KB/s, 1 seconds passed
... 24%, 4480 KB, 3752 KB/s, 1 seconds passed
... 25%, 4512 KB, 3694 KB/s, 1 seconds passed
... 25%, 4544 KB, 3715 KB/s, 1 seconds passed
... 25%, 4576 KB, 3739 KB/s, 1 seconds passed
... 25%, 4608 KB, 3758 KB/s, 1 seconds passed

.. parsed-literal::

    ... 25%, 4640 KB, 3702 KB/s, 1 seconds passed
... 25%, 4672 KB, 3722 KB/s, 1 seconds passed
... 26%, 4704 KB, 3745 KB/s, 1 seconds passed
... 26%, 4736 KB, 3696 KB/s, 1 seconds passed
... 26%, 4768 KB, 3707 KB/s, 1 seconds passed
... 26%, 4800 KB, 3726 KB/s, 1 seconds passed
... 26%, 4832 KB, 3749 KB/s, 1 seconds passed

.. parsed-literal::

    ... 27%, 4864 KB, 3767 KB/s, 1 seconds passed
... 27%, 4896 KB, 3714 KB/s, 1 seconds passed
... 27%, 4928 KB, 3732 KB/s, 1 seconds passed
... 27%, 4960 KB, 3754 KB/s, 1 seconds passed

.. parsed-literal::

    ... 27%, 4992 KB, 3707 KB/s, 1 seconds passed
... 27%, 5024 KB, 3718 KB/s, 1 seconds passed
... 28%, 5056 KB, 3740 KB/s, 1 seconds passed
... 28%, 5088 KB, 3759 KB/s, 1 seconds passed
... 28%, 5120 KB, 3778 KB/s, 1 seconds passed
... 28%, 5152 KB, 3722 KB/s, 1 seconds passed
... 28%, 5184 KB, 3741 KB/s, 1 seconds passed
... 28%, 5216 KB, 3762 KB/s, 1 seconds passed
... 29%, 5248 KB, 3780 KB/s, 1 seconds passed

.. parsed-literal::

    ... 29%, 5280 KB, 3739 KB/s, 1 seconds passed
... 29%, 5312 KB, 3748 KB/s, 1 seconds passed
... 29%, 5344 KB, 3767 KB/s, 1 seconds passed

.. parsed-literal::

    ... 29%, 5376 KB, 3722 KB/s, 1 seconds passed
... 30%, 5408 KB, 3732 KB/s, 1 seconds passed
... 30%, 5440 KB, 3751 KB/s, 1 seconds passed
... 30%, 5472 KB, 3770 KB/s, 1 seconds passed
... 30%, 5504 KB, 3728 KB/s, 1 seconds passed
... 30%, 5536 KB, 3737 KB/s, 1 seconds passed
... 30%, 5568 KB, 3755 KB/s, 1 seconds passed
... 31%, 5600 KB, 3774 KB/s, 1 seconds passed

.. parsed-literal::

    ... 31%, 5632 KB, 3732 KB/s, 1 seconds passed
... 31%, 5664 KB, 3741 KB/s, 1 seconds passed
... 31%, 5696 KB, 3759 KB/s, 1 seconds passed
... 31%, 5728 KB, 3778 KB/s, 1 seconds passed
... 32%, 5760 KB, 3735 KB/s, 1 seconds passed
... 32%, 5792 KB, 3745 KB/s, 1 seconds passed

.. parsed-literal::

    ... 32%, 5824 KB, 3762 KB/s, 1 seconds passed
... 32%, 5856 KB, 3780 KB/s, 1 seconds passed
... 32%, 5888 KB, 3738 KB/s, 1 seconds passed
... 32%, 5920 KB, 3749 KB/s, 1 seconds passed
... 33%, 5952 KB, 3767 KB/s, 1 seconds passed
... 33%, 5984 KB, 3785 KB/s, 1 seconds passed

.. parsed-literal::

    ... 33%, 6016 KB, 3743 KB/s, 1 seconds passed
... 33%, 6048 KB, 3752 KB/s, 1 seconds passed
... 33%, 6080 KB, 3769 KB/s, 1 seconds passed
... 33%, 6112 KB, 3786 KB/s, 1 seconds passed
... 34%, 6144 KB, 3747 KB/s, 1 seconds passed
... 34%, 6176 KB, 3757 KB/s, 1 seconds passed
... 34%, 6208 KB, 3773 KB/s, 1 seconds passed
... 34%, 6240 KB, 3791 KB/s, 1 seconds passed

.. parsed-literal::

    ... 34%, 6272 KB, 3754 KB/s, 1 seconds passed
... 35%, 6304 KB, 3766 KB/s, 1 seconds passed
... 35%, 6336 KB, 3778 KB/s, 1 seconds passed
... 35%, 6368 KB, 3794 KB/s, 1 seconds passed

.. parsed-literal::

    ... 35%, 6400 KB, 3758 KB/s, 1 seconds passed
... 35%, 6432 KB, 3769 KB/s, 1 seconds passed
... 35%, 6464 KB, 3780 KB/s, 1 seconds passed
... 36%, 6496 KB, 3795 KB/s, 1 seconds passed
... 36%, 6528 KB, 3758 KB/s, 1 seconds passed
... 36%, 6560 KB, 3767 KB/s, 1 seconds passed
... 36%, 6592 KB, 3782 KB/s, 1 seconds passed
... 36%, 6624 KB, 3797 KB/s, 1 seconds passed

.. parsed-literal::

    ... 36%, 6656 KB, 3761 KB/s, 1 seconds passed
... 37%, 6688 KB, 3770 KB/s, 1 seconds passed
... 37%, 6720 KB, 3785 KB/s, 1 seconds passed
... 37%, 6752 KB, 3800 KB/s, 1 seconds passed
... 37%, 6784 KB, 3764 KB/s, 1 seconds passed

.. parsed-literal::

    ... 37%, 6816 KB, 3774 KB/s, 1 seconds passed
... 38%, 6848 KB, 3788 KB/s, 1 seconds passed
... 38%, 6880 KB, 3803 KB/s, 1 seconds passed
... 38%, 6912 KB, 3769 KB/s, 1 seconds passed
... 38%, 6944 KB, 3778 KB/s, 1 seconds passed
... 38%, 6976 KB, 3792 KB/s, 1 seconds passed
... 38%, 7008 KB, 3806 KB/s, 1 seconds passed

.. parsed-literal::

    ... 39%, 7040 KB, 3771 KB/s, 1 seconds passed
... 39%, 7072 KB, 3779 KB/s, 1 seconds passed
... 39%, 7104 KB, 3793 KB/s, 1 seconds passed
... 39%, 7136 KB, 3807 KB/s, 1 seconds passed
... 39%, 7168 KB, 3775 KB/s, 1 seconds passed
... 40%, 7200 KB, 3784 KB/s, 1 seconds passed
... 40%, 7232 KB, 3797 KB/s, 1 seconds passed

.. parsed-literal::

    ... 40%, 7264 KB, 3811 KB/s, 1 seconds passed
... 40%, 7296 KB, 3778 KB/s, 1 seconds passed
... 40%, 7328 KB, 3786 KB/s, 1 seconds passed
... 40%, 7360 KB, 3800 KB/s, 1 seconds passed
... 41%, 7392 KB, 3813 KB/s, 1 seconds passed

.. parsed-literal::

    ... 41%, 7424 KB, 3780 KB/s, 1 seconds passed
... 41%, 7456 KB, 3790 KB/s, 1 seconds passed
... 41%, 7488 KB, 3802 KB/s, 1 seconds passed
... 41%, 7520 KB, 3815 KB/s, 1 seconds passed
... 41%, 7552 KB, 3783 KB/s, 1 seconds passed
... 42%, 7584 KB, 3791 KB/s, 2 seconds passed
... 42%, 7616 KB, 3804 KB/s, 2 seconds passed
... 42%, 7648 KB, 3817 KB/s, 2 seconds passed

.. parsed-literal::

    ... 42%, 7680 KB, 3786 KB/s, 2 seconds passed
... 42%, 7712 KB, 3794 KB/s, 2 seconds passed
... 43%, 7744 KB, 3807 KB/s, 2 seconds passed
... 43%, 7776 KB, 3820 KB/s, 2 seconds passed

.. parsed-literal::

    ... 43%, 7808 KB, 3788 KB/s, 2 seconds passed
... 43%, 7840 KB, 3797 KB/s, 2 seconds passed
... 43%, 7872 KB, 3808 KB/s, 2 seconds passed
... 43%, 7904 KB, 3822 KB/s, 2 seconds passed
... 44%, 7936 KB, 3790 KB/s, 2 seconds passed
... 44%, 7968 KB, 3798 KB/s, 2 seconds passed
... 44%, 8000 KB, 3810 KB/s, 2 seconds passed
... 44%, 8032 KB, 3823 KB/s, 2 seconds passed

.. parsed-literal::

    ... 44%, 8064 KB, 3791 KB/s, 2 seconds passed
... 45%, 8096 KB, 3800 KB/s, 2 seconds passed
... 45%, 8128 KB, 3812 KB/s, 2 seconds passed
... 45%, 8160 KB, 3825 KB/s, 2 seconds passed
... 45%, 8192 KB, 3793 KB/s, 2 seconds passed

.. parsed-literal::

    ... 45%, 8224 KB, 3802 KB/s, 2 seconds passed
... 45%, 8256 KB, 3813 KB/s, 2 seconds passed
... 46%, 8288 KB, 3827 KB/s, 2 seconds passed
... 46%, 8320 KB, 3795 KB/s, 2 seconds passed
... 46%, 8352 KB, 3803 KB/s, 2 seconds passed
... 46%, 8384 KB, 3815 KB/s, 2 seconds passed
... 46%, 8416 KB, 3827 KB/s, 2 seconds passed

.. parsed-literal::

    ... 46%, 8448 KB, 3796 KB/s, 2 seconds passed
... 47%, 8480 KB, 3804 KB/s, 2 seconds passed
... 47%, 8512 KB, 3816 KB/s, 2 seconds passed
... 47%, 8544 KB, 3829 KB/s, 2 seconds passed
... 47%, 8576 KB, 3799 KB/s, 2 seconds passed
... 47%, 8608 KB, 3806 KB/s, 2 seconds passed
... 48%, 8640 KB, 3818 KB/s, 2 seconds passed

.. parsed-literal::

    ... 48%, 8672 KB, 3795 KB/s, 2 seconds passed
... 48%, 8704 KB, 3800 KB/s, 2 seconds passed
... 48%, 8736 KB, 3807 KB/s, 2 seconds passed
... 48%, 8768 KB, 3819 KB/s, 2 seconds passed

.. parsed-literal::

    ... 48%, 8800 KB, 3797 KB/s, 2 seconds passed
... 49%, 8832 KB, 3802 KB/s, 2 seconds passed
... 49%, 8864 KB, 3809 KB/s, 2 seconds passed
... 49%, 8896 KB, 3821 KB/s, 2 seconds passed
... 49%, 8928 KB, 3798 KB/s, 2 seconds passed
... 49%, 8960 KB, 3804 KB/s, 2 seconds passed
... 49%, 8992 KB, 3811 KB/s, 2 seconds passed
... 50%, 9024 KB, 3823 KB/s, 2 seconds passed

.. parsed-literal::

    ... 50%, 9056 KB, 3800 KB/s, 2 seconds passed
... 50%, 9088 KB, 3806 KB/s, 2 seconds passed
... 50%, 9120 KB, 3813 KB/s, 2 seconds passed
... 50%, 9152 KB, 3825 KB/s, 2 seconds passed
... 51%, 9184 KB, 3802 KB/s, 2 seconds passed

.. parsed-literal::

    ... 51%, 9216 KB, 3808 KB/s, 2 seconds passed
... 51%, 9248 KB, 3815 KB/s, 2 seconds passed
... 51%, 9280 KB, 3826 KB/s, 2 seconds passed
... 51%, 9312 KB, 3838 KB/s, 2 seconds passed
... 51%, 9344 KB, 3810 KB/s, 2 seconds passed
... 52%, 9376 KB, 3817 KB/s, 2 seconds passed
... 52%, 9408 KB, 3828 KB/s, 2 seconds passed
... 52%, 9440 KB, 3839 KB/s, 2 seconds passed

.. parsed-literal::

    ... 52%, 9472 KB, 3812 KB/s, 2 seconds passed
... 52%, 9504 KB, 3820 KB/s, 2 seconds passed
... 53%, 9536 KB, 3830 KB/s, 2 seconds passed
... 53%, 9568 KB, 3841 KB/s, 2 seconds passed
... 53%, 9600 KB, 3816 KB/s, 2 seconds passed
... 53%, 9632 KB, 3822 KB/s, 2 seconds passed

.. parsed-literal::

    ... 53%, 9664 KB, 3831 KB/s, 2 seconds passed
... 53%, 9696 KB, 3810 KB/s, 2 seconds passed
... 54%, 9728 KB, 3811 KB/s, 2 seconds passed
... 54%, 9760 KB, 3821 KB/s, 2 seconds passed
... 54%, 9792 KB, 3832 KB/s, 2 seconds passed

.. parsed-literal::

    ... 54%, 9824 KB, 3812 KB/s, 2 seconds passed
... 54%, 9856 KB, 3813 KB/s, 2 seconds passed
... 54%, 9888 KB, 3823 KB/s, 2 seconds passed
... 55%, 9920 KB, 3833 KB/s, 2 seconds passed
... 55%, 9952 KB, 3813 KB/s, 2 seconds passed
... 55%, 9984 KB, 3818 KB/s, 2 seconds passed
... 55%, 10016 KB, 3825 KB/s, 2 seconds passed
... 55%, 10048 KB, 3835 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 10080 KB, 3813 KB/s, 2 seconds passed
... 56%, 10112 KB, 3815 KB/s, 2 seconds passed
... 56%, 10144 KB, 3825 KB/s, 2 seconds passed
... 56%, 10176 KB, 3836 KB/s, 2 seconds passed

.. parsed-literal::

    ... 56%, 10208 KB, 3814 KB/s, 2 seconds passed
... 56%, 10240 KB, 3817 KB/s, 2 seconds passed
... 57%, 10272 KB, 3828 KB/s, 2 seconds passed
... 57%, 10304 KB, 3838 KB/s, 2 seconds passed
... 57%, 10336 KB, 3848 KB/s, 2 seconds passed
... 57%, 10368 KB, 3822 KB/s, 2 seconds passed
... 57%, 10400 KB, 3830 KB/s, 2 seconds passed
... 57%, 10432 KB, 3839 KB/s, 2 seconds passed
... 58%, 10464 KB, 3850 KB/s, 2 seconds passed

.. parsed-literal::

    ... 58%, 10496 KB, 3823 KB/s, 2 seconds passed
... 58%, 10528 KB, 3830 KB/s, 2 seconds passed
... 58%, 10560 KB, 3840 KB/s, 2 seconds passed
... 58%, 10592 KB, 3822 KB/s, 2 seconds passed

.. parsed-literal::

    ... 59%, 10624 KB, 3822 KB/s, 2 seconds passed
... 59%, 10656 KB, 3831 KB/s, 2 seconds passed
... 59%, 10688 KB, 3841 KB/s, 2 seconds passed
... 59%, 10720 KB, 3823 KB/s, 2 seconds passed
... 59%, 10752 KB, 3826 KB/s, 2 seconds passed
... 59%, 10784 KB, 3833 KB/s, 2 seconds passed
... 60%, 10816 KB, 3842 KB/s, 2 seconds passed
... 60%, 10848 KB, 3853 KB/s, 2 seconds passed

.. parsed-literal::

    ... 60%, 10880 KB, 3827 KB/s, 2 seconds passed
... 60%, 10912 KB, 3835 KB/s, 2 seconds passed
... 60%, 10944 KB, 3843 KB/s, 2 seconds passed
... 61%, 10976 KB, 3853 KB/s, 2 seconds passed
... 61%, 11008 KB, 3828 KB/s, 2 seconds passed
... 61%, 11040 KB, 3835 KB/s, 2 seconds passed
... 61%, 11072 KB, 3844 KB/s, 2 seconds passed

.. parsed-literal::

    ... 61%, 11104 KB, 3854 KB/s, 2 seconds passed
... 61%, 11136 KB, 3829 KB/s, 2 seconds passed
... 62%, 11168 KB, 3835 KB/s, 2 seconds passed
... 62%, 11200 KB, 3846 KB/s, 2 seconds passed

.. parsed-literal::

    ... 62%, 11232 KB, 3829 KB/s, 2 seconds passed
... 62%, 11264 KB, 3828 KB/s, 2 seconds passed
... 62%, 11296 KB, 3837 KB/s, 2 seconds passed
... 62%, 11328 KB, 3846 KB/s, 2 seconds passed
... 63%, 11360 KB, 3830 KB/s, 2 seconds passed
... 63%, 11392 KB, 3830 KB/s, 2 seconds passed
... 63%, 11424 KB, 3838 KB/s, 2 seconds passed
... 63%, 11456 KB, 3847 KB/s, 2 seconds passed

.. parsed-literal::

    ... 63%, 11488 KB, 3831 KB/s, 2 seconds passed
... 64%, 11520 KB, 3831 KB/s, 3 seconds passed
... 64%, 11552 KB, 3839 KB/s, 3 seconds passed
... 64%, 11584 KB, 3848 KB/s, 3 seconds passed
... 64%, 11616 KB, 3832 KB/s, 3 seconds passed

.. parsed-literal::

    ... 64%, 11648 KB, 3832 KB/s, 3 seconds passed
... 64%, 11680 KB, 3839 KB/s, 3 seconds passed
... 65%, 11712 KB, 3849 KB/s, 3 seconds passed
... 65%, 11744 KB, 3832 KB/s, 3 seconds passed
... 65%, 11776 KB, 3833 KB/s, 3 seconds passed
... 65%, 11808 KB, 3841 KB/s, 3 seconds passed
... 65%, 11840 KB, 3850 KB/s, 3 seconds passed

.. parsed-literal::

    ... 65%, 11872 KB, 3833 KB/s, 3 seconds passed
... 66%, 11904 KB, 3834 KB/s, 3 seconds passed
... 66%, 11936 KB, 3841 KB/s, 3 seconds passed
... 66%, 11968 KB, 3851 KB/s, 3 seconds passed
... 66%, 12000 KB, 3835 KB/s, 3 seconds passed
... 66%, 12032 KB, 3836 KB/s, 3 seconds passed

.. parsed-literal::

    ... 67%, 12064 KB, 3843 KB/s, 3 seconds passed
... 67%, 12096 KB, 3852 KB/s, 3 seconds passed
... 67%, 12128 KB, 3837 KB/s, 3 seconds passed
... 67%, 12160 KB, 3835 KB/s, 3 seconds passed
... 67%, 12192 KB, 3843 KB/s, 3 seconds passed
... 67%, 12224 KB, 3852 KB/s, 3 seconds passed

.. parsed-literal::

    ... 68%, 12256 KB, 3835 KB/s, 3 seconds passed
... 68%, 12288 KB, 3836 KB/s, 3 seconds passed
... 68%, 12320 KB, 3844 KB/s, 3 seconds passed
... 68%, 12352 KB, 3853 KB/s, 3 seconds passed
... 68%, 12384 KB, 3836 KB/s, 3 seconds passed
... 69%, 12416 KB, 3838 KB/s, 3 seconds passed
... 69%, 12448 KB, 3845 KB/s, 3 seconds passed
... 69%, 12480 KB, 3854 KB/s, 3 seconds passed

.. parsed-literal::

    ... 69%, 12512 KB, 3837 KB/s, 3 seconds passed
... 69%, 12544 KB, 3839 KB/s, 3 seconds passed
... 69%, 12576 KB, 3846 KB/s, 3 seconds passed
... 70%, 12608 KB, 3855 KB/s, 3 seconds passed

.. parsed-literal::

    ... 70%, 12640 KB, 3839 KB/s, 3 seconds passed
... 70%, 12672 KB, 3840 KB/s, 3 seconds passed
... 70%, 12704 KB, 3847 KB/s, 3 seconds passed
... 70%, 12736 KB, 3856 KB/s, 3 seconds passed
... 70%, 12768 KB, 3842 KB/s, 3 seconds passed
... 71%, 12800 KB, 3842 KB/s, 3 seconds passed
... 71%, 12832 KB, 3848 KB/s, 3 seconds passed
... 71%, 12864 KB, 3857 KB/s, 3 seconds passed

.. parsed-literal::

    ... 71%, 12896 KB, 3840 KB/s, 3 seconds passed
... 71%, 12928 KB, 3841 KB/s, 3 seconds passed
... 72%, 12960 KB, 3848 KB/s, 3 seconds passed
... 72%, 12992 KB, 3857 KB/s, 3 seconds passed
... 72%, 13024 KB, 3841 KB/s, 3 seconds passed

.. parsed-literal::

    ... 72%, 13056 KB, 3842 KB/s, 3 seconds passed
... 72%, 13088 KB, 3849 KB/s, 3 seconds passed
... 72%, 13120 KB, 3858 KB/s, 3 seconds passed
... 73%, 13152 KB, 3842 KB/s, 3 seconds passed
... 73%, 13184 KB, 3844 KB/s, 3 seconds passed
... 73%, 13216 KB, 3851 KB/s, 3 seconds passed
... 73%, 13248 KB, 3859 KB/s, 3 seconds passed

.. parsed-literal::

    ... 73%, 13280 KB, 3843 KB/s, 3 seconds passed
... 73%, 13312 KB, 3845 KB/s, 3 seconds passed
... 74%, 13344 KB, 3852 KB/s, 3 seconds passed
... 74%, 13376 KB, 3860 KB/s, 3 seconds passed
... 74%, 13408 KB, 3845 KB/s, 3 seconds passed
... 74%, 13440 KB, 3846 KB/s, 3 seconds passed

.. parsed-literal::

    ... 74%, 13472 KB, 3853 KB/s, 3 seconds passed
... 75%, 13504 KB, 3860 KB/s, 3 seconds passed
... 75%, 13536 KB, 3846 KB/s, 3 seconds passed
... 75%, 13568 KB, 3847 KB/s, 3 seconds passed
... 75%, 13600 KB, 3854 KB/s, 3 seconds passed
... 75%, 13632 KB, 3861 KB/s, 3 seconds passed

.. parsed-literal::

    ... 75%, 13664 KB, 3846 KB/s, 3 seconds passed
... 76%, 13696 KB, 3848 KB/s, 3 seconds passed
... 76%, 13728 KB, 3854 KB/s, 3 seconds passed
... 76%, 13760 KB, 3862 KB/s, 3 seconds passed
... 76%, 13792 KB, 3847 KB/s, 3 seconds passed
... 76%, 13824 KB, 3849 KB/s, 3 seconds passed
... 77%, 13856 KB, 3855 KB/s, 3 seconds passed
... 77%, 13888 KB, 3862 KB/s, 3 seconds passed

.. parsed-literal::

    ... 77%, 13920 KB, 3847 KB/s, 3 seconds passed
... 77%, 13952 KB, 3849 KB/s, 3 seconds passed
... 77%, 13984 KB, 3855 KB/s, 3 seconds passed
... 77%, 14016 KB, 3863 KB/s, 3 seconds passed

.. parsed-literal::

    ... 78%, 14048 KB, 3848 KB/s, 3 seconds passed
... 78%, 14080 KB, 3849 KB/s, 3 seconds passed
... 78%, 14112 KB, 3856 KB/s, 3 seconds passed
... 78%, 14144 KB, 3863 KB/s, 3 seconds passed
... 78%, 14176 KB, 3848 KB/s, 3 seconds passed
... 78%, 14208 KB, 3850 KB/s, 3 seconds passed
... 79%, 14240 KB, 3857 KB/s, 3 seconds passed
... 79%, 14272 KB, 3864 KB/s, 3 seconds passed

.. parsed-literal::

    ... 79%, 14304 KB, 3849 KB/s, 3 seconds passed
... 79%, 14336 KB, 3851 KB/s, 3 seconds passed
... 79%, 14368 KB, 3857 KB/s, 3 seconds passed
... 80%, 14400 KB, 3864 KB/s, 3 seconds passed
... 80%, 14432 KB, 3850 KB/s, 3 seconds passed

.. parsed-literal::

    ... 80%, 14464 KB, 3851 KB/s, 3 seconds passed
... 80%, 14496 KB, 3858 KB/s, 3 seconds passed
... 80%, 14528 KB, 3865 KB/s, 3 seconds passed
... 80%, 14560 KB, 3850 KB/s, 3 seconds passed
... 81%, 14592 KB, 3851 KB/s, 3 seconds passed
... 81%, 14624 KB, 3859 KB/s, 3 seconds passed
... 81%, 14656 KB, 3866 KB/s, 3 seconds passed

.. parsed-literal::

    ... 81%, 14688 KB, 3848 KB/s, 3 seconds passed
... 81%, 14720 KB, 3852 KB/s, 3 seconds passed
... 82%, 14752 KB, 3859 KB/s, 3 seconds passed
... 82%, 14784 KB, 3866 KB/s, 3 seconds passed
... 82%, 14816 KB, 3849 KB/s, 3 seconds passed
... 82%, 14848 KB, 3852 KB/s, 3 seconds passed

.. parsed-literal::

    ... 82%, 14880 KB, 3859 KB/s, 3 seconds passed
... 82%, 14912 KB, 3866 KB/s, 3 seconds passed
... 83%, 14944 KB, 3853 KB/s, 3 seconds passed
... 83%, 14976 KB, 3854 KB/s, 3 seconds passed
... 83%, 15008 KB, 3861 KB/s, 3 seconds passed
... 83%, 15040 KB, 3868 KB/s, 3 seconds passed

.. parsed-literal::

    ... 83%, 15072 KB, 3851 KB/s, 3 seconds passed
... 83%, 15104 KB, 3854 KB/s, 3 seconds passed
... 84%, 15136 KB, 3861 KB/s, 3 seconds passed
... 84%, 15168 KB, 3868 KB/s, 3 seconds passed
... 84%, 15200 KB, 3851 KB/s, 3 seconds passed
... 84%, 15232 KB, 3855 KB/s, 3 seconds passed
... 84%, 15264 KB, 3862 KB/s, 3 seconds passed
... 85%, 15296 KB, 3869 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 15328 KB, 3852 KB/s, 3 seconds passed
... 85%, 15360 KB, 3855 KB/s, 3 seconds passed
... 85%, 15392 KB, 3862 KB/s, 3 seconds passed
... 85%, 15424 KB, 3869 KB/s, 3 seconds passed

.. parsed-literal::

    ... 85%, 15456 KB, 3852 KB/s, 4 seconds passed
... 86%, 15488 KB, 3856 KB/s, 4 seconds passed
... 86%, 15520 KB, 3863 KB/s, 4 seconds passed
... 86%, 15552 KB, 3870 KB/s, 4 seconds passed
... 86%, 15584 KB, 3853 KB/s, 4 seconds passed
... 86%, 15616 KB, 3856 KB/s, 4 seconds passed
... 86%, 15648 KB, 3863 KB/s, 4 seconds passed
... 87%, 15680 KB, 3871 KB/s, 4 seconds passed

.. parsed-literal::

    ... 87%, 15712 KB, 3854 KB/s, 4 seconds passed
... 87%, 15744 KB, 3857 KB/s, 4 seconds passed
... 87%, 15776 KB, 3864 KB/s, 4 seconds passed
... 87%, 15808 KB, 3871 KB/s, 4 seconds passed
... 88%, 15840 KB, 3855 KB/s, 4 seconds passed

.. parsed-literal::

    ... 88%, 15872 KB, 3858 KB/s, 4 seconds passed
... 88%, 15904 KB, 3865 KB/s, 4 seconds passed
... 88%, 15936 KB, 3872 KB/s, 4 seconds passed
... 88%, 15968 KB, 3856 KB/s, 4 seconds passed
... 88%, 16000 KB, 3859 KB/s, 4 seconds passed
... 89%, 16032 KB, 3866 KB/s, 4 seconds passed
... 89%, 16064 KB, 3872 KB/s, 4 seconds passed

.. parsed-literal::

    ... 89%, 16096 KB, 3857 KB/s, 4 seconds passed
... 89%, 16128 KB, 3860 KB/s, 4 seconds passed
... 89%, 16160 KB, 3866 KB/s, 4 seconds passed
... 90%, 16192 KB, 3873 KB/s, 4 seconds passed
... 90%, 16224 KB, 3857 KB/s, 4 seconds passed
... 90%, 16256 KB, 3860 KB/s, 4 seconds passed
... 90%, 16288 KB, 3866 KB/s, 4 seconds passed

.. parsed-literal::

    ... 90%, 16320 KB, 3859 KB/s, 4 seconds passed
... 90%, 16352 KB, 3857 KB/s, 4 seconds passed
... 91%, 16384 KB, 3861 KB/s, 4 seconds passed
... 91%, 16416 KB, 3867 KB/s, 4 seconds passed
... 91%, 16448 KB, 3859 KB/s, 4 seconds passed

.. parsed-literal::

    ... 91%, 16480 KB, 3857 KB/s, 4 seconds passed
... 91%, 16512 KB, 3861 KB/s, 4 seconds passed
... 91%, 16544 KB, 3868 KB/s, 4 seconds passed
... 92%, 16576 KB, 3860 KB/s, 4 seconds passed
... 92%, 16608 KB, 3858 KB/s, 4 seconds passed
... 92%, 16640 KB, 3861 KB/s, 4 seconds passed
... 92%, 16672 KB, 3868 KB/s, 4 seconds passed

.. parsed-literal::

    ... 92%, 16704 KB, 3860 KB/s, 4 seconds passed
... 93%, 16736 KB, 3859 KB/s, 4 seconds passed
... 93%, 16768 KB, 3862 KB/s, 4 seconds passed
... 93%, 16800 KB, 3869 KB/s, 4 seconds passed
... 93%, 16832 KB, 3861 KB/s, 4 seconds passed

.. parsed-literal::

    ... 93%, 16864 KB, 3859 KB/s, 4 seconds passed
... 93%, 16896 KB, 3862 KB/s, 4 seconds passed
... 94%, 16928 KB, 3869 KB/s, 4 seconds passed
... 94%, 16960 KB, 3861 KB/s, 4 seconds passed
... 94%, 16992 KB, 3860 KB/s, 4 seconds passed
... 94%, 17024 KB, 3862 KB/s, 4 seconds passed
... 94%, 17056 KB, 3869 KB/s, 4 seconds passed

.. parsed-literal::

    ... 94%, 17088 KB, 3861 KB/s, 4 seconds passed
... 95%, 17120 KB, 3860 KB/s, 4 seconds passed
... 95%, 17152 KB, 3863 KB/s, 4 seconds passed
... 95%, 17184 KB, 3870 KB/s, 4 seconds passed
... 95%, 17216 KB, 3861 KB/s, 4 seconds passed
... 95%, 17248 KB, 3861 KB/s, 4 seconds passed

.. parsed-literal::

    ... 96%, 17280 KB, 3863 KB/s, 4 seconds passed
... 96%, 17312 KB, 3869 KB/s, 4 seconds passed
... 96%, 17344 KB, 3862 KB/s, 4 seconds passed
... 96%, 17376 KB, 3862 KB/s, 4 seconds passed
... 96%, 17408 KB, 3865 KB/s, 4 seconds passed
... 96%, 17440 KB, 3870 KB/s, 4 seconds passed
... 97%, 17472 KB, 3877 KB/s, 4 seconds passed

.. parsed-literal::

    ... 97%, 17504 KB, 3863 KB/s, 4 seconds passed
... 97%, 17536 KB, 3866 KB/s, 4 seconds passed
... 97%, 17568 KB, 3871 KB/s, 4 seconds passed
... 97%, 17600 KB, 3877 KB/s, 4 seconds passed
... 98%, 17632 KB, 3864 KB/s, 4 seconds passed
... 98%, 17664 KB, 3866 KB/s, 4 seconds passed
... 98%, 17696 KB, 3871 KB/s, 4 seconds passed
... 98%, 17728 KB, 3877 KB/s, 4 seconds passed

.. parsed-literal::

    ... 98%, 17760 KB, 3862 KB/s, 4 seconds passed
... 98%, 17792 KB, 3865 KB/s, 4 seconds passed
... 99%, 17824 KB, 3871 KB/s, 4 seconds passed
... 99%, 17856 KB, 3862 KB/s, 4 seconds passed

.. parsed-literal::

    ... 99%, 17888 KB, 3862 KB/s, 4 seconds passed
... 99%, 17920 KB, 3865 KB/s, 4 seconds passed
... 99%, 17952 KB, 3872 KB/s, 4 seconds passed
... 99%, 17984 KB, 3867 KB/s, 4 seconds passed
... 100%, 17990 KB, 3868 KB/s, 4 seconds passed


    ========== Unpacking model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz


Convert Model to OpenVINO IR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



The selected model comes from the public directory, which means it must
be converted into OpenVINO Intermediate Representation (OpenVINO IR). We
use ``omz_converter`` to convert the ONNX format model to the OpenVINO
IR format.

.. code:: ipython3

    if not onnx_path.exists():
        convert_command = (
            f"omz_converter "
            f"--name {model_name} "
            f"--precisions {precision} "
            f"--download_dir {base_model_dir} "
            f"--output_dir {base_model_dir}"
        )
        ! $convert_command


.. parsed-literal::

    ========== Converting human-pose-estimation-3d-0001 to ONNX
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False



.. parsed-literal::

    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin




Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

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



Load the model
~~~~~~~~~~~~~~



Converted models are located in a fixed structure, which indicates
vendor, model name and precision.

First, initialize OpenVINO Runtime. Then, read the
network architecture and model weights from the ``.bin`` and ``.xml``
files to compile for the desired device. An inference request is then
created to infer the compiled model.

.. code:: ipython3

    # initialize OpenVINO
    core = ov.Core()
    # read the network and corresponding weights from file
    model = core.read_model(model=ir_model_path, weights=model_weights_path)
    # load the model on the specified device
    compiled_model = core.compile_model(model=model, device_name=device.value)
    infer_request = compiled_model.create_infer_request()
    input_tensor_name = model.inputs[0].get_any_name()

    # get input and output names of nodes
    input_layer = compiled_model.input(0)
    output_layers = list(compiled_model.outputs)

The input for the model is data from the input image and the outputs are
heat maps, PAF (part affinity fields) and features.

.. code:: ipython3

    input_layer.any_name, [o.any_name for o in output_layers]




.. parsed-literal::

    ('data', ['features', 'heatmaps', 'pafs'])



Processing
----------



Model Inference
~~~~~~~~~~~~~~~



Frames captured from video files or the live webcam are used as the
input for the 3D model. This is how you obtain the output heat maps, PAF
(part affinity fields) and features.

.. code:: ipython3

    def model_infer(scaled_img, stride):
        """
        Run model inference on the input image

        Parameters:
            scaled_img: resized image according to the input size of the model
            stride: int, the stride of the window
        """

        # Remove excess space from the picture
        img = scaled_img[
            0 : scaled_img.shape[0] - (scaled_img.shape[0] % stride),
            0 : scaled_img.shape[1] - (scaled_img.shape[1] % stride),
        ]

        img = np.transpose(img, (2, 0, 1))[
            None,
        ]
        infer_request.infer({input_tensor_name: img})
        # A set of three inference results is obtained
        results = {
            name: infer_request.get_tensor(name).data[:]
            for name in {"features", "heatmaps", "pafs"}
        }
        # Get the results
        results = (results["features"][0], results["heatmaps"][0], results["pafs"][0])

        return results

Draw 2D Pose Overlays
~~~~~~~~~~~~~~~~~~~~~



We need to define some connections between the joints in advance, so
that we can draw the structure of the human body in the resulting image
after obtaining the inference results. Joints are drawn as circles and
limbs are drawn as lines. The code is based on the `3D Human Pose
Estimation
Demo <https://github.com/openvinotoolkit/open_model_zoo/tree/master/demos/human_pose_estimation_3d_demo/python>`__
from Open Model Zoo.

.. code:: ipython3

    # 3D edge index array
    body_edges = np.array(
        [
            [0, 1],
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [1, 15], [15, 16],            # nose - l_eye - l_ear
            [1, 17], [17, 18],            # nose - r_eye - r_ear
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )


    body_edges_2d = np.array(
        [
            [0, 1],                       # neck - nose
            [1, 16], [16, 18],            # nose - l_eye - l_ear
            [1, 15], [15, 17],            # nose - r_eye - r_ear
            [0, 3], [3, 4], [4, 5],       # neck - l_shoulder - l_elbow - l_wrist
            [0, 9], [9, 10], [10, 11],    # neck - r_shoulder - r_elbow - r_wrist
            [0, 6], [6, 7], [7, 8],       # neck - l_hip - l_knee - l_ankle
            [0, 12], [12, 13], [13, 14],  # neck - r_hip - r_knee - r_ankle
        ]
    )


    def draw_poses(frame, poses_2d, scaled_img, use_popup):
        """
        Draw 2D pose overlays on the image to visualize estimated poses.
        Joints are drawn as circles and limbs are drawn as lines.

        :param frame: the input image
        :param poses_2d: array of human joint pairs
        """
        for pose in poses_2d:
            pose = np.array(pose[0:-1]).reshape((-1, 3)).transpose()
            was_found = pose[2] > 0

            pose[0], pose[1] = (
                pose[0] * frame.shape[1] / scaled_img.shape[1],
                pose[1] * frame.shape[0] / scaled_img.shape[0],
            )

            # Draw joints.
            for edge in body_edges_2d:
                if was_found[edge[0]] and was_found[edge[1]]:
                    cv2.line(
                        frame,
                        tuple(pose[0:2, edge[0]].astype(np.int32)),
                        tuple(pose[0:2, edge[1]].astype(np.int32)),
                        (255, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
            # Draw limbs.
            for kpt_id in range(pose.shape[1]):
                if pose[2, kpt_id] != -1:
                    cv2.circle(
                        frame,
                        tuple(pose[0:2, kpt_id].astype(np.int32)),
                        3,
                        (0, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )

        return frame

Main Processing Function
~~~~~~~~~~~~~~~~~~~~~~~~



Run 3D pose estimation on the specified source. It could be either a
webcam feed or a video file.

.. code:: ipython3

    def run_pose_estimation(source=0, flip=False, use_popup=False, skip_frames=0):
        """
        2D image as input, using OpenVINO as inference backend,
        get joints 3D coordinates, and draw 3D human skeleton in the scene

        :param source:      The webcam number to feed the video stream with primary webcam set to "0", or the video path.
        :param flip:        To be used by VideoPlayer function for flipping capture image.
        :param use_popup:   False for showing encoded frames over this notebook, True for creating a popup window.
        :param skip_frames: Number of frames to skip at the beginning of the video.
        """

        focal_length = -1  # default
        stride = 8
        player = None
        skeleton_set = None

        try:
            # create video player to play with target fps  video_path
            # get the frame from camera
            # You can skip first N frames to fast forward video. change 'skip_first_frames'
            player = utils.VideoPlayer(source, flip=flip, fps=30, skip_first_frames=skip_frames)
            # start capturing
            player.start()

            input_image = player.next()
            # set the window size
            resize_scale = 450 / input_image.shape[1]
            windows_width = int(input_image.shape[1] * resize_scale)
            windows_height = int(input_image.shape[0] * resize_scale)

            # use visualization library
            engine3D = engine.Engine3js(grid=True, axis=True, view_width=windows_width, view_height=windows_height)

            if use_popup:
                # display the 3D human pose in this notebook, and origin frame in popup window
                display(engine3D.renderer)
                title = "Press ESC to Exit"
                cv2.namedWindow(title, cv2.WINDOW_KEEPRATIO | cv2.WINDOW_AUTOSIZE)
            else:
                # set the 2D image box, show both human pose and image in the notebook
                imgbox = widgets.Image(
                    format="jpg", height=windows_height, width=windows_width
                )
                display(widgets.HBox([engine3D.renderer, imgbox]))

            skeleton = engine.Skeleton(body_edges=body_edges)

            processing_times = collections.deque()

            while True:
                # grab the frame
                frame = player.next()
                if frame is None:
                    print("Source ended")
                    break

                # resize image and change dims to fit neural network input
                # (see https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001)
                scaled_img = cv2.resize(frame, dsize=(model.inputs[0].shape[3], model.inputs[0].shape[2]))

                if focal_length < 0:  # Focal length is unknown
                    focal_length = np.float32(0.8 * scaled_img.shape[1])

                # inference start
                start_time = time.time()
                # get results
                inference_result = model_infer(scaled_img, stride)

                # inference stop
                stop_time = time.time()
                processing_times.append(stop_time - start_time)
                # Process the point to point coordinates of the data
                poses_3d, poses_2d = parse_poses(inference_result, 1, stride, focal_length, True)

                # use processing times from last 200 frames
                if len(processing_times) > 200:
                    processing_times.popleft()

                processing_time = np.mean(processing_times) * 1000
                fps = 1000 / processing_time

                if len(poses_3d) > 0:
                    # From here, you can rotate the 3D point positions using the function "draw_poses",
                    # or you can directly make the correct mapping below to properly display the object image on the screen
                    poses_3d_copy = poses_3d.copy()
                    x = poses_3d_copy[:, 0::4]
                    y = poses_3d_copy[:, 1::4]
                    z = poses_3d_copy[:, 2::4]
                    poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = (
                        -z + np.ones(poses_3d[:, 2::4].shape) * 200,
                        -y + np.ones(poses_3d[:, 2::4].shape) * 100,
                        -x,
                    )

                    poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
                    people = skeleton(poses_3d=poses_3d)

                    try:
                        engine3D.scene_remove(skeleton_set)
                    except Exception:
                        pass

                    engine3D.scene_add(people)
                    skeleton_set = people

                    # draw 2D
                    frame = draw_poses(frame, poses_2d, scaled_img, use_popup)

                else:
                    try:
                        engine3D.scene_remove(skeleton_set)
                        skeleton_set = None
                    except Exception:
                        pass

                cv2.putText(
                    frame,
                    f"Inference time: {processing_time:.1f}ms ({fps:.1f} FPS)",
                    (10, 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.7,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA,
                )

                if use_popup:
                    cv2.imshow(title, frame)
                    key = cv2.waitKey(1)
                    # escape = 27, use ESC to exit
                    if key == 27:
                        break
                else:
                    # encode numpy array to jpg
                    imgbox.value = cv2.imencode(
                        ".jpg",
                        frame,
                        params=[cv2.IMWRITE_JPEG_QUALITY, 90],
                    )[1].tobytes()

                engine3D.renderer.render(engine3D.scene, engine3D.cam)

        except KeyboardInterrupt:
            print("Interrupted")
        except RuntimeError as e:
            print(e)
        finally:
            clear_output()
            if player is not None:
                # stop capturing
                player.stop()
            if use_popup:
                cv2.destroyAllWindows()
            if skeleton_set:
                engine3D.scene_remove(skeleton_set)

Run
---



Run, using a webcam as the video input. By default, the primary webcam
is set with ``source=0``. If you have multiple webcams, each one will be
assigned a consecutive number starting at 0. Set ``flip=True`` when
using a front-facing camera. Some web browsers, especially Mozilla
Firefox, may cause flickering. If you experience flickering, set
``use_popup=True``.

   **NOTE**:

   *1. To use this notebook with a webcam, you need to run the notebook
   on a computer with a webcam. If you run the notebook on a server
   (e.g. Binder), the webcam will not work.*

   *2. Popup mode may not work if you run this notebook on a remote
   computer (e.g. Binder).*

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
will work.

Using the following method, you can click and move your mouse over the
picture on the left to interact.

.. code:: ipython3

    USE_WEBCAM = False

    cam_id = 0
    video_path = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"

    source = cam_id if USE_WEBCAM else video_path

    run_pose_estimation(source=source, flip=isinstance(source, int), use_popup=False)
