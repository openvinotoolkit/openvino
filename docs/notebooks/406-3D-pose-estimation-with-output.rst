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

    %pip install pythreejs "openvino-dev>=2024.0.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl.metadata (5.4 kB)
    Requirement already satisfied: openvino-dev>=2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.0.0)


.. parsed-literal::

    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.2)


.. parsed-literal::

    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (1.23.5)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.14.2)


.. parsed-literal::

    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (0.7.1)
    Requirement already satisfied: networkx<=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.8.8)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2023.2.1)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (24.0)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2.31.0)
    Requirement already satisfied: openvino==2024.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev>=2024.0.0) (2024.0.0)


.. parsed-literal::

    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl.metadata (1.0 kB)
    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.10)
    Requirement already satisfied: jupyterlab-widgets~=3.0.10 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.10)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.3.2)


.. parsed-literal::

    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (3.6)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2.2.1)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev>=2024.0.0) (2024.2.2)
    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.43)


.. parsed-literal::

    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.17.2)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.10.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.9.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.13)


.. parsed-literal::

    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)
    Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)
    Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)


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

    ... 0%, 32 KB, 912 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 64 KB, 933 KB/s, 0 seconds passed
... 0%, 96 KB, 1381 KB/s, 0 seconds passed
... 0%, 128 KB, 1798 KB/s, 0 seconds passed

.. parsed-literal::

    ... 0%, 160 KB, 1546 KB/s, 0 seconds passed
... 1%, 192 KB, 1845 KB/s, 0 seconds passed
... 1%, 224 KB, 2125 KB/s, 0 seconds passed
... 1%, 256 KB, 2398 KB/s, 0 seconds passed
... 1%, 288 KB, 2660 KB/s, 0 seconds passed
... 1%, 320 KB, 2317 KB/s, 0 seconds passed
... 1%, 352 KB, 2536 KB/s, 0 seconds passed
... 2%, 384 KB, 2748 KB/s, 0 seconds passed
... 2%, 416 KB, 2964 KB/s, 0 seconds passed
... 2%, 448 KB, 3173 KB/s, 0 seconds passed
... 2%, 480 KB, 3379 KB/s, 0 seconds passed
... 2%, 512 KB, 3579 KB/s, 0 seconds passed
... 3%, 544 KB, 3779 KB/s, 0 seconds passed

.. parsed-literal::

    ... 3%, 576 KB, 3977 KB/s, 0 seconds passed
... 3%, 608 KB, 4182 KB/s, 0 seconds passed
... 3%, 640 KB, 3708 KB/s, 0 seconds passed
... 3%, 672 KB, 3878 KB/s, 0 seconds passed
... 3%, 704 KB, 4052 KB/s, 0 seconds passed
... 4%, 736 KB, 4227 KB/s, 0 seconds passed
... 4%, 768 KB, 4401 KB/s, 0 seconds passed
... 4%, 800 KB, 4574 KB/s, 0 seconds passed
... 4%, 832 KB, 4749 KB/s, 0 seconds passed
... 4%, 864 KB, 4877 KB/s, 0 seconds passed
... 4%, 896 KB, 5044 KB/s, 0 seconds passed
... 5%, 928 KB, 5213 KB/s, 0 seconds passed
... 5%, 960 KB, 5381 KB/s, 0 seconds passed
... 5%, 992 KB, 5548 KB/s, 0 seconds passed
... 5%, 1024 KB, 5715 KB/s, 0 seconds passed
... 5%, 1056 KB, 5881 KB/s, 0 seconds passed
... 6%, 1088 KB, 6046 KB/s, 0 seconds passed
... 6%, 1120 KB, 6210 KB/s, 0 seconds passed
... 6%, 1152 KB, 6373 KB/s, 0 seconds passed
... 6%, 1184 KB, 6538 KB/s, 0 seconds passed
... 6%, 1216 KB, 6704 KB/s, 0 seconds passed
... 6%, 1248 KB, 6869 KB/s, 0 seconds passed
... 7%, 1280 KB, 7034 KB/s, 0 seconds passed

.. parsed-literal::

    ... 7%, 1312 KB, 6323 KB/s, 0 seconds passed
... 7%, 1344 KB, 6450 KB/s, 0 seconds passed
... 7%, 1376 KB, 6482 KB/s, 0 seconds passed
... 7%, 1408 KB, 6617 KB/s, 0 seconds passed
... 8%, 1440 KB, 6755 KB/s, 0 seconds passed
... 8%, 1472 KB, 6892 KB/s, 0 seconds passed
... 8%, 1504 KB, 7027 KB/s, 0 seconds passed
... 8%, 1536 KB, 7163 KB/s, 0 seconds passed
... 8%, 1568 KB, 7300 KB/s, 0 seconds passed
... 8%, 1600 KB, 7437 KB/s, 0 seconds passed
... 9%, 1632 KB, 7572 KB/s, 0 seconds passed
... 9%, 1664 KB, 7708 KB/s, 0 seconds passed
... 9%, 1696 KB, 7844 KB/s, 0 seconds passed
... 9%, 1728 KB, 7978 KB/s, 0 seconds passed
... 9%, 1760 KB, 8110 KB/s, 0 seconds passed
... 9%, 1792 KB, 8243 KB/s, 0 seconds passed
... 10%, 1824 KB, 8376 KB/s, 0 seconds passed
... 10%, 1856 KB, 8508 KB/s, 0 seconds passed
... 10%, 1888 KB, 8641 KB/s, 0 seconds passed
... 10%, 1920 KB, 8772 KB/s, 0 seconds passed
... 10%, 1952 KB, 8904 KB/s, 0 seconds passed
... 11%, 1984 KB, 9035 KB/s, 0 seconds passed
... 11%, 2016 KB, 9166 KB/s, 0 seconds passed
... 11%, 2048 KB, 9296 KB/s, 0 seconds passed
... 11%, 2080 KB, 9425 KB/s, 0 seconds passed
... 11%, 2112 KB, 9554 KB/s, 0 seconds passed
... 11%, 2144 KB, 9681 KB/s, 0 seconds passed
... 12%, 2176 KB, 9810 KB/s, 0 seconds passed
... 12%, 2208 KB, 9937 KB/s, 0 seconds passed
... 12%, 2240 KB, 10065 KB/s, 0 seconds passed
... 12%, 2272 KB, 10191 KB/s, 0 seconds passed
... 12%, 2304 KB, 10318 KB/s, 0 seconds passed
... 12%, 2336 KB, 10444 KB/s, 0 seconds passed
... 13%, 2368 KB, 10572 KB/s, 0 seconds passed
... 13%, 2400 KB, 10698 KB/s, 0 seconds passed
... 13%, 2432 KB, 10822 KB/s, 0 seconds passed
... 13%, 2464 KB, 10947 KB/s, 0 seconds passed
... 13%, 2496 KB, 11072 KB/s, 0 seconds passed
... 14%, 2528 KB, 11197 KB/s, 0 seconds passed
... 14%, 2560 KB, 11323 KB/s, 0 seconds passed
... 14%, 2592 KB, 11449 KB/s, 0 seconds passed
... 14%, 2624 KB, 11577 KB/s, 0 seconds passed
... 14%, 2656 KB, 10939 KB/s, 0 seconds passed
... 14%, 2688 KB, 11053 KB/s, 0 seconds passed
... 15%, 2720 KB, 11131 KB/s, 0 seconds passed
... 15%, 2752 KB, 11245 KB/s, 0 seconds passed
... 15%, 2784 KB, 11362 KB/s, 0 seconds passed
... 15%, 2816 KB, 11479 KB/s, 0 seconds passed
... 15%, 2848 KB, 11596 KB/s, 0 seconds passed
... 16%, 2880 KB, 11712 KB/s, 0 seconds passed
... 16%, 2912 KB, 11827 KB/s, 0 seconds passed
... 16%, 2944 KB, 11940 KB/s, 0 seconds passed

.. parsed-literal::

    ... 16%, 2976 KB, 12054 KB/s, 0 seconds passed
... 16%, 3008 KB, 12167 KB/s, 0 seconds passed
... 16%, 3040 KB, 12279 KB/s, 0 seconds passed
... 17%, 3072 KB, 12391 KB/s, 0 seconds passed
... 17%, 3104 KB, 12501 KB/s, 0 seconds passed
... 17%, 3136 KB, 12610 KB/s, 0 seconds passed
... 17%, 3168 KB, 12719 KB/s, 0 seconds passed
... 17%, 3200 KB, 12829 KB/s, 0 seconds passed
... 17%, 3232 KB, 12940 KB/s, 0 seconds passed
... 18%, 3264 KB, 13050 KB/s, 0 seconds passed
... 18%, 3296 KB, 13159 KB/s, 0 seconds passed
... 18%, 3328 KB, 13269 KB/s, 0 seconds passed
... 18%, 3360 KB, 13378 KB/s, 0 seconds passed
... 18%, 3392 KB, 13487 KB/s, 0 seconds passed
... 19%, 3424 KB, 13593 KB/s, 0 seconds passed
... 19%, 3456 KB, 13701 KB/s, 0 seconds passed
... 19%, 3488 KB, 13809 KB/s, 0 seconds passed
... 19%, 3520 KB, 13915 KB/s, 0 seconds passed
... 19%, 3552 KB, 14022 KB/s, 0 seconds passed
... 19%, 3584 KB, 14129 KB/s, 0 seconds passed
... 20%, 3616 KB, 14234 KB/s, 0 seconds passed
... 20%, 3648 KB, 14341 KB/s, 0 seconds passed
... 20%, 3680 KB, 14447 KB/s, 0 seconds passed
... 20%, 3712 KB, 14552 KB/s, 0 seconds passed
... 20%, 3744 KB, 14659 KB/s, 0 seconds passed
... 20%, 3776 KB, 14764 KB/s, 0 seconds passed
... 21%, 3808 KB, 14869 KB/s, 0 seconds passed
... 21%, 3840 KB, 14975 KB/s, 0 seconds passed
... 21%, 3872 KB, 15089 KB/s, 0 seconds passed
... 21%, 3904 KB, 15202 KB/s, 0 seconds passed
... 21%, 3936 KB, 15316 KB/s, 0 seconds passed
... 22%, 3968 KB, 15430 KB/s, 0 seconds passed
... 22%, 4000 KB, 15543 KB/s, 0 seconds passed
... 22%, 4032 KB, 15657 KB/s, 0 seconds passed
... 22%, 4064 KB, 15771 KB/s, 0 seconds passed
... 22%, 4096 KB, 15884 KB/s, 0 seconds passed
... 22%, 4128 KB, 15473 KB/s, 0 seconds passed
... 23%, 4160 KB, 15566 KB/s, 0 seconds passed
... 23%, 4192 KB, 15662 KB/s, 0 seconds passed
... 23%, 4224 KB, 15760 KB/s, 0 seconds passed
... 23%, 4256 KB, 15858 KB/s, 0 seconds passed
... 23%, 4288 KB, 15954 KB/s, 0 seconds passed
... 24%, 4320 KB, 16053 KB/s, 0 seconds passed
... 24%, 4352 KB, 16154 KB/s, 0 seconds passed
... 24%, 4384 KB, 16251 KB/s, 0 seconds passed
... 24%, 4416 KB, 16346 KB/s, 0 seconds passed
... 24%, 4448 KB, 16441 KB/s, 0 seconds passed
... 24%, 4480 KB, 16537 KB/s, 0 seconds passed
... 25%, 4512 KB, 16631 KB/s, 0 seconds passed
... 25%, 4544 KB, 16727 KB/s, 0 seconds passed
... 25%, 4576 KB, 16826 KB/s, 0 seconds passed
... 25%, 4608 KB, 16922 KB/s, 0 seconds passed
... 25%, 4640 KB, 17016 KB/s, 0 seconds passed
... 25%, 4672 KB, 17111 KB/s, 0 seconds passed
... 26%, 4704 KB, 17151 KB/s, 0 seconds passed
... 26%, 4736 KB, 17243 KB/s, 0 seconds passed
... 26%, 4768 KB, 17337 KB/s, 0 seconds passed
... 26%, 4800 KB, 17430 KB/s, 0 seconds passed
... 26%, 4832 KB, 17523 KB/s, 0 seconds passed
... 27%, 4864 KB, 17615 KB/s, 0 seconds passed
... 27%, 4896 KB, 17709 KB/s, 0 seconds passed
... 27%, 4928 KB, 17803 KB/s, 0 seconds passed
... 27%, 4960 KB, 17897 KB/s, 0 seconds passed
... 27%, 4992 KB, 17992 KB/s, 0 seconds passed
... 27%, 5024 KB, 18089 KB/s, 0 seconds passed
... 28%, 5056 KB, 18185 KB/s, 0 seconds passed
... 28%, 5088 KB, 18280 KB/s, 0 seconds passed
... 28%, 5120 KB, 18375 KB/s, 0 seconds passed
... 28%, 5152 KB, 18469 KB/s, 0 seconds passed
... 28%, 5184 KB, 18563 KB/s, 0 seconds passed
... 28%, 5216 KB, 18657 KB/s, 0 seconds passed
... 29%, 5248 KB, 18751 KB/s, 0 seconds passed
... 29%, 5280 KB, 18846 KB/s, 0 seconds passed
... 29%, 5312 KB, 18940 KB/s, 0 seconds passed
... 29%, 5344 KB, 19033 KB/s, 0 seconds passed
... 29%, 5376 KB, 19126 KB/s, 0 seconds passed
... 30%, 5408 KB, 19218 KB/s, 0 seconds passed
... 30%, 5440 KB, 19311 KB/s, 0 seconds passed
... 30%, 5472 KB, 19404 KB/s, 0 seconds passed
... 30%, 5504 KB, 19497 KB/s, 0 seconds passed
... 30%, 5536 KB, 19587 KB/s, 0 seconds passed
... 30%, 5568 KB, 19679 KB/s, 0 seconds passed
... 31%, 5600 KB, 19772 KB/s, 0 seconds passed
... 31%, 5632 KB, 19864 KB/s, 0 seconds passed
... 31%, 5664 KB, 19957 KB/s, 0 seconds passed
... 31%, 5696 KB, 20048 KB/s, 0 seconds passed
... 31%, 5728 KB, 20140 KB/s, 0 seconds passed
... 32%, 5760 KB, 20230 KB/s, 0 seconds passed
... 32%, 5792 KB, 20321 KB/s, 0 seconds passed
... 32%, 5824 KB, 20411 KB/s, 0 seconds passed
... 32%, 5856 KB, 20502 KB/s, 0 seconds passed
... 32%, 5888 KB, 20592 KB/s, 0 seconds passed
... 32%, 5920 KB, 20682 KB/s, 0 seconds passed
... 33%, 5952 KB, 20775 KB/s, 0 seconds passed
... 33%, 5984 KB, 20872 KB/s, 0 seconds passed
... 33%, 6016 KB, 20969 KB/s, 0 seconds passed
... 33%, 6048 KB, 21065 KB/s, 0 seconds passed
... 33%, 6080 KB, 21162 KB/s, 0 seconds passed
... 33%, 6112 KB, 21257 KB/s, 0 seconds passed
... 34%, 6144 KB, 21353 KB/s, 0 seconds passed
... 34%, 6176 KB, 21449 KB/s, 0 seconds passed
... 34%, 6208 KB, 21545 KB/s, 0 seconds passed
... 34%, 6240 KB, 21641 KB/s, 0 seconds passed
... 34%, 6272 KB, 21737 KB/s, 0 seconds passed
... 35%, 6304 KB, 21831 KB/s, 0 seconds passed
... 35%, 6336 KB, 21926 KB/s, 0 seconds passed
... 35%, 6368 KB, 22021 KB/s, 0 seconds passed
... 35%, 6400 KB, 22116 KB/s, 0 seconds passed
... 35%, 6432 KB, 22210 KB/s, 0 seconds passed
... 35%, 6464 KB, 22305 KB/s, 0 seconds passed
... 36%, 6496 KB, 22399 KB/s, 0 seconds passed
... 36%, 6528 KB, 22493 KB/s, 0 seconds passed
... 36%, 6560 KB, 22588 KB/s, 0 seconds passed
... 36%, 6592 KB, 22683 KB/s, 0 seconds passed
... 36%, 6624 KB, 22773 KB/s, 0 seconds passed
... 36%, 6656 KB, 22862 KB/s, 0 seconds passed
... 37%, 6688 KB, 22947 KB/s, 0 seconds passed
... 37%, 6720 KB, 23036 KB/s, 0 seconds passed
... 37%, 6752 KB, 23125 KB/s, 0 seconds passed
... 37%, 6784 KB, 23209 KB/s, 0 seconds passed
... 37%, 6816 KB, 23297 KB/s, 0 seconds passed
... 38%, 6848 KB, 23385 KB/s, 0 seconds passed
... 38%, 6880 KB, 23474 KB/s, 0 seconds passed
... 38%, 6912 KB, 23562 KB/s, 0 seconds passed
... 38%, 6944 KB, 23645 KB/s, 0 seconds passed
... 38%, 6976 KB, 23733 KB/s, 0 seconds passed
... 38%, 7008 KB, 23804 KB/s, 0 seconds passed

.. parsed-literal::

    ... 39%, 7040 KB, 23334 KB/s, 0 seconds passed
... 39%, 7072 KB, 23413 KB/s, 0 seconds passed
... 39%, 7104 KB, 23495 KB/s, 0 seconds passed
... 39%, 7136 KB, 23577 KB/s, 0 seconds passed
... 39%, 7168 KB, 23658 KB/s, 0 seconds passed
... 40%, 7200 KB, 23738 KB/s, 0 seconds passed
... 40%, 7232 KB, 23819 KB/s, 0 seconds passed
... 40%, 7264 KB, 23898 KB/s, 0 seconds passed
... 40%, 7296 KB, 23978 KB/s, 0 seconds passed
... 40%, 7328 KB, 24058 KB/s, 0 seconds passed
... 40%, 7360 KB, 24137 KB/s, 0 seconds passed
... 41%, 7392 KB, 24217 KB/s, 0 seconds passed
... 41%, 7424 KB, 24298 KB/s, 0 seconds passed
... 41%, 7456 KB, 24376 KB/s, 0 seconds passed
... 41%, 7488 KB, 24455 KB/s, 0 seconds passed
... 41%, 7520 KB, 24535 KB/s, 0 seconds passed
... 41%, 7552 KB, 24612 KB/s, 0 seconds passed
... 42%, 7584 KB, 24687 KB/s, 0 seconds passed
... 42%, 7616 KB, 24761 KB/s, 0 seconds passed
... 42%, 7648 KB, 24835 KB/s, 0 seconds passed
... 42%, 7680 KB, 24910 KB/s, 0 seconds passed
... 42%, 7712 KB, 24983 KB/s, 0 seconds passed
... 43%, 7744 KB, 25055 KB/s, 0 seconds passed
... 43%, 7776 KB, 25129 KB/s, 0 seconds passed
... 43%, 7808 KB, 25202 KB/s, 0 seconds passed
... 43%, 7840 KB, 25275 KB/s, 0 seconds passed
... 43%, 7872 KB, 25348 KB/s, 0 seconds passed
... 43%, 7904 KB, 25422 KB/s, 0 seconds passed
... 44%, 7936 KB, 25495 KB/s, 0 seconds passed
... 44%, 7968 KB, 25567 KB/s, 0 seconds passed
... 44%, 8000 KB, 25639 KB/s, 0 seconds passed
... 44%, 8032 KB, 25712 KB/s, 0 seconds passed
... 44%, 8064 KB, 25781 KB/s, 0 seconds passed
... 45%, 8096 KB, 25852 KB/s, 0 seconds passed
... 45%, 8128 KB, 25925 KB/s, 0 seconds passed
... 45%, 8160 KB, 25995 KB/s, 0 seconds passed
... 45%, 8192 KB, 26066 KB/s, 0 seconds passed
... 45%, 8224 KB, 26137 KB/s, 0 seconds passed
... 45%, 8256 KB, 26206 KB/s, 0 seconds passed
... 46%, 8288 KB, 26278 KB/s, 0 seconds passed
... 46%, 8320 KB, 26350 KB/s, 0 seconds passed
... 46%, 8352 KB, 26420 KB/s, 0 seconds passed
... 46%, 8384 KB, 26491 KB/s, 0 seconds passed
... 46%, 8416 KB, 26556 KB/s, 0 seconds passed
... 46%, 8448 KB, 26636 KB/s, 0 seconds passed
... 47%, 8480 KB, 26717 KB/s, 0 seconds passed
... 47%, 8512 KB, 26798 KB/s, 0 seconds passed
... 47%, 8544 KB, 26879 KB/s, 0 seconds passed
... 47%, 8576 KB, 26960 KB/s, 0 seconds passed
... 47%, 8608 KB, 27041 KB/s, 0 seconds passed
... 48%, 8640 KB, 27121 KB/s, 0 seconds passed
... 48%, 8672 KB, 27201 KB/s, 0 seconds passed
... 48%, 8704 KB, 27280 KB/s, 0 seconds passed
... 48%, 8736 KB, 27360 KB/s, 0 seconds passed
... 48%, 8768 KB, 27438 KB/s, 0 seconds passed
... 48%, 8800 KB, 27518 KB/s, 0 seconds passed
... 49%, 8832 KB, 27598 KB/s, 0 seconds passed
... 49%, 8864 KB, 27676 KB/s, 0 seconds passed
... 49%, 8896 KB, 27755 KB/s, 0 seconds passed
... 49%, 8928 KB, 27833 KB/s, 0 seconds passed
... 49%, 8960 KB, 27911 KB/s, 0 seconds passed
... 49%, 8992 KB, 27989 KB/s, 0 seconds passed
... 50%, 9024 KB, 28068 KB/s, 0 seconds passed
... 50%, 9056 KB, 28145 KB/s, 0 seconds passed
... 50%, 9088 KB, 28224 KB/s, 0 seconds passed
... 50%, 9120 KB, 28302 KB/s, 0 seconds passed
... 50%, 9152 KB, 28380 KB/s, 0 seconds passed
... 51%, 9184 KB, 28458 KB/s, 0 seconds passed
... 51%, 9216 KB, 28535 KB/s, 0 seconds passed
... 51%, 9248 KB, 28613 KB/s, 0 seconds passed
... 51%, 9280 KB, 28691 KB/s, 0 seconds passed
... 51%, 9312 KB, 28769 KB/s, 0 seconds passed
... 51%, 9344 KB, 28846 KB/s, 0 seconds passed
... 52%, 9376 KB, 28923 KB/s, 0 seconds passed
... 52%, 9408 KB, 29000 KB/s, 0 seconds passed
... 52%, 9440 KB, 29078 KB/s, 0 seconds passed
... 52%, 9472 KB, 29153 KB/s, 0 seconds passed
... 52%, 9504 KB, 29229 KB/s, 0 seconds passed
... 53%, 9536 KB, 29306 KB/s, 0 seconds passed
... 53%, 9568 KB, 29383 KB/s, 0 seconds passed
... 53%, 9600 KB, 29460 KB/s, 0 seconds passed
... 53%, 9632 KB, 29537 KB/s, 0 seconds passed
... 53%, 9664 KB, 29612 KB/s, 0 seconds passed
... 53%, 9696 KB, 29687 KB/s, 0 seconds passed
... 54%, 9728 KB, 29763 KB/s, 0 seconds passed
... 54%, 9760 KB, 29843 KB/s, 0 seconds passed
... 54%, 9792 KB, 29925 KB/s, 0 seconds passed
... 54%, 9824 KB, 30008 KB/s, 0 seconds passed
... 54%, 9856 KB, 30091 KB/s, 0 seconds passed
... 54%, 9888 KB, 30173 KB/s, 0 seconds passed
... 55%, 9920 KB, 30256 KB/s, 0 seconds passed
... 55%, 9952 KB, 30338 KB/s, 0 seconds passed
... 55%, 9984 KB, 30420 KB/s, 0 seconds passed
... 55%, 10016 KB, 30502 KB/s, 0 seconds passed
... 55%, 10048 KB, 30585 KB/s, 0 seconds passed
... 56%, 10080 KB, 30667 KB/s, 0 seconds passed
... 56%, 10112 KB, 30747 KB/s, 0 seconds passed
... 56%, 10144 KB, 30829 KB/s, 0 seconds passed
... 56%, 10176 KB, 30910 KB/s, 0 seconds passed
... 56%, 10208 KB, 30990 KB/s, 0 seconds passed
... 56%, 10240 KB, 31066 KB/s, 0 seconds passed
... 57%, 10272 KB, 31137 KB/s, 0 seconds passed
... 57%, 10304 KB, 31219 KB/s, 0 seconds passed
... 57%, 10336 KB, 31292 KB/s, 0 seconds passed
... 57%, 10368 KB, 31359 KB/s, 0 seconds passed
... 57%, 10400 KB, 31427 KB/s, 0 seconds passed
... 57%, 10432 KB, 31482 KB/s, 0 seconds passed
... 58%, 10464 KB, 31550 KB/s, 0 seconds passed
... 58%, 10496 KB, 31631 KB/s, 0 seconds passed
... 58%, 10528 KB, 31709 KB/s, 0 seconds passed
... 58%, 10560 KB, 31780 KB/s, 0 seconds passed
... 58%, 10592 KB, 31851 KB/s, 0 seconds passed
... 59%, 10624 KB, 31922 KB/s, 0 seconds passed
... 59%, 10656 KB, 31988 KB/s, 0 seconds passed
... 59%, 10688 KB, 32059 KB/s, 0 seconds passed
... 59%, 10720 KB, 32119 KB/s, 0 seconds passed
... 59%, 10752 KB, 32184 KB/s, 0 seconds passed
... 59%, 10784 KB, 32255 KB/s, 0 seconds passed
... 60%, 10816 KB, 32325 KB/s, 0 seconds passed
... 60%, 10848 KB, 32390 KB/s, 0 seconds passed
... 60%, 10880 KB, 32460 KB/s, 0 seconds passed
... 60%, 10912 KB, 32530 KB/s, 0 seconds passed
... 60%, 10944 KB, 32605 KB/s, 0 seconds passed
... 61%, 10976 KB, 32674 KB/s, 0 seconds passed
... 61%, 11008 KB, 32744 KB/s, 0 seconds passed
... 61%, 11040 KB, 32813 KB/s, 0 seconds passed
... 61%, 11072 KB, 32872 KB/s, 0 seconds passed
... 61%, 11104 KB, 32942 KB/s, 0 seconds passed
... 61%, 11136 KB, 33011 KB/s, 0 seconds passed
... 62%, 11168 KB, 33075 KB/s, 0 seconds passed
... 62%, 11200 KB, 33149 KB/s, 0 seconds passed
... 62%, 11232 KB, 33219 KB/s, 0 seconds passed
... 62%, 11264 KB, 33282 KB/s, 0 seconds passed
... 62%, 11296 KB, 33351 KB/s, 0 seconds passed
... 62%, 11328 KB, 33419 KB/s, 0 seconds passed
... 63%, 11360 KB, 33483 KB/s, 0 seconds passed
... 63%, 11392 KB, 33551 KB/s, 0 seconds passed
... 63%, 11424 KB, 33619 KB/s, 0 seconds passed
... 63%, 11456 KB, 33672 KB/s, 0 seconds passed
... 63%, 11488 KB, 33735 KB/s, 0 seconds passed
... 64%, 11520 KB, 33785 KB/s, 0 seconds passed
... 64%, 11552 KB, 33839 KB/s, 0 seconds passed
... 64%, 11584 KB, 33920 KB/s, 0 seconds passed
... 64%, 11616 KB, 34002 KB/s, 0 seconds passed
... 64%, 11648 KB, 34081 KB/s, 0 seconds passed
... 64%, 11680 KB, 34144 KB/s, 0 seconds passed
... 65%, 11712 KB, 34197 KB/s, 0 seconds passed
... 65%, 11744 KB, 34250 KB/s, 0 seconds passed
... 65%, 11776 KB, 34301 KB/s, 0 seconds passed
... 65%, 11808 KB, 34380 KB/s, 0 seconds passed
... 65%, 11840 KB, 34460 KB/s, 0 seconds passed
... 65%, 11872 KB, 34537 KB/s, 0 seconds passed
... 66%, 11904 KB, 34604 KB/s, 0 seconds passed
... 66%, 11936 KB, 34671 KB/s, 0 seconds passed
... 66%, 11968 KB, 34732 KB/s, 0 seconds passed
... 66%, 12000 KB, 34788 KB/s, 0 seconds passed
... 66%, 12032 KB, 34854 KB/s, 0 seconds passed
... 67%, 12064 KB, 34920 KB/s, 0 seconds passed
... 67%, 12096 KB, 34981 KB/s, 0 seconds passed
... 67%, 12128 KB, 35047 KB/s, 0 seconds passed
... 67%, 12160 KB, 35113 KB/s, 0 seconds passed
... 67%, 12192 KB, 35173 KB/s, 0 seconds passed
... 67%, 12224 KB, 35239 KB/s, 0 seconds passed
... 68%, 12256 KB, 35299 KB/s, 0 seconds passed
... 68%, 12288 KB, 35365 KB/s, 0 seconds passed
... 68%, 12320 KB, 35430 KB/s, 0 seconds passed
... 68%, 12352 KB, 35490 KB/s, 0 seconds passed
... 68%, 12384 KB, 35555 KB/s, 0 seconds passed
... 69%, 12416 KB, 35620 KB/s, 0 seconds passed
... 69%, 12448 KB, 35685 KB/s, 0 seconds passed
... 69%, 12480 KB, 35750 KB/s, 0 seconds passed
... 69%, 12512 KB, 35809 KB/s, 0 seconds passed

.. parsed-literal::

    ... 69%, 12544 KB, 35874 KB/s, 0 seconds passed
... 69%, 12576 KB, 35939 KB/s, 0 seconds passed
... 70%, 12608 KB, 36003 KB/s, 0 seconds passed
... 70%, 12640 KB, 36062 KB/s, 0 seconds passed
... 70%, 12672 KB, 36127 KB/s, 0 seconds passed
... 70%, 12704 KB, 36191 KB/s, 0 seconds passed
... 70%, 12736 KB, 36249 KB/s, 0 seconds passed
... 70%, 12768 KB, 36308 KB/s, 0 seconds passed
... 71%, 12800 KB, 36372 KB/s, 0 seconds passed
... 71%, 12832 KB, 36436 KB/s, 0 seconds passed
... 71%, 12864 KB, 36494 KB/s, 0 seconds passed
... 71%, 12896 KB, 36558 KB/s, 0 seconds passed
... 71%, 12928 KB, 36616 KB/s, 0 seconds passed
... 72%, 12960 KB, 36679 KB/s, 0 seconds passed
... 72%, 12992 KB, 36743 KB/s, 0 seconds passed
... 72%, 13024 KB, 36806 KB/s, 0 seconds passed
... 72%, 13056 KB, 36864 KB/s, 0 seconds passed
... 72%, 13088 KB, 36927 KB/s, 0 seconds passed
... 72%, 13120 KB, 36990 KB/s, 0 seconds passed
... 73%, 13152 KB, 37059 KB/s, 0 seconds passed
... 73%, 13184 KB, 37121 KB/s, 0 seconds passed
... 73%, 13216 KB, 37184 KB/s, 0 seconds passed
... 73%, 13248 KB, 37241 KB/s, 0 seconds passed
... 73%, 13280 KB, 37304 KB/s, 0 seconds passed
... 73%, 13312 KB, 37366 KB/s, 0 seconds passed
... 74%, 13344 KB, 37423 KB/s, 0 seconds passed
... 74%, 13376 KB, 37485 KB/s, 0 seconds passed
... 74%, 13408 KB, 37548 KB/s, 0 seconds passed
... 74%, 13440 KB, 37604 KB/s, 0 seconds passed
... 74%, 13472 KB, 37665 KB/s, 0 seconds passed
... 75%, 13504 KB, 37727 KB/s, 0 seconds passed
... 75%, 13536 KB, 37783 KB/s, 0 seconds passed
... 75%, 13568 KB, 37845 KB/s, 0 seconds passed
... 75%, 13600 KB, 37906 KB/s, 0 seconds passed
... 75%, 13632 KB, 37968 KB/s, 0 seconds passed
... 75%, 13664 KB, 38024 KB/s, 0 seconds passed
... 76%, 13696 KB, 38085 KB/s, 0 seconds passed
... 76%, 13728 KB, 38141 KB/s, 0 seconds passed
... 76%, 13760 KB, 38202 KB/s, 0 seconds passed
... 76%, 13792 KB, 38263 KB/s, 0 seconds passed
... 76%, 13824 KB, 38324 KB/s, 0 seconds passed
... 77%, 13856 KB, 38379 KB/s, 0 seconds passed
... 77%, 13888 KB, 38440 KB/s, 0 seconds passed
... 77%, 13920 KB, 38491 KB/s, 0 seconds passed
... 77%, 13952 KB, 38537 KB/s, 0 seconds passed
... 77%, 13984 KB, 38587 KB/s, 0 seconds passed
... 77%, 14016 KB, 38661 KB/s, 0 seconds passed
... 78%, 14048 KB, 38735 KB/s, 0 seconds passed
... 78%, 14080 KB, 38784 KB/s, 0 seconds passed
... 78%, 14112 KB, 38846 KB/s, 0 seconds passed
... 78%, 14144 KB, 38908 KB/s, 0 seconds passed
... 78%, 14176 KB, 38963 KB/s, 0 seconds passed
... 78%, 14208 KB, 39022 KB/s, 0 seconds passed
... 79%, 14240 KB, 39082 KB/s, 0 seconds passed
... 79%, 14272 KB, 39142 KB/s, 0 seconds passed
... 79%, 14304 KB, 39196 KB/s, 0 seconds passed
... 79%, 14336 KB, 39255 KB/s, 0 seconds passed
... 79%, 14368 KB, 39310 KB/s, 0 seconds passed
... 80%, 14400 KB, 39368 KB/s, 0 seconds passed
... 80%, 14432 KB, 39412 KB/s, 0 seconds passed
... 80%, 14464 KB, 39472 KB/s, 0 seconds passed
... 80%, 14496 KB, 39543 KB/s, 0 seconds passed
... 80%, 14528 KB, 39602 KB/s, 0 seconds passed
... 80%, 14560 KB, 39655 KB/s, 0 seconds passed
... 81%, 14592 KB, 39714 KB/s, 0 seconds passed
... 81%, 14624 KB, 39773 KB/s, 0 seconds passed
... 81%, 14656 KB, 39810 KB/s, 0 seconds passed
... 81%, 14688 KB, 39853 KB/s, 0 seconds passed
... 81%, 14720 KB, 39895 KB/s, 0 seconds passed
... 82%, 14752 KB, 39938 KB/s, 0 seconds passed
... 82%, 14784 KB, 39991 KB/s, 0 seconds passed
... 82%, 14816 KB, 40063 KB/s, 0 seconds passed
... 82%, 14848 KB, 40137 KB/s, 0 seconds passed
... 82%, 14880 KB, 40210 KB/s, 0 seconds passed
... 82%, 14912 KB, 40278 KB/s, 0 seconds passed
... 83%, 14944 KB, 40335 KB/s, 0 seconds passed
... 83%, 14976 KB, 40387 KB/s, 0 seconds passed
... 83%, 15008 KB, 40445 KB/s, 0 seconds passed
... 83%, 15040 KB, 40503 KB/s, 0 seconds passed
... 83%, 15072 KB, 40555 KB/s, 0 seconds passed
... 83%, 15104 KB, 40612 KB/s, 0 seconds passed
... 84%, 15136 KB, 40670 KB/s, 0 seconds passed
... 84%, 15168 KB, 40722 KB/s, 0 seconds passed
... 84%, 15200 KB, 40761 KB/s, 0 seconds passed
... 84%, 15232 KB, 40801 KB/s, 0 seconds passed
... 84%, 15264 KB, 40869 KB/s, 0 seconds passed
... 85%, 15296 KB, 40942 KB/s, 0 seconds passed
... 85%, 15328 KB, 40999 KB/s, 0 seconds passed
... 85%, 15360 KB, 41056 KB/s, 0 seconds passed
... 85%, 15392 KB, 41107 KB/s, 0 seconds passed
... 85%, 15424 KB, 41164 KB/s, 0 seconds passed
... 85%, 15456 KB, 41221 KB/s, 0 seconds passed
... 86%, 15488 KB, 41271 KB/s, 0 seconds passed
... 86%, 15520 KB, 41328 KB/s, 0 seconds passed
... 86%, 15552 KB, 41384 KB/s, 0 seconds passed
... 86%, 15584 KB, 41435 KB/s, 0 seconds passed
... 86%, 15616 KB, 41486 KB/s, 0 seconds passed
... 86%, 15648 KB, 41522 KB/s, 0 seconds passed
... 87%, 15680 KB, 41586 KB/s, 0 seconds passed
... 87%, 15712 KB, 41645 KB/s, 0 seconds passed
... 87%, 15744 KB, 41695 KB/s, 0 seconds passed
... 87%, 15776 KB, 41750 KB/s, 0 seconds passed
... 87%, 15808 KB, 41817 KB/s, 0 seconds passed
... 88%, 15840 KB, 41867 KB/s, 0 seconds passed
... 88%, 15872 KB, 41923 KB/s, 0 seconds passed
... 88%, 15904 KB, 41979 KB/s, 0 seconds passed
... 88%, 15936 KB, 42029 KB/s, 0 seconds passed
... 88%, 15968 KB, 42084 KB/s, 0 seconds passed
... 88%, 16000 KB, 42134 KB/s, 0 seconds passed
... 89%, 16032 KB, 42183 KB/s, 0 seconds passed
... 89%, 16064 KB, 42222 KB/s, 0 seconds passed
... 89%, 16096 KB, 42294 KB/s, 0 seconds passed
... 89%, 16128 KB, 42351 KB/s, 0 seconds passed
... 89%, 16160 KB, 42401 KB/s, 0 seconds passed
... 90%, 16192 KB, 42456 KB/s, 0 seconds passed
... 90%, 16224 KB, 42505 KB/s, 0 seconds passed
... 90%, 16256 KB, 42560 KB/s, 0 seconds passed
... 90%, 16288 KB, 42615 KB/s, 0 seconds passed
... 90%, 16320 KB, 42664 KB/s, 0 seconds passed
... 90%, 16352 KB, 42707 KB/s, 0 seconds passed
... 91%, 16384 KB, 42765 KB/s, 0 seconds passed
... 91%, 16416 KB, 42819 KB/s, 0 seconds passed
... 91%, 16448 KB, 42861 KB/s, 0 seconds passed
... 91%, 16480 KB, 42886 KB/s, 0 seconds passed
... 91%, 16512 KB, 42945 KB/s, 0 seconds passed
... 91%, 16544 KB, 43011 KB/s, 0 seconds passed
... 92%, 16576 KB, 43076 KB/s, 0 seconds passed
... 92%, 16608 KB, 43137 KB/s, 0 seconds passed
... 92%, 16640 KB, 43180 KB/s, 0 seconds passed
... 92%, 16672 KB, 43217 KB/s, 0 seconds passed
... 92%, 16704 KB, 43255 KB/s, 0 seconds passed
... 93%, 16736 KB, 43310 KB/s, 0 seconds passed
... 93%, 16768 KB, 43380 KB/s, 0 seconds passed
... 93%, 16800 KB, 43445 KB/s, 0 seconds passed
... 93%, 16832 KB, 43499 KB/s, 0 seconds passed
... 93%, 16864 KB, 43546 KB/s, 0 seconds passed
... 93%, 16896 KB, 43600 KB/s, 0 seconds passed
... 94%, 16928 KB, 43653 KB/s, 0 seconds passed
... 94%, 16960 KB, 43699 KB/s, 0 seconds passed
... 94%, 16992 KB, 43752 KB/s, 0 seconds passed
... 94%, 17024 KB, 43805 KB/s, 0 seconds passed
... 94%, 17056 KB, 43852 KB/s, 0 seconds passed
... 94%, 17088 KB, 43704 KB/s, 0 seconds passed
... 95%, 17120 KB, 43756 KB/s, 0 seconds passed
... 95%, 17152 KB, 43809 KB/s, 0 seconds passed
... 95%, 17184 KB, 43861 KB/s, 0 seconds passed
... 95%, 17216 KB, 43908 KB/s, 0 seconds passed
... 95%, 17248 KB, 43960 KB/s, 0 seconds passed
... 96%, 17280 KB, 44012 KB/s, 0 seconds passed
... 96%, 17312 KB, 44058 KB/s, 0 seconds passed
... 96%, 17344 KB, 44111 KB/s, 0 seconds passed
... 96%, 17376 KB, 44163 KB/s, 0 seconds passed
... 96%, 17408 KB, 44215 KB/s, 0 seconds passed
... 96%, 17440 KB, 44261 KB/s, 0 seconds passed
... 97%, 17472 KB, 44313 KB/s, 0 seconds passed
... 97%, 17504 KB, 44358 KB/s, 0 seconds passed
... 97%, 17536 KB, 44410 KB/s, 0 seconds passed
... 97%, 17568 KB, 44461 KB/s, 0 seconds passed
... 97%, 17600 KB, 44507 KB/s, 0 seconds passed
... 98%, 17632 KB, 44559 KB/s, 0 seconds passed
... 98%, 17664 KB, 44610 KB/s, 0 seconds passed
... 98%, 17696 KB, 44655 KB/s, 0 seconds passed
... 98%, 17728 KB, 44707 KB/s, 0 seconds passed
... 98%, 17760 KB, 44758 KB/s, 0 seconds passed
... 98%, 17792 KB, 44809 KB/s, 0 seconds passed
... 99%, 17824 KB, 44855 KB/s, 0 seconds passed
... 99%, 17856 KB, 44905 KB/s, 0 seconds passed
... 99%, 17888 KB, 44934 KB/s, 0 seconds passed
... 99%, 17920 KB, 44969 KB/s, 0 seconds passed
... 99%, 17952 KB, 45010 KB/s, 0 seconds passed
... 99%, 17984 KB, 45074 KB/s, 0 seconds passed
... 100%, 17990 KB, 45070 KB/s, 0 seconds passed



.. parsed-literal::


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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/omz_tools/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx



.. parsed-literal::

    ONNX check passed successfully.


.. parsed-literal::


    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False



.. parsed-literal::

    [ INFO ] MO command line tool is considered as the legacy conversion API as of OpenVINO 2023.2 release. Please use OpenVINO Model Converter (OVC). OVC represents a lightweight alternative of MO and provides simplified model conversion API.
    Find more information about transition from MO to OVC at https://docs.openvino.ai/2023.2/openvino_docs_OV_Converter_UG_prepare_model_convert_model_MO_OVC_transition.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-632/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin


Select inference device
~~~~~~~~~~~~~~~~~~~~~~~


Select device from dropdown list for running inference using OpenVINO

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

First, initialize the inference engine, OpenVINO Runtime. Then, read the
network architecture and model weights from the ``.bin`` and ``.xml``
files to compile for the desired device. An inference request is then
created to infer the compiled model.

.. code:: ipython3

    # initialize inference engine
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
