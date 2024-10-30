Convert Detectron2 Models to OpenVINO™
=========================================

`Detectron2 <https://github.com/facebookresearch/detectron2>`__ is
Facebook AI Research’s library that provides state-of-the-art detection
and segmentation algorithms. It is the successor of
`Detectron <https://github.com/facebookresearch/Detectron/>`__ and
`maskrcnn-benchmark <https://github.com/facebookresearch/maskrcnn-benchmark/>`__.
It supports a number of computer vision research projects and production
applications.

In this tutorial we consider how to convert and run Detectron2 models
using OpenVINO™. We will use ``Faster R-CNN FPN x1`` model and
``Mask R-CNN FPN x3`` pretrained on
`COCO <https://cocodataset.org/#home>`__ dataset as examples for object
detection and instance segmentation respectively.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__

   -  `Define helpers for PyTorch model initialization and
      conversion <#define-helpers-for-pytorch-model-initialization-and-conversion>`__
   -  `Prepare input data <#prepare-input-data>`__

-  `Object Detection <#object-detection>`__

   -  `Download PyTorch Detection
      model <#download-pytorch-detection-model>`__
   -  `Convert Detection Model to OpenVINO Intermediate
      Representation <#convert-detection-model-to-openvino-intermediate-representation>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Run Detection model inference <#run-detection-model-inference>`__

-  `Instance Segmentation <#instance-segmentation>`__

   -  `Download Instance Segmentation PyTorch
      model <#download-instance-segmentation-pytorch-model>`__
   -  `Convert Instance Segmentation Model to OpenVINO Intermediate
      Representation <#convert-instance-segmentation-model-to-openvino-intermediate-representation>`__
   -  `Select inference device <#select-inference-device>`__
   -  `Run Instance Segmentation model
      inference <#run-instance-segmentation-model-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



Install required packages for running model

.. code:: ipython3

    import os
    import requests
    from pathlib import Path
    import platform
    
    
    os.environ["GIT_CLONE_PROTECTION_ACTIVE"] = "false"
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        with open("notebook_utils.py", "w") as f:
            f.write(r.text)
    
    if not Path("pip_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/pip_helper.py",
        )
        open("pip_helper.py", "w").write(r.text)
    
    from pip_helper import pip_install
    
    if platform.system() == "Darwin":
        pip_install("numpy<2.0.0")
    pip_install("torch", "torchvision", "opencv-python", "wheel", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    pip_install("git+https://github.com/facebookresearch/detectron2.git", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    pip_install("openvino>=2023.1.0")


.. parsed-literal::

    Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu
    Requirement already satisfied: torch in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2.4.1+cpu)
    Requirement already satisfied: torchvision in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (0.19.1+cpu)
    Requirement already satisfied: opencv-python in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (4.10.0.84)
    Requirement already satisfied: wheel in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (0.44.0)
    Requirement already satisfied: filelock in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (3.16.1)
    Requirement already satisfied: typing-extensions>=4.8.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (4.12.2)
    Requirement already satisfied: sympy in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (1.13.3)
    Requirement already satisfied: networkx in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (3.1)
    Requirement already satisfied: jinja2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (3.1.4)
    Requirement already satisfied: fsspec in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torch) (2024.9.0)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torchvision) (1.23.5)
    Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from torchvision) (10.4.0)
    Requirement already satisfied: MarkupSafe>=2.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jinja2->torch) (2.1.5)
    Requirement already satisfied: mpmath<1.4,>=1.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from sympy->torch) (1.3.0)
    Looking in indexes: https://pypi.org/simple, https://download.pytorch.org/whl/cpu
    Collecting git+https://github.com/facebookresearch/detectron2.git
      Cloning https://github.com/facebookresearch/detectron2.git to /tmp/pip-req-build-we1e_5gi


.. parsed-literal::

      Running command git clone --filter=blob:none --quiet https://github.com/facebookresearch/detectron2.git /tmp/pip-req-build-we1e_5gi


.. parsed-literal::

      Resolved https://github.com/facebookresearch/detectron2.git to commit 8d85329aed8506ea3672e3e208971345973ea761
      Preparing metadata (setup.py): started
      Preparing metadata (setup.py): finished with status 'done'
    Requirement already satisfied: Pillow>=7.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (10.4.0)
    Requirement already satisfied: black in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (24.3.0)
    Collecting cloudpickle (from detectron2==0.6)
      Using cached cloudpickle-3.1.0-py3-none-any.whl.metadata (7.0 kB)
    Collecting fvcore<0.1.6,>=0.1.5 (from detectron2==0.6)
      Using cached fvcore-0.1.5.post20221221-py3-none-any.whl
    Collecting hydra-core>=1.1 (from detectron2==0.6)
      Using cached hydra_core-1.3.2-py3-none-any.whl.metadata (5.5 kB)
    Collecting iopath<0.1.10,>=0.1.7 (from detectron2==0.6)
      Using cached iopath-0.1.9-py3-none-any.whl.metadata (370 bytes)
    Requirement already satisfied: matplotlib in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (3.7.5)
    Requirement already satisfied: omegaconf<2.4,>=2.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (2.3.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (24.1)
    Collecting pycocotools>=2.0.2 (from detectron2==0.6)
      Using cached pycocotools-2.0.7-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.1 kB)
    Requirement already satisfied: tabulate in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (0.9.0)
    Requirement already satisfied: tensorboard in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (2.12.3)
    Requirement already satisfied: termcolor>=1.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (2.4.0)
    Requirement already satisfied: tqdm>4.29.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from detectron2==0.6) (4.66.5)
    Collecting yacs>=0.1.8 (from detectron2==0.6)
      Using cached yacs-0.1.8-py3-none-any.whl.metadata (639 bytes)
    Requirement already satisfied: numpy in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.6) (1.23.5)
    Requirement already satisfied: pyyaml>=5.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from fvcore<0.1.6,>=0.1.5->detectron2==0.6) (6.0.2)
    Requirement already satisfied: antlr4-python3-runtime==4.9.* in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from hydra-core>=1.1->detectron2==0.6) (4.9.3)
    Requirement already satisfied: importlib-resources in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from hydra-core>=1.1->detectron2==0.6) (6.4.5)
    Collecting portalocker (from iopath<0.1.10,>=0.1.7->detectron2==0.6)
      Using cached portalocker-2.10.1-py3-none-any.whl.metadata (8.5 kB)
    Requirement already satisfied: contourpy>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (1.1.1)
    Requirement already satisfied: cycler>=0.10 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (0.12.1)
    Requirement already satisfied: fonttools>=4.22.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (4.54.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (1.4.7)
    Requirement already satisfied: pyparsing>=2.3.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (3.1.4)
    Requirement already satisfied: python-dateutil>=2.7 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from matplotlib->detectron2==0.6) (2.9.0.post0)
    Requirement already satisfied: click>=8.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (8.1.7)
    Requirement already satisfied: mypy-extensions>=0.4.3 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (1.0.0)
    Requirement already satisfied: pathspec>=0.9.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (0.12.1)
    Requirement already satisfied: platformdirs>=2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (4.3.6)
    Requirement already satisfied: tomli>=1.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (2.0.2)
    Requirement already satisfied: typing-extensions>=4.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from black->detectron2==0.6) (4.12.2)
    Requirement already satisfied: absl-py>=0.4 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (1.4.0)
    Requirement already satisfied: grpcio>=1.48.2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (1.67.0)
    Requirement already satisfied: google-auth<3,>=1.6.3 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (2.35.0)
    Requirement already satisfied: google-auth-oauthlib<1.1,>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (1.0.0)
    Requirement already satisfied: markdown>=2.6.8 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (3.7)
    Requirement already satisfied: protobuf>=3.19.6 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (3.20.3)
    Requirement already satisfied: requests<3,>=2.21.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (2.32.3)
    Requirement already satisfied: setuptools>=41.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (44.0.0)
    Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (0.7.2)
    Requirement already satisfied: werkzeug>=1.0.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (3.0.4)
    Requirement already satisfied: wheel>=0.26 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from tensorboard->detectron2==0.6) (0.44.0)
    Requirement already satisfied: cachetools<6.0,>=2.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard->detectron2==0.6) (5.5.0)
    Requirement already satisfied: pyasn1-modules>=0.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard->detectron2==0.6) (0.4.1)
    Requirement already satisfied: rsa<5,>=3.1.4 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from google-auth<3,>=1.6.3->tensorboard->detectron2==0.6) (4.9)
    Requirement already satisfied: requests-oauthlib>=0.7.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from google-auth-oauthlib<1.1,>=0.5->tensorboard->detectron2==0.6) (2.0.0)
    Requirement already satisfied: zipp>=3.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from importlib-resources->hydra-core>=1.1->detectron2==0.6) (3.20.2)
    Requirement already satisfied: importlib-metadata>=4.4 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from markdown>=2.6.8->tensorboard->detectron2==0.6) (8.5.0)
    Requirement already satisfied: six>=1.5 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from python-dateutil>=2.7->matplotlib->detectron2==0.6) (1.16.0)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.6) (3.4.0)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.6) (3.10)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.6) (2.2.3)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard->detectron2==0.6) (2024.8.30)
    Requirement already satisfied: MarkupSafe>=2.1.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from werkzeug>=1.0.1->tensorboard->detectron2==0.6) (2.1.5)
    Requirement already satisfied: pyasn1<0.7.0,>=0.4.6 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard->detectron2==0.6) (0.6.1)
    Requirement already satisfied: oauthlib>=3.0.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<1.1,>=0.5->tensorboard->detectron2==0.6) (3.2.2)
    Using cached hydra_core-1.3.2-py3-none-any.whl (154 kB)
    Using cached iopath-0.1.9-py3-none-any.whl (27 kB)
    Using cached pycocotools-2.0.7-cp38-cp38-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (439 kB)
    Using cached yacs-0.1.8-py3-none-any.whl (14 kB)
    Using cached cloudpickle-3.1.0-py3-none-any.whl (22 kB)
    Using cached portalocker-2.10.1-py3-none-any.whl (18 kB)
    Building wheels for collected packages: detectron2
      Building wheel for detectron2 (setup.py): started
      Building wheel for detectron2 (setup.py): finished with status 'done'
      Created wheel for detectron2: filename=detectron2-0.6-cp38-cp38-linux_x86_64.whl size=8313552 sha256=23ceb6e5b734ecc530172b613be139d732deaa2e962d5a8bc940e6b23a85309d
      Stored in directory: /tmp/pip-ephem-wheel-cache-65iaghs7/wheels/19/ac/65/e48e5e4ec2702274d927c5a6efb75709b24014371d3bb778f2
    Successfully built detectron2
    Installing collected packages: yacs, portalocker, cloudpickle, iopath, hydra-core, pycocotools, fvcore, detectron2
    Successfully installed cloudpickle-3.1.0 detectron2-0.6 fvcore-0.1.5.post20221221 hydra-core-1.3.2 iopath-0.1.9 portalocker-2.10.1 pycocotools-2.0.7 yacs-0.1.8
    Requirement already satisfied: openvino>=2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2024.4.0)
    Requirement already satisfied: numpy<2.1.0,>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (1.23.5)
    Requirement already satisfied: openvino-telemetry>=2023.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (2024.1.0)
    Requirement already satisfied: packaging in /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino>=2023.1.0) (24.1)


Define helpers for PyTorch model initialization and conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Detectron2 provides universal and configurable API for working with
models, it means that all steps required for model creation, conversion
and inference will be common for all models, that is why it is enough to
define helper functions once, then reuse them for different models. For
obtaining models we will use `Detectron2 Model
Zoo <https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md>`__
API. ``detecton_zoo.get`` function allow to download and instantiate
model based on its config file. Configuration file is playing key role
in interaction with models in Detectron2 project and describes model
architecture and training and validation processes.
``detectron_zoo.get_config`` function can be used for finding and
reading model config.

.. code:: ipython3

    import detectron2.model_zoo as detectron_zoo
    
    
    def get_model_and_config(model_name: str):
        """
        Helper function for downloading PyTorch model and its configuration from Detectron2 Model Zoo
    
        Parameters:
          model_name (str): model_id from Detectron2 Model Zoo
        Returns:
          model (torch.nn.Module): Pretrained model instance
          cfg (Config): Configuration for model
        """
        cfg = detectron_zoo.get_config(model_name + ".yaml", trained=True)
        model = detectron_zoo.get(model_name + ".yaml", trained=True)
        return model, cfg

Detectron2 library is based on PyTorch. Starting from 2023.0 release
OpenVINO supports PyTorch models conversion directly via Model
Conversion API. ``ov.convert_model`` function can be used for converting
PyTorch model to OpenVINO Model object instance, that ready to use for
loading on device and then running inference or can be saved on disk for
next deployment using ``ov.save_model`` function.

Detectron2 models use custom complex data structures inside that brings
some difficulties for exporting models in different formats and
frameworks including OpenVINO. For avoid these issues,
``detectron2.export.TracingAdapter`` provided as part of Detectron2
deployment API. ``TracingAdapter`` is a model wrapper class that
simplify model’s structure making it more export-friendly.

.. code:: ipython3

    from detectron2.modeling import GeneralizedRCNN
    from detectron2.export import TracingAdapter
    import torch
    import openvino as ov
    import warnings
    from typing import List, Dict
    
    
    def convert_detectron2_model(model: torch.nn.Module, sample_input: List[Dict[str, torch.Tensor]]):
        """
        Function for converting Detectron2 models, creates TracingAdapter for making model tracing-friendly,
        prepares inputs and converts model to OpenVINO Model
    
        Parameters:
          model (torch.nn.Module): Model object for conversion
          sample_input (List[Dict[str, torch.Tensor]]): sample input for tracing
        Returns:
          ov_model (ov.Model): OpenVINO Model
        """
        # prepare input for tracing adapter
        tracing_input = [{"image": sample_input[0]["image"]}]
    
        # override model forward and disable postprocessing if required
        if isinstance(model, GeneralizedRCNN):
    
            def inference(model, inputs):
                # use do_postprocess=False so it returns ROI mask
                inst = model.inference(inputs, do_postprocess=False)[0]
                return [{"instances": inst}]
    
        else:
            inference = None  # assume that we just call the model directly
    
        # create traceable model
        traceable_model = TracingAdapter(model, tracing_input, inference)
        warnings.filterwarnings("ignore")
        # convert PyTorch model to OpenVINO model
        ov_model = ov.convert_model(traceable_model, example_input=sample_input[0]["image"])
        return ov_model

Prepare input data
~~~~~~~~~~~~~~~~~~



For running model conversion and inference we need to provide example
input. The cells below download sample image and apply preprocessing
steps based on model specific transformations defined in model config.

.. code:: ipython3

    import requests
    from pathlib import Path
    from PIL import Image
    
    MODEL_DIR = Path("model")
    DATA_DIR = Path("data")
    
    MODEL_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    
    input_image_url = "https://farm9.staticflickr.com/8040/8017130856_1b46b5f5fc_z.jpg"
    
    image_file = DATA_DIR / "example_image.jpg"
    
    if not image_file.exists():
        image = Image.open(requests.get(input_image_url, stream=True).raw)
        image.save(image_file)
    else:
        image = Image.open(image_file)
    
    image




.. image:: detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_8_0.png



.. code:: ipython3

    import detectron2.data.transforms as T
    from detectron2.data import detection_utils
    import torch
    
    
    def get_sample_inputs(image_path, cfg):
        # get a sample data
        original_image = detection_utils.read_image(image_path, format=cfg.INPUT.FORMAT)
        # Do same preprocessing as DefaultPredictor
        aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST)
        height, width = original_image.shape[:2]
        image = aug.get_transform(original_image).apply_image(original_image)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
    
        inputs = {"image": image, "height": height, "width": width}
    
        # Sample ready
        sample_inputs = [inputs]
        return sample_inputs

Now, when all components required for model conversion are prepared, we
can consider how to use them on specific examples.

Object Detection
----------------



Download PyTorch Detection model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Download faster_rcnn_R_50_FPN_1x from Detectron Model Zoo.

.. code:: ipython3

    model_name = "COCO-Detection/faster_rcnn_R_50_FPN_1x"
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)

Convert Detection Model to OpenVINO Intermediate Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Convert model using ``convert_detectron2_model`` function and
``sample_input`` prepared above. After conversion, model saved on disk
using ``ov.save_model`` function and can be found in ``model``
directory.

.. code:: ipython3

    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + ".xml")
    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + ".xml"))
    else:
        ov_model = model_xml_path

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    from notebook_utils import device_widget
    
    core = ov.Core()
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Run Detection model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Load our converted model on selected device and run inference on sample
input.

.. code:: ipython3

    compiled_model = core.compile_model(ov_model, device.value)

.. code:: ipython3

    results = compiled_model(sample_input[0]["image"])

Tracing adapter simplifies model input and output format. After
conversion, model has multiple outputs in following format: 1. Predicted
boxes is floating-point tensor in format [``N``, 4], where N is number
of detected boxes. 2. Predicted classes is integer tensor in format
[``N``], where N is number of predicted objects that defines which label
each object belongs. The values range of predicted classes tensor is [0,
``num_labels``], where ``num_labels`` is number of classes supported of
model (in our case 80). 3. Predicted scores is floating-point tensor in
format [``N``], where ``N`` is number of predicted objects that defines
confidence of each prediction. 4. Input image size is integer tensor
with values [``H``, ``W``], where ``H`` is height of input data and
``W`` is width of input data, used for rescaling predictions on
postprocessing step.

For reusing Detectron2 API for postprocessing and visualization, we
provide helpers for wrapping output in original Detectron2 format.

.. code:: ipython3

    from detectron2.structures import Instances, Boxes
    from detectron2.modeling.postprocessing import detector_postprocess
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.data import MetadataCatalog
    import numpy as np
    
    
    def postprocess_detection_result(outputs: Dict, orig_height: int, orig_width: int, conf_threshold: float = 0.0):
        """
        Helper function for postprocessing prediction results
    
        Parameters:
          outputs (Dict): OpenVINO model output dictionary
          orig_height (int): original image height before preprocessing
          orig_width (int): original image width before preprocessing
          conf_threshold (float, optional, defaults 0.0): confidence threshold for valid prediction
        Returns:
          prediction_result (instances): postprocessed predicted instances
        """
        boxes = outputs[0]
        classes = outputs[1]
        has_mask = len(outputs) >= 5
        masks = None if not has_mask else outputs[2]
        scores = outputs[2 if not has_mask else 3]
        model_input_size = (
            int(outputs[3 if not has_mask else 4][0]),
            int(outputs[3 if not has_mask else 4][1]),
        )
        filtered_detections = scores >= conf_threshold
        boxes = Boxes(boxes[filtered_detections])
        scores = scores[filtered_detections]
        classes = classes[filtered_detections]
        out_dict = {"pred_boxes": boxes, "scores": scores, "pred_classes": classes}
        if masks is not None:
            masks = masks[filtered_detections]
            out_dict["pred_masks"] = torch.from_numpy(masks)
        instances = Instances(model_input_size, **out_dict)
        return detector_postprocess(instances, orig_height, orig_width)
    
    
    def draw_instance_prediction(img: np.ndarray, results: Instances, cfg: "Config"):
        """
        Helper function for visualization prediction results
    
        Parameters:
          img (np.ndarray): original image for drawing predictions
          results (instances): model predictions
          cfg (Config): model configuration
        Returns:
           img_with_res: image with results
        """
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])
        visualizer = Visualizer(img, metadata, instance_mode=ColorMode.IMAGE)
        img_with_res = visualizer.draw_instance_predictions(results)
        return img_with_res

.. code:: ipython3

    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=0.05)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    Image.fromarray(img_with_res.get_image())




.. image:: detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_22_0.png



Instance Segmentation
---------------------



As it was discussed above, Detectron2 provides generic approach for
working with models for different use cases. The steps that required to
convert and run models pretrained for Instance Segmentation use case
will be very similar to Object Detection.

Download Instance Segmentation PyTorch model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    model_name = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x"
    model, cfg = get_model_and_config(model_name)
    sample_input = get_sample_inputs(image_file, cfg)

Convert Instance Segmentation Model to OpenVINO Intermediate Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    model_xml_path = MODEL_DIR / (model_name.split("/")[-1] + ".xml")
    
    if not model_xml_path.exists():
        ov_model = convert_detectron2_model(model, sample_input)
        ov.save_model(ov_model, MODEL_DIR / (model_name.split("/")[-1] + ".xml"))
    else:
        ov_model = model_xml_path

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Run Instance Segmentation model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



In comparison with Object Detection, Instance Segmentation models have
additional output that represents instance masks for each object. Our
postprocessing function handle this difference.

.. code:: ipython3

    compiled_model = core.compile_model(ov_model, device.value)

.. code:: ipython3

    results = compiled_model(sample_input[0]["image"])
    results = postprocess_detection_result(results, sample_input[0]["height"], sample_input[0]["width"], conf_threshold=0.05)
    img_with_res = draw_instance_prediction(np.array(image), results, cfg)
    Image.fromarray(img_with_res.get_image())




.. image:: detectron2-to-openvino-with-output_files/detectron2-to-openvino-with-output_32_0.png


