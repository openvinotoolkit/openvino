Live 3D Human Pose Estimation with OpenVINO
===========================================

This notebook demonstrates live 3D Human Pose Estimation with OpenVINO
via a webcam. We utilize the model
`human-pose-estimation-3d-0001 <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/human-pose-estimation-3d-0001>`__
from `Open Model
Zoo <https://github.com/openvinotoolkit/open_model_zoo/>`__. At the end
of this notebook, you will see live inference results from your webcam
(if available). Alternatively, you can also upload a video file to test
out the algorithms. **Make sure you have properly installed the** 
`Jupyter extension <https://github.com/jupyter-widgets/pythreejs#jupyterlab>`__ 
**and been using JupyterLab to run the demo as suggested in the README.md**

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

**Table of contents:**


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
   -  `Draw 2D Pose Overlays <#draw-d-pose-overlays>`__
   -  `Main Processing Function <#main-processing-function>`__

-  `Run <#run>`__

   -  `Run Live Pose Estimation <#run-live-pose-estimation>`__
   -  `Run Pose Estimation on a Video
      File <#run-pose-estimation-on-a-video-file>`__

Prerequisites 
-------------------------------------------------------

**The ``pythreejs`` extension may not display properly when using the
latest Jupyter Notebook release (2.4.1). Therefore, it is recommended to
use Jupyter Lab instead.**

.. code:: ipython3

    %pip install pythreejs "openvino-dev==2023.1.0"


.. parsed-literal::

    Collecting pythreejs
      Using cached pythreejs-2.4.2-py3-none-any.whl (3.4 MB)
    Requirement already satisfied: openvino-dev==2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (2023.1.0)
    Requirement already satisfied: addict>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (2.4.0)
    Requirement already satisfied: defusedxml>=0.7.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (0.7.1)
    Requirement already satisfied: jstyleson>=0.0.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (0.0.2)
    Requirement already satisfied: networkx<=3.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (2.8.2)
    Requirement already satisfied: numpy>=1.16.6 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (1.24.3)
    Requirement already satisfied: opencv-python in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (4.8.1.78)
    Requirement already satisfied: openvino-telemetry>=2022.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (2023.2.1)
    Requirement already satisfied: pillow>=8.1.2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (10.0.1)
    Requirement already satisfied: pyyaml>=5.4.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (6.0.1)
    Requirement already satisfied: requests>=2.25.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (2.31.0)
    Requirement already satisfied: texttable>=1.6.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (1.7.0)
    Requirement already satisfied: tqdm>=4.54.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (4.66.1)
    Requirement already satisfied: openvino==2023.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (2023.1.0)
    Requirement already satisfied: scipy<1.11,>=1.8 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from openvino-dev==2023.1.0) (1.10.1)
    Requirement already satisfied: ipywidgets>=7.2.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (8.1.1)
    Collecting ipydatawidgets>=1.1.1 (from pythreejs)
      Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl.metadata (1.4 kB)
    Requirement already satisfied: traitlets in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pythreejs) (5.13.0)
    Collecting traittypes>=0.2.0 (from ipydatawidgets>=1.1.1->pythreejs)
      Using cached traittypes-0.2.1-py2.py3-none-any.whl (8.6 kB)
    Requirement already satisfied: comm>=0.1.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (0.1.4)
    Requirement already satisfied: ipython>=6.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (8.12.3)
    Requirement already satisfied: widgetsnbextension~=4.0.9 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (4.0.9)
    Requirement already satisfied: jupyterlab-widgets~=3.0.9 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipywidgets>=7.2.1->pythreejs) (3.0.9)
    Requirement already satisfied: charset-normalizer<4,>=2 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev==2023.1.0) (3.3.1)
    Requirement already satisfied: idna<4,>=2.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev==2023.1.0) (3.4)
    Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev==2023.1.0) (2.0.7)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from requests>=2.25.1->openvino-dev==2023.1.0) (2023.7.22)
    Requirement already satisfied: backcall in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.0)
    Requirement already satisfied: decorator in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (5.1.1)
    Requirement already satisfied: jedi>=0.16 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.19.1)
    Requirement already satisfied: matplotlib-inline in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.1.6)
    Requirement already satisfied: pickleshare in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.5)
    Requirement already satisfied: prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (3.0.39)
    Requirement already satisfied: pygments>=2.4.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.16.1)
    Requirement already satisfied: stack-data in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.6.3)
    Requirement already satisfied: typing-extensions in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.5.0)
    Requirement already satisfied: pexpect>4.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (4.8.0)
    Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.8.3)
    Requirement already satisfied: ptyprocess>=0.5 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.7.0)
    Requirement already satisfied: wcwidth in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from prompt-toolkit!=3.0.37,<3.1.0,>=3.0.30->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.9)
    Requirement already satisfied: executing>=1.2.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.0.1)
    Requirement already satisfied: asttokens>=2.1.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (2.4.1)
    Requirement already satisfied: pure-eval in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (0.2.2)
    Requirement already satisfied: six>=1.12.0 in /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets>=7.2.1->pythreejs) (1.16.0)
    Using cached ipydatawidgets-4.3.5-py2.py3-none-any.whl (271 kB)
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Installing collected packages: traittypes, ipydatawidgets, pythreejs
    Successfully installed ipydatawidgets-4.3.5 pythreejs-2.4.2 traittypes-0.2.1
    Note: you may need to restart the kernel to use updated packages.


Imports 
-------------------------------------------------

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
---------------------------------------------------

Download the model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    
    
    ========== Unpacking model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.tar.gz
    


Convert Model to OpenVINO IR format 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
    Conversion to ONNX command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/openvino/model_zoo/internal_scripts/pytorch_to_onnx.py --model-path=model/public/human-pose-estimation-3d-0001 --model-name=PoseEstimationWithMobileNet --model-param=is_convertible_by_mo=True --import-module=model --weights=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.pth --input-shape=1,3,256,448 --input-names=data --output-names=features,heatmaps,pafs --output-file=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx
    
    ONNX check passed successfully.
    
    ========== Converting human-pose-estimation-3d-0001 to IR (FP32)
    Conversion command: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/bin/python -- /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/.venv/bin/mo --framework=onnx --output_dir=model/public/human-pose-estimation-3d-0001/FP32 --model_name=human-pose-estimation-3d-0001 --input=data '--mean_values=data[128.0,128.0,128.0]' '--scale_values=data[255.0,255.0,255.0]' --output=features,heatmaps,pafs --input_model=model/public/human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx '--layout=data(NCHW)' '--input_shape=[1, 3, 256, 448]' --compress_to_fp16=False
    
    [ INFO ] The model was converted to IR v11, the latest model format that corresponds to the source DL framework input/output format. While IR v11 is backwards compatible with OpenVINO Inference Engine API v1.0, please use API v2.0 (as of 2022.1) to take advantage of the latest improvements in IR v11.
    Find more information about API v2.0 and IR v11 at https://docs.openvino.ai/2023.0/openvino_2_0_transition_guide.html
    [ SUCCESS ] Generated IR version 11 model.
    [ SUCCESS ] XML file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.xml
    [ SUCCESS ] BIN file: /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-534/.workspace/scm/ov-notebook/notebooks/406-3D-pose-estimation-webcam/model/public/human-pose-estimation-3d-0001/FP32/human-pose-estimation-3d-0001.bin
    


Select inference device 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
----------------------------------------------------

Model Inference 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
---------------------------------------------

Run Live Pose Estimation 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Using the following method, you can click and move your mouse over the
picture on the left to interact.

.. code:: ipython3

    run_pose_estimation(source=0, flip=True, use_popup=False)

Run Pose Estimation on a Video File 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not have a webcam, you can still run this demo with a video
file. Any `format supported by
OpenCV <https://docs.opencv.org/4.5.1/dd/d43/tutorial_py_video_display.html>`__
will work.

You can click and move your mouse over the picture on the left to
interact.

.. code:: ipython3

    # video url
    video_path = "https://github.com/intel-iot-devkit/sample-videos/raw/master/face-demographics-walking.mp4"
    run_pose_estimation(source=video_path, flip=False, use_popup=False, skip_frames=10)
