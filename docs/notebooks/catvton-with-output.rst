Virtual Try-On with CatVTON and OpenVINO
========================================

Virtual try-on methods based on diffusion models achieve realistic
try-on effects but replicate the backbone network as a ReferenceNet or
leverage additional image encoders to process condition inputs,
resulting in high training and inference costs. `In this
work <http://arxiv.org/abs/2407.15886>`__, authors rethink the necessity
of ReferenceNet and image encoders and innovate the interaction between
garment and person, proposing CatVTON, a simple and efficient virtual
try-on diffusion model. It facilitates the seamless transfer of in-shop
or worn garments of arbitrary categories to target persons by simply
concatenating them in spatial dimensions as inputs. The efficiency of
the model is demonstrated in three aspects: 1. Lightweight network. Only
the original diffusion modules are used, without additional network
modules. The text encoder and cross attentions for text injection in the
backbone are removed, further reducing the parameters by 167.02M. 2.
Parameter-efficient training. We identified the try-on relevant modules
through experiments and achieved high-quality try-on effects by training
only 49.57M parameters (∼5.51% of the backbone network’s parameters). 3.
Simplified inference. CatVTON eliminates all unnecessary conditions and
preprocessing steps, including pose estimation, human parsing, and text
input, requiring only garment reference, target person image, and mask
for the virtual try-on process. Extensive experiments demonstrate that
CatVTON achieves superior qualitative and quantitative results with
fewer prerequisites and trainable parameters than baseline methods.
Furthermore, CatVTON shows good generalization in in-the-wild scenarios
despite using open-source datasets with only 73K samples.

Teaser image from `CatVTON
GitHub <https://github.com/Zheng-Chong/CatVTON>`__ |teaser|

In this tutorial we consider how to convert and run this model using
OpenVINO.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling models <#compiling-models>`__
-  `Interactive demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |teaser| image:: https://github.com/Zheng-Chong/CatVTON/blob/edited/resource/img/teaser.jpg?raw=true

Prerequisites
-------------



.. code:: ipython3

    import platform
    
    
    if platform.system() == "Darwin":
        %pip install -q "numpy<2.0.0"
    %pip install -q "openvino>=2024.4"
    %pip install -q "torch>=2.1" "diffusers>=0.29.1" torchvision opencv_python --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q fvcore "pillow" "tqdm" "gradio>=4.36" "omegaconf==2.4.0.dev3" av pycocotools cloudpickle scipy accelerate "transformers>=4.27.3"


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    import requests
    
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    r = requests.get(
        url="https://raw.githubusercontent.com/aleksandr-mokrov/openvino_notebooks/refs/heads/catvton/utils/cmd_helper.py",
    )
    open("cmd_helper.py", "w").write(r.text)




.. parsed-literal::

    741



.. code:: ipython3

    from cmd_helper import clone_repo
    
    
    clone_repo("https://github.com/Zheng-Chong/CatVTON.git", "3b795364a4d2f3b5adb365f39cdea376d20bc53c")




.. parsed-literal::

    PosixPath('CatVTON')



Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

``ov_catvton_helper.py`` script contains helper function for models
downloading and models conversion, please check its content if you
interested in conversion details.

To download checkpoints and load models, just call the helper function
``download_models``. It takes care about it. Functions
``convert_pipeline_models`` and ``convert_automasker_models`` will
convert models from pipeline and ``automasker`` in OpenVINO format.

The original pipeline contains VAE encoder and decoder and UNET.
|CatVTON-overview|

The ``automasker`` contains ``DensePose`` with
``detectron2.GeneralizedRCNN`` model and ``SCHP`` (``LIP`` and ``ATR``
version).

.. |CatVTON-overview| image:: https://github.com/user-attachments/assets/e35c8dab-1c54-47b1-a73b-2a62e6cdca7c

.. code:: ipython3

    from pathlib import Path
    
    from ov_catvton_helper import download_models, convert_pipeline_models, convert_automasker_models
    
    
    MODEL_DIR = Path("models")
    VAE_ENCODER_PATH = MODEL_DIR / "vae_encoder.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"
    UNET_PATH = MODEL_DIR / "unet.xml"
    DENSEPOSE_PROCESSOR_PATH = MODEL_DIR / "densepose_processor.xml"
    SCHP_PROCESSOR_ATR = MODEL_DIR / "schp_processor_atr.xml"
    SCHP_PROCESSOR_LIP = MODEL_DIR / "schp_processor_lip.xml"
    
    
    pipeline, mask_processor, automasker = download_models(MODEL_DIR)
    convert_pipeline_models(pipeline, VAE_ENCODER_PATH, VAE_DECODER_PATH, UNET_PATH)
    convert_automasker_models(automasker, DENSEPOSE_PROCESSOR_PATH, SCHP_PROCESSOR_ATR, SCHP_PROCESSOR_LIP)


.. parsed-literal::

    Note: switching to '3b795364a4d2f3b5adb365f39cdea376d20bc53c'.
    
    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by switching back to a branch.
    
    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -c with the switch command. Example:
    
      git switch -c <new-branch-name>
    
    Or undo this operation with:
    
      git switch -
    
    Turn off this advice by setting config variable advice.detachedHead to false
    
    HEAD is now at 3b79536 Update default base model path



.. parsed-literal::

    Fetching 10 files:   0%|          | 0/10 [00:00<?, ?it/s]



.. parsed-literal::

    README.md:   0%|          | 0.00/9.66k [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors:   0%|          | 0.00/198M [00:00<?, ?B/s]



.. parsed-literal::

    exp-schp-201908301523-atr.pth:   0%|          | 0.00/267M [00:00<?, ?B/s]



.. parsed-literal::

    .gitattributes:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    exp-schp-201908261155-lip.pth:   0%|          | 0.00/267M [00:00<?, ?B/s]



.. parsed-literal::

    DensePose/Base-DensePose-RCNN-FPN.yaml:   0%|          | 0.00/1.52k [00:00<?, ?B/s]



.. parsed-literal::

    model_final_162be9.pkl:   0%|          | 0.00/256M [00:00<?, ?B/s]



.. parsed-literal::

    (…)nsePose/densepose_rcnn_R_50_FPN_s1x.yaml:   0%|          | 0.00/182 [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors:   0%|          | 0.00/198M [00:00<?, ?B/s]



.. parsed-literal::

    model.safetensors:   0%|          | 0.00/198M [00:00<?, ?B/s]


.. parsed-literal::

    An error occurred while trying to fetch booksforcharlie/stable-diffusion-inpainting: booksforcharlie/stable-diffusion-inpainting does not appear to have a file named diffusion_pytorch_model.safetensors.
    Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/notebooks/catvton/CatVTON/model/SCHP/__init__.py:93: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
      state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:136: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:145: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:147: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/810/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_2d_condition.py:1111: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if dim % default_overall_up_factor != 0:


Compiling models
----------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import openvino as ov
    
    from notebook_utils import device_widget
    
    
    core = ov.Core()
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



``get_compiled_pipeline`` and ``get_compiled_automasker`` functions
defined in ``ov_catvton_helper.py`` provides convenient way for getting
the pipeline and the ``automasker`` with compiled ov-models that are
compatible with the original interface. It accepts the original pipeline
and ``automasker``, inference device and directories with converted
models as arguments. Under the hood we create callable wrapper classes
for compiled models to allow interaction with original pipelines. Note
that all of wrapper classes return ``torch.Tensor``\ s instead of
``np.array``\ s. And then insert wrappers instances in the pipeline.

.. code:: ipython3

    from ov_catvton_helper import get_compiled_pipeline, get_compiled_automasker
    
    
    pipeline = get_compiled_pipeline(pipeline, core, device, VAE_ENCODER_PATH, VAE_DECODER_PATH, UNET_PATH)
    automasker = get_compiled_automasker(automasker, core, device, DENSEPOSE_PROCESSOR_PATH, SCHP_PROCESSOR_ATR, SCHP_PROCESSOR_LIP)

Interactive inference
---------------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    from gradio_helper import make_demo
    
    
    output_dir = "output"
    demo = make_demo(pipeline, mask_processor, automasker, output_dir)
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(debug=False, share=True)


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.







