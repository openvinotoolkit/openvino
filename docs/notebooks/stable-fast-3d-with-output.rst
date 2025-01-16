Stable Fast 3D Mesh Reconstruction and OpenVINO
===============================================

.. warning::

   Important note: This notebook has problems with installation
   `pynim <https://github.com/vork/PyNanoInstantMeshes/issues/2>`__
   library on MacOS. The issue may be environment dependent and may
   occur on other OSes.

`Stable Fast 3D
(SF3D) <https://huggingface.co/stabilityai/stable-fast-3d>`__ is a large
reconstruction model based on
`TripoSR <https://huggingface.co/spaces/stabilityai/TripoSR>`__, which
takes in a single image of an object and generates a textured
UV-unwrapped 3D mesh asset.

You can find `the source code on
GitHub <https://github.com/Stability-AI/stable-fast-3d>`__ and read the
paper `SF3D: Stable Fast 3D Mesh Reconstruction with UV-unwrapping and
Illumination Disentanglement <https://arxiv.org/abs/2408.00653>`__.

.. figure:: https://github.com/Stability-AI/stable-fast-3d/blob/main/demo_files/teaser.gif?raw=true
   :alt: Teaser Video

   Teaser Video

Unlike most existing approaches, SF3D is explicitly trained for mesh
generation, incorporating a fast UV unwrapping technique that enables
swift texture generation rather than relying on vertex colors. The
method also learns to predict material parameters and normal maps to
enhance the visual quality of the reconstructed 3D meshes.

The authors compare their results with TripoSR:

.. figure:: https://github.com/user-attachments/assets/fb1277e5-610f-47d7-97e4-1267624f7f1f
   :alt: sf3d-improvements

   sf3d-improvements

..

   The top shows the effect of light bake-in when relighting the asset.
   SF3D produces a more plausible relighting. By not using vertex
   colors, our method is capable of encoding finer details while also
   having a lower polygon count. Our vertex displacement enables
   estimating smooth shapes, which do not introduce stair-stepping
   artifacts from marching cubes. Lastly, our material property
   prediction allows us to express a variety of different surface types.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Get the original model <#get-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling models and prepare
   pipeline <#compiling-models-and-prepare-pipeline>`__
-  `Interactive inference <#interactive-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/pip_helper.py",
    )
    open("pip_helper.py", "w").write(r.text)
    
    from pip_helper import pip_install
    
    
    pip_install("-q", "gradio>=4.19", "openvino>=2024.3.0", "wheel", "gradio-litmodel3d==0.0.1")
    
    pip_install(
        "-q",
        "torch>=2.2.2",
        "torchvision",
        "transformers>=4.42.3",
        "rembg==2.0.57",
        "trimesh==4.4.1",
        "einops==0.7.0",
        "omegaconf==2.4.0.dev3",
        "jaxtyping==0.2.31",
        "gpytoolbox==0.3.2",
        "open_clip_torch==2.24.0",
        "git+https://github.com/vork/PyNanoInstantMeshes.git",
        "--extra-index-url",
        "https://download.pytorch.org/whl/cpu",
    )

.. code:: ipython3

    import sys
    from pathlib import Path
    
    if not Path("stable-fast-3d").exists():
        !git clone https://github.com/Stability-AI/stable-fast-3d
        %cd stable-fast-3d
        !git checkout "4a8597ad34e5101f307aa8f443b4ce830b205aa8"  # to avoid breaking changes
        %cd ..
    
    sys.path.append("stable-fast-3d")
    pip_install("-q", "stable-fast-3d/texture_baker/")
    pip_install("-q", "stable-fast-3d/uv_unwrapper/")

Get the original model
----------------------

.. code:: ipython3

    from sf3d.system import SF3D
    
    
    model = SF3D.from_pretrained(
        "stabilityai/stable-fast-3d",
        config_name="config.yaml",
        weight_name="model.safetensors",
    )

Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



SF3D is PyTorch model. OpenVINO supports PyTorch models via conversion
to OpenVINO Intermediate Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.
``ov_stable_fast_3d_helper.py`` script contains helper function for
model conversion, please check its content if you interested in
conversion details.

.. raw:: html

   <details>

.. raw:: html

   <summary>

Click here for more detailed explanation of conversion steps

.. raw:: html

   </summary>

.. figure:: https://github.com/user-attachments/assets/8b37e08e-ddda-4dae-b5de-cf3adc4b79c8
   :alt: sf3d-overview

   sf3d-overview

As illustrated in SF3D Overview image, SF3D has 5 main components:

1. An enhanced transformer network that predicts higher resolution
   triplanes, which helps in reducing aliasing artifacts (top left in
   the figure). In this part ``LinearCameraEmbedder``
   (``camera_embedder`` in the implemented pipeline) obtains camera
   embeddings for ``DINOv2`` model (``image_tokenizer``) that obtains
   image tokens. ``TriplaneLearnablePositionalEmbedding`` model
   (``tokenizer``) obtains triplane tokens. The transformer
   ``TwoStreamInterleaveTransformer`` (``backbone``) gets triplane
   tokens (``hidden_states``) and image tokens
   (``encoder_hidden_states``). Then ``PixelShuffleUpsampleNetwork``
   (``post_processor``) processes the output. We will convert all these
   5 models to OpenVINO format and then replace the original models by
   compiled OV-models in the original pipeline. Here is a specific for
   ``DINOv2`` model that calls ``nn.functional.interpolate`` in its
   method ``interpolate_pos_encoding``. This method accepts a tuple of
   floats as ``scale_factor``, but during conversion a tuple of floats
   converts to a tuple of tensors due to conversion specific. It raises
   an error. So, we need to patch it by converting in float.

2. Material Estimation. ``MaterialNet`` is implemented in
   ``ClipBasedHeadEstimator`` model (``image_estimator``). We will
   convert it too.

3. Illumination Modeling. It is not demonstrated in the original demo
   and its results are not used in the original pipeline, so we will not
   use it too. Thus ``global_estimator`` is not needed to be converted.

4. Mesh Extraction and Refinement. In these part ``MaterialMLP``
   (``decoder``) is used. The ``decoder`` accepts lists of include or
   exclude heads in forward method and uses them to choose a part of
   heads. We can’t accept a list of strings in IR-model, but we can
   build 2 decoders with required structures.

5. Fast UV-Unwrapping and Export. It is finalizing step and there are no
   models for conversion.

.. raw:: html

   </details>

.. code:: ipython3

    from ov_stable_fast_3d_helper import (
        convert_image_tokenizer,
        convert_tokenizer,
        convert_backbone,
        convert_post_processor,
        convert_camera_embedder,
        convert_image_estimator,
        convert_decoder,
    )
    
    # uncomment the code below to see the model conversion code of convert_image_tokenizer.
    # replace the function name if you want see the code for another model
    
    # ??convert_image_tokenizer

.. code:: ipython3

    IMAGE_TOKENIZER_OV_PATH = Path("models/image_tokenizer_ir.xml")
    TOKENIZER_OV_PATH = Path("models/tokenizer_ir.xml")
    BACKBONE_OV_PATH = Path("models/backbone_ir.xml")
    POST_PROCESSOR_OV_PATH = Path("models/post_processor_ir.xml")
    CAMERA_EMBEDDER_OV_PATH = Path("models/camera_embedder_ir.xml")
    IMAGE_ESTIMATOR_OV_PATH = Path("models/image_estimator_ir.xml")
    INCLUDE_DECODER_OV_PATH = Path("models/include_decoder_ir.xml")
    EXCLUDE_DECODER_OV_PATH = Path("models/exclude_decoder_ir.xml")
    
    
    convert_image_tokenizer(model.image_tokenizer, IMAGE_TOKENIZER_OV_PATH)
    convert_tokenizer(model.tokenizer, TOKENIZER_OV_PATH)
    convert_backbone(model.backbone, BACKBONE_OV_PATH)
    convert_post_processor(model.post_processor, POST_PROCESSOR_OV_PATH)
    convert_camera_embedder(model.camera_embedder, CAMERA_EMBEDDER_OV_PATH)
    convert_image_estimator(model.image_estimator, IMAGE_ESTIMATOR_OV_PATH)
    convert_decoder(model.decoder, INCLUDE_DECODER_OV_PATH, EXCLUDE_DECODER_OV_PATH)

Compiling models and prepare pipeline
-------------------------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device

``get_compiled_model`` function defined in ``ov_ov_stable_fast_3d.py``
provides convenient way for getting compiled ov-model that is compatible
with the original interface. It accepts the original model, inference
device and directories with converted models as arguments.

.. code:: ipython3

    from ov_stable_fast_3d_helper import get_compiled_model
    
    
    model = get_compiled_model(
        model,
        device,
        IMAGE_TOKENIZER_OV_PATH,
        TOKENIZER_OV_PATH,
        BACKBONE_OV_PATH,
        POST_PROCESSOR_OV_PATH,
        CAMERA_EMBEDDER_OV_PATH,
        IMAGE_ESTIMATOR_OV_PATH,
        INCLUDE_DECODER_OV_PATH,
        EXCLUDE_DECODER_OV_PATH,
    )

Interactive inference
---------------------

It’s taken from the original
``gradio_app.py``, but the model is replaced with the one defined above.

.. code:: ipython3

    import requests
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-fast-3d/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(model=model)
    
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(share=True, debug=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
