TripoSR feedforward 3D reconstruction from a single image and OpenVINO
======================================================================

`TripoSR <https://huggingface.co/spaces/stabilityai/TripoSR>`__ is a
state-of-the-art open-source model for fast feedforward 3D
reconstruction from a single image, developed in collaboration between
`Tripo AI <https://www.tripo3d.ai/>`__ and `Stability
AI <https://stability.ai/news/triposr-3d-generation>`__.

You can find `the source code on
GitHub <https://github.com/VAST-AI-Research/TripoSR>`__ and `demo on
HuggingFace <https://huggingface.co/spaces/stabilityai/TripoSR>`__.
Also, you can read the paper `TripoSR: Fast 3D Object Reconstruction
from a Single Image <https://arxiv.org/abs/2403.02151>`__.

.. figure:: https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/figures/teaser800.gif
   :alt: Teaser Video

   Teaser Video


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
    
    
    pip_install("gradio>=4.19", "openvino>=2024.0.0")
    
    pip_install("torch>=2.2.2", "torchvision", "transformers>=4.35.0", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    pip_install("rembg", "trimesh", "einops", "omegaconf>=2.3.0", "--extra-index-url", "https://download.pytorch.org/whl/cpu")
    pip_install(
        "git+https://github.com/tatsy/torchmcubes.git@cb81cddece46a8a126b08f7fbb9742f8605eefab", "--extra-index-url", "https://download.pytorch.org/whl/cpu"
    )

.. code:: ipython3

    import os
    import sys
    from pathlib import Path
    
    if not Path("TripoSR").exists():
        exit_code = os.system("git clone https://huggingface.co/spaces/stabilityai/TripoSR")
    
        if exit_code != 0:
            raise Exception("Failed to clone the repository!")
    
    sys.path.append("TripoSR")


.. parsed-literal::

    Cloning into 'TripoSR'...
    remote: Enumerating objects: 120, done.[K
    remote: Counting objects: 100% (116/116), done.[K
    remote: Compressing objects: 100% (114/114), done.[K
    remote: Total 120 (delta 38), reused 0 (delta 0), pack-reused 4 (from 1)[K
    Receiving objects: 100% (120/120), 570.19 KiB | 1.07 MiB/s, done.
    Resolving deltas: 100% (38/38), done.
    

Get the original model
----------------------

.. code:: ipython3

    from tsr.system import TSR
    
    
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.renderer.set_chunk_size(131072)
    model.to("cpu");

Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    import torch
    
    import openvino as ov
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                converted_model = ov.convert_model(model, example_input=example_input)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)
    
            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

The original model is a pipeline of several models. There are
``image_tokenizer``, ``tokenizer``, ``backbone`` and ``post_processor``.
``image_tokenizer`` contains ``ViTModel`` that consists of
``ViTPatchEmbeddings``, ``ViTEncoder`` and ``ViTPooler``. ``tokenizer``
is ``Triplane1DTokenizer``, ``backbone`` is ``Transformer1D``,
``post_processor`` is ``TriplaneUpsampleNetwork``. Convert all internal
models one by one.

.. code:: ipython3

    VIT_PATCH_EMBEDDINGS_OV_PATH = Path("models/vit_patch_embeddings_ir.xml")
    
    
    class PatchEmbedingWrapper(torch.nn.Module):
        def __init__(self, patch_embeddings):
            super().__init__()
            self.patch_embeddings = patch_embeddings
    
        def forward(self, pixel_values):
            outputs = self.patch_embeddings(pixel_values=pixel_values)
            return outputs
    
    
    example_input = {
        "pixel_values": torch.rand([1, 3, 224, 224], dtype=torch.float32),
    }
    
    convert(
        PatchEmbedingWrapper(model.image_tokenizer.model.embeddings.patch_embeddings),
        VIT_PATCH_EMBEDDINGS_OV_PATH,
        example_input,
    )


.. parsed-literal::

    /home/ea/work/py311/lib/python3.11/site-packages/transformers/models/vit/modeling_vit.py:163: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if num_channels != self.num_channels:
    

.. code:: ipython3

    VIT_ENCODER_OV_PATH = Path("models/vit_encoder_ir.xml")
    
    
    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
    
        def forward(
            self,
            hidden_states,
        ):
            outputs = self.encoder(
                hidden_states=hidden_states,
            )
    
            return outputs.last_hidden_state
    
    
    example_input = {
        "hidden_states": torch.rand([1, 1025, 768], dtype=torch.float32),
    }
    
    convert(
        EncoderWrapper(model.image_tokenizer.model.encoder),
        VIT_ENCODER_OV_PATH,
        example_input,
    )

.. code:: ipython3

    VIT_POOLER_OV_PATH = Path("models/vit_pooler_ir.xml")
    convert(
        model.image_tokenizer.model.pooler,
        VIT_POOLER_OV_PATH,
        torch.rand([1, 1025, 768], dtype=torch.float32),
    )

.. code:: ipython3

    TOKENIZER_OV_PATH = Path("models/tokenizer_ir.xml")
    convert(model.tokenizer, TOKENIZER_OV_PATH, torch.tensor(1))

.. code:: ipython3

    example_input = {
        "hidden_states": torch.rand([1, 1024, 3072], dtype=torch.float32),
        "encoder_hidden_states": torch.rand([1, 1025, 768], dtype=torch.float32),
    }
    
    BACKBONE_OV_PATH = Path("models/backbone_ir.xml")
    convert(model.backbone, BACKBONE_OV_PATH, example_input)

.. code:: ipython3

    POST_PROCESSOR_OV_PATH = Path("models/post_processor_ir.xml")
    convert(
        model.post_processor,
        POST_PROCESSOR_OV_PATH,
        torch.rand([1, 3, 1024, 32, 32], dtype=torch.float32),
    )

Compiling models and prepare pipeline
-------------------------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    core = ov.Core()
    
    compiled_vit_patch_embeddings = core.compile_model(VIT_PATCH_EMBEDDINGS_OV_PATH, device.value)
    compiled_vit_model_encoder = core.compile_model(VIT_ENCODER_OV_PATH, device.value)
    compiled_vit_model_pooler = core.compile_model(VIT_POOLER_OV_PATH, device.value)
    
    compiled_tokenizer = core.compile_model(TOKENIZER_OV_PATH, device.value)
    compiled_backbone = core.compile_model(BACKBONE_OV_PATH, device.value)
    compiled_post_processor = core.compile_model(POST_PROCESSOR_OV_PATH, device.value)

Let’s create callable wrapper classes for compiled models to allow
interaction with original ``TSR`` class. Note that all of wrapper
classes return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple
    
    
    class VitPatchEmdeddingsWrapper(torch.nn.Module):
        def __init__(self, vit_patch_embeddings, model):
            super().__init__()
            self.vit_patch_embeddings = vit_patch_embeddings
            self.projection = model.projection
    
        def forward(self, pixel_values, interpolate_pos_encoding=False):
            inputs = {
                "pixel_values": pixel_values,
            }
            outs = self.vit_patch_embeddings(inputs)[0]
    
            return torch.from_numpy(outs)
    
    
    class VitModelEncoderWrapper(torch.nn.Module):
        def __init__(self, vit_model_encoder):
            super().__init__()
            self.vit_model_encoder = vit_model_encoder
    
        def forward(
            self,
            hidden_states,
            head_mask,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=False,
        ):
            inputs = {
                "hidden_states": hidden_states.detach().numpy(),
            }
    
            outs = self.vit_model_encoder(inputs)
            outputs = namedtuple("BaseModelOutput", ("last_hidden_state", "hidden_states", "attentions"))
    
            return outputs(torch.from_numpy(outs[0]), None, None)
    
    
    class VitModelPoolerWrapper(torch.nn.Module):
        def __init__(self, vit_model_pooler):
            super().__init__()
            self.vit_model_pooler = vit_model_pooler
    
        def forward(self, hidden_states):
            outs = self.vit_model_pooler(hidden_states.detach().numpy())[0]
    
            return torch.from_numpy(outs)
    
    
    class TokenizerWrapper(torch.nn.Module):
        def __init__(self, tokenizer, model):
            super().__init__()
            self.tokenizer = tokenizer
            self.detokenize = model.detokenize
    
        def forward(self, batch_size):
            outs = self.tokenizer(batch_size)[0]
    
            return torch.from_numpy(outs)
    
    
    class BackboneWrapper(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
    
        def forward(self, hidden_states, encoder_hidden_states):
            inputs = {
                "hidden_states": hidden_states,
                "encoder_hidden_states": encoder_hidden_states.detach().numpy(),
            }
    
            outs = self.backbone(inputs)[0]
    
            return torch.from_numpy(outs)
    
    
    class PostProcessorWrapper(torch.nn.Module):
        def __init__(self, post_processor):
            super().__init__()
            self.post_processor = post_processor
    
        def forward(self, triplanes):
            outs = self.post_processor(triplanes)[0]
    
            return torch.from_numpy(outs)

Replace all models in the original model by wrappers instances:

.. code:: ipython3

    model.image_tokenizer.model.embeddings.patch_embeddings = VitPatchEmdeddingsWrapper(
        compiled_vit_patch_embeddings,
        model.image_tokenizer.model.embeddings.patch_embeddings,
    )
    model.image_tokenizer.model.encoder = VitModelEncoderWrapper(compiled_vit_model_encoder)
    model.image_tokenizer.model.pooler = VitModelPoolerWrapper(compiled_vit_model_pooler)
    
    model.tokenizer = TokenizerWrapper(compiled_tokenizer, model.tokenizer)
    model.backbone = BackboneWrapper(compiled_backbone)
    model.post_processor = PostProcessorWrapper(compiled_post_processor)

Interactive inference
---------------------



.. code:: ipython3

    import tempfile
    import numpy as np
    import rembg
    from PIL import Image
    
    from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
    
    
    rembg_session = rembg.new_session()
    
    
    def preprocess(input_image, do_remove_background, foreground_ratio):
        def fill_background(image):
            image = np.array(image).astype(np.float32) / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
            image = Image.fromarray((image * 255.0).astype(np.uint8))
            return image
    
        if do_remove_background:
            image = input_image.convert("RGB")
            image = remove_background(image, rembg_session)
            image = resize_foreground(image, foreground_ratio)
            image = fill_background(image)
        else:
            image = input_image
            if image.mode == "RGBA":
                image = fill_background(image)
        return image
    
    
    def generate(image):
        scene_codes = model(image, "cpu")  # the device is provided for the image processorit is
        mesh = model.extract_mesh(scene_codes)[0]
        mesh = to_gradio_3d_orientation(mesh)
        mesh_path = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
        mesh.export(mesh_path.name)
        return mesh_path.name

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/triposr-3d-reconstruction/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(preprocess_fn=preprocess, generate_fn=generate)
    
    try:
        demo.launch(debug=True, height=680)
    except Exception:
        demo.launch(share=True, debug=True, height=680)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.
    







.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
