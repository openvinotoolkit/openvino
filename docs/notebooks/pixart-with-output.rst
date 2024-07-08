PixArt-α: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis with OpenVINO
=========================================================================================================

`This paper <https://arxiv.org/abs/2310.00426>`__ introduces
`PIXART-α <https://github.com/PixArt-alpha/PixArt-alpha>`__, a
Transformer-based T2I diffusion model whose image generation quality is
competitive with state-of-the-art image generators, reaching
near-commercial application standards. Additionally, it supports
high-resolution image synthesis up to 1024px resolution with low
training cost. To achieve this goal, three core designs are proposed: 1.
Training strategy decomposition: We devise three distinct training steps
that separately optimize pixel dependency, text-image alignment, and
image aesthetic quality; 2. Efficient T2I Transformer: We incorporate
cross-attention modules into Diffusion Transformer (DiT) to inject text
conditions and streamline the computation-intensive class-condition
branch; 3. High-informative data: We emphasize the significance of
concept density in text-image pairs and leverage a large Vision-Language
model to auto-label dense pseudo-captions to assist text-image alignment
learning.

|image0|

**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load the original model <#load-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Convert text encoder <#convert-text-encoder>`__
   -  `Convert transformer <#convert-transformer>`__
   -  `Convert VAE decoder <#convert-vae-decoder>`__

-  `Compiling models <#compiling-models>`__
-  `Building the pipeline <#building-the-pipeline>`__
-  `Interactive inference <#interactive-inference>`__

.. |image0| image:: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/resolve/main/asset/images/teaser.png

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "diffusers>=0.14.0" sentencepiece "datasets>=2.14.6" "transformers>=4.25.1" "gradio>=4.19" "torch>=2.1" Pillow opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install --pre -Uq openvino --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    openvino-dev 2024.2.0 requires openvino==2024.2.0, but you have openvino 2024.3.0.dev20240627 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Load and run the original pipeline
----------------------------------



We use
`PixArt-LCM-XL-2-1024-MS <https://huggingface.co/PixArt-alpha/PixArt-LCM-XL-2-1024-MS>`__
that uses LCMs. `LCMs <https://arxiv.org/abs/2310.04378>`__ is a
diffusion distillation method which predict ``PF-ODE's`` solution
directly in latent space, achieving super fast inference with few steps.

.. code:: ipython3

    import torch
    from diffusers import PixArtAlphaPipeline


    pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-LCM-XL-2-1024-MS", use_safetensors=True)

    prompt = "A small cactus with a happy face in the Sahara desert."
    generator = torch.Generator().manual_seed(42)

    image = pipe(prompt, guidance_scale=0.0, num_inference_steps=4, generator=generator).images[0]


.. parsed-literal::

    2024-07-02 01:22:15.056189: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-07-02 01:22:15.090603: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-07-02 01:22:15.773162: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    Some weights of the model checkpoint were not used when initializing PixArtTransformer2DModel:
     ['caption_projection.y_embedding']
    You are using the default legacy behaviour of the <class 'transformers.models.t5.tokenization_t5.T5Tokenizer'>. This is expected, and simply means that the `legacy` (previous) behavior will be used so nothing changes for you. If you want to use the new behaviour, set `legacy=False`. This should only be set if you understand what it means, and thoroughly read the reason why this was added as explained in https://github.com/huggingface/transformers/pull/24565



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


.. parsed-literal::

    The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: pixart-with-output_files/pixart-with-output_5_0.png



Convert the model to OpenVINO IR
--------------------------------



Let’s define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    from pathlib import Path

    import numpy as np
    import torch

    import openvino as ov


    def convert(model: torch.nn.Module, xml_path: str, example_input):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            model.eval()
            with torch.no_grad():
                converted_model = ov.convert_model(model, example_input=example_input)
            ov.save_model(converted_model, xml_path)

            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

PixArt-α consists of pure transformer blocks for latent diffusion: It
can directly generate 1024px images from text prompts within a single
sampling process.

|image01|.

During inference it uses text encoder ``T5EncoderModel``, transformer
``Transformer2DModel`` and VAE decoder ``AutoencoderKL``. Let’s convert
the models from the pipeline one by one.

.. |image01| image:: https://huggingface.co/PixArt-alpha/PixArt-XL-2-1024-MS/resolve/main/asset/images/model.png

.. code:: ipython3

    MODEL_DIR = Path("model")

    TEXT_ENCODER_PATH = MODEL_DIR / "text_encoder.xml"
    TRANSFORMER_OV_PATH = MODEL_DIR / "transformer_ir.xml"
    VAE_DECODER_PATH = MODEL_DIR / "vae_decoder.xml"

Convert text encoder
~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    example_input = {
        "input_ids": torch.zeros(1, 120, dtype=torch.int64),
        "attention_mask": torch.zeros(1, 120, dtype=torch.int64),
    }

    convert(pipe.text_encoder, TEXT_ENCODER_PATH, example_input)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4371: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(


Convert transformer
~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, hidden_states=None, timestep=None, encoder_hidden_states=None, encoder_attention_mask=None, resolution=None, aspect_ratio=None):

            return self.transformer.forward(
                hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                added_cond_kwargs={"resolution": resolution, "aspect_ratio": aspect_ratio},
            )


    example_input = {
        "hidden_states": torch.rand([2, 4, 128, 128], dtype=torch.float32),
        "timestep": torch.tensor([999, 999]),
        "encoder_hidden_states": torch.rand([2, 120, 4096], dtype=torch.float32),
        "encoder_attention_mask": torch.rand([2, 120], dtype=torch.float32),
        "resolution": torch.tensor([[1024.0, 1024.0], [1024.0, 1024.0]]),
        "aspect_ratio": torch.tensor([[1.0], [1.0]]),
    }


    w_transformer = TransformerWrapper(pipe.transformer)
    convert(w_transformer, TRANSFORMER_OV_PATH, example_input)


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/embeddings.py:219: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if self.height != height or self.width != width:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/attention_processor.py:682: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if current_length != target_length:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/attention_processor.py:697: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.shape[0] < batch_size * head_size:


Convert VAE decoder
~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class VAEDecoderWrapper(torch.nn.Module):

        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latents):
            return self.vae.decode(latents, return_dict=False)


    convert(VAEDecoderWrapper(pipe.vae), VAE_DECODER_PATH, (torch.zeros((1, 4, 128, 128))))


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:


Compiling models
----------------



Select device from dropdown list for running inference using OpenVINO.

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



.. code:: ipython3

    compiled_model = core.compile_model(TRANSFORMER_OV_PATH)
    compiled_vae = core.compile_model(VAE_DECODER_PATH)
    compiled_text_encoder = core.compile_model(TEXT_ENCODER_PATH)

Building the pipeline
---------------------



Let’s create callable wrapper classes for compiled models to allow
interaction with original pipelines. Note that all of wrapper classes
return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple

    EncoderOutput = namedtuple("EncoderOutput", "last_hidden_state")


    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder, dtype):
            super().__init__()
            self.text_encoder = text_encoder
            self.dtype = dtype

        def forward(self, input_ids=None, attention_mask=None):
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            last_hidden_state = self.text_encoder(inputs)[0]
            return EncoderOutput(torch.from_numpy(last_hidden_state))

.. code:: ipython3

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer, config):
            super().__init__()
            self.transformer = transformer
            self.config = config

        def forward(
            self,
            hidden_states=None,
            timestep=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            resolution=None,
            aspect_ratio=None,
            added_cond_kwargs=None,
            **kwargs
        ):
            inputs = {
                "hidden_states": hidden_states,
                "timestep": timestep,
                "encoder_hidden_states": encoder_hidden_states,
                "encoder_attention_mask": encoder_attention_mask,
            }
            resolution = added_cond_kwargs["resolution"]
            aspect_ratio = added_cond_kwargs["aspect_ratio"]
            if resolution is not None:
                inputs["resolution"] = resolution
                inputs["aspect_ratio"] = aspect_ratio
            outputs = self.transformer(inputs)[0]

            return [torch.from_numpy(outputs)]

.. code:: ipython3

    class VAEWrapper(torch.nn.Module):
        def __init__(self, vae, config):
            super().__init__()
            self.vae = vae
            self.config = config

        def decode(self, latents=None, **kwargs):
            inputs = {
                "latents": latents,
            }

            outs = self.vae(inputs)
            outs = namedtuple("VAE", "sample")(torch.from_numpy(outs[0]))

            return outs

And insert wrappers instances in the pipeline:

.. code:: ipython3

    pipe.__dict__["_internal_dict"]["_execution_device"] = pipe._execution_device  # this is to avoid some problem that can occur in the pipeline

    pipe.register_modules(
        text_encoder=TextEncoderWrapper(compiled_text_encoder, pipe.text_encoder.dtype),
        transformer=TransformerWrapper(compiled_model, pipe.transformer.config),
        vae=VAEWrapper(compiled_vae, pipe.vae.config),
    )

.. code:: ipython3

    generator = torch.Generator().manual_seed(42)

    image = pipe(prompt=prompt, guidance_scale=0.0, num_inference_steps=4, generator=generator).images[0]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-717/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `_execution_device` directly via 'PixArtAlphaPipeline' object attribute is deprecated. Please access '_execution_device' over 'PixArtAlphaPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: pixart-with-output_files/pixart-with-output_26_0.png



Interactive inference
---------------------



.. code:: ipython3

    import gradio as gr


    def generate(prompt, seed, negative_prompt, num_inference_steps):
        generator = torch.Generator().manual_seed(seed)
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, num_inference_steps=num_inference_steps, generator=generator, guidance_scale=0.0).images[0]
        return image


    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Caption"),
            gr.Slider(0, np.iinfo(np.int32).max, label="Seed"),
            gr.Textbox(label="Negative prompt"),
            gr.Slider(2, 20, step=1, label="Number of inference steps", value=4),
        ],
        "image",
        examples=[
            ["A small cactus with a happy face in the Sahara desert.", 42],
            ["an astronaut sitting in a diner, eating fries, cinematic, analog film", 42],
            [
                "Pirate ship trapped in a cosmic maelstrom nebula, rendered in cosmic beach whirlpool engine, volumetric lighting, spectacular, ambient lights, light pollution, cinematic atmosphere, art nouveau style, illustration art artwork by SenseiJaye, intricate detail.",
                0,
            ],
            ["professional portrait photo of an anthropomorphic cat wearing fancy gentleman hat and jacket walking in autumn forest.", 0],
        ],
        allow_flagging="never",
    )
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860

    To create a public link, set `share=True` in `launch()`.







