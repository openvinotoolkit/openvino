Lightweight image generation with aMUSEd and OpenVINO
=====================================================

`Amused <https://huggingface.co/docs/diffusers/api/pipelines/amused>`__
is a lightweight text to image model based off of the
`muse <https://arxiv.org/pdf/2301.00704.pdf>`__ architecture. Amused is
particularly useful in applications that require a lightweight and fast
model such as generating many images quickly at once.

Amused is a VQVAE token based transformer that can generate an image in
fewer forward passes than many diffusion models. In contrast with muse,
it uses the smaller text encoder CLIP-L/14 instead of t5-xxl. Due to its
small parameter count and few forward pass generation process, amused
can generate many images quickly. This benefit is seen particularly at
larger batch sizes.

Table of contents:
^^^^^^^^^^^^^^^^^^

-  `Prerequisites <#prerequisites>`__
-  `Load and run the original
   pipeline <#load-and-run-the-original-pipeline>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Convert the Text Encoder <#convert-the-text-encoder>`__
   -  `Convert the U-ViT transformer <#convert-the-u-vit-transformer>`__
   -  `Convert VQ-GAN decoder
      (VQVAE) <#convert-vq-gan-decoder-vqvae>`__

-  `Compiling models and prepare
   pipeline <#compiling-models-and-prepare-pipeline>`__
-  `Interactive inference <#interactive-inference>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "diffusers>=0.25.0" "openvino>=2023.2.0" "accelerate>=0.20.3" gradio torch --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.1 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.


Load and run the original pipeline
----------------------------------



.. code:: ipython3

    import torch
    from diffusers import AmusedPipeline


    pipe = AmusedPipeline.from_pretrained(
        "amused/amused-256",
    )

    prompt = "kind smiling ghost"
    image = pipe(prompt, generator=torch.Generator('cpu').manual_seed(8)).images[0]
    image.save('text2image_256.png')


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/utils/outputs.py:63: UserWarning: torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.
      torch.utils._pytree._register_pytree_node(



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]


.. parsed-literal::

    2024-02-10 00:42:21.470840: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-02-10 00:42:21.506354: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.


.. parsed-literal::

    2024-02-10 00:42:22.153566: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: 277-amused-lightweight-text-to-image-with-output_files/277-amused-lightweight-text-to-image-with-output_6_0.png



Convert the model to OpenVINO IR
--------------------------------



aMUSEd consists of three separately trained components: a pre-trained
CLIP-L/14 text encoder, a VQ-GAN, and a U-ViT.

.. figure:: https://cdn-uploads.huggingface.co/production/uploads/5dfcb1aada6d0311fd3d5448/97ca2Vqm7jBfCAzq20TtF.png
   :alt: image_png

   image_png

During inference, the U-ViT is conditioned on the text encoder’s hidden
states and iteratively predicts values for all masked tokens. The cosine
masking schedule determines a percentage of the most confident token
predictions to be fixed after every iteration. After 12 iterations, all
tokens have been predicted and are decoded by the VQ-GAN into image
pixels.

Define paths for converted models:

.. code:: ipython3

    from pathlib import Path


    TRANSFORMER_OV_PATH = Path('models/transformer_ir.xml')
    TEXT_ENCODER_OV_PATH = Path('models/text_encoder_ir.xml')
    VQVAE_OV_PATH = Path('models/vqvae_ir.xml')

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

Convert the Text Encoder
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.text_encoder = text_encoder

        def forward(self, input_ids=None, return_dict=None, output_hidden_states=None):

            outputs = self.text_encoder(
                input_ids=input_ids,
                return_dict=return_dict,
                output_hidden_states=output_hidden_states,
            )

            return outputs.text_embeds, outputs.last_hidden_state, outputs.hidden_states


    input_ids = pipe.tokenizer(
        prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
    )

    input_example = {
        'input_ids': input_ids.input_ids,
        'return_dict': torch.tensor(True),
        'output_hidden_states': torch.tensor(True)
    }

    convert(TextEncoderWrapper(pipe.text_encoder), TEXT_ENCODER_OV_PATH, input_example)


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:614: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      encoder_states = () if output_hidden_states else None
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:619: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output_hidden_states:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:273: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:281: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:313: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:642: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if output_hidden_states:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:645: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:736: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:1220: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not return_dict:


Convert the U-ViT transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    class TransformerWrapper(torch.nn.Module):
        def __init__(self, transformer):
            super().__init__()
            self.transformer = transformer

        def forward(self, latents=None, micro_conds=None, pooled_text_emb=None, encoder_hidden_states=None):

            return self.transformer(
                latents,
                micro_conds=micro_conds,
                pooled_text_emb=pooled_text_emb,
                encoder_hidden_states=encoder_hidden_states,
            )


    shape = (1, 16, 16)
    latents = torch.full(
        shape, pipe.scheduler.config.mask_token_id, dtype=torch.long
    )
    latents = torch.cat([latents] * 2)


    example_input = {
        'latents': latents,
        'micro_conds': torch.rand([2, 5], dtype=torch.float32),
        'pooled_text_emb': torch.rand([2, 768], dtype=torch.float32),
        'encoder_hidden_states': torch.rand([2, 77, 768], dtype=torch.float32),
    }


    pipe.transformer.eval()
    w_transformer = TransformerWrapper(pipe.transformer)
    convert(w_transformer, TRANSFORMER_OV_PATH, example_input)

Convert VQ-GAN decoder (VQVAE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Function ``get_latents`` is
needed to return real latents for the conversion. Due to the VQVAE
implementation autogenerated tensor of the required shape is not
suitable. This function repeats part of ``AmusedPipeline``.

.. code:: ipython3

    def get_latents():
        shape = (1, 16, 16)
        latents = torch.full(
            shape, pipe.scheduler.config.mask_token_id, dtype=torch.long
        )
        model_input = torch.cat([latents] * 2)


        model_output = pipe.transformer(
            model_input,
            micro_conds=torch.rand([2, 5], dtype=torch.float32),
            pooled_text_emb=torch.rand([2, 768], dtype=torch.float32),
            encoder_hidden_states=torch.rand([2, 77, 768], dtype=torch.float32),
        )
        guidance_scale = 10.0
        uncond_logits, cond_logits = model_output.chunk(2)
        model_output = uncond_logits + guidance_scale * (cond_logits - uncond_logits)


        latents = pipe.scheduler.step(
            model_output=model_output,
            timestep=torch.tensor(0),
            sample=latents,
        ).prev_sample

        return latents


    class VQVAEWrapper(torch.nn.Module):
        def __init__(self, vqvae):
            super().__init__()
            self.vqvae = vqvae

        def forward(self, latents=None, force_not_quantize=True, shape=None):
            outputs = self.vqvae.decode(
                latents,
                force_not_quantize=force_not_quantize,
                shape=shape.tolist(),
            )

            return outputs


    latents = get_latents()
    example_vqvae_input = {
        'latents': latents,
        'force_not_quantize': torch.tensor(True),
        'shape': torch.tensor((1, 16, 16, 64))
    }

    convert(VQVAEWrapper(pipe.vqvae), VQVAE_OV_PATH, example_vqvae_input)


.. parsed-literal::

    /tmp/ipykernel_2865109/249287788.py:38: TracerWarning: Converting a tensor to a Python list might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      shape=shape.tolist(),
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/vq_model.py:144: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if not force_not_quantize:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:149: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:165: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:


Compiling models and prepare pipeline
-------------------------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets


    core = ov.Core()
    DEVICE = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )

    DEVICE




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_text_encoder = core.compile_model(TEXT_ENCODER_OV_PATH, DEVICE.value)
    ov_transformer = core.compile_model(TRANSFORMER_OV_PATH, DEVICE.value)
    ov_vqvae = core.compile_model(VQVAE_OV_PATH, DEVICE.value)

Let’s create callable wrapper classes for compiled models to allow
interaction with original ``AmusedPipeline`` class. Note that all of
wrapper classes return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple


    class ConvTextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder, config):
            super().__init__()
            self.config = config
            self.text_encoder = text_encoder

        def forward(self, input_ids=None, return_dict=None, output_hidden_states=None):
            inputs = {
                'input_ids': input_ids,
                'return_dict': return_dict,
                'output_hidden_states': output_hidden_states
            }

            outs = self.text_encoder(inputs)

            outputs = namedtuple('CLIPTextModelOutput', ('text_embeds', 'last_hidden_state', 'hidden_states'))

            text_embeds = torch.from_numpy(outs[0])
            last_hidden_state = torch.from_numpy(outs[1])
            hidden_states = list(torch.from_numpy(out) for out in outs.values())[2:]

            return outputs(text_embeds, last_hidden_state, hidden_states)

.. code:: ipython3

    class ConvTransformerWrapper(torch.nn.Module):
        def __init__(self, transformer, config):
            super().__init__()
            self.config = config
            self.transformer = transformer

        def forward(self, latents=None, micro_conds=None, pooled_text_emb=None, encoder_hidden_states=None, **kwargs):
            outputs = self.transformer(
                {
                    'latents': latents,
                    'micro_conds': micro_conds,
                    'pooled_text_emb': pooled_text_emb,
                    'encoder_hidden_states': encoder_hidden_states,
                },
                share_inputs=False
            )

            return torch.from_numpy(outputs[0])

.. code:: ipython3

    class ConvVQVAEWrapper(torch.nn.Module):
        def __init__(self, vqvae, dtype, config):
            super().__init__()
            self.vqvae = vqvae
            self.dtype = dtype
            self.config = config

        def decode(self, latents=None, force_not_quantize=True, shape=None):
            inputs = {
                'latents': latents,
                'force_not_quantize': force_not_quantize,
                'shape': torch.tensor(shape)
            }

            outs = self.vqvae(inputs)
            outs = namedtuple('VQVAE', 'sample')(torch.from_numpy(outs[0]))

            return outs

And insert wrappers instances in the pipeline:

.. code:: ipython3

    prompt = "kind smiling ghost"

    transformer = pipe.transformer
    vqvae = pipe.vqvae
    text_encoder = pipe.text_encoder

    pipe.__dict__["_internal_dict"]['_execution_device'] = pipe._execution_device  # this is to avoid some problem that can occur in the pipeline
    pipe.register_modules(
        text_encoder=ConvTextEncoderWrapper(ov_text_encoder, text_encoder.config),
        transformer=ConvTransformerWrapper(ov_transformer, transformer.config),
        vqvae=ConvVQVAEWrapper(ov_vqvae, vqvae.dtype, vqvae.config),
    )

    image = pipe(prompt, generator=torch.Generator('cpu').manual_seed(8)).images[0]
    image.save('text2image_256.png')


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-609/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/configuration_utils.py:139: FutureWarning: Accessing config attribute `_execution_device` directly via 'AmusedPipeline' object attribute is deprecated. Please access '_execution_device' over 'AmusedPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    image




.. image:: 277-amused-lightweight-text-to-image-with-output_files/277-amused-lightweight-text-to-image-with-output_28_0.png



Interactive inference
---------------------



.. code:: ipython3

    import numpy as np
    import gradio as gr


    def generate(prompt, seed, _=gr.Progress(track_tqdm=True)):
        image = pipe(prompt, generator=torch.Generator('cpu').manual_seed(seed)).images[0]
        return image


    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Prompt"),
            gr.Slider(0, np.iinfo(np.int32).max, label="Seed")
        ],
        "image",
        examples=[
            ["happy snowman", 88],
            ["green ghost rider", 0],
            ["kind smiling ghost", 8],
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



.. .. raw:: html

..    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

