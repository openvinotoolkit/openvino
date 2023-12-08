Image generation with Würstchen and OpenVINO
============================================

.. figure:: 265-wuerstchen-image-generation-with-output_files/499b779a-61d1-4e68-a1c3-437122622ba7.png
   :alt: image.png

   image.png

`Würstchen <https://arxiv.org/abs/2306.00637>`__ is a diffusion model,
whose text-conditional model works in a highly compressed latent space
of images. Why is this important? Compressing data can reduce
computational costs for both training and inference by magnitudes.
Training on 1024x1024 images, is way more expensive than training at
32x32. Usually, other works make use of a relatively small compression,
in the range of 4x - 8x spatial compression. Würstchen takes this to an
extreme. Through its novel design, authors achieve a 42x spatial
compression. This was unseen before because common methods fail to
faithfully reconstruct detailed images after 16x spatial compression.
Würstchen employs a two-stage compression (referred below as *Decoder*).
The first one is a VQGAN, and the second is a Diffusion Autoencoder
(more details can be found in the paper). A third model (referred below
as *Prior*) is learned in that highly compressed latent space. This
training requires fractions of the compute used for current
top-performing models, allowing also cheaper and faster inference.

We will use PyTorch version of Würstchen `model from HuggingFace
Hub <https://huggingface.co/warp-ai/wuerstchen>`__.

**Table of contents:**

-  `Prerequisites <#prerequisites>`__
-  `Load the original model <#load-the-original-model>`__

   -  `Infer the original model <#infer-the-original-model>`__

-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Prior pipeline <#prior-pipeline>`__
   -  `Decoder pipeline <#decoder-pipeline>`__

-  `Compiling models <#compiling-models>`__
-  `Building the pipeline <#building-the-pipeline>`__
-  `Inference <#inference>`__
-  `Interactive inference <#interactive-inference>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q  "diffusers>=0.21.0" transformers accelerate matplotlib gradio
    %pip uninstall -q -y openvino-dev openvino openvino-nightly
    %pip install -q openvino-nightly


.. parsed-literal::

    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.
    Note: you may need to restart the kernel to use updated packages.


.. code:: ipython3

    from pathlib import Path
    from collections import namedtuple
    import gc
    
    import diffusers
    import torch
    import matplotlib.pyplot as plt
    import gradio as gr
    import numpy as np
    
    import openvino as ov

.. code:: ipython3

    MODELS_DIR = Path("models")
    PRIOR_TEXT_ENCODER_PATH = MODELS_DIR / "prior_text_encoder.xml"
    PRIOR_PRIOR_PATH = MODELS_DIR / "prior_prior.xml"
    DECODER_PATH = MODELS_DIR / "decoder.xml"
    TEXT_ENCODER_PATH = MODELS_DIR / "text_encoder.xml"
    VQGAN_PATH = MODELS_DIR / "vqgan.xml"
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

.. code:: ipython3

    BaseModelOutputWithPooling = namedtuple("BaseModelOutputWithPooling", "last_hidden_state")
    DecoderOutput = namedtuple("DecoderOutput", "sample")

Load the original model
-----------------------



We use ``from_pretrained`` method of
``diffusers.AutoPipelineForText2Image`` to load the pipeline.

.. code:: ipython3

    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained("warp-diffusion/wuerstchen")


.. parsed-literal::

    /home/itrushkin/.virtualenvs/wuerstchen/lib/python3.10/site-packages/torch/cuda/__init__.py:611: UserWarning: Can't initialize NVML
      warnings.warn("Can't initialize NVML")



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/4 [00:00<?, ?it/s]


Loaded model has ``WuerstchenCombinedPipeline`` type and consists of 2
parts: prior and decoder.

Infer the original model
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    caption = "Anthropomorphic cat dressed as a fire fighter"
    negative_prompt = ""
    
    output = pipeline(
        prompt=caption,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        prior_guidance_scale=4.0,
        decoder_guidance_scale=0.0,
        output_type="pil",
    ).images



.. parsed-literal::

      0%|          | 0/60 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    plt.figure(figsize=(8 * len(output), 8), dpi=128)
    for i, x in enumerate(output):
        plt.subplot(1, len(output), i + 1)
        plt.imshow(x)
        plt.axis("off")



.. image:: 265-wuerstchen-image-generation-with-output_files/265-wuerstchen-image-generation-with-output_11_0.png


Convert the model to OpenVINO IR
--------------------------------



Main model components: - Prior stage: create low-dimensional latent
space representation of the image using text-conditional LDM - Decoder
stage: using representation from Prior Stage, produce a latent image in
latent space of higher dimensionality using another LDM and using
VQGAN-decoder, decode the latent image to yield a full-resolution output
image

The pipeline consists of 2 sub-pipelines: Prior pipeline accessed by
``prior_pipe`` property, and Decoder Pipeline accessed by
``decoder_pipe`` property.

.. code:: ipython3

    # Prior pipeline
    pipeline.prior_text_encoder.eval()
    pipeline.prior_prior.eval()
    
    # Decoder pipeline
    pipeline.decoder.eval()
    pipeline.text_encoder.eval()
    pipeline.vqgan.eval();

Next, let’s define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    def convert(model: torch.nn.Module, xml_path: Path, **convert_kwargs):
        if not xml_path.exists():
            converted_model = ov.convert_model(model, **convert_kwargs)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)
            del converted_model
    
            # Clean torch jit cache
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()
    
            gc.collect()

Prior pipeline
~~~~~~~~~~~~~~



This pipeline consists of text encoder and prior diffusion model. From
here, we always use fixed shapes in conversion by using an ``input``
parameter to generate a less memory-demanding model.

Text encoder model has 2 inputs: - ``input_ids``: vector of tokenized
input sentence. Default tokenizer vector length is 77. -
``attention_mask``: vector of same length as ``input_ids`` describing
the attention mask.

.. code:: ipython3

    convert(
        pipeline.prior_text_encoder,
        PRIOR_TEXT_ENCODER_PATH,
        example_input={
            "input_ids": torch.zeros(1, 77, dtype=torch.int32),
            "attention_mask": torch.zeros(1, 77),
        },
        input={"input_ids": ((1, 77),), "attention_mask": ((1, 77),)},
    )
    del pipeline.prior_text_encoder
    del pipeline.prior_pipe.text_encoder
    gc.collect()




.. parsed-literal::

    0



Prior model is the canonical unCLIP prior to approximate the image
embedding from the text embedding. Like UNet, it has 3 inputs: sample,
timestep and encoder hidden states.

.. code:: ipython3

    convert(
        pipeline.prior_prior,
        PRIOR_PRIOR_PATH,
        example_input=[torch.zeros(2, 16, 24, 24), torch.zeros(2), torch.zeros(2, 77, 1280)],
        input=[((2, 16, 24, 24),), ((2),), ((2, 77, 1280),)],
    )
    del pipeline.prior_prior
    del pipeline.prior_pipe.prior
    gc.collect()




.. parsed-literal::

    0



Decoder pipeline
~~~~~~~~~~~~~~~~



Decoder pipeline consists of 3 parts: decoder, text encoder and VQGAN.

Decoder model is the WuerstchenDiffNeXt UNet decoder. Inputs are: -
``x``: sample - ``r``: timestep - ``effnet``: interpolation block -
``clip``: encoder hidden states

.. code:: ipython3

    convert(
        pipeline.decoder,
        DECODER_PATH,
        example_input={
            "x": torch.zeros(1, 4, 256, 256),
            "r": torch.zeros(1),
            "effnet": torch.zeros(1, 16, 24, 24),
            "clip": torch.zeros(1, 77, 1024),
        },
        input={
            "x": ((1, 4, 256, 256),),
            "r": ((1),),
            "effnet": ((1, 16, 24, 24),),
            "clip": ((1, 77, 1024),),
        },
    )
    del pipeline.decoder
    del pipeline.decoder_pipe.decoder
    gc.collect()




.. parsed-literal::

    0



The main text encoder has the same input parameters and shapes as text
encoder in `prior pipeline <#prior-pipeline>`__.

.. code:: ipython3

    convert(
        pipeline.text_encoder,
        TEXT_ENCODER_PATH,
        example_input={
            "input_ids": torch.zeros(1, 77, dtype=torch.int32),
            "attention_mask": torch.zeros(1, 77),
        },
        input={"input_ids": ((1, 77),), "attention_mask": ((1, 77),)},
    )
    del pipeline.text_encoder
    del pipeline.decoder_pipe.text_encoder
    gc.collect()




.. parsed-literal::

    0



Pipeline uses VQGAN model ``decode`` method to get the full-size output
image. Here we create the wrapper module for decoding part only. Our
decoder takes as input 4x256x256 latent image.

.. code:: ipython3

    class VqganDecoderWrapper(torch.nn.Module):
        def __init__(self, vqgan):
            super().__init__()
            self.vqgan = vqgan
    
        def forward(self, h):
            return self.vqgan.decode(h)

.. code:: ipython3

    convert(
        VqganDecoderWrapper(pipeline.vqgan),
        VQGAN_PATH,
        example_input=torch.zeros(1, 4, 256, 256),
        input=(1, 4, 256, 256),
    )
    del pipeline.decoder_pipe.vqgan
    gc.collect()




.. parsed-literal::

    0



Compiling models
----------------



.. code:: ipython3

    core = ov.Core()

Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import ipywidgets as widgets
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device

.. code:: ipython3

    ov_prior_text_encoder = core.compile_model(PRIOR_TEXT_ENCODER_PATH, device.value)

.. code:: ipython3

    ov_prior_prior = core.compile_model(PRIOR_PRIOR_PATH, device.value)

.. code:: ipython3

    ov_decoder = core.compile_model(DECODER_PATH, device.value)

.. code:: ipython3

    ov_text_encoder = core.compile_model(TEXT_ENCODER_PATH, device.value)

.. code:: ipython3

    ov_vqgan = core.compile_model(VQGAN_PATH, device.value)

Building the pipeline
---------------------



Let’s create callable wrapper classes for compiled models to allow
interaction with original ``WuerstchenCombinedPipeline`` class. Note
that all of wrapper classes return ``torch.Tensor``\ s instead of
``np.array``\ s.

.. code:: ipython3

    class TextEncoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, text_encoder):
            self.text_encoder = text_encoder
    
        def __call__(self, input_ids, attention_mask):
            output = self.text_encoder({"input_ids": input_ids, "attention_mask": attention_mask})[
                "last_hidden_state"
            ]
            output = torch.tensor(output)
            return BaseModelOutputWithPooling(output)

.. code:: ipython3

    class PriorPriorWrapper:
        config = namedtuple("PriorPriorWrapperConfig", "c_in")(16)  # accessed in the original workflow
    
        def __init__(self, prior):
            self.prior = prior
    
        def __call__(self, x, r, c):
            output = self.prior([x, r, c])[0]
            return torch.tensor(output)

.. code:: ipython3

    class DecoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, decoder):
            self.decoder = decoder
    
        def __call__(self, x, r, effnet, clip):
            output = self.decoder({"x": x, "r": r, "effnet": effnet, "clip": clip})[0]
            output = torch.tensor(output)
            return output

.. code:: ipython3

    class VqganWrapper:
        config = namedtuple("VqganWrapperConfig", "scale_factor")(0.3764)  # accessed in the original workflow
    
        def __init__(self, vqgan):
            self.vqgan = vqgan
    
        def decode(self, h):
            output = self.vqgan(h)[0]
            output = torch.tensor(output)
            return DecoderOutput(output)

And insert wrappers instances in the pipeline:

.. code:: ipython3

    pipeline.prior_pipe.text_encoder = TextEncoderWrapper(ov_prior_text_encoder)
    pipeline.prior_pipe.prior = PriorPriorWrapper(ov_prior_prior)
    
    pipeline.decoder_pipe.decoder = DecoderWrapper(ov_decoder)
    pipeline.decoder_pipe.text_encoder = TextEncoderWrapper(ov_text_encoder)
    pipeline.decoder_pipe.vqgan = VqganWrapper(ov_vqgan)

Inference
---------



.. code:: ipython3

    caption = "Anthropomorphic cat dressed as a fire fighter"
    negative_prompt = ""
    
    output = pipeline(
        prompt=caption,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        prior_guidance_scale=4.0,
        decoder_guidance_scale=0.0,
        output_type="pil",
    ).images



.. parsed-literal::

      0%|          | 0/60 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/12 [00:00<?, ?it/s]


.. code:: ipython3

    plt.figure(figsize=(8 * len(output), 8), dpi=128)
    for i, x in enumerate(output):
        plt.subplot(1, len(output), i + 1)
        plt.imshow(x)
        plt.axis("off")



.. image:: 265-wuerstchen-image-generation-with-output_files/265-wuerstchen-image-generation-with-output_45_0.png


Interactive inference
---------------------



.. code:: ipython3

    def generate(caption, negative_prompt, prior_guidance_scale, seed):
        generator = torch.Generator().manual_seed(seed)
        image = pipeline(
            prompt=caption,
            height=1024,
            width=1024,
            negative_prompt=negative_prompt,
            prior_num_inference_steps=30,
            prior_guidance_scale=prior_guidance_scale,
            generator=generator,
            output_type="pil",
        ).images[0]
        return image

.. code:: ipython3

    demo = gr.Interface(
        generate,
        [
            gr.Textbox(label="Caption"),
            gr.Textbox(label="Negative prompt"),
            gr.Slider(2, 20, step=1, label="Prior guidance scale"),
            gr.Slider(0, np.iinfo(np.int32).max, label="Seed")
        ],
        "image",
        examples=[["Antropomorphic cat dressed as a firefighter", "", 4, 0]],
        allow_flagging="never",
    )
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(debug=False, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
