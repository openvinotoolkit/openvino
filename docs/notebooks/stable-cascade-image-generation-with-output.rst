Image generation with Stable Cascade and OpenVINO
=================================================

`Stable Cascade <https://huggingface.co/stabilityai/stable-cascade>`__
is built upon the
`Würstchen <https://openreview.net/forum?id=gU58d5QeGv>`__ architecture
and its main difference to other models like Stable Diffusion is that it
is working at a much smaller latent space. Why is this important? The
smaller the latent space, the faster you can run inference and the
cheaper the training becomes. How small is the latent space? Stable
Diffusion uses a compression factor of 8, resulting in a 1024x1024 image
being encoded to 128x128. Stable Cascade achieves a compression factor
of 42, meaning that it is possible to encode a 1024x1024 image to 24x24,
while maintaining crisp reconstructions. The text-conditional model is
then trained in the highly compressed latent space.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load the original model <#load-the-original-model>`__

   -  `Infer the original model <#infer-the-original-model>`__

-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Prior pipeline <#prior-pipeline>`__
   -  `Decoder pipeline <#decoder-pipeline>`__

-  `Select inference device <#select-inference-device>`__
-  `Building the pipeline <#building-the-pipeline>`__
-  `Inference <#inference>`__
-  `Interactive inference <#interactive-inference>`__ 
   


This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "diffusers>=0.27.0" accelerate datasets gradio transformers "nncf>=2.10.0" "protobuf>=3.20" "openvino>=2024.1.0" "torch>=2.1" --extra-index-url https://download.pytorch.org/whl/cpu


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    descript-audiotools 0.7.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 5.28.3 which is incompatible.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.19.1+cpu which is incompatible.
    open-clip-torch 2.22.0 requires protobuf<4, but you have protobuf 5.28.3 which is incompatible.
    s3fs 2024.10.0 requires fsspec==2024.10.0.*, but you have fsspec 2024.9.0 which is incompatible.
    tensorflow 2.12.0 requires protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3, but you have protobuf 5.28.3 which is incompatible.
    tensorflow-metadata 1.14.0 requires protobuf<4.21,>=3.20.3, but you have protobuf 5.28.3 which is incompatible.
    wandb 0.15.4 requires protobuf!=4.21.0,<5,>=3.12.0; python_version < "3.9" and sys_platform == "linux", but you have protobuf 5.28.3 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Load and run the original pipeline
----------------------------------



.. code:: ipython3

    import torch
    from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline
    
    prompt = "an image of a shiba inu, donning a spacesuit and helmet"
    negative_prompt = ""
    
    prior = StableCascadePriorPipeline.from_pretrained("stabilityai/stable-cascade-prior", torch_dtype=torch.float32)
    decoder = StableCascadeDecoderPipeline.from_pretrained("stabilityai/stable-cascade", torch_dtype=torch.float32)


.. parsed-literal::

    2024-10-23 04:56:18.757348: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 04:56:18.791430: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 04:56:19.346484: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]


To reduce memory usage, we skip the original inference. If you want run
it, turn it.

.. code:: ipython3

    import ipywidgets as widgets
    
    
    run_original_inference = widgets.Checkbox(
        value=False,
        description="Run original inference",
        disabled=False,
    )
    
    run_original_inference




.. parsed-literal::

    Checkbox(value=False, description='Run original inference')



.. code:: ipython3

    if run_original_inference.value:
        prior.to(torch.device("cpu"))
        prior_output = prior(
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt=negative_prompt,
            guidance_scale=4.0,
            num_images_per_prompt=1,
            num_inference_steps=20,
        )
    
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=0.0,
            output_type="pil",
            num_inference_steps=10,
        ).images[0]
        display(decoder_output)

Convert the model to OpenVINO IR
--------------------------------



Stable Cascade has 2 components: - Prior stage ``prior``: create
low-dimensional latent space representation of the image using
text-conditional LDM - Decoder stage ``decoder``: using representation
from Prior Stage, produce a latent image in latent space of higher
dimensionality using LDM and using VQGAN-decoder, decode the latent
image to yield a full-resolution output image.

Let’s define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file. We use ``nncf.compress_weights`` to `compress model
weights <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html#compress-model-weights>`__
to 8-bit to reduce model size.

.. code:: ipython3

    import gc
    from pathlib import Path
    
    import openvino as ov
    import nncf
    
    
    MODELS_DIR = Path("models")
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input, input_shape=None):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            model.eval()
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                if not input_shape:
                    converted_model = ov.convert_model(model, example_input=example_input)
                else:
                    converted_model = ov.convert_model(model, example_input=example_input, input=input_shape)
            converted_model = nncf.compress_weights(converted_model)
            ov.save_model(converted_model, xml_path)
            del converted_model
    
            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()
    
            gc.collect()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


Prior pipeline
~~~~~~~~~~~~~~



This pipeline consists of text encoder and prior diffusion model. From
here, we always use fixed shapes in conversion by using an
``input_shape`` parameter to generate a less memory-demanding model.

.. code:: ipython3

    PRIOR_TEXT_ENCODER_OV_PATH = MODELS_DIR / "prior_text_encoder_model.xml"
    
    prior.text_encoder.config.output_hidden_states = True
    
    
    class TextEncoderWrapper(torch.nn.Module):
        def __init__(self, text_encoder):
            super().__init__()
            self.text_encoder = text_encoder
    
        def forward(self, input_ids, attention_mask):
            outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            return outputs["text_embeds"], outputs["last_hidden_state"], outputs["hidden_states"]
    
    
    convert(
        TextEncoderWrapper(prior.text_encoder),
        PRIOR_TEXT_ENCODER_OV_PATH,
        example_input={
            "input_ids": torch.zeros(1, 77, dtype=torch.int32),
            "attention_mask": torch.zeros(1, 77),
        },
        input_shape={"input_ids": ((1, 77),), "attention_mask": ((1, 77),)},
    )
    del prior.text_encoder
    gc.collect();


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4779: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:88: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:164: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (194 / 194)            │ 100% (194 / 194)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. code:: ipython3

    PRIOR_PRIOR_MODEL_OV_PATH = MODELS_DIR / "prior_prior_model.xml"
    
    convert(
        prior.prior,
        PRIOR_PRIOR_MODEL_OV_PATH,
        example_input={
            "sample": torch.zeros(2, 16, 24, 24),
            "timestep_ratio": torch.ones(2),
            "clip_text_pooled": torch.zeros(2, 1, 1280),
            "clip_text": torch.zeros(2, 77, 1280),
            "clip_img": torch.zeros(2, 1, 768),
        },
        input_shape=[((-1, 16, 24, 24),), ((-1),), ((-1, 1, 1280),), ((-1, 77, 1280),), (-1, 1, 768)],
    )
    del prior.prior
    gc.collect();


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_stable_cascade.py:548: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if skip is not None and (x.size(-1) != skip.size(-1) or x.size(-2) != skip.size(-2)):


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (711 / 711)            │ 100% (711 / 711)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Decoder pipeline
~~~~~~~~~~~~~~~~



Decoder pipeline consists of 3 parts: decoder, text encoder and VQGAN.

.. code:: ipython3

    DECODER_TEXT_ENCODER_MODEL_OV_PATH = MODELS_DIR / "decoder_text_encoder_model.xml"
    
    convert(
        TextEncoderWrapper(decoder.text_encoder),
        DECODER_TEXT_ENCODER_MODEL_OV_PATH,
        example_input={
            "input_ids": torch.zeros(1, 77, dtype=torch.int32),
            "attention_mask": torch.zeros(1, 77),
        },
        input_shape={"input_ids": ((1, 77),), "attention_mask": ((1, 77),)},
    )
    
    del decoder.text_encoder
    gc.collect();


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (194 / 194)            │ 100% (194 / 194)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. code:: ipython3

    DECODER_DECODER_MODEL_OV_PATH = MODELS_DIR / "decoder_decoder_model.xml"
    
    convert(
        decoder.decoder,
        DECODER_DECODER_MODEL_OV_PATH,
        example_input={
            "sample": torch.zeros(1, 4, 256, 256),
            "timestep_ratio": torch.ones(1),
            "clip_text_pooled": torch.zeros(1, 1, 1280),
            "effnet": torch.zeros(1, 16, 24, 24),
        },
        input_shape=[((-1, 4, 256, 256),), ((-1),), ((-1, 1, 1280),), ((-1, 16, 24, 24),)],
    )
    del decoder.decoder
    gc.collect();


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (779 / 779)            │ 100% (779 / 779)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. code:: ipython3

    VQGAN_PATH = MODELS_DIR / "vqgan_model.xml"
    
    
    class VqganDecoderWrapper(torch.nn.Module):
        def __init__(self, vqgan):
            super().__init__()
            self.vqgan = vqgan
    
        def forward(self, h):
            return self.vqgan.decode(h)
    
    
    convert(
        VqganDecoderWrapper(decoder.vqgan),
        VQGAN_PATH,
        example_input=torch.zeros(1, 4, 256, 256),
        input_shape=(1, 4, 256, 256),
    )
    del decoder.vqgan
    gc.collect();


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (42 / 42)              │ 100% (42 / 42)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Select inference device
-----------------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)
    
    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



Building the pipeline
---------------------



Let’s create callable wrapper classes for compiled models to allow
interaction with original pipelines. Note that all of wrapper classes
return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from collections import namedtuple
    
    core = ov.Core()
    
    
    BaseModelOutputWithPooling = namedtuple("BaseModelOutputWithPooling", ["text_embeds", "last_hidden_state", "hidden_states"])
    
    
    class TextEncoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, text_encoder_path, device):
            self.text_encoder = core.compile_model(text_encoder_path, device.value)
    
        def __call__(self, input_ids, attention_mask, output_hidden_states=True):
            output = self.text_encoder({"input_ids": input_ids, "attention_mask": attention_mask})
            text_embeds = output[0]
            last_hidden_state = output[1]
            hidden_states = list(output.values())[1:]
            return BaseModelOutputWithPooling(torch.from_numpy(text_embeds), torch.from_numpy(last_hidden_state), [torch.from_numpy(hs) for hs in hidden_states])

.. code:: ipython3

    class PriorPriorWrapper:
        def __init__(self, prior_path, device):
            self.prior = core.compile_model(prior_path, device.value)
            self.config = namedtuple("PriorWrapperConfig", ["clip_image_in_channels", "in_channels"])(768, 16)  # accessed in the original workflow
            self.parameters = lambda: (torch.zeros(i, dtype=torch.float32) for i in range(1))  # accessed in the original workflow
    
        def __call__(self, sample, timestep_ratio, clip_text_pooled, clip_text=None, clip_img=None, **kwargs):
            inputs = {
                "sample": sample,
                "timestep_ratio": timestep_ratio,
                "clip_text_pooled": clip_text_pooled,
                "clip_text": clip_text,
                "clip_img": clip_img,
            }
            output = self.prior(inputs)
            return [torch.from_numpy(output[0])]

.. code:: ipython3

    class DecoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, decoder_path, device):
            self.decoder = core.compile_model(decoder_path, device.value)
    
        def __call__(self, sample, timestep_ratio, clip_text_pooled, effnet, **kwargs):
            inputs = {"sample": sample, "timestep_ratio": timestep_ratio, "clip_text_pooled": clip_text_pooled, "effnet": effnet}
            output = self.decoder(inputs)
            return [torch.from_numpy(output[0])]

.. code:: ipython3

    VqganOutput = namedtuple("VqganOutput", "sample")
    
    
    class VqganWrapper:
        config = namedtuple("VqganWrapperConfig", "scale_factor")(0.3764)  # accessed in the original workflow
    
        def __init__(self, vqgan_path, device):
            self.vqgan = core.compile_model(vqgan_path, device.value)
    
        def decode(self, h):
            output = self.vqgan(h)[0]
            output = torch.tensor(output)
            return VqganOutput(output)

And insert wrappers instances in the pipeline:

.. code:: ipython3

    prior.text_encoder = TextEncoderWrapper(PRIOR_TEXT_ENCODER_OV_PATH, device)
    prior.prior = PriorPriorWrapper(PRIOR_PRIOR_MODEL_OV_PATH, device)
    decoder.decoder = DecoderWrapper(DECODER_DECODER_MODEL_OV_PATH, device)
    decoder.text_encoder = TextEncoderWrapper(DECODER_TEXT_ENCODER_MODEL_OV_PATH, device)
    decoder.vqgan = VqganWrapper(VQGAN_PATH, device)

Inference
---------



.. code:: ipython3

    prior_output = prior(
        prompt=prompt,
        height=1024,
        width=1024,
        negative_prompt=negative_prompt,
        guidance_scale=4.0,
        num_images_per_prompt=1,
        num_inference_steps=20,
    )
    
    decoder_output = decoder(
        image_embeddings=prior_output.image_embeddings,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=0.0,
        output_type="pil",
        num_inference_steps=10,
    ).images[0]
    display(decoder_output)



.. parsed-literal::

      0%|          | 0/20 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/10 [00:00<?, ?it/s]



.. image:: stable-cascade-image-generation-with-output_files/stable-cascade-image-generation-with-output_29_2.png


Interactive inference
---------------------



.. code:: ipython3

    def generate(prompt, negative_prompt, prior_guidance_scale, decoder_guidance_scale, seed):
        generator = torch.Generator().manual_seed(seed)
        prior_output = prior(
            prompt=prompt,
            height=1024,
            width=1024,
            negative_prompt=negative_prompt,
            guidance_scale=prior_guidance_scale,
            num_images_per_prompt=1,
            num_inference_steps=20,
            generator=generator,
        )
    
        decoder_output = decoder(
            image_embeddings=prior_output.image_embeddings,
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=decoder_guidance_scale,
            output_type="pil",
            num_inference_steps=10,
            generator=generator,
        ).images[0]
    
        return decoder_output

.. code:: ipython3

    import requests
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-cascade-image-generation/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(generate)
    
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







