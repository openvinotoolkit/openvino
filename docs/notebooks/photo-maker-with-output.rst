Text-to-image generation using PhotoMaker and OpenVINO
======================================================

PhotoMaker is an efficient personalized text-to-image generation method,
which mainly encodes an arbitrary number of input ID images into a stack
ID embedding for preserving ID information. Such an embedding, serving
as a unified ID representation, can not only encapsulate the
characteristics of the same input ID comprehensively, but also
accommodate the characteristics of different IDs for subsequent
integration. This paves the way for more intriguing and practically
valuable applications. Users can input one or a few face photos, along
with a text prompt, to receive a customized photo or painting (no
training required!). Additionally, this model can be adapted to any base
model based on ``SDXL`` or used in conjunction with other ``LoRA``
modules.More details about PhotoMaker can be found in the `technical
report <https://arxiv.org/pdf/2312.04461.pdf>`__.

This notebook explores how to speed up PhotoMaker pipeline using
OpenVINO.


**Table of contents:**


-  `PhotoMaker pipeline
   introduction <#photomaker-pipeline-introduction>`__
-  `Prerequisites <#prerequisites>`__
-  `Load original pipeline and prepare models for
   conversion <#load-original-pipeline-and-prepare-models-for-conversion>`__
-  `Convert models to OpenVINO Intermediate representation (IR)
   format <#convert-models-to-openvino-intermediate-representation-ir-format>`__

   -  `ID Encoder <#id-encoder>`__
   -  `Text Encoder <#text-encoder>`__
   -  `U-Net <#u-net>`__
   -  `VAE Decoder <#vae-decoder>`__

-  `Prepare Inference pipeline <#prepare-inference-pipeline>`__

   -  `Select inference device for Stable Diffusion
      pipeline <#select-inference-device-for-stable-diffusion-pipeline>`__
   -  `Compile models and create their Wrappers for
      inference <#compile-models-and-create-their-wrappers-for-inference>`__

-  `Running Text-to-Image Generation with
   OpenVINO <#running-text-to-image-generation-with-openvino>`__
-  `Interactive Demo <#interactive-demo>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

PhotoMaker pipeline introduction
--------------------------------



For the proposed PhotoMaker, we first obtain the text embedding and
image embeddings from ``text encoder(s)`` and ``image(ID) encoder``,
respectively. Then, we extract the fused embedding by merging the
corresponding class embedding (e.g., man and woman) and each image
embedding. Next, we concatenate all fused embeddings along the length
dimension to form the stacked ID embedding. Finally, we feed the stacked
ID embedding to all cross-attention layers for adaptively merging the ID
content in the ``diffusion model``. Note that although we use images of
the same ID with the masked background during training, we can directly
input images of different IDs without background distortion to create a
new ID during inference.

Prerequisites
-------------



Clone PhotoMaker repository

.. code:: ipython3

    from pathlib import Path
    
    if not Path("PhotoMaker").exists():
        !git clone https://github.com/TencentARC/PhotoMaker.git
        %cd PhotoMaker
        !git checkout "1e78aa6514c11a84ef1be27b56c7c72d6c70f8fc"
        %cd ..


.. parsed-literal::

    Cloning into 'PhotoMaker'...
    remote: Enumerating objects: 303, done.[K
    remote: Counting objects: 100% (148/148), done.[K
    remote: Compressing objects: 100% (95/95), done.[K
    remote: Total 303 (delta 130), reused 53 (delta 53), pack-reused 155 (from 1)[K
    Receiving objects: 100% (303/303), 10.23 MiB | 29.77 MiB/s, done.
    Resolving deltas: 100% (162/162), done.
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/photo-maker/PhotoMaker
    Note: switching to '1e78aa6514c11a84ef1be27b56c7c72d6c70f8fc'.
    
    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by switching back to a branch.
    
    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -c with the switch command. Example:
    
      git switch -c <new-branch-name>
    
    Or undo this operation with:
    
      git switch -
    
    Turn off this advice by setting config variable advice.detachedHead to false
    
    HEAD is now at 1e78aa6 Update README.md
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/photo-maker


Install required packages

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu \
    transformers "torch>=2.1" "diffusers>=0.26,<0.30" "gradio>=4.19" "openvino>=2024.0.0" "einops" torchvision "peft>=0.6.2" "nncf>=2.9.0" "protobuf==3.20.3" "insightface" "onnxruntime"


.. parsed-literal::

    ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    descript-audiotools 0.7.2 requires protobuf<3.20,>=3.9.2, but you have protobuf 3.20.3 which is incompatible.
    mobileclip 0.1.0 requires torchvision==0.14.1, but you have torchvision 0.17.2+cpu which is incompatible.
    paddleclas 2.5.2 requires gast==0.3.3, but you have gast 0.4.0 which is incompatible.
    paddleclas 2.5.2 requires opencv-python==4.6.0.66, but you have opencv-python 4.10.0.84 which is incompatible.
    supervision 0.24.0 requires numpy<1.23.3,>=1.21.2; python_full_version <= "3.10.0", but you have numpy 1.24.4 which is incompatible.
    tensorflow 2.12.0 requires numpy<1.24,>=1.22, but you have numpy 1.24.4 which is incompatible.
    Note: you may need to restart the kernel to use updated packages.


Prepare PyTorch models

.. code:: ipython3

    adapter_id = "TencentARC/PhotoMaker"
    base_model_id = "SG161222/RealVisXL_V3.0"
    
    TEXT_ENCODER_OV_PATH = Path("model/text_encoder.xml")
    TEXT_ENCODER_2_OV_PATH = Path("model/text_encoder_2.xml")
    UNET_OV_PATH = Path("model/unet.xml")
    ID_ENCODER_OV_PATH = Path("model/id_encoder.xml")
    VAE_DECODER_OV_PATH = Path("model/vae_decoder.xml")

Load original pipeline and prepare models for conversion
--------------------------------------------------------



For exporting each PyTorch model, we will download the ``ID encoder``
weight, ``LoRa`` weight from HuggingFace hub, then using the
``PhotoMakerStableDiffusionXLPipeline`` object from repository of
PhotoMaker to generate the original PhotoMaker pipeline.

.. code:: ipython3

    import torch
    import numpy as np
    import os
    from PIL import Image
    from pathlib import Path
    from PhotoMaker.photomaker.model import PhotoMakerIDEncoder
    from PhotoMaker.photomaker.pipeline import PhotoMakerStableDiffusionXLPipeline
    from diffusers import EulerDiscreteScheduler
    import gc
    
    trigger_word = "img"
    
    
    def load_original_pytorch_pipeline_components(photomaker_path: str, base_model_id: str):
        # Load base model
        pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(base_model_id, use_safetensors=True).to("cpu")
    
        # Load PhotoMaker checkpoint
        pipe.load_photomaker_adapter(
            os.path.dirname(photomaker_path),
            subfolder="",
            weight_name=os.path.basename(photomaker_path),
            trigger_word=trigger_word,
        )
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.fuse_lora()
        gc.collect()
        return pipe


.. parsed-literal::

    2024-10-23 02:19:25.748160: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-10-23 02:19:25.783265: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-10-23 02:19:26.449413: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    from huggingface_hub import hf_hub_download
    
    photomaker_path = hf_hub_download(repo_id=adapter_id, filename="photomaker-v1.bin", repo_type="model")
    
    pipe = load_original_pytorch_pipeline_components(photomaker_path, base_model_id)



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]


.. parsed-literal::

    Loading PhotoMaker components [1] id_encoder from [/opt/home/k8sworker/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/f68f8e6309bf213d28d68230abff0ccc92de9f30]...
    Loading PhotoMaker components [2] lora_weights from [/opt/home/k8sworker/.cache/huggingface/hub/models--TencentARC--PhotoMaker/snapshots/f68f8e6309bf213d28d68230abff0ccc92de9f30]


Convert models to OpenVINO Intermediate representation (IR) format
------------------------------------------------------------------



Starting from 2023.0 release, OpenVINO supports PyTorch models
conversion directly. We need to provide a model object, input data for
model tracing to ``ov.convert_model`` function to obtain OpenVINO
``ov.Model`` object instance. Model can be saved on disk for next
deployment using ``ov.save_model`` function.

The pipeline consists of five important parts:

-  ID Encoder for generating image embeddings to condition by image
   annotation.
-  Text Encoders for creating text embeddings to generate an image from
   a text prompt.
-  Unet for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of models, which require
extensive memory to store the weights during inference, can benefit from
weight compression in the following ways:

-  enabling the inference of exceptionally large models that cannot be
   accommodated in the memory of the device;

-  improving the inference performance of the models by reducing the
   latency of the memory access when computing the operations with
   weights, for example, Linear layers.

`Neural Network Compression Framework
(NNCF) <https://github.com/openvinotoolkit/nncf>`__ provides 4-bit /
8-bit mixed weight quantization as a compression method. The main
difference between weights compression and full model quantization
(post-training quantization) is that activations remain floating-point
in the case of weights compression which leads to a better accuracy.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters.

More details about weights compression can be found in `OpenVINO
documentation <https://docs.openvino.ai/2023.3/weight_compression.html>`__.

.. code:: ipython3

    import openvino as ov
    import nncf
    
    
    def flattenize_inputs(inputs):
        """
        Helper function for resolve nested input structure (e.g. lists or tuples of tensors)
        """
        flatten_inputs = []
        for input_data in inputs:
            if input_data is None:
                continue
            if isinstance(input_data, (list, tuple)):
                flatten_inputs.extend(flattenize_inputs(input_data))
            else:
                flatten_inputs.append(input_data)
        return flatten_inputs
    
    
    dtype_mapping = {
        torch.float32: ov.Type.f32,
        torch.float64: ov.Type.f64,
        torch.int32: ov.Type.i32,
        torch.int64: ov.Type.i64,
        torch.bool: ov.Type.boolean,
    }
    
    
    def prepare_input_info(input_dict):
        """
        Helper function for preparing input info (shapes and data types) for conversion based on example inputs
        """
        flatten_inputs = flattenize_inputs(input_dict.values())
        input_info = []
        for input_data in flatten_inputs:
            updated_shape = list(input_data.shape)
            if input_data.ndim == 5:
                updated_shape[1] = -1
            input_info.append((dtype_mapping[input_data.dtype], updated_shape))
        return input_info
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input, input_info):
        """
        Helper function for converting PyTorch model to OpenVINO IR
        """
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                ov_model = ov.convert_model(model, example_input=example_input, input=input_info)
            ov_model = nncf.compress_weights(ov_model)
            ov.save_model(ov_model, xml_path)
    
            del ov_model
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


ID Encoder
~~~~~~~~~~



PhotoMaker merged image encoder and fuse module to create an ID Encoder.
It will used to generate image embeddings to update text encoder’s
output(text embeddings) which will be the input for U-Net model.

.. code:: ipython3

    id_encoder = pipe.id_encoder
    id_encoder.eval()
    
    
    def create_bool_tensor(*size):
        new_tensor = torch.zeros((size), dtype=torch.bool)
        return new_tensor
    
    
    inputs = {
        "id_pixel_values": torch.randn((1, 1, 3, 224, 224)),
        "prompt_embeds": torch.randn((1, 77, 2048)),
        "class_tokens_mask": create_bool_tensor(1, 77),
    }
    
    input_info = prepare_input_info(inputs)
    
    convert(id_encoder, ID_ENCODER_OV_PATH, inputs, input_info)
    
    del id_encoder
    gc.collect();


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.


.. parsed-literal::

    WARNING:nncf:NNCF provides best results with torch==2.4.*, while current torch version is 2.2.2+cpu. If you encounter issues, consider switching to torch==2.4.*


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_utils.py:4664: FutureWarning: `_is_quantized_training_enabled` is going to be deprecated in transformers 4.39.0. Please use `model.hf_quantizer.is_trainable` instead
      warnings.warn(
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/notebooks/photo-maker/PhotoMaker/photomaker/model.py:84: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert class_tokens_mask.sum() == stacked_id_embeds.shape[0], f"{class_tokens_mask.sum()} != {stacked_id_embeds.shape[0]}"


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (151 / 151)            │ 100% (151 / 151)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Text Encoder
~~~~~~~~~~~~



The text-encoder is responsible for transforming the input prompt, for
example, “a photo of an astronaut riding a horse” into an embedding
space that can be understood by the U-Net. It is usually a simple
transformer-based encoder that maps a sequence of input tokens to a
sequence of latent text embeddings.

.. code:: ipython3

    import types
    
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    text_encoder_2 = pipe.text_encoder_2
    text_encoder_2.eval()
    
    text_encoder.config.output_hidden_states = True
    text_encoder.config.return_dict = False
    
    inputs = {"input_ids": torch.ones((1, 77), dtype=torch.long)}
    
    input_info = prepare_input_info(inputs)
    
    convert(text_encoder, TEXT_ENCODER_OV_PATH, inputs, input_info)
    
    text_encoder_2._orig_forward = text_encoder_2.forward
    
    
    def text_encoder_fwd_wrapper(self, input_ids):
        res = self._orig_forward(input_ids, return_dict=True, output_hidden_states=True)
        return tuple([v for v in res.values() if v is not None])
    
    
    text_encoder_2.forward = types.MethodType(text_encoder_fwd_wrapper, text_encoder_2)
    
    convert(text_encoder_2, TEXT_ENCODER_2_OV_PATH, inputs, input_info)
    
    del text_encoder
    del text_encoder_2
    gc.collect();


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:86: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (73 / 73)              │ 100% (73 / 73)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (194 / 194)            │ 100% (194 / 194)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









U-Net
~~~~~



The process of U-Net model conversion remains the same, like for
original Stable Diffusion XL model.

.. code:: ipython3

    unet = pipe.unet
    unet.eval()
    
    
    class UnetWrapper(torch.nn.Module):
        def __init__(self, unet):
            super().__init__()
            self.unet = unet
    
        def forward(
            self,
            sample=None,
            timestep=None,
            encoder_hidden_states=None,
            text_embeds=None,
            time_ids=None,
        ):
            return self.unet.forward(
                sample,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
            )
    
    
    inputs = {
        "sample": torch.rand([2, 4, 128, 128], dtype=torch.float32),
        "timestep": torch.from_numpy(np.array(1, dtype=float)),
        "encoder_hidden_states": torch.rand([2, 77, 2048], dtype=torch.float32),
        "text_embeds": torch.rand([2, 1280], dtype=torch.float32),
        "time_ids": torch.rand([2, 6], dtype=torch.float32),
    }
    
    input_info = prepare_input_info(inputs)
    
    w_unet = UnetWrapper(unet)
    convert(w_unet, UNET_OV_PATH, inputs, input_info)
    
    del w_unet, unet
    gc.collect();


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unets/unet_2d_condition.py:1103: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if dim % default_overall_up_factor != 0:
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:136: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/downsampling.py:145: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:146: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/jobs/ov-notebook/jobs/OVNotebookOps/builds/801/archive/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/upsampling.py:162: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (794 / 794)            │ 100% (794 / 794)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









VAE Decoder
~~~~~~~~~~~



The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

When running Text-to-Image pipeline, we will see that we only need the
VAE decoder.

.. code:: ipython3

    vae_decoder = pipe.vae
    vae_decoder.eval()
    
    
    class VAEDecoderWrapper(torch.nn.Module):
        def __init__(self, vae_decoder):
            super().__init__()
            self.vae = vae_decoder
    
        def forward(self, latents):
            return self.vae.decode(latents)
    
    
    w_vae_decoder = VAEDecoderWrapper(vae_decoder)
    inputs = torch.zeros((1, 4, 128, 128))
    
    convert(w_vae_decoder, VAE_DECODER_OV_PATH, inputs, input_info=[1, 4, 128, 128])
    
    del w_vae_decoder, vae_decoder
    gc.collect();


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (40 / 40)              │ 100% (40 / 40)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









Prepare Inference pipeline
--------------------------



In this example, we will reuse ``PhotoMakerStableDiffusionXLPipeline``
pipeline to generate the image with OpenVINO, so each model’s object in
this pipeline should be replaced with new OpenVINO model object.

Select inference device for Stable Diffusion pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



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



Compile models and create their Wrappers for inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To access original PhotoMaker workflow, we have to create a new wrapper
for each OpenVINO compiled model. For matching original pipeline, part
of OpenVINO model wrapper’s attributes should be reused from original
model objects and inference output must be converted from numpy to
``torch.tensor``.



.. code:: ipython3

    import openvino as ov
    
    core = ov.Core()
    
    compiled_id_encoder = core.compile_model(ID_ENCODER_OV_PATH, device.value)
    compiled_unet = core.compile_model(UNET_OV_PATH, device.value)
    compiled_text_encoder = core.compile_model(TEXT_ENCODER_OV_PATH, device.value)
    compiled_text_encoder_2 = core.compile_model(TEXT_ENCODER_2_OV_PATH, device.value)
    compiled_vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device.value)

.. code:: ipython3

    from collections import namedtuple
    
    
    class OVIDEncoderWrapper(PhotoMakerIDEncoder):
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, id_encoder, orig_id_encoder):
            super().__init__()
            self.id_encoder = id_encoder
            self.modules = orig_id_encoder.modules  # accessed in the original workflow
            self.config = orig_id_encoder.config  # accessed in the original workflow
    
        def __call__(
            self,
            *args,
        ):
            id_pixel_values, prompt_embeds, class_tokens_mask = args
            inputs = {
                "id_pixel_values": id_pixel_values,
                "prompt_embeds": prompt_embeds,
                "class_tokens_mask": class_tokens_mask,
            }
            output = self.id_encoder(inputs)[0]
            return torch.from_numpy(output)

.. code:: ipython3

    class OVTextEncoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, text_encoder, orig_text_encoder):
            self.text_encoder = text_encoder
            self.modules = orig_text_encoder.modules  # accessed in the original workflow
            self.config = orig_text_encoder.config  # accessed in the original workflow
    
        def __call__(self, input_ids, **kwargs):
            inputs = {"input_ids": input_ids}
            output = self.text_encoder(inputs)
    
            hidden_states = []
            hidden_states_len = len(output)
            for i in range(1, hidden_states_len):
                hidden_states.append(torch.from_numpy(output[i]))
            if hidden_states_len - 1 < 2:
                hidden_states.append(torch.from_numpy(output[i]))
            BaseModelOutputWithPooling = namedtuple("BaseModelOutputWithPooling", "last_hidden_state hidden_states")
            output = BaseModelOutputWithPooling(torch.from_numpy(output[0]), hidden_states)
            return output

.. code:: ipython3

    class OVUnetWrapper:
        def __init__(self, unet, unet_orig):
            self.unet = unet
            self.config = unet_orig.config  # accessed in the original workflow
            self.add_embedding = unet_orig.add_embedding  # accessed in the original workflow
    
        def __call__(self, *args, **kwargs):
            latent_model_input, t = args
            inputs = {
                "sample": latent_model_input,
                "timestep": t,
                "encoder_hidden_states": kwargs["encoder_hidden_states"],
                "text_embeds": kwargs["added_cond_kwargs"]["text_embeds"],
                "time_ids": kwargs["added_cond_kwargs"]["time_ids"],
            }
    
            output = self.unet(inputs)
    
            return [torch.from_numpy(output[0])]

.. code:: ipython3

    class OVVAEDecoderWrapper:
        dtype = torch.float32  # accessed in the original workflow
    
        def __init__(self, vae, vae_orig):
            self.vae = vae
            self.config = vae_orig.config  # accessed in the original workflow
    
        def decode(self, latents, return_dict=False):
            output = self.vae(latents)[0]
            output = torch.from_numpy(output)
    
            return [output]

Replace the PyTorch model objects in original pipeline with OpenVINO
models

.. code:: ipython3

    pipe.id_encoder = OVIDEncoderWrapper(compiled_id_encoder, pipe.id_encoder)
    pipe.unet = OVUnetWrapper(compiled_unet, pipe.unet)
    pipe.text_encoder = OVTextEncoderWrapper(compiled_text_encoder, pipe.text_encoder)
    pipe.text_encoder_2 = OVTextEncoderWrapper(compiled_text_encoder_2, pipe.text_encoder_2)
    pipe.vae = OVVAEDecoderWrapper(compiled_vae_decoder, pipe.vae)

Running Text-to-Image Generation with OpenVINO
----------------------------------------------



.. code:: ipython3

    from diffusers.utils import load_image
    
    prompt = "sci-fi, closeup portrait photo of a man img in Iron man suit, face"
    negative_prompt = "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, cartoons, sketch), open mouth"
    generator = torch.Generator("cpu").manual_seed(42)
    
    input_id_images = []
    original_image = load_image("./PhotoMaker/examples/newton_man/newton_0.jpg")
    input_id_images.append(original_image)
    
    ## Parameter setting
    num_steps = 20
    style_strength_ratio = 20
    start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
    if start_merge_step > 30:
        start_merge_step = 30
    
    images = pipe(
        prompt=prompt,
        input_id_images=input_id_images,
        negative_prompt=negative_prompt,
        num_images_per_prompt=1,
        num_inference_steps=num_steps,
        start_merge_step=start_merge_step,
        generator=generator,
    ).images



.. parsed-literal::

      0%|          | 0/20 [00:00<?, ?it/s]


.. code:: ipython3

    import matplotlib.pyplot as plt
    
    
    def visualize_results(orig_img: Image.Image, output_img: Image.Image):
        """
        Helper function for pose estimationresults visualization
    
        Parameters:
           orig_img (Image.Image): original image
           output_img (Image.Image): processed image with PhotoMaker
        Returns:
           fig (matplotlib.pyplot.Figure): matplotlib generated figure
        """
        orig_img = orig_img.resize(output_img.size)
        orig_title = "Original image"
        output_title = "Output image"
        im_w, im_h = orig_img.size
        is_horizontal = im_h < im_w
        fig, axs = plt.subplots(
            2 if is_horizontal else 1,
            1 if is_horizontal else 2,
            sharex="all",
            sharey="all",
        )
        fig.suptitle(f"Prompt: '{prompt}'", fontweight="bold")
        fig.patch.set_facecolor("white")
        list_axes = list(axs.flat)
        for a in list_axes:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.grid(False)
        list_axes[0].imshow(np.array(orig_img))
        list_axes[1].imshow(np.array(output_img))
        list_axes[0].set_title(orig_title, fontsize=15)
        list_axes[1].set_title(output_title, fontsize=15)
        fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
        fig.tight_layout()
        return fig
    
    
    fig = visualize_results(original_image, images[0])



.. image:: photo-maker-with-output_files/photo-maker-with-output_33_0.png


Interactive Demo
----------------



.. code:: ipython3

    def generate_from_text(text_promt, input_image, neg_prompt, seed, num_steps, style_strength_ratio):
        """
        Helper function for generating result image from prompt text
    
        Parameters:
           text_promt (String): positive prompt
           input_image (Image.Image): original image
           neg_prompt (String): negative prompt
           seed (Int):  seed for random generator state initialization
           num_steps (Int): number of sampling steps
           style_strength_ratio (Int):  the percentage of step when merging the ID embedding to text embedding
    
        Returns:
           result (Image.Image): generation result
        """
        start_merge_step = int(float(style_strength_ratio) / 100 * num_steps)
        if start_merge_step > 30:
            start_merge_step = 30
        result = pipe(
            text_promt,
            input_id_images=input_image,
            negative_prompt=neg_prompt,
            num_inference_steps=num_steps,
            num_images_per_prompt=1,
            start_merge_step=start_merge_step,
            generator=torch.Generator().manual_seed(seed),
            height=1024,
            width=1024,
        ).images[0]
    
        return result

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/photo-maker/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=generate_from_text)
    
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








.. code:: ipython3

    # please uncomment and run this cell for stopping gradio interface
    # demo.close()
