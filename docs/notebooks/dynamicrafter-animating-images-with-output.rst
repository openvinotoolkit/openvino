Animating Open-domain Images with DynamiCrafter and OpenVINO
============================================================

Animating a still image offers an engaging visual experience.
Traditional image animation techniques mainly focus on animating natural
scenes with stochastic dynamics (e.g. clouds and fluid) or
domain-specific motions (e.g. human hair or body motions), and thus
limits their applicability to more general visual content. To overcome
this limitation, `DynamiCrafter
team <https://doubiiu.github.io/projects/DynamiCrafter/>`__ explores the
synthesis of dynamic content for open-domain images, converting them
into animated videos. The key idea is to utilize the motion prior of
text-to-video diffusion models by incorporating the image into the
generative process as guidance. Given an image, DynamiCrafter team first
projects it into a text-aligned rich context representation space using
a query transformer, which facilitates the video model to digest the
image content in a compatible fashion. However, some visual details
still struggle to be preserved in the resultant videos. To supplement
with more precise image information, DynamiCrafter team further feeds
the full image to the diffusion model by concatenating it with the
initial noises. Experimental results show that the proposed method can
produce visually convincing and more logical & natural motions, as well
as higher conformity to the input image.

In this tutorial, we consider how to use DynamiCrafter with OpenVINO. An
additional part demonstrates how to run optimization with
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to speed up model.

.. raw:: html

   <table class="center">

.. raw:: html

   <tr>

.. raw:: html

   <td colspan="2">

“bear playing guitar happily, snowing”

.. raw:: html

   </td>

.. raw:: html

   <td colspan="2">

“boy walking on the street”

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   <tr>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   <td>

.. raw:: html

   </td>

.. raw:: html

   </tr>

.. raw:: html

   </table >


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load the original model <#load-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__

   -  `Convert CLIP text encoder <#convert-clip-text-encoder>`__
   -  `Convert CLIP image encoder <#convert-clip-image-encoder>`__
   -  `Convert AE encoder <#convert-ae-encoder>`__
   -  `Convert Diffusion U-Net model <#convert-diffusion-u-net-model>`__
   -  `Convert AE decoder <#convert-ae-decoder>`__

-  `Compiling models <#compiling-models>`__
-  `Building the pipeline <#building-the-pipeline>`__
-  `Run OpenVINO pipeline
   inference <#run-openvino-pipeline-inference>`__
-  `Quantization <#quantization>`__

   -  `Prepare calibration dataset <#prepare-calibration-dataset>`__
   -  `Run Quantization <#run-quantization>`__
   -  `Run Weights Compression <#run-weights-compression>`__
   -  `Compare model file sizes <#compare-model-file-sizes>`__
   -  `Compare inference time of the FP32 and INT8
      pipelines <#compare-inference-time-of-the-fp32-and-int8-pipelines>`__

-  `Interactive inference <#interactive-inference>`__ 
   


This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "openvino>=2024.2.0" "nncf>=2.11.0" "datasets>=2.20.0"
    %pip install -q "gradio>=4.19" omegaconf einops pytorch_lightning kornia "open_clip_torch==2.22.0" transformers av opencv-python "torch==2.2.2" --extra-index-url https://download.pytorch.org/whl/cpu

.. code:: ipython3

    from pathlib import Path
    import requests
    
    
    if not Path("cmd_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/cmd_helper.py",
        )
        open("cmd_helper.py", "w").write(r.text)
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

.. code:: ipython3

    from cmd_helper import clone_repo
    
    
    clone_repo("https://github.com/Doubiiu/DynamiCrafter.git", "26e665cd6c174234238d2ded661e2e56f875d360")

Load and run the original pipeline
----------------------------------



We will use model for 256x256 resolution as example. Also, models for
320x512 and 576x1024 are
`available <https://github.com/Doubiiu/DynamiCrafter?tab=readme-ov-file#-models>`__.

.. code:: ipython3

    import os
    from collections import OrderedDict
    
    import torch
    from huggingface_hub import hf_hub_download
    from omegaconf import OmegaConf
    
    from utils.utils import instantiate_from_config
    
    
    def load_model_checkpoint(model, ckpt):
        def load_checkpoint(model, ckpt, full_strict):
            state_dict = torch.load(ckpt, map_location="cpu")
            if "state_dict" in list(state_dict.keys()):
                state_dict = state_dict["state_dict"]
                try:
                    model.load_state_dict(state_dict, strict=full_strict)
                except Exception:
                    ## rename the keys for 256x256 model
                    new_pl_sd = OrderedDict()
                    for k, v in state_dict.items():
                        new_pl_sd[k] = v
    
                    for k in list(new_pl_sd.keys()):
                        if "framestride_embed" in k:
                            new_key = k.replace("framestride_embed", "fps_embedding")
                            new_pl_sd[new_key] = new_pl_sd[k]
                            del new_pl_sd[k]
                    model.load_state_dict(new_pl_sd, strict=full_strict)
            else:
                ## deepspeed
                new_pl_sd = OrderedDict()
                for key in state_dict["module"].keys():
                    new_pl_sd[key[16:]] = state_dict["module"][key]
                model.load_state_dict(new_pl_sd, strict=full_strict)
    
            return model
    
        load_checkpoint(model, ckpt, full_strict=True)
        print(">>> model checkpoint loaded.")
        return model
    
    
    def download_model():
        REPO_ID = "Doubiiu/DynamiCrafter"
        if not os.path.exists("./checkpoints/dynamicrafter_256_v1/"):
            os.makedirs("./checkpoints/dynamicrafter_256_v1/")
        local_file = os.path.join("./checkpoints/dynamicrafter_256_v1/model.ckpt")
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename="model.ckpt", local_dir="./checkpoints/dynamicrafter_256_v1/", local_dir_use_symlinks=False)
    
        ckpt_path = "checkpoints/dynamicrafter_256_v1/model.ckpt"
        config_file = "dynamicrafter/configs/inference_256_v1.0.yaml"
        config = OmegaConf.load(config_file)
        model_config = config.pop("model", OmegaConf.create())
        model_config["params"]["unet_config"]["params"]["use_checkpoint"] = False
        model = instantiate_from_config(model_config)
        model = load_model_checkpoint(model, ckpt_path)
        model.eval()
    
        return model
    
    
    model = download_model()


.. parsed-literal::

    AE working on z of shape (1, 4, 32, 32) = 4096 dimensions.
    >>> model checkpoint loaded.
    

Convert the model to OpenVINO IR
--------------------------------



Let’s define the conversion function for PyTorch modules. We use
``ov.convert_model`` function to obtain OpenVINO Intermediate
Representation object and ``ov.save_model`` function to save it as XML
file.

.. code:: ipython3

    import gc
    
    import openvino as ov
    
    
    def convert(model: torch.nn.Module, xml_path: str, example_input, input_shape=None):
        xml_path = Path(xml_path)
        if not xml_path.exists():
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                if not input_shape:
                    converted_model = ov.convert_model(model, example_input=example_input)
                else:
                    converted_model = ov.convert_model(model, example_input=example_input, input=input_shape)
            ov.save_model(converted_model, xml_path, compress_to_fp16=False)
    
            # cleanup memory
            torch._C._jit_clear_class_registry()
            torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
            torch.jit._state._clear_class_state()

Flowchart of DynamiCrafter proposed in `the
paper <https://arxiv.org/abs/2310.12190>`__:

|schema| Description: > During inference, our model can generate
animation clips from noise conditioned on the input still image.

Let’s convert models from the pipeline one by one.

.. |schema| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/76171391/d1033876-c664-4345-a254-0649edbf1906

Convert CLIP text encoder
~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    from lvdm.modules.encoders.condition import FrozenOpenCLIPEmbedder
    
    MODEL_DIR = Path("models")
    
    COND_STAGE_MODEL_OV_PATH = MODEL_DIR / "cond_stage_model.xml"
    
    
    class FrozenOpenCLIPEmbedderWrapper(FrozenOpenCLIPEmbedder):
        def forward(self, tokens):
            z = self.encode_with_transformer(tokens.to(self.device))
            return z
    
    
    cond_stage_model = FrozenOpenCLIPEmbedderWrapper(device="cpu")
    
    if not COND_STAGE_MODEL_OV_PATH.exists():
        convert(
            cond_stage_model,
            COND_STAGE_MODEL_OV_PATH,
            example_input=torch.ones([1, 77], dtype=torch.long),
        )
    
    del cond_stage_model
    gc.collect();

Convert CLIP image encoder
~~~~~~~~~~~~~~~~~~~~~~~~~~


``FrozenOpenCLIPImageEmbedderV2`` model accepts images of various
resolutions.

.. code:: ipython3

    EMBEDDER_OV_PATH = MODEL_DIR / "embedder_ir.xml"
    
    
    dummy_input = torch.rand([1, 3, 767, 767], dtype=torch.float32)
    
    model.embedder.model.visual.input_patchnorm = None  # fix error: visual model has not  attribute 'input_patchnorm'
    if not EMBEDDER_OV_PATH.exists():
        convert(model.embedder, EMBEDDER_OV_PATH, example_input=dummy_input, input_shape=[1, 3, -1, -1])
    
    
    del model.embedder
    gc.collect();

Convert AE encoder
~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    ENCODER_FIRST_STAGE_OV_PATH = MODEL_DIR / "encoder_first_stage_ir.xml"
    
    
    dummy_input = torch.rand([1, 3, 256, 256], dtype=torch.float32)
    
    if not ENCODER_FIRST_STAGE_OV_PATH.exists():
        convert(
            model.first_stage_model.encoder,
            ENCODER_FIRST_STAGE_OV_PATH,
            example_input=dummy_input,
        )
    
    del model.first_stage_model.encoder
    gc.collect();

Convert Diffusion U-Net model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    MODEL_OV_PATH = MODEL_DIR / "model_ir.xml"
    
    
    class ModelWrapper(torch.nn.Module):
        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model
    
        def forward(self, xc, t, context=None, fs=None, temporal_length=None):
            outputs = self.diffusion_model(xc, t, context=context, fs=fs, temporal_length=temporal_length)
            return outputs
    
    
    if not MODEL_OV_PATH.exists():
        convert(
            ModelWrapper(model.model.diffusion_model),
            MODEL_OV_PATH,
            example_input={
                "xc": torch.rand([1, 8, 16, 32, 32], dtype=torch.float32),
                "t": torch.tensor([1]),
                "context": torch.rand([1, 333, 1024], dtype=torch.float32),
                "fs": torch.tensor([3]),
                "temporal_length": torch.tensor([16]),
            },
        )
    
    out_channels = model.model.diffusion_model.out_channels
    del model.model.diffusion_model
    gc.collect();

Convert AE decoder
~~~~~~~~~~~~~~~~~~

``Decoder`` receives a
``bfloat16`` tensor. numpy doesn’t support this type. To avoid problems
with the conversion lets replace ``decode`` method to convert bfloat16
to float32.

.. code:: ipython3

    import types
    
    
    def decode(self, z, **kwargs):
        z = self.post_quant_conv(z)
        z = z.float()
        dec = self.decoder(z)
        return dec
    
    
    model.first_stage_model.decode = types.MethodType(decode, model.first_stage_model)

.. code:: ipython3

    DECODER_FIRST_STAGE_OV_PATH = MODEL_DIR / "decoder_first_stage_ir.xml"
    
    
    dummy_input = torch.rand([16, 4, 32, 32], dtype=torch.float32)
    
    if not DECODER_FIRST_STAGE_OV_PATH.exists():
        convert(
            model.first_stage_model.decoder,
            DECODER_FIRST_STAGE_OV_PATH,
            example_input=dummy_input,
        )
    
    del model.first_stage_model.decoder
    gc.collect();

Compiling models
----------------



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    from notebook_utils import device_widget
    
    core = ov.Core()
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    cond_stage_model = core.read_model(COND_STAGE_MODEL_OV_PATH)
    encoder_first_stage = core.read_model(ENCODER_FIRST_STAGE_OV_PATH)
    embedder = core.read_model(EMBEDDER_OV_PATH)
    model_ov = core.read_model(MODEL_OV_PATH)
    decoder_first_stage = core.read_model(DECODER_FIRST_STAGE_OV_PATH)
    
    compiled_cond_stage_model = core.compile_model(cond_stage_model, device.value)
    compiled_encode_first_stage = core.compile_model(encoder_first_stage, device.value)
    compiled_embedder = core.compile_model(embedder, device.value)
    compiled_model = core.compile_model(model_ov, device.value)
    compiled_decoder_first_stage = core.compile_model(decoder_first_stage, device.value)

Building the pipeline
---------------------



Let’s create callable wrapper classes for compiled models to allow
interaction with original pipelines. Note that all of wrapper classes
return ``torch.Tensor``\ s instead of ``np.array``\ s.

.. code:: ipython3

    from typing import Any
    import open_clip
    
    
    class CondStageModelWrapper(torch.nn.Module):
        def __init__(self, cond_stage_model):
            super().__init__()
            self.cond_stage_model = cond_stage_model
    
        def encode(self, tokens):
            if isinstance(tokens, list):
                tokens = open_clip.tokenize(tokens[0])
            outs = self.cond_stage_model(tokens)[0]
    
            return torch.from_numpy(outs)
    
    
    class EncoderFirstStageModelWrapper(torch.nn.Module):
        def __init__(self, encode_first_stage):
            super().__init__()
            self.encode_first_stage = encode_first_stage
    
        def forward(self, x):
            outs = self.encode_first_stage(x)[0]
    
            return torch.from_numpy(outs)
    
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)
    
    
    class EmbedderWrapper(torch.nn.Module):
        def __init__(self, embedder):
            super().__init__()
            self.embedder = embedder
    
        def forward(self, x):
            outs = self.embedder(x)[0]
    
            return torch.from_numpy(outs)
    
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)
    
    
    class CModelWrapper(torch.nn.Module):
        def __init__(self, diffusion_model, out_channels):
            super().__init__()
            self.diffusion_model = diffusion_model
            self.out_channels = out_channels
    
        def forward(self, xc, t, context, fs, temporal_length):
            inputs = {
                "xc": xc,
                "t": t,
                "context": context,
                "fs": fs,
            }
            outs = self.diffusion_model(inputs)[0]
    
            return torch.from_numpy(outs)
    
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)
    
    
    class DecoderFirstStageModelWrapper(torch.nn.Module):
        def __init__(self, decoder_first_stage):
            super().__init__()
            self.decoder_first_stage = decoder_first_stage
    
        def forward(self, x):
            x.float()
            outs = self.decoder_first_stage(x)[0]
    
            return torch.from_numpy(outs)
    
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self.forward(*args, **kwargs)

And insert wrappers instances in the pipeline:

.. code:: ipython3

    model.cond_stage_model = CondStageModelWrapper(compiled_cond_stage_model)
    model.first_stage_model.encoder = EncoderFirstStageModelWrapper(compiled_encode_first_stage)
    model.embedder = EmbedderWrapper(compiled_embedder)
    model.model.diffusion_model = CModelWrapper(compiled_model, out_channels)
    model.first_stage_model.decoder = DecoderFirstStageModelWrapper(compiled_decoder_first_stage)

Run OpenVINO pipeline inference
-------------------------------



.. code:: ipython3

    from einops import repeat, rearrange
    import torchvision.transforms as transforms
    
    
    transform = transforms.Compose(
        [
            transforms.Resize(min((256, 256))),
            transforms.CenterCrop((256, 256)),
        ]
    )
    
    
    def get_latent_z(model, videos):
        b, c, t, h, w = videos.shape
        x = rearrange(videos, "b c t h w -> (b t) c h w")
        z = model.encode_first_stage(x)
        z = rearrange(z, "(b t) c h w -> b c t h w", b=b, t=t)
        return z
    
    
    def process_input(model, prompt, image, transform=transform, fs=3):
        text_emb = model.get_learned_conditioning([prompt])
    
        # img cond
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
        img_tensor = (img_tensor / 255.0 - 0.5) * 2
    
        image_tensor_resized = transform(img_tensor)  # 3,h,w
        videos = image_tensor_resized.unsqueeze(0)  # bchw
    
        z = get_latent_z(model, videos.unsqueeze(2))  # bc,1,hw
        frames = model.temporal_length
        img_tensor_repeat = repeat(z, "b c t h w -> b c (repeat t) h w", repeat=frames)
    
        cond_images = model.embedder(img_tensor.unsqueeze(0))  # blc
        img_emb = model.image_proj_model(cond_images)
        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
    
        fs = torch.tensor([fs], dtype=torch.long, device=model.device)
        cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
        return cond

.. code:: ipython3

    import time
    from PIL import Image
    import numpy as np
    from lvdm.models.samplers.ddim import DDIMSampler
    from pytorch_lightning import seed_everything
    import torchvision
    
    
    def register_buffer(self, name, attr):
        if isinstance(attr, torch.Tensor):
            if attr.device != torch.device("cpu"):
                attr = attr.to(torch.device("cpu"))
        setattr(self, name, attr)
    
    
    def batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1.0, cfg_scale=1.0, temporal_cfg_scale=None, **kwargs):
        ddim_sampler = DDIMSampler(model)
        uncond_type = model.uncond_type
        batch_size = noise_shape[0]
        fs = cond["fs"]
        del cond["fs"]
        if noise_shape[-1] == 32:
            timestep_spacing = "uniform"
            guidance_rescale = 0.0
        else:
            timestep_spacing = "uniform_trailing"
            guidance_rescale = 0.7
        # construct unconditional guidance
        if cfg_scale != 1.0:
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = model.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
    
            # process image embedding token
            if hasattr(model, "embedder"):
                uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
                ## img: b c h w >> b l c
                uc_img = model.embedder(uc_img)
                uc_img = model.image_proj_model(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)
    
            if isinstance(cond, dict):
                uc = {key: cond[key] for key in cond.keys()}
                uc.update({"c_crossattn": [uc_emb]})
            else:
                uc = uc_emb
        else:
            uc = None
    
        x_T = None
        batch_variants = []
    
        for _ in range(n_samples):
            if ddim_sampler is not None:
                kwargs.update({"clean_cond": True})
                samples, _ = ddim_sampler.sample(
                    S=ddim_steps,
                    conditioning=cond,
                    batch_size=noise_shape[0],
                    shape=noise_shape[1:],
                    verbose=False,
                    unconditional_guidance_scale=cfg_scale,
                    unconditional_conditioning=uc,
                    eta=ddim_eta,
                    temporal_length=noise_shape[2],
                    conditional_guidance_scale_temporal=temporal_cfg_scale,
                    x_T=x_T,
                    fs=fs,
                    timestep_spacing=timestep_spacing,
                    guidance_rescale=guidance_rescale,
                    **kwargs,
                )
            # reconstruct from latent to pixel space
            batch_images = model.decode_first_stage(samples)
            batch_variants.append(batch_images)
        # batch, <samples>, c, t, h, w
        batch_variants = torch.stack(batch_variants, dim=1)
        return batch_variants
    
    
    # monkey patching to replace the original method 'register_buffer' that uses CUDA
    DDIMSampler.register_buffer = types.MethodType(register_buffer, DDIMSampler)
    
    
    def save_videos(batch_tensors, savedir, filenames, fps=10):
        # b,samples,c,t,h,w
        n_samples = batch_tensors.shape[1]
        for idx, vid_tensor in enumerate(batch_tensors):
            video = vid_tensor.detach().cpu()
            video = torch.clamp(video.float(), -1.0, 1.0)
            video = video.permute(2, 0, 1, 3, 4)  # t,n,c,h,w
            frame_grids = [torchvision.utils.make_grid(framesheet, nrow=int(n_samples)) for framesheet in video]  # [3, 1*h, n*w]
            grid = torch.stack(frame_grids, dim=0)  # stack in temporal dim [t, 3, n*h, w]
            grid = (grid + 1.0) / 2.0
            grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)
            savepath = os.path.join(savedir, f"{filenames[idx]}.mp4")
            torchvision.io.write_video(savepath, grid, fps=fps, video_codec="h264", options={"crf": "10"})
    
    
    def get_image(image, prompt, steps=5, cfg_scale=7.5, eta=1.0, fs=3, seed=123, model=model, result_dir="results"):
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
    
        seed_everything(seed)
    
        # torch.cuda.empty_cache()
        print("start:", prompt, time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
        start = time.time()
        if steps > 60:
            steps = 60
        model = model.cpu()
        batch_size = 1
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = 256 // 8, 256 // 8
        noise_shape = [batch_size, channels, frames, h, w]
    
        # text cond
        with torch.no_grad(), torch.cpu.amp.autocast():
            cond = process_input(model, prompt, image, transform, fs=3)
    
            ## inference
            batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
            ## b,samples,c,t,h,w
            prompt_str = prompt.replace("/", "_slash_") if "/" in prompt else prompt
            prompt_str = prompt_str.replace(" ", "_") if " " in prompt else prompt_str
            prompt_str = prompt_str[:40]
            if len(prompt_str) == 0:
                prompt_str = "empty_prompt"
    
        save_videos(batch_samples, result_dir, filenames=[prompt_str], fps=8)
        print(f"Saved in {prompt_str}.mp4. Time used: {(time.time() - start):.2f} seconds")
    
        return os.path.join(result_dir, f"{prompt_str}.mp4")

.. code:: ipython3

    image_path = "dynamicrafter/prompts/256/art.png"
    prompt = "man fishing in a boat at sunset"
    seed = 234
    image = Image.open(image_path)
    image = np.asarray(image)
    result_dir = "results"
    video_path = get_image(image, prompt, steps=20, seed=seed, model=model, result_dir=result_dir)


.. parsed-literal::

    Seed set to 234
    /tmp/ipykernel_971108/2451984876.py:25: UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at ../torch/csrc/utils/tensor_numpy.cpp:206.)
      img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
    

.. parsed-literal::

    start: man fishing in a boat at sunset 2024-08-06 13:54:24
    Saved in man_fishing_in_a_boat_at_sunset.mp4. Time used: 164.28 seconds
    

.. code:: ipython3

    from IPython.display import HTML
    
    HTML(
        f"""
        <video alt="video" controls>
            <source src="{video_path}" type="video/mp4">
        </video>
    """
    )




.. raw:: html

    
    <video alt="video" controls>
        <source src="results/man_fishing_in_a_boat_at_sunset.mp4" type="video/mp4">
    </video>
    



Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``DynamiCrafter`` structure, denoising UNet model is used
in the cycle repeating inference on each diffusion step, while other
parts of pipeline take part only once. Now we will show you how to
optimize pipeline using
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce memory and
computation cost.

Please select below whether you would like to run quantization to
improve model inference speed.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    from notebook_utils import quantization_widget
    
    to_quantize = quantization_widget()
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



Let’s load ``skip magic`` extension to skip quantization if
``to_quantize`` is not selected

.. code:: ipython3

    # Fetch `skip_kernel_extension` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    int8_model = None
    MODEL_INT8_OV_PATH = MODEL_DIR / "model_ir_int8.xml"
    COND_STAGE_MODEL_INT8_OV_PATH = MODEL_DIR / "cond_stage_model_int8.xml"
    DECODER_FIRST_STAGE_INT8_OV_PATH = MODEL_DIR / "decoder_first_stage_ir_int8.xml"
    ENCODER_FIRST_STAGE_INT8_OV_PATH = MODEL_DIR / "encoder_first_stage_ir_int8.xml"
    EMBEDDER_INT8_OV_PATH = MODEL_DIR / "embedder_ir_int8.xml"
    
    %load_ext skip_kernel_extension

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`jovianzm/Pexels-400k <https://huggingface.co/datasets/jovianzm/Pexels-400k>`__
dataset from Hugging Face as calibration data.

.. code:: ipython3

    from io import BytesIO
    
    
    def download_image(url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            # Convert the image to a NumPy array
            img_array = np.array(img)
            return img_array
        except Exception as err:
            print(f"Error occurred: {err}")
            return None

To collect intermediate model inputs for calibration we should customize
``CompiledModel``.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    from tqdm.notebook import tqdm
    import pickle
    
    
    class CompiledModelDecorator(ov.CompiledModel):
        def __init__(self, compiled_model, keep_prob, data_cache = None):
            super().__init__(compiled_model)
            self.data_cache = data_cache if data_cache else []
            self.keep_prob = np.clip(keep_prob, 0, 1)
    
        def __call__(self, *args, **kwargs):
            if np.random.rand() <= self.keep_prob:
                self.data_cache.append(*args)
            return super().__call__(*args, **kwargs)
    
    def collect_calibration_data(model, subset_size):
        calibration_dataset_filepath = Path("calibration_data")/f"{subset_size}.pkl"
        calibration_dataset_filepath.parent.mkdir(exist_ok=True, parents=True)
        if not calibration_dataset_filepath.exists():
            original_diffusion_model = model.model.diffusion_model.diffusion_model
            modified_model = CompiledModelDecorator(original_diffusion_model, keep_prob=1)
            model.model.diffusion_model = CModelWrapper(modified_model, model.model.diffusion_model.out_channels)
        
            dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", trust_remote_code=True, split="train", streaming=True).shuffle(seed=42).take(subset_size)
        
            pbar = tqdm(total=subset_size)
            channels = model.model.diffusion_model.out_channels
            frames = model.temporal_length
            h, w = 256 // 8, 256 // 8
            noise_shape = [1, channels, frames, h, w]
            for batch in dataset:
                prompt = batch["caption"]
                image_path = batch["image_url"]
                image = download_image(image_path)
                if image is None:
                    continue
        
                cond = process_input(model, prompt, image)
                batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=20, ddim_eta=1.0, cfg_scale=7.5)
        
                collected_subset_size = len(model.model.diffusion_model.diffusion_model.data_cache)
                if collected_subset_size >= subset_size:
                    pbar.update(subset_size - pbar.n)
                    break
                pbar.update(collected_subset_size - pbar.n)
        
            calibration_dataset = model.model.diffusion_model.diffusion_model.data_cache[:subset_size]
            model.model.diffusion_model.diffusion_model = original_diffusion_model
            with open(calibration_dataset_filepath, 'wb') as f:
                pickle.dump(calibration_dataset, f)
        with open(calibration_dataset_filepath, 'rb') as f:
            calibration_dataset = pickle.load(f)
        return calibration_dataset

.. code:: ipython3

    %%skip not $to_quantize.value
    
    
    if not MODEL_INT8_OV_PATH.exists():
        subset_size = 300
        calibration_data = collect_calibration_data(model, subset_size=subset_size)



.. parsed-literal::

      0%|          | 0/300 [00:00<?, ?it/s]


Run Quantization
~~~~~~~~~~~~~~~~



Quantization of the first and last ``Convolution`` layers impacts the
generation results. We recommend using ``IgnoredScope`` to keep accuracy
sensitive layers in FP16 precision. ``FastBiasCorrection`` algorithm is
disabled due to minimal accuracy improvement in SD models and increased
quantization time.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    
    
    if MODEL_INT8_OV_PATH.exists():
        print("Model already quantized")
    else:
        ov_model_ir = core.read_model(MODEL_OV_PATH)
        quantized_model = nncf.quantize(
            model=ov_model_ir,
            subset_size=subset_size,
            calibration_dataset=nncf.Dataset(calibration_data),
            model_type=nncf.ModelType.TRANSFORMER,
            ignored_scope=nncf.IgnoredScope(names=[
                "__module.diffusion_model.input_blocks.0.0/aten::_convolution/Convolution",
                "__module.diffusion_model.out.2/aten::_convolution/Convolution",
            ]),
            advanced_parameters=nncf.AdvancedQuantizationParameters(disable_bias_correction=True)
        )
        ov.save_model(quantized_model, MODEL_INT8_OV_PATH)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino
    


.. parsed-literal::

    Output()






    







    



.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:2 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:269 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 69 __module.diffusion_model.input_blocks.0.0/aten::_convolution/Convolution
    165 __module.diffusion_model.input_blocks.0.0/aten::_convolution/Add
    
    INFO:nncf:Not adding activation input quantizer for operation: 4107 __module.diffusion_model.out.2/aten::_convolution/Convolution
    4411 __module.diffusion_model.out.2/aten::_convolution/Add
    
    


.. parsed-literal::

    Output()






    







    


Run Weights Compression
~~~~~~~~~~~~~~~~~~~~~~~



Quantizing of the remaining components of the pipeline does not
significantly improve inference performance but can lead to a
substantial degradation of accuracy. The weight compression will be
applied to footprint reduction.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    def compress_model_weights(fp_model_path, int8_model_path):
        if not int8_model_path.exists():
            model = core.read_model(fp_model_path)
            compressed_model = nncf.compress_weights(model)
            ov.save_model(compressed_model, int8_model_path)
    
    
    compress_model_weights(COND_STAGE_MODEL_OV_PATH, COND_STAGE_MODEL_INT8_OV_PATH)
    compress_model_weights(DECODER_FIRST_STAGE_OV_PATH, DECODER_FIRST_STAGE_INT8_OV_PATH)
    compress_model_weights(ENCODER_FIRST_STAGE_OV_PATH, ENCODER_FIRST_STAGE_INT8_OV_PATH)
    compress_model_weights(EMBEDDER_OV_PATH, EMBEDDER_INT8_OV_PATH)


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (97 / 97)              │ 100% (97 / 97)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    


.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (39 / 39)              │ 100% (39 / 39)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    


.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (31 / 31)              │ 100% (31 / 31)                         │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    


.. parsed-literal::

    Output()






    







    


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │   Num bits (N) │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │              8 │ 100% (129 / 129)            │ 100% (129 / 129)                       │
    ┕━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙
    


.. parsed-literal::

    Output()






    







    


Let’s run the optimized pipeline

.. code:: ipython3

    %%skip not $to_quantize.value
    
    compiled_cond_stage_model = core.compile_model(core.read_model(COND_STAGE_MODEL_INT8_OV_PATH), device.value)
    compiled_encode_first_stage = core.compile_model(core.read_model(ENCODER_FIRST_STAGE_INT8_OV_PATH), device.value)
    compiled_embedder = core.compile_model(core.read_model(EMBEDDER_INT8_OV_PATH), device.value)
    compiled_model = core.compile_model(core.read_model(MODEL_INT8_OV_PATH), device.value)
    compiled_decoder_first_stage = core.compile_model(core.read_model(DECODER_FIRST_STAGE_INT8_OV_PATH), device.value)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    int8_model = download_model()
    int8_model.first_stage_model.decode = types.MethodType(decode, int8_model.first_stage_model)
    int8_model.embedder.model.visual.input_patchnorm = None  # fix error: visual model has not  attribute 'input_patchnorm'
    
    int8_model.cond_stage_model = CondStageModelWrapper(compiled_cond_stage_model)
    int8_model.first_stage_model.encoder = EncoderFirstStageModelWrapper(compiled_encode_first_stage)
    int8_model.embedder = EmbedderWrapper(compiled_embedder)
    int8_model.model.diffusion_model = CModelWrapper(compiled_model, out_channels)
    int8_model.first_stage_model.decoder = DecoderFirstStageModelWrapper(compiled_decoder_first_stage)


.. parsed-literal::

    AE working on z of shape (1, 4, 32, 32) = 4096 dimensions.
    >>> model checkpoint loaded.
    

.. code:: ipython3

    %%skip not $to_quantize.value
    
    image_path = "dynamicrafter/prompts/256/art.png"
    prompt = "man fishing in a boat at sunset"
    seed = 234
    image = Image.open(image_path)
    image = np.asarray(image)
    
    result_dir = "results_int8"
    video_path = get_image(image, prompt, steps=20, seed=seed, model=int8_model, result_dir=result_dir)


.. parsed-literal::

    Seed set to 234
    

.. parsed-literal::

    start: man fishing in a boat at sunset 2024-08-06 15:09:26
    Saved in man_fishing_in_a_boat_at_sunset.mp4. Time used: 81.47 seconds
    

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from IPython.display import display, HTML
    
    display(HTML(f"""
        <video alt="video" controls>
            <source src={video_path} type="video/mp4">
        </video>
    """))



.. raw:: html

    
    <video alt="video" controls>
        <source src=results_int8/man_fishing_in_a_boat_at_sunset.mp4 type="video/mp4">
    </video>
    


Compare model file sizes
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp32_model_paths = [COND_STAGE_MODEL_OV_PATH, DECODER_FIRST_STAGE_OV_PATH, ENCODER_FIRST_STAGE_OV_PATH, EMBEDDER_OV_PATH, MODEL_OV_PATH]
    int8_model_paths = [COND_STAGE_MODEL_INT8_OV_PATH, DECODER_FIRST_STAGE_INT8_OV_PATH, ENCODER_FIRST_STAGE_INT8_OV_PATH, EMBEDDER_INT8_OV_PATH, MODEL_INT8_OV_PATH]
    
    for fp16_path, int8_path in zip(fp32_model_paths, int8_model_paths):
        fp32_ir_model_size = fp16_path.with_suffix(".bin").stat().st_size
        int8_model_size = int8_path.with_suffix(".bin").stat().st_size
        print(f"{fp16_path.stem} compression rate: {fp32_ir_model_size / int8_model_size:.3f}")


.. parsed-literal::

    cond_stage_model compression rate: 3.977
    decoder_first_stage_ir compression rate: 3.987
    encoder_first_stage_ir compression rate: 3.986
    embedder_ir compression rate: 3.977
    model_ir compression rate: 3.981
    

Compare inference time of the FP32 and INT8 models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To measure the inference performance of the ``FP32`` and ``INT8``
models, we use median inference time on calibration subset.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    
    
    def calculate_inference_time(model, validation_size=3):
        calibration_dataset = datasets.load_dataset("jovianzm/Pexels-400k", split="train", streaming=True).take(validation_size)
        inference_time = []
        channels = model.model.diffusion_model.out_channels
        frames = model.temporal_length
        h, w = 256 // 8, 256 // 8
        noise_shape = [1, channels, frames, h, w]
        for batch in calibration_dataset:
            prompt = batch["title"]
            image_path = batch["thumbnail"]
            image = download_image(image_path)
            cond = process_input(model, prompt, image, transform, fs=3)
    
            start = time.perf_counter()
            _ = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=20, ddim_eta=1.0, cfg_scale=7.5)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp_latency = calculate_inference_time(model)
    print(f"FP32 latency: {fp_latency:.3f}")
    int8_latency = calculate_inference_time(int8_model)
    print(f"INT8 latency: {int8_latency:.3f}")
    print(f"Performance speed up: {fp_latency / int8_latency:.3f}")


.. parsed-literal::

    FP32 latency: 162.304
    INT8 latency: 79.590
    Performance speed up: 2.039
    

Interactive inference
---------------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    from ipywidgets import widgets
    
    quantized_models_present = int8_model is not None
    
    use_quantized_models = widgets.Checkbox(
        value=quantized_models_present,
        description="Use quantized models",
        disabled=not quantized_models_present,
    )
    
    use_quantized_models

.. code:: ipython3

    from functools import partial
    
    demo_model = int8_model if use_quantized_models.value else model
    get_image_fn = partial(get_image, model=demo_model)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/dynamicrafter-animating-images/gradio_helper.py"
        )
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    demo = make_demo(fn=get_image_fn)
    
    try:
        demo.queue().launch(debug=True)
    except Exception:
        demo.queue().launch(debug=True, share=True)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
