Text-to-Image Generation with Stable Diffusion v2 and OpenVINO™
===============================================================

Stable Diffusion v2 is the next generation of Stable Diffusion model a
Text-to-Image latent diffusion model created by the researchers and
engineers from `Stability AI <https://stability.ai/>`__ and
`LAION <https://laion.ai/>`__.

General diffusion models are machine learning systems that are trained
to denoise random gaussian noise step by step, to get to a sample of
interest, such as an image. Diffusion models have shown to achieve
state-of-the-art results for generating image data. But one downside of
diffusion models is that the reverse denoising process is slow. In
addition, these models consume a lot of memory because they operate in
pixel space, which becomes unreasonably expensive when generating
high-resolution images. Therefore, it is challenging to train these
models and also use them for inference. OpenVINO brings capabilities to
run model inference on Intel hardware and opens the door to the
fantastic world of diffusion models for everyone!

In previous notebooks, we already discussed how to run `Text-to-Image
generation and Image-to-Image generation using Stable Diffusion
v1 <stable-diffusion-text-to-image-with-output.html>`__
and `controlling its generation process using
ControlNet <controlnet-stable-diffusion-with-output.html>`__.
Now is turn of Stable Diffusion v2.

Stable Diffusion v2: What’s new?
--------------------------------

The new stable diffusion model offers a bunch of new features inspired
by the other models that have emerged since the introduction of the
first iteration. Some of the features that can be found in the new model
are:

-  The model comes with a new robust encoder, OpenCLIP, created by LAION
   and aided by Stability AI; this version v2 significantly enhances the
   produced photos over the V1 versions.
-  The model can now generate images in a 768x768 resolution, offering
   more information to be shown in the generated images.
-  The model finetuned with
   `v-objective <https://arxiv.org/abs/2202.00512>`__. The
   v-parameterization is particularly useful for numerical stability
   throughout the diffusion process to enable progressive distillation
   for models. For models that operate at higher resolution, it is also
   discovered that the v-parameterization avoids color shifting
   artifacts that are known to affect high resolution diffusion models,
   and in the video setting it avoids temporal color shifting that
   sometimes appears with epsilon-prediction used in Stable Diffusion
   v1.
-  The model also comes with a new diffusion model capable of running
   upscaling on the images generated. Upscaled images can be adjusted up
   to 4 times the original image. Provided as separated model, for more
   details please check
   `stable-diffusion-x4-upscaler <https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler>`__
-  The model comes with a new refined depth architecture capable of
   preserving context from prior generation layers in an image-to-image
   setting. This structure preservation helps generate images that
   preserving forms and shadow of objects, but with different content.
-  The model comes with an updated inpainting module built upon the
   previous model. This text-guided inpainting makes switching out parts
   in the image easier than before.

This notebook demonstrates how to convert and run Stable Diffusion v2
model using OpenVINO.

Notebook contains the following steps:

1. Create PyTorch models pipeline using Diffusers library.
2. Convert PyTorch models to OpenVINO IR format, using model conversion
   API.
3. Apply hybrid post-training quantization to UNet model with
   `NNCF <https://github.com/openvinotoolkit/nncf/>`__.
4. Run Stable Diffusion v2 Text-to-Image pipeline with OpenVINO.

**Note:** This is the full version of the Stable Diffusion text-to-image
implementation. If you would like to get started and run the notebook
quickly, check out `stable-diffusion-v2-text-to-image-demo
notebook <stable-diffusion-v2-with-output.html>`__.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Stable Diffusion v2 for Text-to-Image
   Generation <#stable-diffusion-v2-for-text-to-image-generation>`__

   -  `Stable Diffusion in Diffusers
      library <#stable-diffusion-in-diffusers-library>`__
   -  `Convert models to OpenVINO Intermediate representation (IR)
      format <#convert-models-to-openvino-intermediate-representation-ir-format>`__
   -  `Text Encoder <#text-encoder>`__
   -  `U-Net <#u-net>`__
   -  `VAE <#vae>`__
   -  `Prepare Inference Pipeline <#prepare-inference-pipeline>`__
   -  `Configure Inference Pipeline <#configure-inference-pipeline>`__

-  `Quantization <#quantization>`__

   -  `Prepare calibration dataset <#prepare-calibration-dataset>`__
   -  `Run Hybrid Model Quantization <#run-hybrid-model-quantization>`__
   -  `Compare inference time of the FP16 and INT8
      pipelines <#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-  `Run Text-to-Image generation <#run-text-to-image-generation>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

Prerequisites
-------------



install required packages

.. code:: ipython3

    %pip install -q "diffusers>=0.14.0" "openvino>=2023.1.0" "datasets>=2.14.6" "transformers>=4.25.1" "gradio>=4.19" "torch>=2.1" Pillow opencv-python --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "nncf>=2.9.0"

Stable Diffusion v2 for Text-to-Image Generation
------------------------------------------------



To start, let’s look on Text-to-Image process for Stable Diffusion v2.
We will use `Stable Diffusion
v2-1 <https://huggingface.co/stabilityai/stable-diffusion-2-1>`__ model
for these purposes. The main difference from Stable Diffusion v2 and
Stable Diffusion v2.1 is usage of more data, more training, and less
restrictive filtering of the dataset, that gives promising results for
selecting wide range of input text prompts. More details about model can
be found in `Stability AI blog
post <https://stability.ai/blog/stablediffusion2-1-release7-dec-2022>`__
and original model
`repository <https://github.com/Stability-AI/stablediffusion>`__.

Stable Diffusion in Diffusers library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To work with Stable Diffusion
v2, we will use Hugging Face
`Diffusers <https://github.com/huggingface/diffusers>`__ library. To
experiment with Stable Diffusion models, Diffusers exposes the
`StableDiffusionPipeline <https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation>`__
similar to the `other Diffusers
pipelines <https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
The code below demonstrates how to create ``StableDiffusionPipeline``
using ``stable-diffusion-2-1``:

.. code:: ipython3

    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base").to("cpu")
    
    # for reducing memory consumption get all components from pipeline independently
    text_encoder = pipe.text_encoder
    text_encoder.eval()
    unet = pipe.unet
    unet.eval()
    vae = pipe.vae
    vae.eval()
    
    conf = pipe.scheduler.config
    
    del pipe



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/6 [00:00<?, ?it/s]


Convert models to OpenVINO Intermediate representation (IR) format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Starting from 2023.0 release, OpenVINO supports PyTorch models directly
via Model Conversion API. ``ov.convert_model`` function accepts instance
of PyTorch model and example inputs for tracing and returns object of
``ov.Model`` class, ready to use or save on disk using ``ov.save_model``
function.

The pipeline consists of three important parts:

-  Text Encoder to create condition to generate an image from a text
   prompt.
-  U-Net for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

Let us convert each part:

Text Encoder
~~~~~~~~~~~~



The text-encoder is responsible for transforming the input prompt, for
example, “a photo of an astronaut riding a horse” into an embedding
space that can be understood by the U-Net. It is usually a simple
transformer-based encoder that maps a sequence of input tokens to a
sequence of latent text embeddings.

The input of the text encoder is tensor ``input_ids``, which contains
indexes of tokens from text processed by the tokenizer and padded to the
maximum length accepted by the model. Model outputs are two tensors:
``last_hidden_state`` - hidden state from the last MultiHeadAttention
layer in the model and ``pooler_out`` - pooled output for whole model
hidden states.

.. code:: ipython3

    from pathlib import Path
    
    sd2_1_model_dir = Path("sd2.1")
    sd2_1_model_dir.mkdir(exist_ok=True)

.. code:: ipython3

    import gc
    import torch
    import openvino as ov
    
    TEXT_ENCODER_OV_PATH = sd2_1_model_dir / "text_encoder.xml"
    
    
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    
    
    def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
        """
        Convert Text Encoder model to IR.
        Function accepts pipeline, prepares example inputs for conversion
        Parameters:
            text_encoder (torch.nn.Module): text encoder PyTorch model
            ir_path (Path): File for storing model
        Returns:
            None
        """
        if not ir_path.exists():
            input_ids = torch.ones((1, 77), dtype=torch.long)
            # switch model to inference mode
            text_encoder.eval()
    
            # disable gradients calculation for reducing memory consumption
            with torch.no_grad():
                # export model
                ov_model = ov.convert_model(
                    text_encoder,  # model instance
                    example_input=input_ids,  # example inputs for model tracing
                    input=([1, 77],),  # input shape for conversion
                )
                ov.save_model(ov_model, ir_path)
                del ov_model
                cleanup_torchscript_cache()
            print("Text Encoder successfully converted to IR")
    
    
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder(text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    
    del text_encoder
    gc.collect();


.. parsed-literal::

    Text encoder will be loaded from sd2.1/text_encoder.xml


U-Net
~~~~~



U-Net model gradually denoises latent image representation guided by
text encoder hidden state.

U-Net model has three inputs:

-  ``sample`` - latent image sample from previous step. Generation
   process has not been started yet, so you will use random noise.
-  ``timestep`` - current scheduler step.
-  ``encoder_hidden_state`` - hidden state of text encoder.

Model predicts the ``sample`` state for the next step.

Generally, U-Net model conversion process remain the same like in Stable
Diffusion v1, expect small changes in input sample size. Our model was
pretrained to generate images with resolution 768x768, initial latent
sample size for this case is 96x96. Besides that, for different use
cases like inpainting and depth to image generation model also can
accept additional image information: depth map or mask as channel-wise
concatenation with initial latent sample. For converting U-Net model for
such use cases required to modify number of input channels.

.. code:: ipython3

    import numpy as np
    
    UNET_OV_PATH = sd2_1_model_dir / "unet.xml"
    
    
    def convert_unet(
        unet: torch.nn.Module,
        ir_path: Path,
        num_channels: int = 4,
        width: int = 64,
        height: int = 64,
    ):
        """
        Convert Unet model to IR format.
        Function accepts pipeline, prepares example inputs for conversion
        Parameters:
            unet (torch.nn.Module): UNet PyTorch model
            ir_path (Path): File for storing model
            num_channels (int, optional, 4): number of input channels
            width (int, optional, 64): input width
            height (int, optional, 64): input height
        Returns:
            None
        """
        dtype_mapping = {torch.float32: ov.Type.f32, torch.float64: ov.Type.f64}
        if not ir_path.exists():
            # prepare inputs
            encoder_hidden_state = torch.ones((2, 77, 1024))
            latents_shape = (2, num_channels, width, height)
            latents = torch.randn(latents_shape)
            t = torch.from_numpy(np.array(1, dtype=np.float32))
            unet.eval()
            dummy_inputs = (latents, t, encoder_hidden_state)
            input_info = []
            for input_tensor in dummy_inputs:
                shape = ov.PartialShape(tuple(input_tensor.shape))
                element_type = dtype_mapping[input_tensor.dtype]
                input_info.append((shape, element_type))
    
            with torch.no_grad():
                ov_model = ov.convert_model(unet, example_input=dummy_inputs, input=input_info)
            ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
            print("U-Net successfully converted to IR")
    
    
    if not UNET_OV_PATH.exists():
        convert_unet(unet, UNET_OV_PATH, width=96, height=96)
        del unet
        gc.collect()
    else:
        del unet
    gc.collect();

VAE
~~~



The VAE model has two parts, an encoder and a decoder. The encoder is
used to convert the image into a low dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder. When you
run inference for Text-to-Image, there is no initial image as a starting
point. You can skip this step and directly generate initial random
noise.

When running Text-to-Image pipeline, we will see that we **only need the
VAE decoder**, but preserve VAE encoder conversion, it will be useful in
next chapter of our tutorial.

Note: This process will take a few minutes and use significant amount of
RAM (recommended at least 32GB).

.. code:: ipython3

    VAE_ENCODER_OV_PATH = sd2_1_model_dir / "vae_encoder.xml"
    
    
    def convert_vae_encoder(vae: torch.nn.Module, ir_path: Path, width: int = 512, height: int = 512):
        """
        Convert VAE model to IR format.
        VAE model, creates wrapper class for export only necessary for inference part,
        prepares example inputs for onversion
        Parameters:
            vae (torch.nn.Module): VAE PyTorch model
            ir_path (Path): File for storing model
            width (int, optional, 512): input width
            height (int, optional, 512): input height
        Returns:
            None
        """
    
        class VAEEncoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
    
            def forward(self, image):
                return self.vae.encode(x=image)["latent_dist"].sample()
    
        if not ir_path.exists():
            vae_encoder = VAEEncoderWrapper(vae)
            vae_encoder.eval()
            image = torch.zeros((1, 3, width, height))
            with torch.no_grad():
                ov_model = ov.convert_model(vae_encoder, example_input=image, input=([1, 3, width, height],))
            ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
            print("VAE encoder successfully converted to IR")
    
    
    def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path, width: int = 64, height: int = 64):
        """
        Convert VAE decoder model to IR format.
        Function accepts VAE model, creates wrapper class for export only necessary for inference part,
        prepares example inputs for conversion
        Parameters:
            vae (torch.nn.Module): VAE model
            ir_path (Path): File for storing model
            width (int, optional, 64): input width
            height (int, optional, 64): input height
        Returns:
            None
        """
    
        class VAEDecoderWrapper(torch.nn.Module):
            def __init__(self, vae):
                super().__init__()
                self.vae = vae
    
            def forward(self, latents):
                return self.vae.decode(latents)
    
        if not ir_path.exists():
            vae_decoder = VAEDecoderWrapper(vae)
            latents = torch.zeros((1, 4, width, height))
    
            vae_decoder.eval()
            with torch.no_grad():
                ov_model = ov.convert_model(vae_decoder, example_input=latents, input=([1, 4, width, height],))
            ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
            print("VAE decoder successfully converted to IR")
    
    
    if not VAE_ENCODER_OV_PATH.exists():
        convert_vae_encoder(vae, VAE_ENCODER_OV_PATH, 768, 768)
    else:
        print(f"VAE encoder will be loaded from {VAE_ENCODER_OV_PATH}")
    
    VAE_DECODER_OV_PATH = sd2_1_model_dir / "vae_decoder.xml"
    
    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(vae, VAE_DECODER_OV_PATH, 96, 96)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")
    
    del vae
    gc.collect();


.. parsed-literal::

    VAE encoder will be loaded from sd2.1/vae_encoder.xml
    VAE decoder will be loaded from sd2.1/vae_decoder.xml


Prepare Inference Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~



Putting it all together, let us now take a closer look at how the model
works in inference by illustrating the logical flow.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/22090501/ec454103-0d28-48e3-a18e-b55da3fab381
   :alt: text2img-stable-diffusion v2

   text2img-stable-diffusion v2

The stable diffusion model takes both a latent seed and a text prompt as
input. The latent seed is then used to generate random latent image
representations of size :math:`96 \times 96` where as the text prompt is
transformed to text embeddings of size :math:`77 \times 1024` via
OpenCLIP’s text encoder.

Next, the U-Net iteratively *denoises* the random latent image
representations while being conditioned on the text embeddings. The
output of the U-Net, being the noise residual, is used to compute a
denoised latent image representation via a scheduler algorithm. Many
different scheduler algorithms can be used for this computation, each
having its pros and cons. For Stable Diffusion, it is recommended to use
one of:

-  `PNDM
   scheduler <https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_pndm.py>`__
-  `DDIM
   scheduler <https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py>`__
-  `K-LMS
   scheduler <https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_lms_discrete.py>`__

Theory on how the scheduler algorithm function works is out of scope for
this notebook, but in short, you should remember that they compute the
predicted denoised image representation from the previous noise
representation and the predicted noise residual. For more information,
it is recommended to look into `Elucidating the Design Space of
Diffusion-Based Generative Models <https://arxiv.org/abs/2206.00364>`__.

The chart above looks very similar to Stable Diffusion V1 from
`notebook <stable-diffusion-text-to-image-with-output.html>`__,
but there is some small difference in details:

-  Changed input resolution for U-Net model.
-  Changed text encoder and as the result size of its hidden state
   embeddings.
-  Additionally, to improve image generation quality authors introduced
   negative prompting. Technically, positive prompt steers the diffusion
   toward the images associated with it, while negative prompt steers
   the diffusion away from it.In other words, negative prompt declares
   undesired concepts for generation image, e.g. if we want to have
   colorful and bright image, gray scale image will be result which we
   want to avoid, in this case gray scale can be treated as negative
   prompt. The positive and negative prompt are in equal footing. You
   can always use one with or without the other. More explanation of how
   it works can be found in this
   `article <https://stable-diffusion-art.com/how-negative-prompt-work/>`__.

.. code:: ipython3

    import inspect
    from typing import List, Optional, Union, Dict
    
    import PIL
    import cv2
    import torch
    
    from transformers import CLIPTokenizer
    from diffusers import DiffusionPipeline
    from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
    
    
    def scale_fit_to_window(dst_width: int, dst_height: int, image_width: int, image_height: int):
        """
        Preprocessing helper function for calculating image size for resize with peserving original aspect ratio
        and fitting image to specific window size
    
        Parameters:
          dst_width (int): destination window width
          dst_height (int): destination window height
          image_width (int): source image width
          image_height (int): source image height
        Returns:
          result_width (int): calculated width for resize
          result_height (int): calculated height for resize
        """
        im_scale = min(dst_height / image_height, dst_width / image_width)
        return int(im_scale * image_width), int(im_scale * image_height)
    
    
    def preprocess(image: PIL.Image.Image):
        """
        Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
        then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
        converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
        The function returns preprocessed input tensor and padding size, which can be used in postprocessing.
    
        Parameters:
          image (PIL.Image.Image): input image
        Returns:
           image (np.ndarray): preprocessed image tensor
           meta (Dict): dictionary with preprocessing metadata info
        """
        src_width, src_height = image.size
        dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
        image = np.array(image.resize((dst_width, dst_height), resample=PIL.Image.Resampling.LANCZOS))[None, :]
        pad_width = 512 - dst_width
        pad_height = 512 - dst_height
        pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        image = np.pad(image, pad, mode="constant")
        image = image.astype(np.float32) / 255.0
        image = 2.0 * image - 1.0
        image = image.transpose(0, 3, 1, 2)
        return image, {"padding": pad, "src_width": src_width, "src_height": src_height}
    
    
    class OVStableDiffusionPipeline(DiffusionPipeline):
        def __init__(
            self,
            vae_decoder: ov.Model,
            text_encoder: ov.Model,
            tokenizer: CLIPTokenizer,
            unet: ov.Model,
            scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
            vae_encoder: ov.Model = None,
        ):
            """
            Pipeline for text-to-image generation using Stable Diffusion.
            Parameters:
                vae_decoder (Model):
                    Variational Auto-Encoder (VAE) Model to decode images to and from latent representations.
                text_encoder (Model):
                    Frozen text-encoder. Stable Diffusion uses the text portion of
                    [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
                    the clip-vit-large-patch14(https://huggingface.co/openai/clip-vit-large-patch14) variant.
                tokenizer (CLIPTokenizer):
                    Tokenizer of class CLIPTokenizer(https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
                unet (Model): Conditional U-Net architecture to denoise the encoded image latents.
                vae_encoder (Model):
                    Variational Auto-Encoder (VAE) Model to encode images to latent representation.
                scheduler (SchedulerMixin):
                    A scheduler to be used in combination with unet to denoise the encoded image latents. Can be one of
                    DDIMScheduler, LMSDiscreteScheduler, or PNDMScheduler.
            """
            super().__init__()
            self.scheduler = scheduler
            self.vae_decoder = vae_decoder
            self.vae_encoder = vae_encoder
            self.text_encoder = text_encoder
            self.unet = unet
            self.register_to_config(unet=unet)
            self._text_encoder_output = text_encoder.output(0)
            self._unet_output = unet.output(0)
            self._vae_d_output = vae_decoder.output(0)
            self._vae_e_output = vae_encoder.output(0) if vae_encoder is not None else None
            self.height = self.unet.input(0).shape[2] * 8
            self.width = self.unet.input(0).shape[3] * 8
            self.tokenizer = tokenizer
    
        def __call__(
            self,
            prompt: Union[str, List[str]],
            image: PIL.Image.Image = None,
            negative_prompt: Union[str, List[str]] = None,
            num_inference_steps: Optional[int] = 50,
            guidance_scale: Optional[float] = 7.5,
            eta: Optional[float] = 0.0,
            output_type: Optional[str] = "pil",
            seed: Optional[int] = None,
            strength: float = 1.0,
        ):
            """
            Function invoked when calling the pipeline for generation.
            Parameters:
                prompt (str or List[str]):
                    The prompt or prompts to guide the image generation.
                image (PIL.Image.Image, *optional*, None):
                     Intinal image for generation.
                negative_prompt (str or List[str]):
                    The negative prompt or prompts to guide the image generation.
                num_inference_steps (int, *optional*, defaults to 50):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                guidance_scale (float, *optional*, defaults to 7.5):
                    Guidance scale as defined in Classifier-Free Diffusion Guidance(https://arxiv.org/abs/2207.12598).
                    guidance_scale is defined as `w` of equation 2.
                    Higher guidance scale encourages to generate images that are closely linked to the text prompt,
                    usually at the expense of lower image quality.
                eta (float, *optional*, defaults to 0.0):
                    Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                    [DDIMScheduler], will be ignored for others.
                output_type (`str`, *optional*, defaults to "pil"):
                    The output format of the generate image. Choose between
                    [PIL](https://pillow.readthedocs.io/en/stable/): PIL.Image.Image or np.array.
                seed (int, *optional*, None):
                    Seed for random generator state initialization.
                strength (int, *optional*, 1.0):
                    strength between initial image and generated in Image-to-Image pipeline, do not used in Text-to-Image
            Returns:
                Dictionary with keys:
                    sample - the last generated image PIL.Image.Image or np.array
            """
            if seed is not None:
                np.random.seed(seed)
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # get prompt text embeddings
            text_embeddings = self._encode_prompt(
                prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                negative_prompt=negative_prompt,
            )
            # set timesteps
            accepts_offset = "offset" in set(inspect.signature(self.scheduler.set_timesteps).parameters.keys())
            extra_set_kwargs = {}
            if accepts_offset:
                extra_set_kwargs["offset"] = 1
    
            self.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)
            latent_timestep = timesteps[:1]
    
            # get the initial random noise unless the user supplied it
            latents, meta = self.prepare_latents(image, latent_timestep)
    
            # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
            # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
            # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
            # and should be between [0, 1]
            accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
            extra_step_kwargs = {}
            if accepts_eta:
                extra_step_kwargs["eta"] = eta
    
            for t in self.progress_bar(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                # predict the noise residual
                noise_pred = self.unet([latent_model_input, np.array(t, dtype=np.float32), text_embeddings])[self._unet_output]
                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)["prev_sample"].numpy()
            # scale and decode the image latents with vae
            image = self.vae_decoder(latents * (1 / 0.18215))[self._vae_d_output]
    
            image = self.postprocess_image(image, meta, output_type)
            return {"sample": image}
    
        def _encode_prompt(
            self,
            prompt: Union[str, List[str]],
            num_images_per_prompt: int = 1,
            do_classifier_free_guidance: bool = True,
            negative_prompt: Union[str, List[str]] = None,
        ):
            """
            Encodes the prompt into text encoder hidden states.
    
            Parameters:
                prompt (str or list(str)): prompt to be encoded
                num_images_per_prompt (int): number of images that should be generated per prompt
                do_classifier_free_guidance (bool): whether to use classifier free guidance or not
                negative_prompt (str or list(str)): negative prompt to be encoded
            Returns:
                text_embeddings (np.ndarray): text encoder hidden states
            """
            batch_size = len(prompt) if isinstance(prompt, list) else 1
    
            # tokenize input prompts
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="np",
            )
            text_input_ids = text_inputs.input_ids
    
            text_embeddings = self.text_encoder(text_input_ids)[self._text_encoder_output]
    
            # duplicate text embeddings for each generation per prompt
            if num_images_per_prompt != 1:
                bs_embed, seq_len, _ = text_embeddings.shape
                text_embeddings = np.tile(text_embeddings, (1, num_images_per_prompt, 1))
                text_embeddings = np.reshape(text_embeddings, (bs_embed * num_images_per_prompt, seq_len, -1))
    
            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance:
                uncond_tokens: List[str]
                max_length = text_input_ids.shape[-1]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                else:
                    uncond_tokens = negative_prompt
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="np",
                )
    
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self._text_encoder_output]
    
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
                uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))
    
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
    
            return text_embeddings
    
        def prepare_latents(self, image: PIL.Image.Image = None, latent_timestep: torch.Tensor = None):
            """
            Function for getting initial latents for starting generation
    
            Parameters:
                image (PIL.Image.Image, *optional*, None):
                    Input image for generation, if not provided randon noise will be used as starting point
                latent_timestep (torch.Tensor, *optional*, None):
                    Predicted by scheduler initial step for image generation, required for latent image mixing with nosie
            Returns:
                latents (np.ndarray):
                    Image encoded in latent space
            """
            latents_shape = (1, 4, self.height // 8, self.width // 8)
            noise = np.random.randn(*latents_shape).astype(np.float32)
            if image is None:
                # if we use LMSDiscreteScheduler, let's make sure latents are mulitplied by sigmas
                if isinstance(self.scheduler, LMSDiscreteScheduler):
                    noise = noise * self.scheduler.sigmas[0].numpy()
                return noise, {}
            input_image, meta = preprocess(image)
            latents = self.vae_encoder(input_image)[self._vae_e_output]
            latents = latents * 0.18215
            latents = self.scheduler.add_noise(torch.from_numpy(latents), torch.from_numpy(noise), latent_timestep).numpy()
            return latents, meta
    
        def postprocess_image(self, image: np.ndarray, meta: Dict, output_type: str = "pil"):
            """
            Postprocessing for decoded image. Takes generated image decoded by VAE decoder, unpad it to initila image size (if required),
            normalize and convert to [0, 255] pixels range. Optionally, convertes it from np.ndarray to PIL.Image format
    
            Parameters:
                image (np.ndarray):
                    Generated image
                meta (Dict):
                    Metadata obtained on latents preparing step, can be empty
                output_type (str, *optional*, pil):
                    Output format for result, can be pil or numpy
            Returns:
                image (List of np.ndarray or PIL.Image.Image):
                    Postprocessed images
            """
            if "padding" in meta:
                pad = meta["padding"]
                (_, end_h), (_, end_w) = pad[1:3]
                h, w = image.shape[2:]
                unpad_h = h - end_h
                unpad_w = w - end_w
                image = image[:, :, :unpad_h, :unpad_w]
            image = np.clip(image / 2 + 0.5, 0, 1)
            image = np.transpose(image, (0, 2, 3, 1))
            # 9. Convert to PIL
            if output_type == "pil":
                image = self.numpy_to_pil(image)
                if "src_height" in meta:
                    orig_height, orig_width = meta["src_height"], meta["src_width"]
                    image = [img.resize((orig_width, orig_height), PIL.Image.Resampling.LANCZOS) for img in image]
            else:
                if "src_height" in meta:
                    orig_height, orig_width = meta["src_height"], meta["src_width"]
                    image = [cv2.resize(img, (orig_width, orig_width)) for img in image]
            return image
    
        def get_timesteps(self, num_inference_steps: int, strength: float):
            """
            Helper function for getting scheduler timesteps for generation
            In case of image-to-image generation, it updates number of steps according to strength
    
            Parameters:
               num_inference_steps (int):
                  number of inference steps for generation
               strength (float):
                   value between 0.0 and 1.0, that controls the amount of noise that is added to the input image.
                   Values that approach 1.0 allow for lots of variations but will also produce images that are not semantically consistent with the input.
            """
            # get the original timestep using init_timestep
            init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    
            t_start = max(num_inference_steps - init_timestep, 0)
            timesteps = self.scheduler.timesteps[t_start:]
    
            return timesteps, num_inference_steps - t_start

Configure Inference Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



First, you should create instances of OpenVINO Model.

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

    Dropdown(description='Device:', index=4, options=('CPU', 'GPU.0', 'GPU.1', 'GPU.2', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_config = {"INFERENCE_PRECISION_HINT": "f32"} if device.value != "CPU" else {}
    
    
    core = ov.Core()
    text_enc = core.compile_model(TEXT_ENCODER_OV_PATH, device.value)
    unet_model = core.compile_model(UNET_OV_PATH, device.value)
    vae_decoder = core.compile_model(VAE_DECODER_OV_PATH, device.value, ov_config)
    vae_encoder = core.compile_model(VAE_ENCODER_OV_PATH, device.value, ov_config)

Model tokenizer and scheduler are also important parts of the pipeline.
Let us define them and put all components together.

.. code:: ipython3

    from transformers import CLIPTokenizer
    
    scheduler = DDIMScheduler.from_config(conf)  # DDIMScheduler is used because UNet quantization produces better results with it
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    ov_pipe = OVStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=unet_model,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        scheduler=scheduler,
    )

Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``Stable Diffusion v2`` structure, the UNet model takes up
significant portion of the overall pipeline execution time. Now we will
show you how to optimize the UNet part using
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce
computation cost and speed up the pipeline. Quantizing the rest of the
pipeline does not significantly improve inference performance but can
lead to a substantial degradation of accuracy.

For this model we apply quantization in hybrid mode which means that we
quantize: (1) weights of MatMul and Embedding layers and (2) activations
of other layers. The steps are the following:

1. Create a calibration dataset for quantization.
2. Collect operations with weights.
3. Run ``nncf.compress_model()`` to compress only the model weights.
4. Run ``nncf.quantize()`` on the compressed model with weighted
   operations ignored by providing ``ignored_scope`` parameter.
5. Save the ``INT8`` model using ``openvino.save_model()`` function.

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



.. code:: ipython3

    # Fetch `skip_kernel_extension` module
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
    )
    open("skip_kernel_extension.py", "w").write(r.text)
    
    int8_ov_pipe = None
    
    %load_ext skip_kernel_extension

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`conceptual_captions <https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
dataset from Hugging Face as calibration data. To collect intermediate
model inputs for calibration we should customize ``CompiledModel``.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    import numpy as np
    from tqdm.notebook import tqdm
    from typing import Any, Dict, List
    
    
    def disable_progress_bar(pipeline, disable=True):
        if not hasattr(pipeline, "_progress_bar_config"):
            pipeline._progress_bar_config = {'disable': disable}
        else:
            pipeline._progress_bar_config['disable'] = disable
    
    
    class CompiledModelDecorator(ov.CompiledModel):
        def __init__(self, compiled_model: ov.CompiledModel, data_cache: List[Any] = None, keep_prob: float = 0.5):
            super().__init__(compiled_model)
            self.data_cache = data_cache if data_cache is not None else []
            self.keep_prob = keep_prob
    
        def __call__(self, *args, **kwargs):
            if np.random.rand() <= self.keep_prob:
                self.data_cache.append(*args)
            return super().__call__(*args, **kwargs)
    
    
    def collect_calibration_data(ov_pipe, calibration_dataset_size: int, num_inference_steps: int) -> List[Dict]:
        original_unet = ov_pipe.unet
        calibration_data = []
        ov_pipe.unet = CompiledModelDecorator(original_unet, calibration_data, keep_prob=0.7)
        disable_progress_bar(ov_pipe)
    
        dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", trust_remote_code=True).shuffle(seed=42)
    
        # Run inference for data collection
        pbar = tqdm(total=calibration_dataset_size)
        for batch in dataset:
            prompt = batch["caption"]
            if len(prompt) > ov_pipe.tokenizer.model_max_length:
                continue
            ov_pipe(prompt, num_inference_steps=num_inference_steps, seed=1)
            pbar.update(len(calibration_data) - pbar.n)
            if pbar.n >= calibration_dataset_size:
                break
    
        disable_progress_bar(ov_pipe, disable=False)
        ov_pipe.unet = original_unet
        return calibration_data

Run Hybrid Model Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    from collections import deque
    from transformers import set_seed
    import nncf
    
    def get_operation_const_op(operation, const_port_id: int):
        node = operation.input_value(const_port_id).get_node()
        queue = deque([node])
        constant_node = None
        allowed_propagation_types_list = ["Convert", "FakeQuantize", "Reshape"]
    
        while len(queue) != 0:
            curr_node = queue.popleft()
            if curr_node.get_type_name() == "Constant":
                constant_node = curr_node
                break
            if len(curr_node.inputs()) == 0:
                break
            if curr_node.get_type_name() in allowed_propagation_types_list:
                queue.append(curr_node.input_value(0).get_node())
    
        return constant_node
    
    
    def is_embedding(node) -> bool:
        allowed_types_list = ["f16", "f32", "f64"]
        const_port_id = 0
        input_tensor = node.input_value(const_port_id)
        if input_tensor.get_element_type().get_type_name() in allowed_types_list:
            const_node = get_operation_const_op(node, const_port_id)
            if const_node is not None:
                return True
    
        return False
    
    
    def collect_ops_with_weights(model):
        ops_with_weights = []
        for op in model.get_ops():
            if op.get_type_name() == "MatMul":
                constant_node_0 = get_operation_const_op(op, const_port_id=0)
                constant_node_1 = get_operation_const_op(op, const_port_id=1)
                if constant_node_0 or constant_node_1:
                    ops_with_weights.append(op.get_friendly_name())
            if op.get_type_name() == "Gather" and is_embedding(op):
                ops_with_weights.append(op.get_friendly_name())
    
        return ops_with_weights
    
    UNET_INT8_OV_PATH = sd2_1_model_dir / 'unet_optimized.xml'
    if not UNET_INT8_OV_PATH.exists():
        calibration_dataset_size = 300
        set_seed(1)
        unet_calibration_data = collect_calibration_data(ov_pipe,
                                                         calibration_dataset_size=calibration_dataset_size,
                                                         num_inference_steps=50)
    
        unet = core.read_model(UNET_OV_PATH)
        
        # Collect operations which weights will be compressed
        unet_ignored_scope = collect_ops_with_weights(unet)
        
        # Compress model weights
        compressed_unet = nncf.compress_weights(unet, ignored_scope=nncf.IgnoredScope(types=['Convolution']))
        
        # Quantize both weights and activations of Convolution layers
        quantized_unet = nncf.quantize(
            model=compressed_unet,
            calibration_dataset=nncf.Dataset(unet_calibration_data),
            subset_size=calibration_dataset_size,
            model_type=nncf.ModelType.TRANSFORMER,
            ignored_scope=nncf.IgnoredScope(names=unet_ignored_scope),
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=-1)
        )
        
        ov.save_model(quantized_unet, UNET_INT8_OV_PATH)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, onnx, openvino


.. code:: ipython3

    %%skip not $to_quantize.value
    
    int8_unet_model = core.compile_model(UNET_INT8_OV_PATH, device.value)
    int8_ov_pipe = OVStableDiffusionPipeline(
        tokenizer=tokenizer,
        text_encoder=text_enc,
        unet=int8_unet_model,
        vae_encoder=vae_encoder,
        vae_decoder=vae_decoder,
        scheduler=scheduler
    )

Compare UNet file size
~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_ir_model_size = UNET_OV_PATH.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 1024
    
    print(f"FP16 model size: {fp16_ir_model_size:.2f} KB")
    print(f"INT8 model size: {quantized_model_size:.2f} KB")
    print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 model size: 1691232.51 KB
    INT8 model size: 846918.58 KB
    Model compression rate: 1.997


Compare inference time of the FP16 and INT8 pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To measure the inference performance of the ``FP16`` and ``INT8``
pipelines, we use median inference time on calibration subset.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    
    def calculate_inference_time(pipeline, validation_data):
        inference_time = []
        pipeline.set_progress_bar_config(disable=True)
        for prompt in validation_data:
            start = time.perf_counter()
            _ = pipeline(prompt, num_inference_steps=10, seed=0)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        return np.median(inference_time)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    validation_size = 10
    validation_dataset = datasets.load_dataset("google-research-datasets/conceptual_captions", split="train", streaming=True, trust_remote_code=True).take(validation_size)
    validation_data = [batch["caption"] for batch in validation_dataset]
    
    fp_latency = calculate_inference_time(ov_pipe, validation_data)
    int8_latency = calculate_inference_time(int8_ov_pipe, validation_data)
    print(f"Performance speed-up: {fp_latency / int8_latency:.3f}")


.. parsed-literal::

    /home/nsavel/venvs/ov_notebooks_tmp/lib/python3.8/site-packages/datasets/load.py:1429: FutureWarning: The repository for conceptual_captions contains custom code which must be executed to correctly load the dataset. You can inspect the repository content at https://hf.co/datasets/conceptual_captions
    You can avoid this message in future by passing the argument `trust_remote_code=True`.
    Passing `trust_remote_code=True` will be mandatory to load this dataset from the next major release of `datasets`.
      warnings.warn(


.. parsed-literal::

    Performance speed-up: 1.232


Run Text-to-Image generation
----------------------------



Now, you can define a text prompts for image generation and run
inference pipeline. Optionally, you can also change the random generator
seed for latent state initialization and number of steps.

   **Note**: Consider increasing ``steps`` to get more precise results.
   A suggested value is ``50``, but it will take longer time to process.

Please select below whether you would like to use the quantized model to
launch the interactive demo.

.. code:: ipython3

    import ipywidgets as widgets
    
    quantized_model_present = int8_ov_pipe is not None
    
    use_quantized_model = widgets.Checkbox(
        value=quantized_model_present,
        description="Use quantized model",
        disabled=not quantized_model_present,
    )
    
    use_quantized_model




.. parsed-literal::

    Checkbox(value=True, description='Use quantized model')



.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v2/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    pipeline = int8_ov_pipe if use_quantized_model.value else ov_pipe
    
    demo = make_demo(pipeline)
    
    try:
        demo.queue().launch()
    except Exception:
        demo.queue().launch(share=True)
