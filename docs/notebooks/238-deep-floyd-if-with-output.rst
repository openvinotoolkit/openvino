Image generation with DeepFloyd IF and OpenVINO™
================================================

DeepFloyd IF is an advanced open-source text-to-image model that
delivers remarkable photorealism and language comprehension. DeepFloyd
IF consists of a frozen text encoder and three cascaded pixel diffusion
modules: a base model that creates 64x64 pixel images based on text
prompts and two super-resolution models, each designed to generate
images with increasing resolution: 256x256 pixel and 1024x1024 pixel.
All stages of the model employ a frozen text encoder, built on the T5
transformer, to derive text embeddings, which are then passed to a UNet
architecture enhanced with cross-attention and attention pooling.

Text encoder impact
~~~~~~~~~~~~~~~~~~~

-  **Profound text prompt comprehension.** The generation pipeline
   leverages the T5-XXL-1.1 Large Language Model (LLM) as a text
   encoder. Its intelligence is backed by a substantial number of
   text-image cross-attention layers, this ensures superior alignment
   between the prompt and the generated image.

-  **Realistic text in generated images.** Capitalizing on the
   capabilities of the T5 model, DeepFloyd IF produces readable text
   depictions alongside objects with distinct attributes, which have
   typically been a challenge for most existing text-to-image models.

DeepFloyd IF Distinctive Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, it is **Modular**. DeepFloyd IF pipeline is a consecutive
inference of several neural networks.

Which makes it **Cascaded**. The base model generates low-resolution
samples, then super-resolution models upsample the images to produce
high-resolution results. The models were individually trained at
different resolutions.

DeepFloyd IF employs **Diffusion** models. Diffusion models are machine
learning systems that are trained to denoise random Gaussian noise step
by step, to get to a sample of interest, such as an image. Diffusion
models have been shown to achieve state-of-the-art results for
generating image data.

And finally, DeepFloyd IF operates in **Pixel** space. Unlike latent
diffusion models (Stable Diffusion for instance), the diffusion is
implemented on a pixel level.

.. figure:: https://github.com/deep-floyd/IF/raw/develop/pics/deepfloyd_if_scheme.jpg
   :alt: deepfloyd_if_scheme

   deepfloyd_if_scheme

The graph above depicts the three-stage generation pipeline: A text
prompt is passed through the frozen T5-XXL LLM to convert it into a
vector in embedded space.

1. Stage 1: The first diffusion model in the cascade transforms the
   embedding vector into a 64x64 image. The DeepFloyd team has trained
   **three versions** of the base model, each with different parameters:
   IF-I 400M, IF-I 900M, and IF-I 4.3B. The smallest one is used by
   default, but users are free to change the checkpoint name to
   `“DeepFloyd/IF-I-L-v1.0” <https://huggingface.co/DeepFloyd/IF-I-L-v1.0>`__
   or
   `“DeepFloyd/IF-I-XL-v1.0” <https://huggingface.co/DeepFloyd/IF-I-XL-v1.0>`__

2. Stage 2: To upscale the image, two text-conditional super-resolution
   models (Efficient U-Net) are applied to the output of the first
   diffusion model. The first of these upscales the sample from 64x64
   pixel to 256x256 pixel resolution. Again, several versions of this
   model are available: IF-II 400M (default) and IF-II 1.2B (checkpoint
   name “DeepFloyd/IF-II-L-v1.0”).

3. Stage 3: Follows the same path as Stage 2 and upscales the image to
   1024x1024 pixel resolution. It is not released yet, so we will use a
   conventional Super Resolution network to get hi-res results. 
   



.. _top:

**Table of contents**:

- `Prerequisites <#prerequisites>`__

  - `Authentication <#authentication>`__

- `DeepFloyd IF in Diffusers library <#deepfloyd-if-in-diffusers-library>`__
- `Convert models to OpenVINO Intermediate representation (IR) format <#convert-models-to-openvino-intermediate-representation-ir-format>`__
- `Convert Text Encoder <#convert-text-encoder>`__
- `Convert the first Pixel Diffusion module’s UNet <#convert-the-first-pixel-diffusion-modules-unet>`__
- `Convert the second pixel diffusion module <#convert-the-second-pixel-diffusion-module>`__
- `Prepare Inference pipeline <#prepare-inference-pipeline>`__
- `Run Text-to-Image generation <#run-text-to-image-generation>`__

  - `Text Encoder inference <#text-encoder-inference>`__
  - `First Stage diffusion block inference <#first-stage-diffusion-block-inference>`__
  - `Second Stage diffusion block inference <#second-stage-diffusion-block-inference>`__
  - `Third Stage diffusion block <#third-stage-diffusion-block>`__
  - `Upscale the generated image using a Super Resolution network <#upscale-the-generated-image-using-a-super-resolution-network>`__

    - `Download the Super Resolution model weights <#download-the-super-resolution-model-weights>`__
    - `Reshape the model’s inputs <#reshape-the-models-inputs>`__
    - `Prepare the input images and run the model <#prepare-the-input-images-and-run-the-model>`__
    - `Display the result <#display-the-result>`__

.. note::

   - *This example requires the download of roughly 27 GB of model
     checkpoints, which could take some time depending on your internet
     connection speed. Additionally, the converted models will consume
     another 27 GB of disk space.*
   - *Please be aware that a minimum of 32 GB of RAM is necessary to
     convert and run inference on the models. There may be instances
     where the notebook appears to freeze or stop responding.*
   - *To access the model checkpoints, you’ll need a Hugging Face
     account. You’ll also be prompted to explicitly accept the*\ `model
     license <https://huggingface.co/DeepFloyd/IF-I-M-v1.0>`__\ *.*

Prerequisites `⇑ <#top>`__
###############################################################################################################################

Install required packages.

.. code:: ipython3

    # Set up requirements
    
    !pip install -q --upgrade pip
    !pip install -q "diffusers>=0.16.1" accelerate transformers safetensors sentencepiece huggingface_hub
    !pip install -q "openvino-dev>=2023.0.0"

.. code:: ipython3

    from collections import namedtuple
    import gc
    from pathlib import Path
    from typing import Union, Tuple
    
    import diffusers
    from diffusers import DiffusionPipeline
    from diffusers.utils import pt_to_pil
    from openvino.runtime import Core, PartialShape, serialize
    from openvino.tools import mo
    from openvino.tools.mo.convert import InputCutInfo
    import torch


.. parsed-literal::

    2023-05-29 11:26:42.788524: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-05-29 11:26:42.825669: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-05-29 11:26:43.383859: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


.. code:: ipython3

    checkpoint_variant = 'fp16'
    model_dtype = torch.float32
    ir_input_type = 'f32'
    compress_to_fp16 = False
    
    models_dir = Path('./models')
    models_dir.mkdir(exist_ok=True)
    
    encoder_ir_path = models_dir / 'encoder_ir.xml'
    first_stage_unet_ir_path = models_dir / 'unet_ir_I.xml'
    second_stage_unet_ir_path = models_dir / 'unet_ir_II.xml'

Authentication `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

In order to access IF checkpoints, users need to provide an authentication token.

If you already have a token, you can input it into the provided form in
the next cell. If not, please proceed according to the following
instructions:

1. Make sure to have a `Hugging Face <https://huggingface.co/>`__
   account and be logged in
2. Accept the license on the model card of
   `DeepFloyd/IF-I-M-v1.0 <https://huggingface.co/DeepFloyd/IF-I-M-v1.0>`__
3. To generate a token, proceed to `this
   page <https://huggingface.co/settings/tokens>`__

Uncheck the ``Add token as git credential?`` box.

.. code:: ipython3

    from huggingface_hub import login
    
    # Execute this cell to access the authentication form
    login()



.. parsed-literal::

    VBox(children=(HTML(value='<center> <img\nsrc=https://huggingface.co/front/assets/huggingface_logo-noborder.sv…


DeepFloyd IF in Diffusers library `⇑ <#top>`__
###############################################################################################################################

To work with IF by DeepFloyd Lab, we will use `Hugging Face Diffusers
package <https://github.com/huggingface/diffusers>`__. Diffusers package
exposes the ``DiffusionPipeline`` class, simplifying experiments with
diffusion models. The code below demonstrates how to create a
``DiffusionPipeline`` using IF configs:

.. code:: ipython3

    %%time
    
    # Downloading the model weights may take some time. The approximate total checkpoints size is 27GB.
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-M-v1.0",
        variant=checkpoint_variant,
        torch_dtype=model_dtype
    )
    
    stage_2 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-II-M-v1.0",
        text_encoder=None,
        variant=checkpoint_variant,
        torch_dtype=model_dtype
    )


.. parsed-literal::

    safety_checker/model.safetensors not found
    
    A mixture of fp16 and non-fp16 filenames will be loaded.
    Loaded fp16 filenames:
    [unet/diffusion_pytorch_model.fp16.bin, text_encoder/pytorch_model.fp16-00002-of-00002.bin, text_encoder/pytorch_model.fp16-00001-of-00002.bin]
    Loaded non-fp16 filenames:
    [watermarker/diffusion_pytorch_model.bin, safety_checker/pytorch_model.bin
    If this behavior is not expected, please check your folder structure.
    The config attributes {'lambda_min_clipped': -5.1} were passed to DDPMScheduler, but are not expected and will be ignored. Please verify your scheduler_config.json configuration file.



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/2 [00:00<?, ?it/s]


.. parsed-literal::

    
    A mixture of fp16 and non-fp16 filenames will be loaded.
    Loaded fp16 filenames:
    [text_encoder/model.fp16-00002-of-00002.safetensors, safety_checker/model.fp16.safetensors, unet/diffusion_pytorch_model.fp16.safetensors, text_encoder/model.fp16-00001-of-00002.safetensors]
    Loaded non-fp16 filenames:
    [watermarker/diffusion_pytorch_model.safetensors
    If this behavior is not expected, please check your folder structure.
    The config attributes {'lambda_min_clipped': -5.1} were passed to DDPMScheduler, but are not expected and will be ignored. Please verify your scheduler_config.json configuration file.


.. parsed-literal::

    CPU times: user 11.2 s, sys: 33.5 s, total: 44.7 s
    Wall time: 16.1 s


Convert models to OpenVINO Intermediate representation (IR) format. `⇑ <#top>`__
###############################################################################################################################

Model conversion API enables direct conversion of PyTorch
models. We will utilize the ``mo.convert_model`` method to acquire
OpenVINO IR versions of the models. This requires providing a model
object, input data for model tracing, and other relevant parameters. The
``use_legacy_frontend=True`` parameter instructs model conversion API to
employ the ONNX model format as an intermediate step, as opposed to
using the PyTorch JIT compiler, which is not optimal for our situation.

The pipeline consists of three important parts:

-  A Text Encoder that translates user prompts to vectors in the latent
   space that the Diffusion model can understand.
-  A Stage 1 U-Net for step-by-step denoising latent image
   representation.
-  A Stage 2 U-Net that takes low resolution output from the previous
   step and the latent representations to upscale the resulting image.

Let us convert each part.

1. Convert Text Encoder `⇑ <#top>`__
###############################################################################################################################


The text encoder is responsible for converting the input prompt, such as
“ultra close-up color photo portrait of rainbow owl with deer horns in
the woods” into an embedding space that can be fed to the next stage’s
U-Net. Typically, it is a transformer-based encoder that maps a sequence
of input tokens to a sequence of text embeddings.

The input for the text encoder consists of a tensor ``input_ids``, which
contains token indices from the text processed by the tokenizer and
padded to the maximum length accepted by the model.

*Note* the ``input`` argument passed to the ``convert_model`` method.
The ``convert_model`` can be called with the ``input shape`` argument
and/or the PyTorch-specific ``example_input`` argument. However, in this
case, the ``InputCutInfo`` class was utilized to describe the model
input and provide it as the ``input`` argument. Using the
``InputCutInfo`` class offers a framework-agnostic solution and enables
the definition of complex inputs. It allows specifying the input name,
shape, type, and value within a single argument, providing greater
flexibility.

To learn more, refer to this
`page <https://docs.openvino.ai/2023.1/openvino_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html>`__

.. code:: ipython3

    %%time
    
    if not encoder_ir_path.exists():
        encoder_ir = mo.convert_model(
            stage_1.text_encoder,
            input=[InputCutInfo(shape=PartialShape([1,77]), type='i64'),],
            compress_to_fp16=compress_to_fp16,
        )
        
        # Serialize the IR model to disk, we will load it at inference time
        serialize(encoder_ir, encoder_ir_path)
        del encoder_ir
        
    del stage_1.text_encoder
    gc.collect();


.. parsed-literal::

    CPU times: user 306 ms, sys: 1.05 s, total: 1.36 s
    Wall time: 1.37 s


Convert the first Pixel Diffusion module’s UNet `⇑ <#top>`__
###############################################################################################################################


U-Net model gradually denoises latent image representation guided by
text encoder hidden state.

U-Net model has three inputs:

``sample`` - latent image sample from previous step. Generation process
has not been started yet, so you will use random noise. ``timestep`` -
current scheduler step. ``encoder_hidden_state`` - hidden state of text
encoder. Model predicts the sample state for the next step.

The first Diffusion module in the cascade generates 64x64 pixel low
resolution images.

.. code:: ipython3

    %%time
    
    if not first_stage_unet_ir_path.exists():
        unet_1_ir = mo.convert_model(
            stage_1.unet,
            input=[InputCutInfo(shape=PartialShape([2, 3, 64, 64]), type=ir_input_type),
                   InputCutInfo(shape=PartialShape([]), type='i32'),
                   InputCutInfo(shape=PartialShape([2, 77, 4096]), type=ir_input_type)],
            compress_to_fp16=compress_to_fp16,
        )
    
        serialize(unet_1_ir, first_stage_unet_ir_path)
        
        del unet_1_ir
    
    stage_1_config = stage_1.unet.config
    del stage_1.unet
    gc.collect();


.. parsed-literal::

    CPU times: user 282 ms, sys: 16.7 ms, total: 298 ms
    Wall time: 298 ms


Convert the second pixel diffusion module `⇑ <#top>`__
###############################################################################################################################


The second Diffusion module in the cascade generates 256x256 pixel
images.

The second stage pipeline will use bilinear interpolation to upscale the
64x64 image that was generated in the previous stage to a higher 256x256
resolution. Then it will denoise the image taking into account the
encoded user prompt.

.. code:: ipython3

    %%time
    
    if not second_stage_unet_ir_path.exists():
        unet_2_ir = mo.convert_model(
            stage_2.unet,
            input=[InputCutInfo(shape=PartialShape([2, 6, 256, 256]), type=ir_input_type),
                   InputCutInfo(shape=PartialShape([]), type='i32'),
                   InputCutInfo(shape=PartialShape([2, 77, 4096]), type=ir_input_type),
                   InputCutInfo(shape=PartialShape([2]), type='i32'),],
            compress_to_fp16=compress_to_fp16,
        )
    
        serialize(unet_2_ir, second_stage_unet_ir_path)
        
        del unet_2_ir
        
    stage_2_config = stage_2.unet.config
    del stage_2.unet
    gc.collect();


.. parsed-literal::

    CPU times: user 240 ms, sys: 33 ms, total: 273 ms
    Wall time: 273 ms


Prepare Inference pipeline `⇑ <#top>`__
###############################################################################################################################


The original pipeline from the source repository will be reused in this
example. In order to achieve this, adapter classes were created to
enable OpenVINO models to replace Pytorch models and integrate
seamlessly into the pipeline.

.. code:: ipython3

    core = Core()

Select inference device
~~~~~~~~~~~~~~~~~~~~~~~

Select device from dropdown list for running inference using OpenVINO:

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

    class TextEncoder:
        """
        Text Encoder Adapter Class.
        
        This class is designed to seamlessly integrate the OpenVINO compiled model
        into the `stage_1.encode_prompt` routine.
        """
    
        def __init__(self, ir_path: Union[str, Path], dtype: torch.dtype, device: str = 'CPU') -> None:
            """
            Init the adapter with the IR model path.
            
            Parameters: 
                ir_path (str, Path): text encoder IR model path
                dtype (torch.dtype): result dtype
                device (str): inference device
            Returns:
                None
            """
            self.ir_path = ir_path 
            self.dtype = dtype
            self.encoder_openvino = core.compile_model(self.ir_path, device)
            
        def __call__(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor = None):
            """Adapt the network call."""
            result = self.encoder_openvino(input_ids)
            result_numpy = result[self.encoder_openvino.outputs[0]]
            return [torch.tensor(result_numpy, dtype=self.dtype)]

.. code:: ipython3

    # The pipelines for Stages 1 and 2 expect the UNet models to return an object containing a sample attribute.
    result_tuple = namedtuple('result', 'sample')
    
    
    class UnetFirstStage:
        """
        IF Stage-1 Unet Adapter Class.
        
        This class is designed to seamlessly integrate the OpenVINO compiled model into
        the `stage_1` diffusion pipeline.
        """
    
        def __init__(self, unet_ir_path: Union[str, Path],
                     config: diffusers.configuration_utils.FrozenDict,
                     dtype: torch.dtype,
                     device: str = 'CPU'
                     ) -> None:
            """
            Init the adapter with the IR model path and model config.
            
            Parameters: 
                unet_ir_path (str, Path): unet IR model path
                config (diffusers.configuration_utils.FrozenDict): original model config
                dtype (torch.dtype): result dtype
                device (str): inference device
            Returns:
                None
            """
            self.unet_openvino = core.compile_model(unet_ir_path, device)
            self.config = config
            self.dtype = dtype
            
        def __call__(self,
                     sample: torch.FloatTensor,
                     timestamp: int,
                     encoder_hidden_states: torch.Tensor,
                     class_labels: torch.Tensor = None,
                     cross_attention_kwargs: int = None
                    ) -> Tuple:
            """
            Adapt the network call.
            
            To learn more abould the model parameters please refer to
            its source code: https://github.com/huggingface/diffusers/blob/7200985eab7126801fffcf8251fd149c1cf1f291/src/diffusers/models/unet_2d_condition.py#L610
            """
            result = self.unet_openvino([sample, timestamp, encoder_hidden_states])
            result_numpy = result[self.unet_openvino.outputs[0]]
            return result_tuple(torch.tensor(result_numpy, dtype=self.dtype))
    
    
    class UnetSecondStage:
        """
        IF Stage-2 Unet Adapter Class.
        
        This class is designed to seamlessly integrate the OpenVINO compiled model into
        the `stage_2` diffusion pipeline.
        """
    
        def __init__(self, unet_ir_path: Union[str, Path],
                     config: diffusers.configuration_utils.FrozenDict,
                     dtype: torch.dtype,
                     device: str = 'CPU'
                     ) -> None:
            """
            Init the adapter with the IR model path and model config.
            
            Parameters: 
                unet_ir_path (str, Path): unet IR model path
                config (diffusers.configuration_utils.FrozenDict): original model config
                dtype (torch.dtype): result dtype
                device (str): inference device
            Returns:
                None
            """
            self.unet_openvino = core.compile_model(unet_ir_path, device)
            self.config = config
            self.dtype = dtype
            
        def __call__(self,
                     sample: torch.FloatTensor,
                     timestamp: int,
                     encoder_hidden_states: torch.Tensor,
                     class_labels: torch.Tensor = None,
                     cross_attention_kwargs: int = None
                    ) -> Tuple:
            """
            Adapt the network call.
            
            To learn more abould the model parameters please refer to
            its source code: https://github.com/huggingface/diffusers/blob/7200985eab7126801fffcf8251fd149c1cf1f291/src/diffusers/models/unet_2d_condition.py#L610
            """
            result = self.unet_openvino([sample, timestamp, encoder_hidden_states, class_labels])
            result_numpy = result[self.unet_openvino.outputs[0]]
            return result_tuple(torch.tensor(result_numpy, dtype=self.dtype))

Run Text-to-Image generation `⇑ <#top>`__
###############################################################################################################################


Now, we can set a text prompt for image generation and execute the
inference pipeline. Optionally, you can also modify the random generator
seed for latent state initialization and adjust the number of images to
be generated for the given prompt.

Text Encoder inference `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. code:: ipython3

    %%time
    
    prompt = 'ultra close color photo portrait of rainbow owl with deer horns in the woods'
    negative_prompt = 'blurred unreal uncentered occluded'
    
    # Initialize TextEncoder wrapper class
    stage_1.text_encoder = TextEncoder(encoder_ir_path, dtype=model_dtype, device=device.value)
    print('The model has been loaded')
    
    # Generate text embeddings
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt=negative_prompt)
    
    # Delete the encoder to free up memory
    del stage_1.text_encoder.encoder_openvino
    gc.collect()


.. parsed-literal::

    The model has been loaded


.. parsed-literal::

    /home/ea/work/notebooks_convert/notebooks_conv_env/lib/python3.8/site-packages/diffusers/configuration_utils.py:135: FutureWarning: Accessing config attribute `unet` directly via 'IFPipeline' object attribute is deprecated. Please access 'unet' over 'IFPipeline's config object instead, e.g. 'scheduler.config.unet'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)


.. parsed-literal::

    CPU times: user 52.8 s, sys: 38.2 s, total: 1min 31s
    Wall time: 30.2 s




.. parsed-literal::

    0



First Stage diffusion block inference `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. code:: ipython3

    %%time
    
    # Changing the following parameters will affect the model output
    # Note that increasing the number of diffusion steps will increase the inference time linearly.
    RANDOM_SEED = 42
    N_DIFFUSION_STEPS = 50
    
    # Initialize the First Stage UNet wrapper class
    stage_1.unet = UnetFirstStage(
        first_stage_unet_ir_path,
        stage_1_config,
        dtype=model_dtype,
        device=device.value
    )
    print('The model has been loaded')
    
    # Fix PRNG seed
    generator = torch.manual_seed(RANDOM_SEED)
    
    # Inference
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
                    generator=generator, output_type="pt", num_inference_steps=N_DIFFUSION_STEPS).images
    
    # Delete the model to free up memory
    del stage_1.unet.unet_openvino
    gc.collect()
    
    # Show the image
    pt_to_pil(image)[0]


.. parsed-literal::

    The model has been loaded



.. parsed-literal::

      0%|          | 0/50 [00:00<?, ?it/s]


.. parsed-literal::

    CPU times: user 4min 35s, sys: 5.63 s, total: 4min 41s
    Wall time: 20.6 s




.. image:: 238-deep-floyd-if-with-output_files/238-deep-floyd-if-with-output_29_3.png



Second Stage diffusion block inference `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


.. code:: ipython3

    %%time
    
    # Initialize the Second Stage UNet wrapper class
    stage_2.unet = UnetSecondStage(
        second_stage_unet_ir_path,
        stage_2_config,
        dtype=model_dtype,
        device=device.value
    )
    print('The model has been loaded')
    
    image = stage_2(
        image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds,
        generator=generator, output_type="pt", num_inference_steps=20).images
    
    # Delete the model to free up memory
    del stage_2.unet.unet_openvino
    gc.collect()
    
    # Show the image
    pil_image = pt_to_pil(image)[0]
    pil_image


.. parsed-literal::

    The model has been loaded



.. parsed-literal::

      0%|          | 0/20 [00:00<?, ?it/s]


.. parsed-literal::

    CPU times: user 13min 12s, sys: 10.6 s, total: 13min 22s
    Wall time: 55.7 s




.. image:: 238-deep-floyd-if-with-output_files/238-deep-floyd-if-with-output_31_3.png



Third Stage diffusion block `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The final block, which
upscales images to a higher resolution (1024x1024 px), has not been
released by DeepFloyd yet. Stay tuned!

Upscale the generated image using a Super Resolution network. `⇑ <#top>`__
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Though the third stage has not been officially released, we’ll employ
the Super Resolution network from `Example
#202 <https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/202-vision-superresolution/202-vision-superresolution-image.ipynb>`__
to enhance our low-resolution result!

Note, this step will be substituted with the Third IF stage upon its
release!

.. code:: ipython3

    # Temporary requirement
    !pip install -q matplotlib

Download the Super Resolution model weights `⇑ <#top>`__
-------------------------------------------------------------------------------------------------------------------------------


.. code:: ipython3

    import sys
    sys.path.append("../utils")
    
    import cv2
    import numpy as np
    from PIL import Image
    
    from notebook_utils import download_file
    
    # 1032: 4x superresolution, 1033: 3x superresolution
    model_name = 'single-image-super-resolution-1032'
    
    sr_model_xml_name = f'{model_name}.xml'
    sr_model_bin_name = f'{model_name}.bin'
    
    sr_model_xml_path = models_dir / sr_model_xml_name
    sr_model_bin_path = models_dir / sr_model_bin_name
    
    if not sr_model_xml_path.exists():
        base_url = f'https://storage.openvinotoolkit.org/repositories/open_model_zoo/2023.0/models_bin/1/{model_name}/FP16/'
        model_xml_url = base_url + sr_model_xml_name
        model_bin_url = base_url + sr_model_bin_name
    
        download_file(model_xml_url, sr_model_xml_name, models_dir)
        download_file(model_bin_url, sr_model_bin_name, models_dir)
    else:
        print(f'{model_name} already downloaded to {models_dir}')


.. parsed-literal::

    single-image-super-resolution-1032 already downloaded to models


Reshape the model’s inputs `⇑ <#top>`__
-------------------------------------------------------------------------------------------------------------------------------

We need to reshape the inputs for the model. This is necessary because the IR model was converted with
a different target input resolution. The Second IF stage returns 256x256
pixel images. Using the 4x Super Resolution model makes our target image
size 1024x1024 pixel.

.. code:: ipython3

    model = core.read_model(model=sr_model_xml_path)
    model.reshape({
        0: [1, 3, 256, 256],
        1: [1, 3, 1024, 1024]
    })
    compiled_model = core.compile_model(model=model, device_name=device.value)

Prepare the input images and run the model `⇑ <#top>`__
-------------------------------------------------------------------------------------------------------------------------------


.. code:: ipython3

    original_image = np.array(pil_image)
    bicubic_image = cv2.resize(
        src=original_image, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC
    )
    
    # Reshape the images from (H,W,C) to (N,C,H,W) as expected by the model.
    input_image_original = np.expand_dims(original_image.transpose(2, 0, 1), axis=0)
    input_image_bicubic = np.expand_dims(bicubic_image.transpose(2, 0, 1), axis=0)
    
    # Model Inference
    result = compiled_model(
        [input_image_original, input_image_bicubic]
    )[compiled_model.output(0)]

Display the result `⇑ <#top>`__
-------------------------------------------------------------------------------------------------------------------------------


.. code:: ipython3

    def convert_result_to_image(result) -> np.ndarray:
        """
        Convert network result of floating point numbers to image with integer
        values from 0-255. Values outside this range are clipped to 0 and 255.
    
        :param result: a single superresolution network result in N,C,H,W shape
        """
        result = 255 * result.squeeze(0).transpose(1, 2, 0)
        result[result < 0] = 0
        result[result > 255] = 255
        return Image.fromarray(result.astype(np.uint8), 'RGB')
    
    img = convert_result_to_image(result)
    img




.. image:: 238-deep-floyd-if-with-output_files/238-deep-floyd-if-with-output_41_0.png


