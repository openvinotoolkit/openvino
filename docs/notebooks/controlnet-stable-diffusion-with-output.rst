Text-to-Image Generation with ControlNet Conditioning
=====================================================

Diffusion models make a revolution in AI-generated art. This technology
enables creation of high-quality images simply by writing a text prompt.
Even though this technology gives very promising results, the diffusion
process, in the first order, is the process of generating images from
random noise and text conditions, which do not always clarify how
desired content should look, which forms it should have and where it is
located in relation to other objects on the image. Researchers have been
looking for ways to have more control over the results of the generation
process. ControlNet provides a minimal interface allowing users to
customize the generation process to a great extent.

ControlNet was introduced in `Adding Conditional Control to
Text-to-Image Diffusion Models <https://arxiv.org/abs/2302.05543>`__
paper. It provides a framework that enables support for various spatial
contexts such as a depth map, a segmentation map, a scribble, and key
points that can serve as additional conditionings to Diffusion models
such as Stable Diffusion.

This notebook explores ControlNet in depth, especially a new technique
for imparting high levels of control over the shape of synthesized
images. It demonstrates how to run it, using OpenVINO. An additional
part demonstrates how to run quantization with
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to speed up
pipeline. Let us get “controlling”!

Background
----------

Stable Diffusion
~~~~~~~~~~~~~~~~

`Stable Diffusion <https://github.com/CompVis/stable-diffusion>`__ is a
text-to-image latent diffusion model created by researchers and
engineers from CompVis, Stability AI, and LAION. Diffusion models as
mentioned above can generate high-quality images. Stable Diffusion is
based on a particular type of diffusion model called Latent Diffusion,
proposed in `High-Resolution Image Synthesis with Latent Diffusion
Models <https://arxiv.org/abs/2112.10752>`__ paper. Generally speaking,
diffusion models are machine learning systems that are trained to
denoise random Gaussian noise step by step, to get to a sample of
interest, such as an image. Diffusion models have been shown to achieve
state-of-the-art results for generating image data. But one downside of
diffusion models is that the reverse denoising process is slow because
of its repeated, sequential nature. In addition, these models consume a
lot of memory because they operate in pixel space, which becomes huge
when generating high-resolution images. Latent diffusion can reduce the
memory and compute complexity by applying the diffusion process over a
lower dimensional latent space, instead of using the actual pixel space.
This is the key difference between standard diffusion and latent
diffusion models: in latent diffusion, the model is trained to generate
latent (compressed) representations of the images.

There are three main components in latent diffusion:

-  A text-encoder, for example `CLIP’s Text
   Encoder <https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel>`__
   for creation condition to generate image from text prompt.
-  A U-Net for step-by-step denoising latent image representation.
-  An autoencoder (VAE) for encoding input image to latent space (if
   required) and decoding latent space to image back after generation.

For more details regarding Stable Diffusion work, refer to the `project
website <https://ommer-lab.com/research/latent-diffusion-models/>`__.
There is a tutorial for Stable Diffusion Text-to-Image generation with
OpenVINO, see the following
`notebook <stable-diffusion-text-to-image-with-output.html>`__.

ControlNet
~~~~~~~~~~

ControlNet is a neural network structure to control diffusion models by
adding extra conditions. Using this new framework, we can capture a
scene, structure, object, or subject pose from an inputted image, and
then transfer that quality to the generation process. In practice, this
enables the model to completely retain the original input shape, and
create a novel image that conserves the shape, pose, or outline while
using the novel features from the inputted prompt.

.. figure:: https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/he.png
   :alt: controlnet block

   controlnet block

Functionally, ControlNet operates by wrapping around an image synthesis
process to impart attention to the shape required to operate the model
using either its inbuilt prediction or one of many additional annotator
models. Referring to the diagram above, we can see, on a rudimentary
level, how ControlNet uses a trainable copy in conjunction with the
original network to modify the final output with respect to the shape of
the input control source.

By repeating the above simple structure 14 times, we can control stable
diffusion in the following way:

.. figure:: https://raw.githubusercontent.com/lllyasviel/ControlNet/main/github_page/sd.png
   :alt: sd + controlnet

   sd + controlnet

The input is simultaneously passed through the SD blocks, represented on
the left, while simultaneously being processed by the ControlNet blocks
on the right. This process is almost the same during encoding. When
denoising the image, at each step the SD decoder blocks will receive
control adjustments from the parallel processing path from ControlNet.

In the end, we are left with a very similar image synthesis pipeline
with an additional control added for the shape of the output features in
the final image.

Training ControlNet consists of the following steps:

1. Cloning the pre-trained parameters of a Diffusion model, such as
   Stable Diffusion’s latent UNet, (referred to as “trainable copy”)
   while also maintaining the pre-trained parameters separately (”locked
   copy”). It is done so that the locked parameter copy can preserve the
   vast knowledge learned from a large dataset, whereas the trainable
   copy is employed to learn task-specific aspects.
2. The trainable and locked copies of the parameters are connected via
   “zero convolution” layers (see here for more information) which are
   optimized as a part of the ControlNet framework. This is a training
   trick to preserve the semantics already learned by a frozen model as
   the new conditions are trained.

The process of extracting specific information from the input image is
called an annotation. ControlNet comes pre-packaged with compatibility
with several annotators-models that help it to identify the shape/form
of the target in the image:

-  Canny Edge Detection
-  M-LSD Lines
-  HED Boundary
-  Scribbles
-  Normal Map
-  Human Pose Estimation
-  Semantic Segmentation
-  Depth Estimation

This tutorial focuses mainly on conditioning by pose. However, the
discussed steps are also applicable to other annotation modes.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Instantiating Generation
   Pipeline <#instantiating-generation-pipeline>`__

   -  `ControlNet in Diffusers
      library <#controlnet-in-diffusers-library>`__
   -  `OpenPose <#openpose>`__

-  `Convert models to OpenVINO Intermediate representation (IR)
   format <#convert-models-to-openvino-intermediate-representation-ir-format>`__

   -  `OpenPose conversion <#openpose-conversion>`__

-  `Select inference device <#select-inference-device>`__

   -  `ControlNet conversion <#controlnet-conversion>`__
   -  `UNet conversion <#unet-conversion>`__
   -  `Text Encoder <#text-encoder>`__
   -  `VAE Decoder conversion <#vae-decoder-conversion>`__

-  `Prepare Inference pipeline <#prepare-inference-pipeline>`__
-  `Running Text-to-Image Generation with ControlNet Conditioning and
   OpenVINO <#running-text-to-image-generation-with-controlnet-conditioning-and-openvino>`__
-  `Select inference device for Stable Diffusion
   pipeline <#select-inference-device-for-stable-diffusion-pipeline>`__
-  `Quantization <#quantization>`__

   -  `Prepare calibration datasets <#prepare-calibration-datasets>`__
   -  `Run quantization <#run-quantization>`__
   -  `Compare model file sizes <#compare-model-file-sizes>`__
   -  `Compare inference time of the FP16 and INT8
      pipelines <#compare-inference-time-of-the-fp16-and-int8-pipelines>`__

-  `Interactive demo <#interactive-demo>`__

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

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu "torch>=2.1" "torchvision"
    %pip install -q "diffusers>=0.14.0" "matplotlib>=3.4" "transformers>=4.30.2" "controlnet-aux>=0.0.6" "gradio>=3.36" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -q "openvino>=2023.1.0" "datasets>=2.14.6" "nncf>=2.7.0"
    
    import requests
    
    r = requests.get(
        url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
    )
    open("notebook_utils.py", "w").write(r.text)

Instantiating Generation Pipeline
---------------------------------



ControlNet in Diffusers library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



For working with Stable Diffusion and ControlNet models, we will use
Hugging Face `Diffusers <https://github.com/huggingface/diffusers>`__
library. To experiment with ControlNet, Diffusers exposes the
`StableDiffusionControlNetPipeline <https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/controlnet>`__
similar to the `other Diffusers
pipelines <https://huggingface.co/docs/diffusers/api/pipelines/overview>`__.
Central to the ``StableDiffusionControlNetPipeline`` is the
``controlnet`` argument which enables providing a particularly trained
`ControlNetModel <https://huggingface.co/docs/diffusers/main/en/api/models#diffusers.ControlNetModel>`__
instance while keeping the pre-trained diffusion model weights the same.
The code below demonstrates how to create
``StableDiffusionControlNetPipeline``, using the ``controlnet-openpose``
controlnet model and ``stable-diffusion-v1-5``:

.. code:: ipython3

    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
    
    controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float32)
    pipe = StableDiffusionControlNetPipeline.from_pretrained("botp/stable-diffusion-v1-5", controlnet=controlnet)

OpenPose
~~~~~~~~



Annotation is an important part of working with ControlNet.
`OpenPose <https://github.com/CMU-Perceptual-Computing-Lab/openpose>`__
is a fast keypoint detection model that can extract human poses like
positions of hands, legs, and head. Below is the ControlNet workflow
using OpenPose. Keypoints are extracted from the input image using
OpenPose and saved as a control map containing the positions of
keypoints. It is then fed to Stable Diffusion as an extra conditioning
together with the text prompt. Images are generated based on these two
conditionings.

.. figure:: https://user-images.githubusercontent.com/29454499/224248986-eedf6492-dd7a-402b-b65d-36de952094ec.png
   :alt: controlnet-openpose-pipe

   controlnet-openpose-pipe

The code below demonstrates how to instantiate the OpenPose model.

.. code:: ipython3

    from controlnet_aux import OpenposeDetector
    
    pose_estimator = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

Now, let us check its result on example image:

.. code:: ipython3

    import requests
    from PIL import Image
    import matplotlib.pyplot as plt
    import numpy as np
    
    
    example_url = "https://user-images.githubusercontent.com/29454499/224540208-c172c92a-9714-4a7b-857a-b1e54b4d4791.jpg"
    img = Image.open(requests.get(example_url, stream=True).raw)
    pose = pose_estimator(img)
    
    
    def visualize_pose_results(
        orig_img: Image.Image,
        skeleton_img: Image.Image,
        left_title: str = "Original image",
        right_title: str = "Pose",
    ):
        """
        Helper function for pose estimationresults visualization
    
        Parameters:
           orig_img (Image.Image): original image
           skeleton_img (Image.Image): processed image with body keypoints
           left_title (str): title for the left image
           right_title (str): title for the right image
        Returns:
           fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
        """
        orig_img = orig_img.resize(skeleton_img.size)
        im_w, im_h = orig_img.size
        is_horizontal = im_h <= im_w
        figsize = (20, 10) if is_horizontal else (10, 20)
        fig, axs = plt.subplots(
            2 if is_horizontal else 1,
            1 if is_horizontal else 2,
            figsize=figsize,
            sharex="all",
            sharey="all",
        )
        fig.patch.set_facecolor("white")
        list_axes = list(axs.flat)
        for a in list_axes:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.grid(False)
        list_axes[0].imshow(np.array(orig_img))
        list_axes[1].imshow(np.array(skeleton_img))
        list_axes[0].set_title(left_title, fontsize=15)
        list_axes[1].set_title(right_title, fontsize=15)
        fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
        fig.tight_layout()
        return fig
    
    
    fig = visualize_pose_results(img, pose)



.. image:: controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_8_0.png


Convert models to OpenVINO Intermediate representation (IR) format
------------------------------------------------------------------



Starting from 2023.0 release, OpenVINO supports PyTorch models
conversion directly. We need to provide a model object, input data for
model tracing to ``ov.convert_model`` function to obtain OpenVINO
``ov.Model`` object instance. Model can be saved on disk for next
deployment using ``ov.save_model`` function.

The pipeline consists of five important parts:

-  OpenPose for obtaining annotation based on an estimated pose.
-  ControlNet for conditioning by image annotation.
-  Text Encoder for creation condition to generate an image from a text
   prompt.
-  Unet for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

Let us convert each part:

OpenPose conversion
~~~~~~~~~~~~~~~~~~~



OpenPose model is represented in the pipeline as a wrapper on the
PyTorch model which not only detects poses on an input image but is also
responsible for drawing pose maps. We need to convert only the pose
estimation part, which is located inside the wrapper
``pose_estimator.body_estimation.model``.

.. code:: ipython3

    from pathlib import Path
    import torch
    import openvino as ov
    
    OPENPOSE_OV_PATH = Path("openpose.xml")
    
    
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    
    
    if not OPENPOSE_OV_PATH.exists():
        with torch.no_grad():
            ov_model = ov.convert_model(
                pose_estimator.body_estimation.model,
                example_input=torch.zeros([1, 3, 184, 136]),
                input=[[1, 3, 184, 136]],
            )
            ov.save_model(ov_model, OPENPOSE_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("OpenPose successfully converted to IR")
    else:
        print(f"OpenPose will be loaded from {OPENPOSE_OV_PATH}")


.. parsed-literal::

    OpenPose will be loaded from openpose.xml


To reuse the original drawing procedure, we replace the PyTorch OpenPose
model with the OpenVINO model, using the following code:

.. code:: ipython3

    from collections import namedtuple
    
    
    class OpenPoseOVModel:
        """Helper wrapper for OpenPose model inference"""
    
        def __init__(self, core, model_path, device="AUTO"):
            self.core = core
            self.model = core.read_model(model_path)
            self._device = device
            self.compiled_model = core.compile_model(self.model, device)
    
        def __call__(self, input_tensor: torch.Tensor):
            """
            inference step
    
            Parameters:
              input_tensor (torch.Tensor): tensor with prerpcessed input image
            Returns:
               predicted keypoints heatmaps
            """
            h, w = input_tensor.shape[2:]
            input_shape = self.model.input(0).shape
            if h != input_shape[2] or w != input_shape[3]:
                self.reshape_model(h, w)
            results = self.compiled_model(input_tensor)
            return torch.from_numpy(results[self.compiled_model.output(0)]), torch.from_numpy(results[self.compiled_model.output(1)])
    
        def reshape_model(self, height: int, width: int):
            """
            helper method for reshaping model to fit input data
    
            Parameters:
              height (int): input tensor height
              width (int): input tensor width
            Returns:
              None
            """
            self.model.reshape({0: [1, 3, height, width]})
            self.compiled_model = self.core.compile_model(self.model, self._device)
    
        def parameters(self):
            Device = namedtuple("Device", ["device"])
            return [Device(torch.device("cpu"))]
    
    
    core = ov.Core()

Select inference device
-----------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    ov_openpose = OpenPoseOVModel(core, OPENPOSE_OV_PATH, device=device.value)
    pose_estimator.body_estimation.model = ov_openpose

.. code:: ipython3

    pose = pose_estimator(img)
    fig = visualize_pose_results(img, pose)



.. image:: controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_17_0.png


Great! As we can see, it works perfectly.

ControlNet conversion
~~~~~~~~~~~~~~~~~~~~~



The ControlNet model accepts the same inputs like UNet in Stable
Diffusion pipeline and additional condition sample - skeleton key points
map predicted by pose estimator:

-  ``sample`` - latent image sample from the previous step, generation
   process has not been started yet, so we will use random noise,
-  ``timestep`` - current scheduler step,
-  ``encoder_hidden_state`` - hidden state of text encoder,
-  ``controlnet_cond`` - condition input annotation.

The output of the model is attention hidden states from down and middle
blocks, which serves additional context for the UNet model.

.. code:: ipython3

    import gc
    from functools import partial
    
    inputs = {
        "sample": torch.randn((2, 4, 64, 64)),
        "timestep": torch.tensor(1),
        "encoder_hidden_states": torch.randn((2, 77, 768)),
        "controlnet_cond": torch.randn((2, 3, 512, 512)),
    }
    
    input_info = [(name, ov.PartialShape(inp.shape)) for name, inp in inputs.items()]
    
    CONTROLNET_OV_PATH = Path("controlnet-pose.xml")
    controlnet.eval()
    with torch.no_grad():
        down_block_res_samples, mid_block_res_sample = controlnet(**inputs, return_dict=False)
    
    if not CONTROLNET_OV_PATH.exists():
        with torch.no_grad():
            controlnet.forward = partial(controlnet.forward, return_dict=False)
            ov_model = ov.convert_model(controlnet, example_input=inputs, input=input_info)
            ov.save_model(ov_model, CONTROLNET_OV_PATH)
            del ov_model
            cleanup_torchscript_cache()
        print("ControlNet successfully converted to IR")
    else:
        print(f"ControlNet will be loaded from {CONTROLNET_OV_PATH}")
    
    del controlnet
    gc.collect()


.. parsed-literal::

    ControlNet will be loaded from controlnet-pose.xml




.. parsed-literal::

    4890



UNet conversion
~~~~~~~~~~~~~~~



The process of UNet model conversion remains the same, like for original
Stable Diffusion model, but with respect to the new inputs generated by
ControlNet.

.. code:: ipython3

    from typing import Tuple
    
    UNET_OV_PATH = Path("unet_controlnet.xml")
    
    dtype_mapping = {
        torch.float32: ov.Type.f32,
        torch.float64: ov.Type.f64,
        torch.int32: ov.Type.i32,
        torch.int64: ov.Type.i64,
    }
    
    
    class UnetWrapper(torch.nn.Module):
        def __init__(
            self,
            unet,
            sample_dtype=torch.float32,
            timestep_dtype=torch.int64,
            encoder_hidden_states=torch.float32,
            down_block_additional_residuals=torch.float32,
            mid_block_additional_residual=torch.float32,
        ):
            super().__init__()
            self.unet = unet
            self.sample_dtype = sample_dtype
            self.timestep_dtype = timestep_dtype
            self.encoder_hidden_states_dtype = encoder_hidden_states
            self.down_block_additional_residuals_dtype = down_block_additional_residuals
            self.mid_block_additional_residual_dtype = mid_block_additional_residual
    
        def forward(
            self,
            sample: torch.Tensor,
            timestep: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            down_block_additional_residuals: Tuple[torch.Tensor],
            mid_block_additional_residual: torch.Tensor,
        ):
            sample.to(self.sample_dtype)
            timestep.to(self.timestep_dtype)
            encoder_hidden_states.to(self.encoder_hidden_states_dtype)
            down_block_additional_residuals = [res.to(self.down_block_additional_residuals_dtype) for res in down_block_additional_residuals]
            mid_block_additional_residual.to(self.mid_block_additional_residual_dtype)
            return self.unet(
                sample,
                timestep,
                encoder_hidden_states,
                down_block_additional_residuals=down_block_additional_residuals,
                mid_block_additional_residual=mid_block_additional_residual,
            )
    
    
    def flattenize_inputs(inputs):
        flatten_inputs = []
        for input_data in inputs:
            if input_data is None:
                continue
            if isinstance(input_data, (list, tuple)):
                flatten_inputs.extend(flattenize_inputs(input_data))
            else:
                flatten_inputs.append(input_data)
        return flatten_inputs
    
    
    if not UNET_OV_PATH.exists():
        inputs.pop("controlnet_cond", None)
        inputs["down_block_additional_residuals"] = down_block_res_samples
        inputs["mid_block_additional_residual"] = mid_block_res_sample
    
        unet = UnetWrapper(pipe.unet)
        unet.eval()
    
        with torch.no_grad():
            ov_model = ov.convert_model(unet, example_input=inputs)
    
        flatten_inputs = flattenize_inputs(inputs.values())
        for input_data, input_tensor in zip(flatten_inputs, ov_model.inputs):
            input_tensor.get_node().set_partial_shape(ov.PartialShape(input_data.shape))
            input_tensor.get_node().set_element_type(dtype_mapping[input_data.dtype])
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, UNET_OV_PATH)
        del ov_model
        cleanup_torchscript_cache()
        del unet
        del pipe.unet
        gc.collect()
        print("Unet successfully converted to IR")
    else:
        del pipe.unet
        print(f"Unet will be loaded from {UNET_OV_PATH}")
    gc.collect()


.. parsed-literal::

    Unet will be loaded from unet_controlnet.xml




.. parsed-literal::

    0



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

    TEXT_ENCODER_OV_PATH = Path("text_encoder.xml")
    
    
    def convert_encoder(text_encoder: torch.nn.Module, ir_path: Path):
        """
        Convert Text Encoder model to OpenVINO IR.
        Function accepts text encoder model, prepares example inputs for conversion, and convert it to OpenVINO Model
        Parameters:
            text_encoder (torch.nn.Module): text_encoder model
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
                ov_model = ov.convert_model(
                    text_encoder,  # model instance
                    example_input=input_ids,  # inputs for model tracing
                    input=([1, 77],),
                )
                ov.save_model(ov_model, ir_path)
                del ov_model
            cleanup_torchscript_cache()
            print("Text Encoder successfully converted to IR")
    
    
    if not TEXT_ENCODER_OV_PATH.exists():
        convert_encoder(pipe.text_encoder, TEXT_ENCODER_OV_PATH)
    else:
        print(f"Text encoder will be loaded from {TEXT_ENCODER_OV_PATH}")
    del pipe.text_encoder
    gc.collect()


.. parsed-literal::

    Text encoder will be loaded from text_encoder.xml




.. parsed-literal::

    0



VAE Decoder conversion
~~~~~~~~~~~~~~~~~~~~~~



The VAE model has two parts, an encoder, and a decoder. The encoder is
used to convert the image into a low-dimensional latent representation,
which will serve as the input to the U-Net model. The decoder,
conversely, transforms the latent representation back into an image.

During latent diffusion training, the encoder is used to get the latent
representations (latents) of the images for the forward diffusion
process, which applies more and more noise at each step. During
inference, the denoised latents generated by the reverse diffusion
process are converted back into images using the VAE decoder. During
inference, we will see that we **only need the VAE decoder**. You can
find instructions on how to convert the encoder part in a stable
diffusion
`notebook <stable-diffusion-text-to-image-with-output.html>`__.

.. code:: ipython3

    VAE_DECODER_OV_PATH = Path("vae_decoder.xml")
    
    
    def convert_vae_decoder(vae: torch.nn.Module, ir_path: Path):
        """
        Convert VAE model to IR format.
        Function accepts pipeline, creates wrapper class for export only necessary for inference part,
        prepares example inputs for convert,
        Parameters:
            vae (torch.nn.Module): VAE model
            ir_path (Path): File for storing model
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
            latents = torch.zeros((1, 4, 64, 64))
    
            vae_decoder.eval()
            with torch.no_grad():
                ov_model = ov.convert_model(
                    vae_decoder,
                    example_input=latents,
                    input=[
                        (1, 4, 64, 64),
                    ],
                )
                ov.save_model(ov_model, ir_path)
            del ov_model
            cleanup_torchscript_cache()
            print("VAE decoder successfully converted to IR")
    
    
    if not VAE_DECODER_OV_PATH.exists():
        convert_vae_decoder(pipe.vae, VAE_DECODER_OV_PATH)
    else:
        print(f"VAE decoder will be loaded from {VAE_DECODER_OV_PATH}")


.. parsed-literal::

    VAE decoder will be loaded from vae_decoder.xml


Prepare Inference pipeline
--------------------------



Putting it all together, let us now take a closer look at how the model
works in inference by illustrating the logical flow. |detailed workflow|

The stable diffusion model takes both a latent seed and a text prompt as
input. The latent seed is then used to generate random latent image
representations of size :math:`64 \times 64` where as the text prompt is
transformed to text embeddings of size :math:`77 \times 768` via CLIP’s
text encoder.

Next, the U-Net iteratively *denoises* the random latent image
representations while being conditioned on the text embeddings. In
comparison with the original stable-diffusion pipeline, latent image
representation, encoder hidden states, and control condition annotation
passed via ControlNet on each denoising step for obtaining middle and
down blocks attention parameters, these attention blocks results
additionally will be provided to the UNet model for the control
generation process. The output of the U-Net, being the noise residual,
is used to compute a denoised latent image representation via a
scheduler algorithm. Many different scheduler algorithms can be used for
this computation, each having its pros and cons. For Stable Diffusion,
it is recommended to use one of:

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
Diffusion-Based Generative Models <https://arxiv.org/abs/2206.00364>`__

In this tutorial, instead of using Stable Diffusion’s default
`PNDMScheduler <https://huggingface.co/docs/diffusers/main/en/api/schedulers/pndm>`__,
we use one of the currently fastest diffusion model schedulers, called
`UniPCMultistepScheduler <https://huggingface.co/docs/diffusers/main/en/api/schedulers/unipc>`__.
Choosing an improved scheduler can drastically reduce inference time -
in this case, we can reduce the number of inference steps from 50 to 20
while more or less keeping the same image generation quality. More
information regarding schedulers can be found
`here <https://huggingface.co/docs/diffusers/main/en/using-diffusers/schedulers>`__.

The *denoising* process is repeated a given number of times (by default
50) to step-by-step retrieve better latent image representations. Once
complete, the latent image representation is decoded by the decoder part
of the variational auto-encoder.

Similarly to Diffusers ``StableDiffusionControlNetPipeline``, we define
our own ``OVContrlNetStableDiffusionPipeline`` inference pipeline based
on OpenVINO.

.. |detailed workflow| image:: https://user-images.githubusercontent.com/29454499/224261720-2d20ca42-f139-47b7-b8b9-0b9f30e1ae1e.png

.. code:: ipython3

    from diffusers import DiffusionPipeline
    from transformers import CLIPTokenizer
    from typing import Union, List, Optional, Tuple
    import cv2
    
    
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
    
    
    def preprocess(image: Image.Image):
        """
        Image preprocessing function. Takes image in PIL.Image format, resizes it to keep aspect ration and fits to model input window 512x512,
        then converts it to np.ndarray and adds padding with zeros on right or bottom side of image (depends from aspect ratio), after that
        converts data to float32 data type and change range of values from [0, 255] to [-1, 1], finally, converts data layout from planar NHWC to NCHW.
        The function returns preprocessed input tensor and padding size, which can be used in postprocessing.
    
        Parameters:
          image (Image.Image): input image
        Returns:
           image (np.ndarray): preprocessed image tensor
           pad (Tuple[int]): pading size for each dimension for restoring image size in postprocessing
        """
        src_width, src_height = image.size
        dst_width, dst_height = scale_fit_to_window(512, 512, src_width, src_height)
        image = np.array(image.resize((dst_width, dst_height), resample=Image.Resampling.LANCZOS))[None, :]
        pad_width = 512 - dst_width
        pad_height = 512 - dst_height
        pad = ((0, 0), (0, pad_height), (0, pad_width), (0, 0))
        image = np.pad(image, pad, mode="constant")
        image = image.astype(np.float32) / 255.0
        image = image.transpose(0, 3, 1, 2)
        return image, pad
    
    
    def randn_tensor(
        shape: Union[Tuple, List],
        dtype: Optional[np.dtype] = np.float32,
    ):
        """
        Helper function for generation random values tensor with given shape and data type
    
        Parameters:
          shape (Union[Tuple, List]): shape for filling random values
          dtype (np.dtype, *optiona*, np.float32): data type for result
        Returns:
          latents (np.ndarray): tensor with random values with given data type and shape (usually represents noise in latent space)
        """
        latents = np.random.randn(*shape).astype(dtype)
    
        return latents
    
    
    class OVContrlNetStableDiffusionPipeline(DiffusionPipeline):
        """
        OpenVINO inference pipeline for Stable Diffusion with ControlNet guidence
        """
    
        def __init__(
            self,
            tokenizer: CLIPTokenizer,
            scheduler,
            core: ov.Core,
            controlnet: ov.Model,
            text_encoder: ov.Model,
            unet: ov.Model,
            vae_decoder: ov.Model,
            device: str = "AUTO",
        ):
            super().__init__()
            self.tokenizer = tokenizer
            self.vae_scale_factor = 8
            self.scheduler = scheduler
            self.load_models(core, device, controlnet, text_encoder, unet, vae_decoder)
            self.set_progress_bar_config(disable=True)
    
        def load_models(
            self,
            core: ov.Core,
            device: str,
            controlnet: ov.Model,
            text_encoder: ov.Model,
            unet: ov.Model,
            vae_decoder: ov.Model,
        ):
            """
            Function for loading models on device using OpenVINO
    
            Parameters:
              core (Core): OpenVINO runtime Core class instance
              device (str): inference device
              controlnet (Model): OpenVINO Model object represents ControlNet
              text_encoder (Model): OpenVINO Model object represents text encoder
              unet (Model): OpenVINO Model object represents UNet
              vae_decoder (Model): OpenVINO Model object represents vae decoder
            Returns
              None
            """
            self.text_encoder = core.compile_model(text_encoder, device)
            self.text_encoder_out = self.text_encoder.output(0)
            self.register_to_config(controlnet=core.compile_model(controlnet, device))
            self.register_to_config(unet=core.compile_model(unet, device))
            self.unet_out = self.unet.output(0)
            self.vae_decoder = core.compile_model(vae_decoder, device)
            self.vae_decoder_out = self.vae_decoder.output(0)
    
        def __call__(
            self,
            prompt: Union[str, List[str]],
            image: Image.Image,
            num_inference_steps: int = 10,
            negative_prompt: Union[str, List[str]] = None,
            guidance_scale: float = 7.5,
            controlnet_conditioning_scale: float = 1.0,
            eta: float = 0.0,
            latents: Optional[np.array] = None,
            output_type: Optional[str] = "pil",
        ):
            """
            Function invoked when calling the pipeline for generation.
    
            Parameters:
                prompt (`str` or `List[str]`):
                    The prompt or prompts to guide the image generation.
                image (`Image.Image`):
                    `Image`, or tensor representing an image batch which will be repainted according to `prompt`.
                num_inference_steps (`int`, *optional*, defaults to 100):
                    The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                    expense of slower inference.
                negative_prompt (`str` or `List[str]`):
                    negative prompt or prompts for generation
                guidance_scale (`float`, *optional*, defaults to 7.5):
                    Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                    `guidance_scale` is defined as `w` of equation 2. of [Imagen
                    Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                    1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                    usually at the expense of lower image quality. This pipeline requires a value of at least `1`.
                latents (`np.ndarray`, *optional*):
                    Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                    generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                    tensor will ge generated by sampling using the supplied random `generator`.
                output_type (`str`, *optional*, defaults to `"pil"`):
                    The output format of the generate image. Choose between
                    [PIL](https://pillow.readthedocs.io/en/stable/): `Image.Image` or `np.array`.
            Returns:
                image ([List[Union[np.ndarray, Image.Image]]): generaited images
    
            """
    
            # 1. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0
            # 2. Encode input prompt
            text_embeddings = self._encode_prompt(prompt, negative_prompt=negative_prompt)
    
            # 3. Preprocess image
            orig_width, orig_height = image.size
            image, pad = preprocess(image)
            height, width = image.shape[-2:]
            if do_classifier_free_guidance:
                image = np.concatenate(([image] * 2))
    
            # 4. set timesteps
            self.scheduler.set_timesteps(num_inference_steps)
            timesteps = self.scheduler.timesteps
    
            # 6. Prepare latent variables
            num_channels_latents = 4
            latents = self.prepare_latents(
                batch_size,
                num_channels_latents,
                height,
                width,
                text_embeddings.dtype,
                latents,
            )
    
            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # Expand the latents if we are doing classifier free guidance.
                    # The latents are expanded 3 times because for pix2pix the guidance\
                    # is applied for both the text and the input image.
                    latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
    
                    result = self.controlnet([latent_model_input, t, text_embeddings, image])
                    down_and_mid_blok_samples = [sample * controlnet_conditioning_scale for _, sample in result.items()]
    
                    # predict the noise residual
                    noise_pred = self.unet([latent_model_input, t, text_embeddings, *down_and_mid_blok_samples])[self.unet_out]
    
                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred[0], noise_pred[1]
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(torch.from_numpy(noise_pred), t, torch.from_numpy(latents)).prev_sample.numpy()
    
                    # update progress
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
    
            # 8. Post-processing
            image = self.decode_latents(latents, pad)
    
            # 9. Convert to PIL
            if output_type == "pil":
                image = self.numpy_to_pil(image)
                image = [img.resize((orig_width, orig_height), Image.Resampling.LANCZOS) for img in image]
            else:
                image = [cv2.resize(img, (orig_width, orig_width)) for img in image]
    
            return image
    
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
    
            text_embeddings = self.text_encoder(text_input_ids)[self.text_encoder_out]
    
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
    
                uncond_embeddings = self.text_encoder(uncond_input.input_ids)[self.text_encoder_out]
    
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = uncond_embeddings.shape[1]
                uncond_embeddings = np.tile(uncond_embeddings, (1, num_images_per_prompt, 1))
                uncond_embeddings = np.reshape(uncond_embeddings, (batch_size * num_images_per_prompt, seq_len, -1))
    
                # For classifier free guidance, we need to do two forward passes.
                # Here we concatenate the unconditional and text embeddings into a single batch
                # to avoid doing two forward passes
                text_embeddings = np.concatenate([uncond_embeddings, text_embeddings])
    
            return text_embeddings
    
        def prepare_latents(
            self,
            batch_size: int,
            num_channels_latents: int,
            height: int,
            width: int,
            dtype: np.dtype = np.float32,
            latents: np.ndarray = None,
        ):
            """
            Preparing noise to image generation. If initial latents are not provided, they will be generated randomly,
            then prepared latents scaled by the standard deviation required by the scheduler
    
            Parameters:
               batch_size (int): input batch size
               num_channels_latents (int): number of channels for noise generation
               height (int): image height
               width (int): image width
               dtype (np.dtype, *optional*, np.float32): dtype for latents generation
               latents (np.ndarray, *optional*, None): initial latent noise tensor, if not provided will be generated
            Returns:
               latents (np.ndarray): scaled initial noise for diffusion
            """
            shape = (
                batch_size,
                num_channels_latents,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            )
            if latents is None:
                latents = randn_tensor(shape, dtype=dtype)
            else:
                latents = latents
    
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
            return latents
    
        def decode_latents(self, latents: np.array, pad: Tuple[int]):
            """
            Decode predicted image from latent space using VAE Decoder and unpad image result
    
            Parameters:
               latents (np.ndarray): image encoded in diffusion latent space
               pad (Tuple[int]): each side padding sizes obtained on preprocessing step
            Returns:
               image: decoded by VAE decoder image
            """
            latents = 1 / 0.18215 * latents
            image = self.vae_decoder(latents)[self.vae_decoder_out]
            (_, end_h), (_, end_w) = pad[1:3]
            h, w = image.shape[2:]
            unpad_h = h - end_h
            unpad_w = w - end_w
            image = image[:, :, :unpad_h, :unpad_w]
            image = np.clip(image / 2 + 0.5, 0, 1)
            image = np.transpose(image, (0, 2, 3, 1))
            return image

.. code:: ipython3

    from transformers import CLIPTokenizer
    from diffusers import UniPCMultistepScheduler
    
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    
    
    def visualize_results(orig_img: Image.Image, skeleton_img: Image.Image, result_img: Image.Image):
        """
        Helper function for results visualization
    
        Parameters:
           orig_img (Image.Image): original image
           skeleton_img (Image.Image): image with body pose keypoints
           result_img (Image.Image): generated image
        Returns:
           fig (matplotlib.pyplot.Figure): matplotlib generated figure contains drawing result
        """
        orig_title = "Original image"
        skeleton_title = "Pose"
        orig_img = orig_img.resize(result_img.size)
        im_w, im_h = orig_img.size
        is_horizontal = im_h <= im_w
        figsize = (20, 20)
        fig, axs = plt.subplots(
            3 if is_horizontal else 1,
            1 if is_horizontal else 3,
            figsize=figsize,
            sharex="all",
            sharey="all",
        )
        fig.patch.set_facecolor("white")
        list_axes = list(axs.flat)
        for a in list_axes:
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.get_xaxis().set_visible(False)
            a.get_yaxis().set_visible(False)
            a.grid(False)
        list_axes[0].imshow(np.array(orig_img))
        list_axes[1].imshow(np.array(skeleton_img))
        list_axes[2].imshow(np.array(result_img))
        list_axes[0].set_title(orig_title, fontsize=15)
        list_axes[1].set_title(skeleton_title, fontsize=15)
        list_axes[2].set_title("Result", fontsize=15)
        fig.subplots_adjust(wspace=0.01 if is_horizontal else 0.00, hspace=0.01 if is_horizontal else 0.1)
        fig.tight_layout()
        fig.savefig("result.png", bbox_inches="tight")
        return fig

Running Text-to-Image Generation with ControlNet Conditioning and OpenVINO
--------------------------------------------------------------------------



Now, we are ready to start generation. For improving the generation
process, we also introduce an opportunity to provide a
``negative prompt``. Technically, positive prompt steers the diffusion
toward the images associated with it, while negative prompt steers the
diffusion away from it. More explanation of how it works can be found in
this
`article <https://stable-diffusion-art.com/how-negative-prompt-work/>`__.
We can keep this field empty if we want to generate image without
negative prompting.

Select inference device for Stable Diffusion pipeline
-----------------------------------------------------



select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    core = ov.Core()
    
    device = device_widget("CPU")
    
    device




.. parsed-literal::

    Dropdown(description='Device:', options=('CPU', 'AUTO'), value='CPU')



.. code:: ipython3

    ov_pipe = OVContrlNetStableDiffusionPipeline(
        tokenizer,
        scheduler,
        core,
        CONTROLNET_OV_PATH,
        TEXT_ENCODER_OV_PATH,
        UNET_OV_PATH,
        VAE_DECODER_OV_PATH,
        device=device.value,
    )

.. code:: ipython3

    np.random.seed(42)
    
    pose = pose_estimator(img)
    
    prompt = "Dancing Darth Vader, best quality, extremely detailed"
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    result = ov_pipe(prompt, pose, 20, negative_prompt=negative_prompt)
    
    result[0]


.. parsed-literal::

    /home/ltalamanova/tmp_venv/lib/python3.11/site-packages/diffusers/configuration_utils.py:139: FutureWarning: Accessing config attribute `controlnet` directly via 'OVContrlNetStableDiffusionPipeline' object attribute is deprecated. Please access 'controlnet' over 'OVContrlNetStableDiffusionPipeline's config object instead, e.g. 'scheduler.config.controlnet'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)




.. image:: controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_34_1.png



Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``OVContrlNetStableDiffusionPipeline`` structure,
ControlNet and UNet are used in the cycle repeating inference on each
diffusion step, while other parts of pipeline take part only once. That
is why computation cost and speed of ControlNet and UNet become the
critical path in the pipeline. Quantizing the rest of the SD pipeline
does not significantly improve inference performance but can lead to a
substantial degradation of accuracy.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize()`` to obtain quantized model.
3. Save the ``INT8`` model using ``openvino.save_model()`` function.

Please select below whether you would like to run quantization to
improve model inference speed.

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
    
    int8_pipe = None
    
    %load_ext skip_kernel_extension

Prepare calibration datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`jschoormans/humanpose_densepose <https://huggingface.co/datasets/jschoormans/humanpose_densepose>`__
dataset from Hugging Face as calibration data. We use a prompts below as
negative prompts for ControlNet and UNet. To collect intermediate model
inputs for calibration we should customize ``CompiledModel``.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    negative_prompts = [
        "blurry unreal occluded",
        "low contrast disfigured uncentered mangled",
        "amateur out of frame low quality nsfw",
        "ugly underexposed jpeg artifacts",
        "low saturation disturbing content",
        "overexposed severe distortion",
        "amateur NSFW",
        "ugly mutilated out of frame disfigured",
    ]

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    
    num_inference_steps = 20
    subset_size = 200
    
    dataset = datasets.load_dataset("jschoormans/humanpose_densepose", split="train", streaming=True).shuffle(seed=42)
    input_data = []
    for batch in dataset:
        caption = batch["caption"]
        if len(caption) > tokenizer.model_max_length:
            continue
        img = batch["file_name"]
        input_data.append((caption, pose_estimator(img)))
        if len(input_data) >= subset_size // num_inference_steps:
            break

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    from tqdm.notebook import tqdm
    from transformers import set_seed
    from typing import Any, Dict, List
    
    set_seed(42)
    
    class CompiledModelDecorator(ov.CompiledModel):
        def __init__(self, compiled_model: ov.CompiledModel, keep_prob: float = 1.0):
            super().__init__(compiled_model)
            self.data_cache = []
            self.keep_prob = np.clip(keep_prob, 0, 1)
    
        def __call__(self, *args, **kwargs):
            if np.random.rand() <= self.keep_prob:
                self.data_cache.append(*args)
            return super().__call__(*args, **kwargs)
    
    def collect_calibration_data(pipeline: OVContrlNetStableDiffusionPipeline, subset_size: int) -> List[Dict]:
        original_unet = pipeline.unet
        pipeline.unet = CompiledModelDecorator(original_unet)
        pipeline.set_progress_bar_config(disable=True)
    
        pbar = tqdm(total=subset_size)
        for prompt, pose in input_data:
            img = batch["file_name"]
            negative_prompt = np.random.choice(negative_prompts)
            _ = pipeline(prompt, pose, num_inference_steps, negative_prompt=negative_prompt)
            collected_subset_size = len(pipeline.unet.data_cache)
            pbar.update(collected_subset_size - pbar.n)
            if collected_subset_size >= subset_size:
                break
    
        calibration_dataset = pipeline.unet.data_cache[:subset_size]
        pipeline.set_progress_bar_config(disable=False)
        pipeline.unet = original_unet
        return calibration_dataset

.. code:: ipython3

    %%skip not $to_quantize.value
    
    CONTROLNET_INT8_OV_PATH = Path("controlnet-pose_int8.xml")
    UNET_INT8_OV_PATH = Path("unet_controlnet_int8.xml")
    
    if not (CONTROLNET_INT8_OV_PATH.exists() and UNET_INT8_OV_PATH.exists()):
        unet_calibration_data = collect_calibration_data(ov_pipe, subset_size=subset_size)



.. parsed-literal::

      0%|          | 0/200 [00:00<?, ?it/s]


.. code:: ipython3

    %%skip not $to_quantize.value
    
    if not CONTROLNET_INT8_OV_PATH.exists():
        control_calibration_data = []
        prev_idx = 0
        for _, pose_img in input_data:
            preprocessed_image, _ = preprocess(pose_img)
            preprocessed_image = np.concatenate(([preprocessed_image] * 2))
            for i in range(prev_idx, prev_idx + num_inference_steps):
                control_calibration_data.append(unet_calibration_data[i][:3] + [preprocessed_image])
            prev_idx += num_inference_steps

Run quantization
~~~~~~~~~~~~~~~~



Create a quantized model from the pre-trained converted OpenVINO model.
``FastBiasCorrection`` algorithm is disabled due to minimal accuracy
improvement in SD models and increased quantization time.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    
    if not UNET_INT8_OV_PATH.exists():
        unet = core.read_model(UNET_OV_PATH)
        quantized_unet = nncf.quantize(
            model=unet,
            calibration_dataset=nncf.Dataset(unet_calibration_data),
            subset_size=subset_size,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                disable_bias_correction=True
            )
        )
        ov.save_model(quantized_unet, UNET_INT8_OV_PATH)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if not CONTROLNET_INT8_OV_PATH.exists():
        controlnet = core.read_model(CONTROLNET_OV_PATH)
        quantized_controlnet = nncf.quantize(
            model=controlnet,
            calibration_dataset=nncf.Dataset(control_calibration_data),
            subset_size=subset_size,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                disable_bias_correction=True
            )
        )
        ov.save_model(quantized_controlnet, CONTROLNET_INT8_OV_PATH)

Let’s compare the images generated by the original and optimized
pipelines.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    int8_pipe = OVContrlNetStableDiffusionPipeline(
        tokenizer,
        scheduler,
        core,
        CONTROLNET_INT8_OV_PATH,
        TEXT_ENCODER_OV_PATH,
        UNET_INT8_OV_PATH,
        VAE_DECODER_OV_PATH,
        device=device.value
    )

.. code:: ipython3

    %%skip not $to_quantize.value
    
    np.random.seed(42)
    int8_image = int8_pipe(prompt, pose, 20, negative_prompt=negative_prompt)[0]
    fig = visualize_pose_results(result[0], int8_image, left_title="FP16 pipeline", right_title="INT8 pipeline")



.. image:: controlnet-stable-diffusion-with-output_files/controlnet-stable-diffusion-with-output_50_0.png


Compare model file sizes
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_ir_model_size = UNET_OV_PATH.with_suffix(".bin").stat().st_size / 2**20
    quantized_model_size = UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 2**20
    
    print(f"FP16 UNet size: {fp16_ir_model_size:.2f} MB")
    print(f"INT8 UNet size: {quantized_model_size:.2f} MB")
    print(f"UNet compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 UNet size: 1639.41 MB
    INT8 UNet size: 820.96 MB
    UNet compression rate: 1.997


.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_ir_model_size = CONTROLNET_OV_PATH.with_suffix(".bin").stat().st_size / 2**20
    quantized_model_size = CONTROLNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 2**20
    
    print(f"FP16 ControlNet size: {fp16_ir_model_size:.2f} MB")
    print(f"INT8 ControlNet size: {quantized_model_size:.2f} MB")
    print(f"ControlNet compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 ControlNet size: 689.07 MB
    INT8 ControlNet size: 345.12 MB
    ControlNet compression rate: 1.997


Compare inference time of the FP16 and INT8 pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To measure the inference performance of the ``FP16`` and ``INT8``
pipelines, we use mean inference time on 3 samples.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    
    def calculate_inference_time(pipeline):
        inference_time = []
        pipeline.set_progress_bar_config(disable=True)
        for i in range(3):
            prompt, pose = input_data[i]
            negative_prompt = np.random.choice(negative_prompts)
            start = time.perf_counter()
            _ = pipeline(prompt, pose, num_inference_steps=num_inference_steps, negative_prompt=negative_prompt)
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        pipeline.set_progress_bar_config(disable=False)
        return np.mean(inference_time)

.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp_latency = calculate_inference_time(ov_pipe)
    print(f"FP16 pipeline: {fp_latency:.3f} seconds")
    int8_latency = calculate_inference_time(int8_pipe)
    print(f"INT8 pipeline: {int8_latency:.3f} seconds")
    print(f"Performance speed-up: {fp_latency / int8_latency:.3f}")


.. parsed-literal::

    FP16 pipeline: 31.296 seconds


.. parsed-literal::

    /home/ltalamanova/tmp_venv/lib/python3.11/site-packages/diffusers/configuration_utils.py:139: FutureWarning: Accessing config attribute `unet` directly via 'OVContrlNetStableDiffusionPipeline' object attribute is deprecated. Please access 'unet' over 'OVContrlNetStableDiffusionPipeline's config object instead, e.g. 'scheduler.config.unet'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)


.. parsed-literal::

    INT8 pipeline: 24.183 seconds
    Performance speed-up: 1.294


Interactive demo
----------------



Please select below whether you would like to use the quantized model to
launch the interactive demo.

.. code:: ipython3

    import ipywidgets as widgets
    
    quantized_model_present = int8_pipe is not None
    
    use_quantized_model = widgets.Checkbox(
        value=True if quantized_model_present else False,
        description="Use quantized model",
        disabled=not quantized_model_present,
    )
    
    use_quantized_model

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/controlnet-stable-diffusion/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    from gradio_helper import make_demo
    
    pipeline = int8_pipe if use_quantized_model.value else ov_pipe
    
    demo = make_demo(pipeline=pipeline, pose_estimator=pose_estimator)
    
    try:
        demo.queue().launch(debug=False)
    except Exception:
        demo.queue().launch(share=True, debug=False)
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
