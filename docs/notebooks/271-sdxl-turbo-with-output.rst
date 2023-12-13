Single step image generation using SDXL-turbo and OpenVINO
==========================================================

SDXL-Turbo is a fast generative text-to-image model that can synthesize
photorealistic images from a text prompt in a single network evaluation.
SDXL-Turbo is a distilled version of `SDXL
1.0 <https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0>`__,
trained for real-time synthesis. SDXL Turbo is based on a novel
distillation technique called Adversarial Diffusion Distillation (ADD),
which enables the model to synthesize image outputs in a single step and
generate real-time text-to-image outputs while maintaining high sampling
fidelity. More details about this distillation approach can be found in
`technical
report <https://stability.ai/research/adversarial-diffusion-distillation>`__.
More details about model can be found in `Stability AI blog
post <https://stability.ai/news/stability-ai-sdxl-turbo>`__.

Previously, we already discussed how to launch Stable Diffusion XL model
using OpenVINO in the following
`notebook <../248-stable-diffusion-xl>`__, in this tutorial we will
focus on the
`SDXL-turbo <https://huggingface.co/stabilityai/sdxl-turbo>`__ version.
Additionally, to improve image decoding speed, we will use `Tiny
Autoencoder <https://github.com/madebyollin/taesd>`__, which is useful
for real-time previewing of the SDXL generation process.

We will use a pre-trained model from the `Hugging Face
Diffusers <https://huggingface.co/docs/diffusers/index>`__ library. To
simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library is
used to convert the models to OpenVINO™ IR format.

Prerequisites
-------------

.. code:: ipython3

    %pip uninstall -q -y openvino-dev openvino openvino-nightly
    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    torch transformers diffusers "git+https://github.com/huggingface/optimum-intel.git" gradio openvino-nightly

Convert model to OpenVINO format
--------------------------------

`sdxl-turbo <https://huggingface.co/stabilityai/sdxl-turbo>`__ is
available for downloading via the `HuggingFace
hub <https://huggingface.co/models>`__. We will use optimum-cli
interface for exporting it into OpenVINO Intermediate Representation
(IR) format.

Optimum CLI interface for converting models supports export to OpenVINO
(supported starting optimum-intel 1.12 version). General command format:

.. code:: bash

   optimum-cli export openvino --model <model_id_or_path> --task <task> <output_dir>

where task is task to export the model for, if not specified, the task
will be auto-inferred based on the model. Available tasks depend on the
model, for sdxl should be selected ``stable-diffusion-xl``

You can find a mapping between tasks and model classes in Optimum
TaskManager
`documentation <https://huggingface.co/docs/optimum/exporters/task_manager>`__.

Additionally, you can specify weights compression ``--fp16`` for the
compression model to FP16 and ``--int8`` for the compression model to
INT8. Please note, that for INT8, it is necessary to install nncf.

Full list of supported arguments available via ``--help`` For more
details and examples of usage, please check `optimum
documentation <https://huggingface.co/docs/optimum/intel/inference#export>`__.

For Tiny Autoencoder, we will use ``ov.convert_model`` function for
obtaining ``ov.Model`` and save it using ``ov.save_model``. Model
consists of 2 parts that used in pipeline separately: ``vae_encoder``
for encoding input image in latent space in image-to-image generation
task and ``vae_decoder`` that responsible for decoding diffusion result
back to image format.

.. code:: ipython3

    from pathlib import Path
    
    model_dir = Path("./model")
    sdxl_model_id = "stabilityai/sdxl-turbo"
    tae_id = "madebyollin/taesdxl"
    skip_convert_model = model_dir.exists()

.. code:: ipython3

    import torch
    import openvino as ov
    from diffusers import AutoencoderTiny
    import gc
    
    class VAEEncoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
    
        def forward(self, sample):
            return self.vae.encode(sample)
        
    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
    
        def forward(self, latent_sample):
            return self.vae.decode(latent_sample)
    
    def convert_tiny_vae(model_id, output_path):
        tiny_vae = AutoencoderTiny.from_pretrained(model_id)
        tiny_vae.eval()
        vae_encoder = VAEEncoder(tiny_vae)
        ov_model = ov.convert_model(vae_encoder, example_input=torch.zeros((1,3,512,512)))
        ov.save_model(ov_model, output_path / "vae_encoder/openvino_model.xml")
        tiny_vae.save_config(output_path / "vae_encoder")
        vae_decoder = VAEDecoder(tiny_vae)
        ov_model = ov.convert_model(vae_decoder, example_input=torch.zeros((1,4,64,64)))
        ov.save_model(ov_model, output_path / "vae_decoder/openvino_model.xml")
        tiny_vae.save_config(output_path / "vae_decoder")    
        
    
    if not skip_convert_model:
        !optimum-cli export openvino --model $sdxl_model_id --task stable-diffusion-xl $model_dir --fp16
        convert_tiny_vae(tae_id, model_dir)

Text-to-image generation
------------------------

Text-to-image generation lets you create images using text description.
To start generating images, we need to load models first. To load an
OpenVINO model and run an inference with Optimum and OpenVINO Runtime,
you need to replace diffusers ``StableDiffusionXLPipeline`` with Optimum
``OVStableDiffusionXLPipeline``. Pipeline initialization starts with
using ``from_pretrained`` method, where a directory with OpenVINO models
should be passed. Additionally, you can specify an inference device.

Select inference device for text-to-image generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import ipywidgets as widgets
    
    core = ov.Core()
    
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='AUTO',
        description='Device:',
        disabled=False,
    )
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    from optimum.intel.openvino import OVStableDiffusionXLPipeline
    text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    2023-12-01 11:21:33.190808: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-12-01 11:21:33.194252: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-12-01 11:21:33.260150: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2023-12-01 11:21:33.261916: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-12-01 11:21:33.994990: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    Compiling the vae_decoder to AUTO ...
    Compiling the unet to AUTO ...
    Compiling the text_encoder_2 to AUTO ...
    Compiling the text_encoder to AUTO ...
    Compiling the vae_encoder to AUTO ...


The pipeline interface is similar to original
``StableDiffusionXLPipeline``. We should provide text prompt. The
default number of steps is 50, while sdxl-turbo required only 1 step.
According to the information provided in model card, model does not use
negative prompt and guidance scale and this parameters should be
disabled using ``guidance_scale = 0``

.. code:: ipython3

    import numpy as np
    
    prompt = "cute cat"
    image = text2image_pipe(prompt, num_inference_steps=1, height=512, width=512, guidance_scale=0.0, generator=np.random.RandomState(987)).images[0]
    image.save("cat.png")
    image


.. parsed-literal::

    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:565: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:599: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)
    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:615: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)




.. image:: 271-sdxl-turbo-with-output_files/271-sdxl-turbo-with-output_11_3.png



.. code:: ipython3

    del text2image_pipe
    gc.collect();

Image-to-Image generation
-------------------------

Image-to-image generation lets you transform images to match the
characteristics provided in the text description. We can reuse the
already converted model for running the Image2Image generation pipeline.
For that, we should replace ``OVStableDiffusionXLPipeline`` with
``OVStableDiffusionXLImage2ImagePipeline``.

.. code:: ipython3

    from optimum.intel import OVStableDiffusionXLImg2ImgPipeline
    
    image2image_pipe = OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir, device=device.value)


.. parsed-literal::

    Compiling the vae_decoder to AUTO ...
    Compiling the unet to AUTO ...
    Compiling the vae_encoder to AUTO ...
    Compiling the text_encoder to AUTO ...
    Compiling the text_encoder_2 to AUTO ...


.. code:: ipython3

    photo_prompt = "a cute cat with bow tie"

``strength`` parameter is important for the image-to-image generation
pipeline. It is a value between 0.0 and 1.0, that controls the amount of
noise that is added to the input image. Values that approach 1.0 enable
lots of variations but will also produce images that are not
semantically consistent with the input, then close to 0, less noise will
be added and the target image will preserve source image content.
strength has an impact not only on a number of noise but also the number
of generation steps. The number of denoising iterations in the
image-to-image generation pipeline is calculated as
``int(num_inference_steps * strength)``. With sdxl-turbo we should be
careful with selecting ``num_inference_steps`` and ``strength`` to
produce the correct result and make sure that the number of steps used
in pipeline >= 1 after applying strength multiplication. e.g. in example
below, we will use ``num_inference_steps=2`` and ``stength=0.5``,
finally, we get 0.5 \* 2.0 = 1 step in our pipeline.

.. code:: ipython3

    photo_image = image2image_pipe(photo_prompt, image=image, num_inference_steps=2, generator=np.random.RandomState(511), guidance_scale=0.0, strength=0.5).images[0]
    photo_image.save("cat_tie.png")
    photo_image


.. parsed-literal::

    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:636: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]




.. image:: 271-sdxl-turbo-with-output_files/271-sdxl-turbo-with-output_17_2.png



.. code:: ipython3

    del image2image_pipe
    gc.collect();

Quantization
------------

`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``SDXL-Turbo Model`` structure, the UNet model takes up
significant portion of the overall pipeline execution time. Now we will
show you how to optimize the UNet part using
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce
computation cost and speed up the pipeline. Quantizing the rest of the
SDXL pipeline does not significantly improve inference performance but
can lead to a substantial degradation of accuracy.

The optimization process contains the following steps:

1. Create a calibration dataset for quantization.
2. Run ``nncf.quantize()`` to obtain quantized model.
3. Save the ``INT8`` model using ``openvino.save_model()`` function.

Please select below whether you would like to run quantization to
improve model inference speed.

.. code:: ipython3

    to_quantize = widgets.Checkbox(
        value=True,
        description='Quantization',
        disabled=False,
    )
    
    to_quantize




.. parsed-literal::

    Checkbox(value=True, description='Quantization')



.. code:: ipython3

    import sys
    sys.path.append("../utils")
    
    int8_pipe = None
    
    if to_quantize.value and "GPU" in device.value:
        to_quantize.value = False
    
    %load_ext skip_kernel_extension

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use a portion of
`laion/laion2B-en <https://huggingface.co/datasets/laion/laion2B-en>`__
dataset from Hugging Face as calibration data. To collect intermediate
model inputs for calibration we should customize ``CompiledModel``.

.. code:: ipython3

    UNET_INT8_OV_PATH = model_dir / "optimized_unet" / "openvino_model.xml"
    
    def disable_progress_bar(pipeline, disable=True):
        if not hasattr(pipeline, "_progress_bar_config"):
            pipeline._progress_bar_config = {'disable': disable}
        else:
            pipeline._progress_bar_config['disable'] = disable

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import datasets
    import numpy as np
    from tqdm.notebook import tqdm
    from transformers import set_seed
    from typing import Any, Dict, List
    
    set_seed(1)
    
    class CompiledModelDecorator(ov.CompiledModel):
        def __init__(self, compiled_model: ov.CompiledModel, data_cache: List[Any] = None):
            super().__init__(compiled_model)
            self.data_cache = data_cache if data_cache else []
    
        def __call__(self, *args, **kwargs):
            self.data_cache.append(*args)
            return super().__call__(*args, **kwargs)
    
    def collect_calibration_data(pipe, subset_size: int) -> List[Dict]:
        original_unet = pipe.unet.request
        pipe.unet.request = CompiledModelDecorator(original_unet)
    
        dataset = datasets.load_dataset("laion/laion2B-en", split="train", streaming=True).shuffle(seed=42)
        disable_progress_bar(pipe)
    
        # Run inference for data collection
        pbar = tqdm(total=subset_size)
        diff = 0
        for batch in dataset:
            prompt = batch["TEXT"]
            if len(prompt) > pipe.tokenizer.model_max_length:
                continue
            _ = pipe(
                prompt,
                num_inference_steps=1,
                height=512,
                width=512,
                guidance_scale=0.0,
                generator=np.random.RandomState(987)
            )
            collected_subset_size = len(pipe.unet.request.data_cache)
            if collected_subset_size >= subset_size:
                pbar.update(subset_size - pbar.n)
                break
            pbar.update(collected_subset_size - diff)
            diff = collected_subset_size
    
        calibration_dataset = pipe.unet.request.data_cache
        disable_progress_bar(pipe, disable=False)
        pipe.unet.request = original_unet
        return calibration_dataset

.. code:: ipython3

    %%skip not $to_quantize.value
    
    if not UNET_INT8_OV_PATH.exists():
        text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value)
        unet_calibration_data = collect_calibration_data(text2image_pipe, subset_size=200)


.. parsed-literal::

    Compiling the vae_decoder to AUTO ...
    Compiling the unet to AUTO ...
    Compiling the text_encoder_2 to AUTO ...
    Compiling the vae_encoder to AUTO ...
    Compiling the text_encoder to AUTO ...



.. parsed-literal::

    Resolving data files:   0%|          | 0/128 [00:00<?, ?it/s]



.. parsed-literal::

      0%|          | 0/200 [00:00<?, ?it/s]


.. parsed-literal::

    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:565: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)
    <string>:17: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:615: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)


Run quantization
~~~~~~~~~~~~~~~~

Create a quantized model from the pre-trained converted OpenVINO model.
Quantization of the first and last ``Convolution`` layers impacts the
generation results. We recommend using ``IgnoredScope`` to keep accuracy
sensitive ``Convolution`` layers in FP16 precision.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    from nncf.scopes import IgnoredScope
    
    UNET_OV_PATH = model_dir / "unet" / "openvino_model.xml"
    if not UNET_INT8_OV_PATH.exists():
        unet = core.read_model(UNET_OV_PATH)
        quantized_unet = nncf.quantize(
            model=unet,
            model_type=nncf.ModelType.TRANSFORMER,
            calibration_dataset=nncf.Dataset(unet_calibration_data),
            ignored_scope=IgnoredScope(
                names=[
                    "__module.model.conv_in/aten::_convolution/Convolution",
                    "__module.model.up_blocks.2.resnets.2.conv_shortcut/aten::_convolution/Convolution",
                    "__module.model.conv_out/aten::_convolution/Convolution"
                ],
            ),
        )
        ov.save_model(quantized_unet, UNET_INT8_OV_PATH)



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



.. parsed-literal::

    INFO:nncf:3 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:420 ignored nodes were found by name in the NNCFGraph
    INFO:nncf:Not adding activation input quantizer for operation: 5 __module.model.conv_in/aten::_convolution/Convolution
    13 __module.model.conv_in/aten::_convolution/Add_87
    
    INFO:nncf:Not adding activation input quantizer for operation: 460 __module.model.up_blocks.2.resnets.2.conv_shortcut/aten::_convolution/Convolution
    899 __module.model.up_blocks.2.resnets.2.conv_shortcut/aten::_convolution/Add_16859
    
    INFO:nncf:Not adding activation input quantizer for operation: 3911 __module.model.conv_out/aten::_convolution/Convolution
    4032 __module.model.conv_out/aten::_convolution/Add_16873
    



.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>




.. parsed-literal::

    Output()



.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"></pre>




.. raw:: html

    <pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
    </pre>



Let us check predictions with the quantized UNet using the same input
data.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from IPython.display import display
    
    int8_text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value, compile=False)
    int8_text2image_pipe.unet.model = core.read_model(UNET_INT8_OV_PATH)
    int8_text2image_pipe.unet.request = None
    
    prompt = "cute cat"
    image = int8_text2image_pipe(prompt, num_inference_steps=1, height=512, width=512, guidance_scale=0.0, generator=np.random.RandomState(987)).images[0]
    display(image)


.. parsed-literal::

    Compiling the text_encoder to AUTO ...
    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:565: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)
    Compiling the text_encoder_2 to AUTO ...



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]


.. parsed-literal::

    Compiling the unet to AUTO ...
    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:599: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)
    Compiling the vae_decoder to AUTO ...
    /home/ltalamanova/env_ci/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:615: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)



.. image:: 271-sdxl-turbo-with-output_files/271-sdxl-turbo-with-output_29_3.png


.. code:: ipython3

    %%skip not $to_quantize.value
    
    int8_image2image_pipe = OVStableDiffusionXLImg2ImgPipeline.from_pretrained(model_dir, device=device.value, compile=False)
    int8_image2image_pipe.unet.model = core.read_model(UNET_INT8_OV_PATH)
    int8_image2image_pipe.unet.request = None
    
    photo_prompt = "a cute cat with bow tie"
    photo_image = int8_image2image_pipe(photo_prompt, image=image, num_inference_steps=2, generator=np.random.RandomState(511), guidance_scale=0.0, strength=0.5).images[0]
    display(photo_image)



.. parsed-literal::

      0%|          | 0/1 [00:00<?, ?it/s]



.. image:: 271-sdxl-turbo-with-output_files/271-sdxl-turbo-with-output_30_1.png


Compare UNet file size
^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    %%skip not $to_quantize.value
    
    fp16_ir_model_size = UNET_OV_PATH.with_suffix(".bin").stat().st_size / 1024
    quantized_model_size = UNET_INT8_OV_PATH.with_suffix(".bin").stat().st_size / 1024
    
    print(f"FP16 model size: {fp16_ir_model_size:.2f} KB")
    print(f"INT8 model size: {quantized_model_size:.2f} KB")
    print(f"Model compression rate: {fp16_ir_model_size / quantized_model_size:.3f}")


.. parsed-literal::

    FP16 model size: 5014578.27 KB
    INT8 model size: 2513501.39 KB
    Model compression rate: 1.995


Compare inference time of the FP16 and INT8 models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To measure the inference performance of the ``FP16`` and ``INT8``
pipelines, we use median inference time on calibration subset.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import time
    
    validation_size = 7
    calibration_dataset = datasets.load_dataset("laion/laion2B-en", split="train", streaming=True).take(validation_size)
    validation_data = []
    for batch in calibration_dataset:
        prompt = batch["TEXT"]
        validation_data.append(prompt)
    
    def calculate_inference_time(pipe, dataset):
        inference_time = []
        disable_progress_bar(pipe)
    
        for prompt in dataset:
            start = time.perf_counter()
            image = pipe(
                prompt,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=np.random.RandomState(23)
            ).images[0]
            end = time.perf_counter()
            delta = end - start
            inference_time.append(delta)
        disable_progress_bar(pipe, disable=False)
        return np.median(inference_time)



.. parsed-literal::

    Resolving data files:   0%|          | 0/128 [00:00<?, ?it/s]


.. code:: ipython3

    %%skip not $to_quantize.value
    
    int8_latency = calculate_inference_time(int8_text2image_pipe, validation_data)
    text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value)
    fp_latency = calculate_inference_time(text2image_pipe, validation_data)
    print(f"FP16 pipeline latency: {fp_latency:.3f}")
    print(f"INT8 pipeline latency: {int8_latency:.3f}")
    print(f"Text-to-Image generation speed up: {fp_latency / int8_latency:.3f}")


.. parsed-literal::

    FP16 pipeline latency: 1.115
    INT8 pipeline latency: 0.575
    Text-to-Image generation speed up: 1.938


Interactive Demo
----------------

Now, you can check model work using own text descriptions. Provide text
prompt in the text box and launch generation using Run button.
Additionally you can control generation with additional parameters: \*
Seed - random seed for initialization \* Steps - number of generation
steps \* Height and Width - size of generated image

   Please note that increasing image size may require to increasing
   number of steps for accurate result. We recommend running 104x1024
   resolution image generation using 4 steps.

Please select below whether you would like to use the quantized model to
launch the interactive demo.

.. code:: ipython3

    quantized_model_present = UNET_INT8_OV_PATH.exists()
    
    use_quantized_model = widgets.Checkbox(
        value=True if quantized_model_present else False,
        description='Use quantized model',
        disabled=False,
    )
    
    use_quantized_model




.. parsed-literal::

    Checkbox(value=True, description='Use quantized model')



.. code:: ipython3

    import gradio as gr
    
    text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value)
    if use_quantized_model.value:
        if not quantized_model_present:
            raise RuntimeError("Quantized model not found.")
        text2image_pipe.unet.model = core.read_model(UNET_INT8_OV_PATH)
        text2image_pipe.unet.request = core.compile_model(text2image_pipe.unet.model)
    
    
    def generate_from_text(text, seed, num_steps, height, width):
        result = text2image_pipe(text, num_inference_steps=num_steps, guidance_scale=0.0, generator=np.random.RandomState(seed), height=height, width=width).images[0]
        return result
    
    
    with gr.Blocks() as demo:
        with gr.Column():
            positive_input = gr.Textbox(label="Text prompt")
            with gr.Row():
                seed_input = gr.Number(precision=0, label="Seed", value=42, minimum=0)
                steps_input = gr.Slider(label="Steps", value=1, minimum=1, maximum=4, step=1)
                height_input = gr.Slider(label="Height", value=512, minimum=256, maximum=1024, step=32)
                width_input = gr.Slider(label="Width", value=512, minimum=256, maximum=1024, step=32)
                btn = gr.Button()
            out = gr.Image(label="Result (Quantized)" if use_quantized_model.value else "Result (Original)", type="pil", width=512)
            btn.click(generate_from_text, [positive_input, seed_input, steps_input, height_input, width_input], out)
            gr.Examples([
                ["cute cat", 999], 
                ["underwater world coral reef, colorful jellyfish, 35mm, cinematic lighting, shallow depth of field,  ultra quality, masterpiece, realistic", 89],
                ["a photo realistic happy white poodle dog ​​playing in the grass, extremely detailed, high res, 8k, masterpiece, dynamic angle", 1569],
                ["Astronaut on Mars watching sunset, best quality, cinematic effects,", 65245],
                ["Black and white street photography of a rainy night in New York, reflections on wet pavement", 48199]
            ], [positive_input, seed_input])
    
    # if you are launching remotely, specify server_name and server_port
    # demo.launch(server_name='your server name', server_port='server port in int')
    # Read more in the docs: https://gradio.app/docs/
    # if you want create public link for sharing demo, please add share=True
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(share=True, debug=False)
