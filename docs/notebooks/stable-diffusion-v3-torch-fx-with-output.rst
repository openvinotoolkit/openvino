Image generation with Torch.FX Stable Diffusion v3 and OpenVINO
===============================================================

Stable Diffusion V3 is next generation of latent diffusion image Stable
Diffusion models family that outperforms state-of-the-art text-to-image
generation systems in typography and prompt adherence, based on human
preference evaluations. In comparison with previous versions, it based
on Multimodal Diffusion Transformer (MMDiT) text-to-image model that
features greatly improved performance in image quality, typography,
complex prompt understanding, and resource-efficiency.

.. figure:: https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd079427-89f2-4d28-a10e-c80792d750bf
   :alt: mmdit.png

   mmdit.png

More details about model can be found in `model
card <https://huggingface.co/stabilityai/stable-diffusion-3-medium>`__,
`research
paper <https://stability.ai/news/stable-diffusion-3-research-paper>`__
and `Stability.AI blog
post <https://stability.ai/news/stable-diffusion-3-medium>`__. In this
tutorial, we will demonstrate the optimize stable diffusion 3 in a Torch
FX representation using NNCF
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ for model
optimization. Additionally, we will accelerate the pipeline further by
running with torch.compile using the openvino backend. If you want to
run previous Stable Diffusion versions, please check our other
notebooks:

-  `Stable Diffusion <stable-diffusion-text-to-image-with-output.html>`__
-  `Stable Diffusion v2 <stable-diffusion-v2-with-output.html>`__
-  `Stable Diffusion v3 <stable-diffusion-v3-with-output.html>`__
-  `Stable Diffusion XL <stable-diffusion-xl-with-output.html>`__
-  `LCM Stable
   Diffusion <latent-consistency-models-image-generation-with-output.html>`__
-  `Turbo SDXL <sdxl-turbo-with-output.html>`__
-  `Turbo SD <sketch-to-image-pix2pix-turbo-with-output.html>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Build PyTorch pipeline <#build-pytorch-pipeline>`__

   -  `Store the Configs <#store-the-configs>`__

-  `Run FP Inference <#run-fp-inference>`__
-  `Convert models to Torch FX <#convert-models-to-torch-fx>`__
-  `Quantization <#quantization>`__

   -  `Collect Calibration Dataset <#collect-calibration-dataset>`__
   -  `Compress and Quantize models <#compress-and-quantize-models>`__
   -  `Create Optimized Pipeline <#create-optimized-pipeline>`__
   -  `Check File Size <#check-file-size>`__
   -  `Optimized pipeline inference <#optimized-pipeline-inference>`__
   -  `Visualize Results <#visualize-results>`__

-  `Interactive demo <#interactive-demo>`__

Prerequisites
-------------



.. code:: ipython3

    %pip install -q "gradio>=4.19" "torch>=2.5" "torchvision>=0.20" "numpy<2.0" "transformers" "datasets>=2.14.6" "opencv-python" "pillow" "peft>=0.7.0" "diffusers>=0.31.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -qU "openvino>=2024.3.0"
    %pip install -q "nncf>=2.14.0" "typing_extensions>=4.11"

.. code:: ipython3

    from pathlib import Path
    
    import requests
    
    if not Path("sd3_torch_fx_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v3/sd3_torch_fx_helper.py")
        open("sd3_torch_fx_helper.py", "w").write(r.text)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v3/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)
    
    if not Path("skip_kernel_extension.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/skip_kernel_extension.py",
        )
        open("skip_kernel_extension.py", "w").write(r.text)
    
    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry
    
    collect_telemetry("stable-diffusion-v3-torch-fx.ipynb")

Build PyTorch pipeline
----------------------



   **Note**: run model with notebook, you will need to accept license
   agreement. You must be a registered user in Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: ipython3

    # uncomment these lines to login to huggingfacehub to get access to pretrained model
    
    # from huggingface_hub import notebook_login, whoami
    
    # try:
    #     whoami()
    #     print('Authorization token already provided')
    # except OSError:
    #     notebook_login()

.. code:: ipython3

    from sd3_torch_fx_helper import get_sd3_pipeline, init_pipeline
    
    pipe = get_sd3_pipeline()
    pipe.to("cpu")

Store the Configs
~~~~~~~~~~~~~~~~~



This will be used later when wrapping the Torch FX models to insert back
into the pipeline

.. code:: ipython3

    configs_dict = {}
    configs_dict["text_encoder"] = pipe.text_encoder.config
    configs_dict["text_encoder_2"] = pipe.text_encoder_2.config
    configs_dict["transformer"] = pipe.transformer.config
    configs_dict["vae"] = pipe.vae.config
    
    pipe_config = pipe.config

Run FP Inference
----------------



.. code:: ipython3

    import numpy as np
    import torch
    
    generator = torch.Generator(device="cpu").manual_seed(42)
    prompt = "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors"
    num_inference_steps = 28
    with torch.no_grad():
        image = pipe(
            prompt=prompt, negative_prompt="", num_inference_steps=num_inference_steps, generator=generator, guidance_scale=5, height=512, width=512
        ).images[0]
    image

.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device

Convert models to Torch FX
--------------------------



This step converts the pytorch models in the hf pipeline to Torch FX
representation using the ``capture_pre_autograd()`` function.

The pipeline consists of four important parts:

-  Clip and T5 Text Encoders to create condition to generate an image
   from a text prompt.
-  Transformer for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

.. code:: ipython3

    import torch
    from nncf.torch.dynamic_graph.patch_pytorch import disable_patching
    
    text_encoder_input = torch.ones((1, 77), dtype=torch.long)
    text_encoder_kwargs = {}
    text_encoder_kwargs["output_hidden_states"] = True
    
    vae_encoder_input = torch.ones((1, 3, 64, 64))
    vae_decoder_input = torch.ones((1, 16, 64, 64))
    
    unet_kwargs = {}
    unet_kwargs["hidden_states"] = torch.ones((2, 16, 64, 64))
    unet_kwargs["timestep"] = torch.from_numpy(np.array([1, 2], dtype=np.float32))
    unet_kwargs["encoder_hidden_states"] = torch.ones((2, 154, 4096))
    unet_kwargs["pooled_projections"] = torch.ones((2, 2048))
    
    with torch.no_grad():
        with disable_patching():
            text_encoder = torch.export.export_for_training(
                pipe.text_encoder.eval(),
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            ).module()
            text_encoder_2 = torch.export.export_for_training(
                pipe.text_encoder_2.eval(),
                args=(text_encoder_input,),
                kwargs=(text_encoder_kwargs),
            ).module()
            pipe.vae.decoder = torch.export.export_for_training(pipe.vae.decoder.eval(), args=(vae_decoder_input,)).module()
            pipe.vae.encoder = torch.export.export_for_training(pipe.vae.encoder.eval(), args=(vae_encoder_input,)).module()
            vae = pipe.vae
            transformer = torch.export.export_for_training(pipe.transformer.eval(), args=(), kwargs=(unet_kwargs)).module()
    models_dict = {}
    models_dict["transformer"] = transformer
    models_dict["vae"] = vae
    models_dict["text_encoder"] = text_encoder
    models_dict["text_encoder_2"] = text_encoder_2
    del unet_kwargs
    del vae_encoder_input
    del vae_decoder_input
    del text_encoder_input
    del text_encoder_kwargs
    del pipe

Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``StableDiffusion3Pipeline`` structure, the ``transformer``
model takes up significant portion of the overall pipeline execution
time. Now we will show you how to optimize the transformer part using
`NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce
computation cost and speed up the pipeline. Quantizing the rest of the
pipeline does not significantly improve inference performance but can
lead to a substantial degradation of accuracy. That’s why we use 8-bit
weight compression for the rest of the pipeline to reduce memory
footprint.

Please select below whether you would like to run quantization to
improve model inference speed.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    from notebook_utils import quantization_widget
    
    to_quantize = quantization_widget()
    
    to_quantize

Let’s load ``skip magic`` extension to skip quantization if
``to_quantize`` is not selected

.. code:: ipython3

    # Fetch `skip_kernel_extension` module
    
    %load_ext skip_kernel_extension

Collect Calibration Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    from typing import Any, Dict, List
    
    import datasets
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from tqdm.notebook import tqdm
    
    
    def disable_progress_bar(pipeline, disable=True):
        if not hasattr(pipeline, "_progress_bar_config"):
            pipeline._progress_bar_config = {"disable": disable}
        else:
            pipeline._progress_bar_config["disable"] = disable
    
    
    class UNetWrapper(SD3Transformer2DModel):
        def __init__(self, transformer, config):
            super().__init__(**config)
            self.transformer = transformer
            self.captured_args = []
    
        def forward(self, *args, **kwargs):
            del kwargs["joint_attention_kwargs"]
            del kwargs["return_dict"]
            self.captured_args.append((*args, *tuple(kwargs.values())))
            return self.transformer(*args, **kwargs)
    
    
    def collect_calibration_data(
        pipe, calibration_dataset_size: int, num_inference_steps: int
    ) -> List[Dict]:
    
        original_unet = pipe.transformer
        calibration_data = []
        disable_progress_bar(pipe)
    
        dataset = datasets.load_dataset(
            "google-research-datasets/conceptual_captions",
            split="train",
            trust_remote_code=True,
        ).shuffle(seed=42)
    
        transformer_config = dict(pipe.transformer.config)
        if "model" in transformer_config:
            del transformer_config["model"]
        wrapped_unet = UNetWrapper(pipe.transformer.model, transformer_config)
        pipe.transformer = wrapped_unet
        # Run inference for data collection
        pbar = tqdm(total=calibration_dataset_size)
        for i, batch in enumerate(dataset):
            prompt = batch["caption"]
            if len(prompt) > pipe.tokenizer.model_max_length:
                continue
            # Run the pipeline
            pipe(prompt, num_inference_steps=num_inference_steps, height=512, width=512)
            calibration_data.extend(wrapped_unet.captured_args)
            wrapped_unet.captured_args = []
            pbar.update(len(calibration_data) - pbar.n)
            if pbar.n >= calibration_dataset_size:
                break
    
        disable_progress_bar(pipe, disable=False)
        pipe.transformer = original_unet
        return calibration_data
    
    
    if to_quantize:
        pipe = init_pipeline(models_dict, configs_dict)
        calibration_dataset_size = 200
        unet_calibration_data = collect_calibration_data(
            pipe, calibration_dataset_size=calibration_dataset_size, num_inference_steps=28
        )
        del pipe

Compress and Quantize models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    from nncf.quantization.advanced_parameters import AdvancedSmoothQuantParameters
    from nncf.quantization.range_estimator import RangeEstimatorParametersSet
    
    text_encoder = models_dict["text_encoder"]
    text_encoder_2 = models_dict["text_encoder_2"]
    vae_encoder = models_dict["vae"].encoder
    vae_decoder = models_dict["vae"].decoder
    original_transformer = models_dict["transformer"]
    if to_quantize:
        with disable_patching():
            with torch.no_grad():
                nncf.compress_weights(text_encoder)
                nncf.compress_weights(text_encoder_2)
                nncf.compress_weights(vae_encoder)
                nncf.compress_weights(vae_decoder)
                quantized_transformer = nncf.quantize(
                    model=original_transformer,
                    calibration_dataset=nncf.Dataset(unet_calibration_data),
                    subset_size=len(unet_calibration_data),
                    model_type=nncf.ModelType.TRANSFORMER,
                    ignored_scope=nncf.IgnoredScope(names=["conv2d"]),
                    advanced_parameters=nncf.AdvancedQuantizationParameters(
                        weights_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
                        activations_range_estimator_params=RangeEstimatorParametersSet.MINMAX,
                    ),
                )
    
    optimized_models_dict = {}
    optimized_models_dict["transformer"] = quantized_transformer
    optimized_models_dict["vae"] = vae
    optimized_models_dict["text_encoder"] = text_encoder
    optimized_models_dict["text_encoder_2"] = text_encoder_2
    del models_dict

.. code:: ipython3

    %%skip not $to_quantize.value
    import openvino.torch
    
    optimized_models_dict["text_encoder"] = torch.compile(
        optimized_models_dict["text_encoder"], backend="openvino"
    )
    optimized_models_dict["text_encoder_2"] = torch.compile(
        optimized_models_dict["text_encoder_2"], backend="openvino"
    )
    optimized_models_dict["vae"].encoder = torch.compile(
        optimized_models_dict["vae"].encoder, backend="openvino"
    )
    optimized_models_dict["vae"].decoder = torch.compile(
        optimized_models_dict["vae"].decoder, backend="openvino"
    )
    optimized_models_dict["transformer"] = torch.compile(
        optimized_models_dict["transformer"], backend="openvino"
    )

Create Optimized Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~



Initialize the optimized pipeline using the optimized models

.. code:: ipython3

    %%skip not $to_quantize.value
    
    opt_pipe = init_pipeline(optimized_models_dict, configs_dict)

Check File Size
~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    
    def get_model_size(models):
        total_size = 0
        for model in models:
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
    
            model_size_mb = (param_size + buffer_size) / 1024**2
    
            total_size += model_size_mb
        return total_size
    
    
    optimized_model_size = get_model_size([opt_pipe.transformer])
    original_model_size = get_model_size([original_transformer])
    
    print(f"Original Transformer Size: {original_model_size} MB")
    print(f"Optimized Transformer Size: {optimized_model_size} MB")
    print(f"Compression Rate: {original_model_size / optimized_model_size:.3f}")

Optimized pipeline inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



Run inference with single step to compile the model.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    # Warmup the model for initial compile
    with torch.no_grad():
        opt_pipe(
            prompt=prompt, negative_prompt="", num_inference_steps=1, generator=generator, height=512, width=512
        ).images[0]

Visualize Results
~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    
    from sd3_torch_fx_helper import visualize_results
    
    generator = torch.Generator(device="cpu").manual_seed(42)
    opt_image = opt_pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=5,
        generator=generator,
        height=512,
        width=512
    ).images[0]
    
    visualize_results(image, opt_image)

Interactive demo
----------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    use_quantized_models = quantization_widget()
    
    use_quantized_models

.. code:: ipython3

    from gradio_helper import make_demo
    
    fx_pipe = init_pipeline(models_dict if not to_quantize.value else optimized_models_dict, configs_dict)
    demo = make_demo(fx_pipe, False)
    
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    try:
        demo.launch(debug=True)
    except Exception:
        demo.launch(debug=True, share=True)
