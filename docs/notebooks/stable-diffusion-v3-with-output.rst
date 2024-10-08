Image generation with Stable Diffusion v3 and OpenVINO
======================================================

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
tutorial, we will consider how to convert Stable Diffusion v3 for
running with OpenVINO. An additional part demonstrates how to run
optimization with `NNCF <https://github.com/openvinotoolkit/nncf/>`__ to
speed up pipeline. If you want to run previous Stable Diffusion
versions, please check our other notebooks:

-  `Stable Diffusion <stable-diffusion-text-to-image-with-output.html>`__
-  `Stable Diffusion v2 <stable-diffusion-v2-with-output.html>`__
-  `Stable Diffusion XL <stable-diffusion-xl-with-output.html>`__
-  `LCM Stable
   Diffusion <latent-consistency-models-image-generation-with-output.html>`__
-  `Turbo SDXL <sdxl-turbo-with-output.html>`__
-  `Turbo SD <sketch-to-image-pix2pix-turbo-with-output.html>`__


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Build PyTorch pipeline <#build-pytorch-pipeline>`__
-  `Convert models with OpenVINO <#convert-models-with-openvino>`__

   -  `Transformer <#transformer>`__
   -  `T5 Text Encoder <#t5-text-encoder>`__
   -  `Clip text encoders <#clip-text-encoders>`__
   -  `VAE <#vae>`__

-  `Prepare OpenVINO inference
   pipeline <#prepare-openvino-inference-pipeline>`__
-  `Run OpenVINO model <#run-openvino-model>`__
-  `Quantization <#quantization>`__

   -  `Prepare calibration dataset <#prepare-calibration-dataset>`__
   -  `Run Quantization <#run-quantization>`__
   -  `Run Weights Compression <#run-weights-compression>`__
   -  `Compare model file sizes <#compare-model-file-sizes>`__
   -  `Compare inference time of the FP16 and optimized
      pipelines <#compare-inference-time-of-the-fp16-and-optimized-pipelines>`__

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

    %pip install -q "git+https://github.com/initml/diffusers.git@clement/feature/flash_sd3" "gradio>=4.19" "torch>=2.1"  "transformers" "nncf>=2.12.0" "datasets>=2.14.6" "opencv-python" "pillow" "peft>=0.7.0" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install -qU "openvino>=2024.3.0"

.. code:: ipython3

    import requests
    from pathlib import Path
    
    if not Path("sd3_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v3/sd3_helper.py")
        open("sd3_helper.py", "w").write(r.text)
    
    if not Path("sd3_quantization_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v3/sd3_quantization_helper.py")
        open("sd3_quantization_helper.py", "w").write(r.text)
    
    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/stable-diffusion-v3/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)
    
    if not Path("notebook_utils.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py")
        open("notebook_utils.py", "w").write(r.text)

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

We will use
`Diffusers <https://huggingface.co/docs/diffusers/main/en/index>`__
library integration for running Stable Diffusion v3 model. You can find
more details in Diffusers
`documentation <https://huggingface.co/docs/diffusers/main/en/api/pipelines/stable_diffusion/stable_diffusion_3>`__.
Additionally, we can apply optimization for pipeline performance and
memory consumption:

-  **Use flash SD3**. Flash Diffusion is a diffusion distillation method
   proposed in `Flash Diffusion: Accelerating Any Conditional Diffusion
   Model for Few Steps Image
   Generation <http://arxiv.org/abs/2406.02347>`__. The model
   represented as a 90.4M LoRA distilled version of SD3 model that is
   able to generate 1024x1024 images in 4 steps. If you want disable it,
   you can unset checkbox **Use flash SD3**
-  **Remove T5 text encoder**. Removing the memory-intensive 4.7B
   parameter T5-XXL text encoder during inference can significantly
   decrease the memory requirements for SD3 with only a slight loss in
   performance. If you want to use this model in pipeline, please set
   **use t5 text encoder** checkbox.

.. code:: ipython3

    from sd3_helper import get_pipeline_options
    
    pt_pipeline_options, use_flash_lora, load_t5 = get_pipeline_options()
    
    display(pt_pipeline_options)


.. parsed-literal::

    /home/ea/work/my_optimum_intel/optimum_env/lib/python3.8/site-packages/diffusers/models/transformers/transformer_2d.py:34: FutureWarning: `Transformer2DModelOutput` is deprecated and will be removed in version 1.0.0. Importing `Transformer2DModelOutput` from `diffusers.models.transformer_2d` is deprecated and this will be removed in a future version. Please use `from diffusers.models.modeling_outputs import Transformer2DModelOutput`, instead.
      deprecate("Transformer2DModelOutput", "1.0.0", deprecation_message)
    2024-08-08 08:15:46.648328: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2024-08-08 08:15:46.650527: I tensorflow/tsl/cuda/cudart_stub.cc:28] Could not find cuda drivers on your machine, GPU will not be used.
    2024-08-08 08:15:46.687530: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2024-08-08 08:15:47.368728: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT



.. parsed-literal::

    VBox(children=(Checkbox(value=True, description='Use flash SD3'), Checkbox(value=False, description='Use t5 te…


Convert models with OpenVINO
----------------------------



Starting from 2023.0 release, OpenVINO supports PyTorch models directly
via Model Conversion API. ``ov.convert_model`` function accepts instance
of PyTorch model and example inputs for tracing and returns object of
``ov.Model`` class, ready to use or save on disk using ``ov.save_model``
function.

The pipeline consists of four important parts:

-  Clip and T5 Text Encoders to create condition to generate an image
   from a text prompt.
-  Transformer for step-by-step denoising latent image representation.
-  Autoencoder (VAE) for decoding latent space to image.

We will use ``convert_sd3`` helper function defined in
`sd3_helper.py <sd3_helper.py-with-output.html>`__ that create original PyTorch model
and convert each part of pipeline using ``ov.convert_model``.

.. code:: ipython3

    from sd3_helper import convert_sd3
    
    # Uncomment the line beolow to see model conversion code
    # ??convert_sd3

.. code:: ipython3

    convert_sd3(load_t5.value, use_flash_lora.value)


.. parsed-literal::

    SD3 model already converted


Prepare OpenVINO inference pipeline
-----------------------------------



.. code:: ipython3

    from sd3_helper import OVStableDiffusion3Pipeline, init_pipeline  # noqa: F401
    
    # Uncomment line below to see pipeline code
    # ??OVStableDiffusion3Pipeline

Run OpenVINO model
------------------



.. code:: ipython3

    from notebook_utils import device_widget
    
    device = device_widget()
    
    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



.. code:: ipython3

    from sd3_helper import TEXT_ENCODER_PATH, TEXT_ENCODER_2_PATH, TEXT_ENCODER_3_PATH, TRANSFORMER_PATH, VAE_DECODER_PATH
    
    models_dict = {"transformer": TRANSFORMER_PATH, "vae": VAE_DECODER_PATH, "text_encoder": TEXT_ENCODER_PATH, "text_encoder_2": TEXT_ENCODER_2_PATH}
    
    if load_t5.value:
        models_dict["text_encoder_3"] = TEXT_ENCODER_3_PATH
    
    ov_pipe = init_pipeline(models_dict, device.value, use_flash_lora.value)


.. parsed-literal::

    Models compilation
    transformer - Done!
    vae - Done!
    text_encoder - Done!
    text_encoder_2 - Done!


.. code:: ipython3

    import torch
    
    image = ov_pipe(
        "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
        negative_prompt="",
        num_inference_steps=28 if not use_flash_lora.value else 4,
        guidance_scale=5 if not use_flash_lora.value else 0,
        height=512,
        width=512,
        generator=torch.Generator().manual_seed(141),
    ).images[0]
    image



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]




.. image:: stable-diffusion-v3-with-output_files/stable-diffusion-v3-with-output_16_1.png



Quantization
------------



`NNCF <https://github.com/openvinotoolkit/nncf/>`__ enables
post-training quantization by adding quantization layers into model
graph and then using a subset of the training dataset to initialize the
parameters of these additional quantization layers. Quantized operations
are executed in ``INT8`` instead of ``FP32``/``FP16`` making model
inference faster.

According to ``OVStableDiffusion3Pipeline`` structure, the
``transformer`` model takes up significant portion of the overall
pipeline execution time. Now we will show you how to optimize the UNet
part using `NNCF <https://github.com/openvinotoolkit/nncf/>`__ to reduce
computation cost and speed up the pipeline. Quantizing the rest of the
pipeline does not significantly improve inference performance but can
lead to a substantial degradation of accuracy. That’s why we use 4-bit
weight compression for the rest of the pipeline to reduce memory
footprint.

Please select below whether you would like to run quantization to
improve model inference speed.

   **NOTE**: Quantization is time and memory consuming operation.
   Running quantization code below may take some time.

.. code:: ipython3

    from notebook_utils import quantization_widget
    from sd3_quantization_helper import TRANSFORMER_INT8_PATH, TEXT_ENCODER_INT4_PATH, TEXT_ENCODER_2_INT4_PATH, TEXT_ENCODER_3_INT4_PATH, VAE_DECODER_INT4_PATH
    
    to_quantize = quantization_widget()
    
    to_quantize


.. parsed-literal::

    huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
    To disable this warning, you can either:
    	- Avoid using `tokenizers` before the fork if possible
    	- Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino




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
    
    optimized_pipe = None
    
    opt_models_dict = {
        "transformer": TRANSFORMER_INT8_PATH,
        "text_encoder": TEXT_ENCODER_INT4_PATH,
        "text_encoder_2": TEXT_ENCODER_2_INT4_PATH,
        "vae": VAE_DECODER_INT4_PATH,
    }
    
    if TEXT_ENCODER_3_PATH.exists():
        opt_models_dict["text_encoder_3"] = TEXT_ENCODER_3_INT4_PATH
    
    %load_ext skip_kernel_extension

Prepare calibration dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~



We use a portion of
`google-research-datasets/conceptual_captions <https://huggingface.co/datasets/google-research-datasets/conceptual_captions>`__
dataset from Hugging Face as calibration data. We use prompts below to
guide image generation and to determine what not to include in the
resulting image.

To collect intermediate model inputs for calibration we should customize
``CompiledModel``. We should set the height and width of the image to
512 to reduce memory consumption during quantization.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from sd3_quantization_helper import collect_calibration_data, TRANSFORMER_INT8_PATH
    
    # Uncomment the line to see calibration data collection code
    # ??collect_calibration_data


Run Quantization
~~~~~~~~~~~~~~~~



Quantization of the first ``Convolution`` layer impacts the generation
results. We recommend using ``IgnoredScope`` to keep accuracy sensitive
layers in FP16 precision.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    import nncf
    import gc
    import openvino as ov
    
    core = ov.Core()
    
    
    if not TRANSFORMER_INT8_PATH.exists():
        calibration_dataset_size = 200
        print("Calibration data collection started")
        unet_calibration_data = collect_calibration_data(ov_pipe,
                                                         calibration_dataset_size=calibration_dataset_size,
                                                         num_inference_steps=28 if not use_flash_lora.value else 4,
                                                         guidance_scale=5 if not use_flash_lora.value else 0
                                                         )
        print("Calibration data collection finished")
        
        del ov_pipe
        gc.collect()
        ov_pipe = None
    
        transformer = core.read_model(TRANSFORMER_PATH)
        quantized_model = nncf.quantize(
            model=transformer,
            calibration_dataset=nncf.Dataset(unet_calibration_data),
            subset_size=calibration_dataset_size,
            model_type=nncf.ModelType.TRANSFORMER,
            ignored_scope=nncf.IgnoredScope(names=["__module.model.base_model.model.pos_embed.proj.base_layer/aten::_convolution/Convolution"]),
        )
    
        ov.save_model(quantized_model, TRANSFORMER_INT8_PATH)

Run Weights Compression
~~~~~~~~~~~~~~~~~~~~~~~



Quantizing of the ``Text Encoders`` and ``Autoencoder`` does not
significantly improve inference performance but can lead to a
substantial degradation of accuracy.

For reducing model memory consumption we will use weights compression.
The `Weights
Compression <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__
algorithm is aimed at compressing the weights of the models and can be
used to optimize the model footprint and performance of large models
where the size of weights is relatively larger than the size of
activations, for example, Large Language Models (LLM). Compared to INT8
compression, INT4 compression improves performance even more, but
introduces a minor drop in prediction quality.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from sd3_quantization_helper import compress_models
    
    compress_models()


.. parsed-literal::

    Compressed text_encoder can be found in stable-diffusion-3/text_encoder_int4.xml
    Compressed text_encoder_2 can be found in stable-diffusion-3/text_encoder_2_int4.xml
    Compressed vae_decoder can be found in stable-diffusion-3/vae_decoder_int4.xml


Let’s compare the images generated by the original and optimized
pipelines.

.. code:: ipython3

    %%skip not $to_quantize.value
    optimized_pipe = init_pipeline(opt_models_dict, device.value, use_flash_lora.value)


.. parsed-literal::

    Models compilation
    transformer - Done!
    text_encoder - Done!
    text_encoder_2 - Done!
    vae - Done!


.. code:: ipython3

    %%skip not $to_quantize.value
    
    from sd3_quantization_helper import visualize_results
    
    opt_image = optimized_pipe(
        "A raccoon trapped inside a glass jar full of colorful candies, the background is steamy with vivid colors",
        negative_prompt="",
        num_inference_steps=28 if not use_flash_lora.value else 4,
        guidance_scale=5 if not use_flash_lora.value else 0,
        height=512,
        width=512,
        generator=torch.Generator().manual_seed(141),
    ).images[0]
    
    visualize_results(image, opt_image)



.. parsed-literal::

      0%|          | 0/4 [00:00<?, ?it/s]



.. image:: stable-diffusion-v3-with-output_files/stable-diffusion-v3-with-output_30_1.png


Compare model file sizes
~~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    %%skip not $to_quantize.value
    from sd3_quantization_helper import compare_models_size
    
    del optimized_pipe
    gc.collect()
    
    compare_models_size()


.. parsed-literal::

    transformer compression rate: 1.939
    text_encoder compression rate: 2.714
    text_encoder_2 compression rate: 3.057
    vae_decoder compression rate: 2.007


Compare inference time of the FP16 and optimized pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



To measure the inference performance of the ``FP16`` and optimized
pipelines, we use mean inference time on 5 samples.

   **NOTE**: For the most accurate performance estimation, it is
   recommended to run ``benchmark_app`` in a terminal/command prompt
   after closing other applications.

.. code:: ipython3

    %%skip not $to_quantize.value
    
    from sd3_quantization_helper import compare_perf
    
    compare_perf(models_dict, opt_models_dict, device.value, use_flash_lora.value, validation_size=5)


.. parsed-literal::

    Load FP16 pipeline
    Models compilation
    transformer - Done!
    vae - Done!
    text_encoder - Done!
    text_encoder_2 - Done!
    Load Optimized pipeline
    Models compilation
    transformer - Done!
    text_encoder - Done!
    text_encoder_2 - Done!
    vae - Done!
    Performance speed-up: 1.540


Interactive demo
----------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    from sd3_helper import get_pipeline_selection_option
    
    use_quantized_models = get_pipeline_selection_option(opt_models_dict)
    
    use_quantized_models




.. parsed-literal::

    Checkbox(value=True, description='Use quantized models')



.. code:: ipython3

    from gradio_helper import make_demo
    
    ov_pipe = init_pipeline(models_dict if not use_quantized_models.value else opt_models_dict, device.value, use_flash_lora.value)
    demo = make_demo(ov_pipe, use_flash_lora.value)
    
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    try:
        demo.launch(debug=False)
    except Exception:
        demo.launch(debug=False, share=True)
