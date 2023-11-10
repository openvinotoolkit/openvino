Stable Diffusion v2.1 using Optimum-Intel OpenVINO and multiple Intel Hardware
==============================================================================

This notebook will provide you a way to see different precision models
performing in different hardware. This notebook was done for showing
case the use of Optimum-Intel-OpenVINO and it is not optimized for
running multiple times.

|image0|

**Table of contents:**


-  `Showing Info Available
   Devices <#showing-info-available-devices>`__
-  `Using full precision model in CPU with
   ``StableDiffusionPipeline`` <#using-full-precision-model-in-cpu-with-stablediffusionpipeline>`__
-  `Using full precision model in CPU with
   ``OVStableDiffusionPipeline`` <#using-full-precision-model-in-cpu-with-ovstablediffusionpipeline>`__
-  `Using full precision model in dGPU with
   ``OVStableDiffusionPipeline`` <#using-full-precision-model-in-dgpu-with-ovstablediffusionpipeline>`__

.. |image0| image:: https://github.com/openvinotoolkit/openvino_notebooks/assets/10940214/1858dae4-72fd-401e-b055-66d503d82446

Optimum Intel is the interface between the Transformers and Diffusers
libraries and the different tools and libraries provided by Intel to
accelerate end-to-end pipelines on Intel architectures. More details in
this
`repository <https://github.com/huggingface/optimum-intel#openvino>`__.

``Note: We suggest you to create a different environment and run the following installation command there.``

.. code:: ipython3

    %pip install -q "optimum-intel[openvino,diffusers]" "ipywidgets" "transformers >= 4.31"

.. code:: ipython3

    import warnings
    warnings.filterwarnings('ignore')

Showing Info Available Devices 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``available_devices`` property shows the available devices in your
system. The “FULL_DEVICE_NAME” option to ``ie.get_property()`` shows the
name of the device. Check what is the ID name for the discrete GPU, if
you have integrated GPU (iGPU) and discrete GPU (dGPU), it will show
``device_name="GPU.0"`` for iGPU and ``device_name="GPU.1"`` for dGPU.
If you just have either an iGPU or dGPU that will be assigned to
``"GPU"``

Note: For more details about GPU with OpenVINO visit this
`link <https://docs.openvino.ai/nightly/openvino_docs_install_guides_configurations_for_intel_gpu.html>`__.
If you have been facing any issue in Ubuntu 20.04 or Windows 11 read
this
`blog <https://blog.openvino.ai/blog-posts/install-gpu-drivers-windows-ubuntu>`__.

.. code:: ipython3

    from openvino.runtime import Core
    
    ie = Core()
    devices = ie.available_devices
    
    for device in devices:
        device_name = ie.get_property(device, "FULL_DEVICE_NAME")
        print(f"{device}: {device_name}")


.. parsed-literal::

    CPU: Intel(R) Xeon(R) Gold 6348 CPU @ 2.60GHz
    GPU: Intel(R) Data Center GPU Flex 170 (dGPU)


Using full precision model in CPU with ``StableDiffusionPipeline`` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from diffusers import StableDiffusionPipeline
    
    import gc
    
    model_id = "stabilityai/stable-diffusion-2-1-base"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe.save_pretrained("./stabilityai_cpu")
    prompt = "red car in snowy forest"
    output_cpu = pipe(prompt, num_inference_steps=17).images[0]
    output_cpu.save("image_cpu.png")
    output_cpu
    
    del pipe
    gc.collect()



.. parsed-literal::

    Fetching 13 files:   0%|          | 0/13 [00:00<?, ?it/s]



.. parsed-literal::

    Downloading model.safetensors:   0%|          | 0.00/1.36G [00:00<?, ?B/s]



.. parsed-literal::

    Downloading (…)ch_model.safetensors:   0%|          | 0.00/335M [00:00<?, ?B/s]



.. parsed-literal::

    Downloading (…)ch_model.safetensors:   0%|          | 0.00/3.46G [00:00<?, ?B/s]



.. parsed-literal::

      0%|          | 0/17 [00:00<?, ?it/s]




.. image:: 236-stable-diffusion-v2-optimum-demo-comparison-with-output_files/236-stable-diffusion-v2-optimum-demo-comparison-with-output_7_5.png



Using full precision model in CPU with ``OVStableDiffusionPipeline`` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    from optimum.intel.openvino import OVStableDiffusionPipeline
    
    model_id = "stabilityai/stable-diffusion-2-1-base"
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True, compile=False)
    ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)
    ov_pipe.save_pretrained("./openvino_ir")
    ov_pipe.compile()



.. parsed-literal::

    Framework not specified. Using pt to export to ONNX.
    Keyword arguments {'subfolder': '', 'config': {'_class_name': 'StableDiffusionPipeline', '_diffusers_version': '0.10.0.dev0', 'feature_extractor': ['transformers', 'CLIPImageProcessor'], 'requires_safety_checker': False, 'safety_checker': [None, None], 'scheduler': ['diffusers', 'PNDMScheduler'], 'text_encoder': ['transformers', 'CLIPTextModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'], 'unet': ['diffusers', 'UNet2DConditionModel'], 'vae': ['diffusers', 'AutoencoderKL']}} are not expected by StableDiffusionPipeline and will be ignored.
    Using framework PyTorch: 2.0.1+cu117


.. parsed-literal::

    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
    


.. parsed-literal::

    Using framework PyTorch: 2.0.1+cu117
    Saving external data to one file...


.. parsed-literal::

    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
    


.. parsed-literal::

    Using framework PyTorch: 2.0.1+cu117
    Using framework PyTorch: 2.0.1+cu117


.. parsed-literal::

    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
    
    ============= Diagnostic Run torch.onnx.export version 2.0.1+cu117 =============
    verbose: False, log level: Level.ERROR
    ======================= 0 NONE 0 NOTE 0 WARNING 0 ERROR ========================
    


.. parsed-literal::

    Compiling the text_encoder...
    Compiling the vae_decoder...
    Compiling the unet...


.. code:: ipython3

    prompt = "red car in snowy forest"
    output_cpu_ov = ov_pipe(prompt, num_inference_steps=17).images[0]
    output_cpu_ov.save("image_ov_cpu.png")
    output_cpu_ov



.. parsed-literal::

      0%|          | 0/18 [00:00<?, ?it/s]




.. image:: 236-stable-diffusion-v2-optimum-demo-comparison-with-output_files/236-stable-diffusion-v2-optimum-demo-comparison-with-output_10_1.png



Using full precision model in dGPU with ``OVStableDiffusionPipeline`` 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model in this notebook is FP32 precision. And thanks to the new
feature of OpenVINO 2023.0 you do not need to convert the model to FP16
for running the inference on GPU.

.. code:: ipython3

    ov_pipe.to("GPU")
    ov_pipe.compile()


.. parsed-literal::

    Compiling the text_encoder...
    Compiling the vae_decoder...
    Compiling the unet...


.. code:: ipython3

    prompt = "red car in snowy forest"
    output_gpu_ov = ov_pipe(prompt, num_inference_steps=17).images[0]
    output_gpu_ov.save("image_ov_gpu.png")
    output_gpu_ov
    
    del ov_pipe
    gc.collect()


