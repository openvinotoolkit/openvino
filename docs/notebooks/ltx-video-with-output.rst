LTX Video and OpenVINO™
=======================

`LTX-Video <https://github.com/Lightricks/LTX-Video>`__ is a
transformer-based latent diffusion model that adopts a holistic approach
to video generation by seamlessly integrating the responsibilities of
the Video-VAE and the denoising transformer. Unlike existing methods,
which treat these components as independent, LTX-Video aims to optimize
their interaction for improved efficiency and quality. At its core is a
carefully designed Video-VAE that achieves a high compression ratio of
1:192, with spatiotemporal downscaling of 32×32×8 pixels per token,
enabled by relocating the patchifying operation from the transformer’s
input to the VAE’s input. Operating in this highly compressed latent
space enables the transformer to efficiently perform full spatiotemporal
selfattention, which is essential for generating high-resolution videos
with temporal consistency. However, the high compression inherently
limits the representation of fine details. To address this, this VAE
decoder is tasked with both latent-to-pixel conversion and the final
denoising step, producing the clean result directly in pixel space. This
approach preserves the ability to generate fine details without
incurring the runtime cost of a separate upsampling module. The model
supports diverse use cases, including text-to-video and image-to-video
generation, with both capabilities trained simultaneously.

In this example we show how to convert text-to-video pipeline in
OpenVINO format and run inference. For reducing memory consumption,
weights compression optimization can be applied using
`NNCF <https://github.com/openvinotoolkit/nncf>`__.

+-----------------+-----------------+-----------------+-----------------+
| |example1|\     | |example2|\     | |example3|\     | |example4|\     |
+-----------------+-----------------+-----------------+-----------------+
| |example5|\     | |example6|\     | |example7|\     | |example8|\     |
+-----------------+-----------------+-----------------+-----------------+
| |example9|\     | |example10|\    | |example11|\    | |example12|\    |
+-----------------+-----------------+-----------------+-----------------+
| |example13|\    | |example14|\    | |example15|\    | |example16|\    |
+-----------------+-----------------+-----------------+-----------------+


**Table of contents:**


-  `Prerequisites <#prerequisites>`__
-  `Load and run the original model <#load-the-original-model>`__
-  `Convert the model to OpenVINO
   IR <#convert-the-model-to-openvino-ir>`__
-  `Compiling the model <#compiling-the-model>`__
-  `Run OpenVINO model inference <#run-openvino-model-inference>`__
-  `Interactive inference <#interactive-inference>`__

Installation Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~

This is a self-contained example that relies solely on its own code.

We recommend running the notebook in a virtual environment. You only
need a Jupyter server to start. For details, please refer to
`Installation
Guide <https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/README.md#-installation-guide>`__.

.. |example1| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main/docs/_static/ltx-video_example_00001.gif
.. |example2| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00002.gif
.. |example3| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00003.gif
.. |example4| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00004.gif
.. |example5| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00005.gif
.. |example6| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00006.gif
.. |example7| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00007.gif
.. |example8| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00008.gif
.. |example9| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00009.gif
.. |example10| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00010.gif
.. |example11| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00011.gif
.. |example12| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00012.gif
.. |example13| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00013.gif
.. |example14| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00014.gif
.. |example15| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00015.gif
.. |example16| image:: https://media.githubusercontent.com/media/Lightricks/LTX-Video/refs/heads/main//docs/_static/ltx-video_example_00016.gif

Prerequisites
-------------



.. code:: ipython3

    from pathlib import Path
    import requests


    if not Path("notebook_utils.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/utils/notebook_utils.py",
        )
        open("notebook_utils.py", "w").write(r.text)


    if not Path("ov_ltx_video_helper.py").exists():
        r = requests.get(
            url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/ltx-video/ov_ltx_video_helper.py",
        )
        open("ov_ltx_video_helper.py", "w").write(r.text)


    %pip install -qU "torch>=2.1.0" "torchvision>=0.16" "diffusers>=0.28.2" "transformers>=4.44.2" "sentencepiece>=0.1.96" "huggingface-hub~=0.25.2" "einops" "accelerate" "matplotlib" "imageio[ffmpeg]" "nncf>=2.14" "gradio>=4.26" --extra-index-url https://download.pytorch.org/whl/cpu
    %pip install --pre -qU "openvino>=2024.6.0" --extra-index-url https://storage.openvinotoolkit.org/simple/wheels/nightly

    # Read more about telemetry collection at https://github.com/openvinotoolkit/openvino_notebooks?tab=readme-ov-file#-telemetry
    from notebook_utils import collect_telemetry

    collect_telemetry("ltx-video.ipynb")

Load the original model
~~~~~~~~~~~~~~~~~~~~~~~



.. code:: ipython3

    import torch
    from diffusers import LTXPipeline
    from diffusers.utils import export_to_video

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video")



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/5 [00:00<?, ?it/s]



.. parsed-literal::

    Loading checkpoint shards:   0%|          | 0/4 [00:00<?, ?it/s]


Convert the model to OpenVINO IR
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



OpenVINO supports PyTorch models via conversion to OpenVINO Intermediate
Representation (IR). `OpenVINO model conversion
API <https://docs.openvino.ai/2024/openvino-workflow/model-preparation.html#convert-a-model-with-python-convert-model>`__
should be used for these purposes. ``ov.convert_model`` function accepts
original PyTorch model instance and example input for tracing and
returns ``ov.Model`` representing this model in OpenVINO framework.
Converted model can be used for saving on disk using ``ov.save_model``
function or directly loading on device using ``core.complie_model``.

``ov_ltx_video_helper.py`` script contains helper function for models
downloading and models conversion, please check its content if you
interested in conversion details. Note that we delete the original
models after conversion from pipeline to free memory.

LTX Video text-to-video pipeline consists of 3 models: ``Text Encoder``
converts input text into embeddings, ``Transformer`` processes these
embeddings to generate latents from noise step by step, ``VAEDecoder``
performs the last denoising step in conjunction with converting latents
to pixels.

For reducing memory consumption, weights compression optimization can be
applied using `NNCF <https://github.com/openvinotoolkit/nncf>`__. Weight
compression aims to reduce the memory footprint of a model. models,
which require extensive memory to store the weights during inference,
can benefit from weight compression in the following ways:

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
in the case of weights compression which leads to a better accuracy. In
addition, weight compression is data-free and does not require a
calibration dataset, making it easy to use.

``nncf.compress_weights`` function can be used for performing weights
compression. The function accepts an OpenVINO model and other
compression parameters.

More details about weights compression can be found in `OpenVINO
documentation <https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/weight-compression.html>`__.

.. code:: ipython3

    import ipywidgets as widgets


    to_compress_weights = widgets.Checkbox(
        value=True,
        description="Apply Weight Compression",
        disabled=False,
    )

    to_compress_weights

.. code:: ipython3

    from ov_ltx_video_helper import convert_text_encoder, convert_transformer, convert_vae_decoder


    # Uncomment line below to see model conversion code (replace to convert_transformer and convert_vae_decoder to see them too)
    # convert_text_encoder??

.. code:: ipython3

    import gc


    TEXT_ENCODER_PATH = Path("models/text_encoder_ir.xml")
    TRANSFORMER_OV_PATH = Path("models/transformer_ir.xml")
    VAE_DECODER_PATH = Path("models/vae_ir.xml")

    text_encoder_dtype = pipe.text_encoder.dtype
    transformer_config = pipe.transformer.config
    vae_config = pipe.vae.config
    vae_latents_mean = pipe.vae.latents_mean
    vae_latents_std = pipe.vae.latents_std

    convert_text_encoder(pipe.text_encoder, TEXT_ENCODER_PATH, to_compress_weights.value)
    del pipe.text_encoder
    convert_transformer(pipe.transformer, TRANSFORMER_OV_PATH, to_compress_weights.value)
    del pipe.transformer
    convert_vae_decoder(pipe.vae, VAE_DECODER_PATH, to_compress_weights.value)
    del pipe.vae
    gc.collect()


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, openvino
    ⌛ text_encoder conversion started


.. parsed-literal::

    `loss_type=None` was set in the config but it is unrecognised.Using the default loss: `ForCausalLMLoss`.
    /usr/lib/python3.10/importlib/util.py:247: DeprecationWarning: The `openvino.runtime` module is deprecated and will be removed in the 2026.0 release. Please replace `openvino.runtime` with `openvino`.
      self.__spec__.loader.exec_module(self)


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_asym                 │ 100% (170 / 170)            │ 100% (170 / 170)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ text_encoder model conversion finished
    ⌛ transformer conversion started


.. parsed-literal::

    /home/maleksandr/test_notebooks/ltx-video/openvino_notebooks/notebooks/venv-ltx/lib/python3.10/site-packages/diffusers/models/attention_processor.py:711: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if current_length != target_length:
    /home/maleksandr/test_notebooks/ltx-video/openvino_notebooks/notebooks/venv-ltx/lib/python3.10/site-packages/diffusers/models/attention_processor.py:726: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attention_mask.shape[0] < batch_size * head_size:


.. parsed-literal::

    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_asym                 │ 100% (287 / 287)            │ 100% (287 / 287)                       │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ transformer model conversion finished
    ⌛ vae conversion started
    INFO:nncf:Statistics of the bitwidth distribution:
    ┍━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┯━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┑
    │ Weight compression mode   │ % all parameters (layers)   │ % ratio-defining parameters (layers)   │
    ┝━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┿━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┥
    │ int8_asym                 │ 100% (45 / 45)              │ 100% (45 / 45)                         │
    ┕━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┷━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┙



.. parsed-literal::

    Output()









.. parsed-literal::

    ✅ vae model conversion finished


Compiling the model
~~~~~~~~~~~~~~~~~~~



Select device from dropdown list for running inference using OpenVINO.

.. code:: ipython3

    import openvino as ov

    from notebook_utils import device_widget


    core = ov.Core()

    device = device_widget()

    device




.. parsed-literal::

    Dropdown(description='Device:', index=1, options=('CPU', 'AUTO'), value='AUTO')



``ov_catvton_helper.py`` provides wrapper classes that wrap the compiled
models to keep the original interface. Note that all of wrapper classes
return ``torch.Tensor``\ s instead of ``np.array``\ s. Then we insert
wrappers instances in the pipeline.

.. code:: ipython3

    from ov_ltx_video_helper import TextEncoderWrapper, ConvTransformerWrapper, VAEWrapper


    # Uncomment the line below to see the wrapper class code (replace to ConvTransformerWrapper and VAEWrapper to see them too)
    # TextEncoderWrapper??

.. code:: ipython3

    compiled_transformer = core.compile_model(TRANSFORMER_OV_PATH, device.value)
    compiled_vae = core.compile_model(VAE_DECODER_PATH, device.value)
    compiled_text_encoder = core.compile_model(TEXT_ENCODER_PATH, device.value)

    pipe.__dict__["_internal_dict"]["_execution_device"] = pipe._execution_device  # this is to avoid some problem that can occur in the pipeline

    pipe.register_modules(
        text_encoder=TextEncoderWrapper(compiled_text_encoder, text_encoder_dtype),
        transformer=ConvTransformerWrapper(compiled_transformer, transformer_config),
        vae=VAEWrapper(compiled_vae, vae_config, vae_latents_mean, vae_latents_std),
    )

Run OpenVINO model inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~



`General
tips <https://huggingface.co/Lightricks/LTX-Video#general-tips>`__:

- The model works on resolutions that are divisible by 32 and number of
  frames that are divisible by 8 + 1 (e.g. 257). In case the input is not
  satisfied to the described conditions, the input will be padded with -1
  and then cropped to the desired resolution and number of frames.
- The model works best on resolutions under 720 x 1280 and number of frames
 below 257.
- Prompts should be in English. The more elaborate the
  better. Good prompt looks like The turquoise waves crash against the
  dark, jagged rocks of the shore, sending white foam spraying into the
  air. The scene is dominated by the stark contrast between the bright
  blue water and the dark, almost black rocks. The water is a clear,
  turquoise color, and the waves are capped with white foam. The rocks are
  dark and jagged, and they are covered in patches of green moss. The
  shore is lined with lush green vegetation, including trees and bushes.
  In the background, there are rolling hills covered in dense forest. The
  sky is cloudy, and the light is dim.

.. code:: ipython3

    prompt = "A clear, turquoise river flows through a rocky canyon, cascading over a small waterfall and forming a pool of water at the bottom.The river is the main focus of the scene, with its clear water reflecting the surrounding trees and rocks. The canyon walls are steep and rocky, with some vegetation growing on them. The trees are mostly pine trees, with their green needles contrasting with the brown and gray rocks. The overall tone of the scene is one of peace and tranquility."
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
    generator = torch.Generator(device="cpu").manual_seed(42)


    video = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=704,
        height=480,
        num_frames=24,
        num_inference_steps=40,
        generator=generator,
    ).frames[0]
    export_to_video(video, "output1_ov.mp4", fps=24)


.. parsed-literal::

    /home/maleksandr/test_notebooks/ltx-video/openvino_notebooks/notebooks/venv-ltx/lib/python3.10/site-packages/diffusers/configuration_utils.py:140: FutureWarning: Accessing config attribute `_execution_device` directly via 'LTXPipeline' object attribute is deprecated. Please access '_execution_device' over 'LTXPipeline's config object instead, e.g. 'scheduler.config._execution_device'.
      deprecate("direct config name access", "1.0.0", deprecation_message, standard_warn=False)



.. parsed-literal::

      0%|          | 0/40 [00:00<?, ?it/s]


.. code:: ipython3

    from IPython.display import Video

    Video("output1_ov.mp4")

Interactive inference
---------------------



Please select below whether you would like to use the quantized models
to launch the interactive demo.

.. code:: ipython3

    import gradio as gr


    def generate(prompt, negative_prompt, width, height, num_frames, num_inference_steps, seed, _=gr.Progress(track_tqdm=True)):
        generator = torch.Generator().manual_seed(seed)
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]
        file_name = "output.mp4"
        export_to_video(video, file_name, fps=24)
        return file_name

.. code:: ipython3

    if not Path("gradio_helper.py").exists():
        r = requests.get(url="https://raw.githubusercontent.com/openvinotoolkit/openvino_notebooks/latest/notebooks/ltx-video/gradio_helper.py")
        open("gradio_helper.py", "w").write(r.text)

    from gradio_helper import make_demo

    demo = make_demo(fn=generate)

    try:
        demo.queue().launch(debug=True)
    except Exception:
        demo.queue().launch(share=True, debug=True)
    # If you are launching remotely, specify server_name and server_port
    # EXAMPLE: `demo.launch(server_name='your server name', server_port='server port in int')`
    # To learn more please refer to the Gradio docs: https://gradio.app/docs/
