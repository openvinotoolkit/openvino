Image generation with Segmind Stable Diffusion 1B (SSD-1B) model and OpenVINO
=============================================================================

The `Segmind Stable Diffusion Model
(SSD-1B) <https://github.com/segmind/SSD-1B?ref=blog.segmind.com>`__ is
a distilled 50% smaller version of the Stable Diffusion XL (SDXL),
offering a 60% speedup while maintaining high-quality text-to-image
generation capabilities. It has been trained on diverse datasets,
including Grit and Midjourney scrape data, to enhance its ability to
create a wide range of visual content based on textual prompts. This
model employs a knowledge distillation strategy, where it leverages the
teachings of several expert models in succession, including SDXL,
ZavyChromaXL, and JuggernautXL, to combine their strengths and produce
impressive visual outputs.

.. figure:: https://user-images.githubusercontent.com/82945616/277419571-a5583e8a-6a05-4680-a540-f80502feed0b.png
   :alt: image

   image

In this tutorial, we consider how to run the SSD-1B model using
OpenVINO.

We will use a pre-trained model from the `Hugging Face
Diffusers <https://huggingface.co/docs/diffusers/index>`__ library. To
simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library is
used to convert the models to OpenVINO™ IR format.

**Table of contents:**


-  `Install Prerequisites <#install-prerequisites>`__
-  `SSD-1B Base model <#ssd-b-base-model>`__
-  `Select inference device SSD-1B Base
   model <#select-inference-device-ssd-b-base-model>`__
-  `Text2image Generation Interactive
   Demo <#textimage-generation-interactive-demo>`__

Install prerequisites 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    %pip install -q "git+https://github.com/huggingface/optimum-intel.git"
    %pip install -q "openvino>=2023.1.0"
    %pip install -q --upgrade-strategy eager "invisible-watermark>=0.2.0" "transformers>=4.33" "accelerate" "onnx" "onnxruntime" safetensors "diffusers>=0.22.0"
    %pip install -q gradio


.. parsed-literal::

    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.
    DEPRECATION: pytorch-lightning 1.6.5 has a non-standard dependency specifier torch>=1.8.*. pip 24.0 will enforce this behaviour change. A possible replacement is to upgrade to a newer version of pytorch-lightning or contact the author to suggest that they release a version with a conforming dependency specifiers. Discussion can be found at https://github.com/pypa/pip/issues/12063
    Note: you may need to restart the kernel to use updated packages.


SSD-1B Base model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We will start with the base model part, which is responsible for the
generation of images of the desired output size.
`SSD-1B <https://huggingface.co/segmind/SSD-1B>`__ is available for
downloading via the `HuggingFace hub <https://huggingface.co/models>`__.
It already provides a ready-to-use model in OpenVINO format compatible
with `Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__.

To load an OpenVINO model and run an inference with OpenVINO Runtime,
you need to replace diffusers ``StableDiffusionXLPipeline`` with Optimum
``OVStableDiffusionXLPipeline``. In case you want to load a PyTorch
model and convert it to the OpenVINO format on the fly, you can set
``export=True``.

You can save the model on disk using the ``save_pretrained`` method.

.. code:: ipython3

    from pathlib import Path
    from optimum.intel.openvino import OVStableDiffusionXLPipeline
    
    
    model_id = "segmind/SSD-1B"
    model_dir = Path("openvino-ssd-1b")


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    2023-12-07 00:09:54.638748: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-12-07 00:09:54.672777: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-12-07 00:09:55.202678: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT


Select inference device SSD-1B Base model 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

select device from dropdown list for running inference using OpenVINO

.. code:: ipython3

    import ipywidgets as widgets
    import openvino as ov
    
    
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

    import gc
    
    
    if not model_dir.exists():
        text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_id, compile=False, device=device.value, export=True)
        text2image_pipe.half()
        text2image_pipe.save_pretrained(model_dir)
        text2image_pipe.compile()
        gc.collect()
    else:
        text2image_pipe = OVStableDiffusionXLPipeline.from_pretrained(model_dir, device=device.value)


.. parsed-literal::

    Framework not specified. Using pt to export to ONNX.
    Keyword arguments {'subfolder': '', 'trust_remote_code': False} are not expected by StableDiffusionXLImg2ImgPipeline and will be ignored.



.. parsed-literal::

    Loading pipeline components...:   0%|          | 0/7 [00:00<?, ?it/s]


.. parsed-literal::

    Using framework PyTorch: 1.13.1+cpu


.. parsed-literal::

    WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.base has been moved to tensorflow.python.trackable.base. The old module will be deleted in version 2.11.


.. parsed-literal::

    [ WARNING ]  Please fix your imports. Module %s has been moved to %s. The old module will be deleted in version %s.
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:66: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if input_shape[-1] > 1 or self.sliding_window is not None:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/modeling_attn_mask_utils.py:137: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if past_key_values_length > 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:273: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:281: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/transformers/models/clip/modeling_clip.py:313: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
    Using framework PyTorch: 1.13.1+cpu
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/unet_2d_condition.py:878: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if dim % default_overall_up_factor != 0:
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/resnet.py:265: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/resnet.py:271: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/resnet.py:173: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      assert hidden_states.shape[1] == self.channels
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/diffusers/models/resnet.py:186: TracerWarning: Converting a tensor to a Python boolean might cause the trace to be incorrect. We can't record the data flow of Python values, so this value will be treated as a constant in the future. This means that the trace might not generalize to other inputs!
      if hidden_states.shape[0] >= 64:
    Using framework PyTorch: 1.13.1+cpu
    Using framework PyTorch: 1.13.1+cpu
    Using framework PyTorch: 1.13.1+cpu
    Compiling the vae_decoder to AUTO ...
    Compiling the unet to AUTO ...
    Compiling the vae_encoder to AUTO ...
    Compiling the text_encoder_2 to AUTO ...
    Compiling the text_encoder to AUTO ...


Run Text2Image generation pipeline 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, we can run the model for the generation of images using text
prompts. To speed up evaluation and reduce the required memory we
decrease ``num_inference_steps`` and image size (using ``height`` and
``width``). You can modify them to suit your needs and depend on the
target hardware. We also specified a ``generator`` parameter based on a
numpy random state with a specific seed for results reproducibility.
>\ **Note**: Generating a default size 1024x1024 image requires about
53GB for the SSD-1B model in case if the converted model is loaded from
disk and up to 64GB RAM for the SDXL model after exporting.

.. code:: ipython3

    prompt = "An astronaut riding a green horse"  # Your prompt here
    neg_prompt = "ugly, blurry, poor quality"  # Negative prompt here
    image = text2image_pipe(prompt=prompt, num_inference_steps=15, negative_prompt=neg_prompt).images[0]
    image


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:565: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)



.. parsed-literal::

      0%|          | 0/15 [00:00<?, ?it/s]


.. parsed-literal::

    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:599: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)
    /opt/home/k8sworker/ci-ai/cibuilds/ov-notebook/OVNotebookOps-561/.workspace/scm/ov-notebook/.venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_diffusion.py:615: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      outputs = self.request(inputs, shared_memory=True)




.. image:: 248-ssd-b1-with-output_files/248-ssd-b1-with-output_9_3.png



Generating a 512x512 image requires about 27GB for the SSD-1B model and
about 42GB RAM for the SDXL model in case if the converted model is
loaded from disk.

.. code:: ipython3

    import numpy as np
    
    
    prompt = "cute cat 4k, high-res, masterpiece, best quality, soft lighting, dynamic angle"
    image = text2image_pipe(prompt, num_inference_steps=15, height=512, width=512, generator=np.random.RandomState(314)).images[0]
    image



.. parsed-literal::

      0%|          | 0/15 [00:00<?, ?it/s]




.. image:: 248-ssd-b1-with-output_files/248-ssd-b1-with-output_11_1.png



Image2Image Generation Interactive Demo 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    import gradio as gr
    import numpy as np
    
    
    prompt = "An astronaut riding a green horse"
    neg_prompt = "ugly, blurry, poor quality"
    
    def generate_from_text(text_promt, neg_prompt, seed, num_steps):
        result = text2image_pipe(text_promt, negative_prompt=neg_prompt, num_inference_steps=num_steps, generator=np.random.RandomState(seed), height=512, width=512).images[0]
        return result
    
    
    with gr.Blocks() as demo:
        with gr.Column():
            positive_input = gr.Textbox(label="Text prompt")
            neg_input = gr.Textbox(label="Negative prompt")
            with gr.Row():
                seed_input = gr.Slider(0, 10_000_000, value=42, label="Seed")
                steps_input = gr.Slider(label="Steps", value=10)
                btn = gr.Button()
            out = gr.Image(label="Result", type="pil", width=512)
            btn.click(generate_from_text, [positive_input, neg_input, seed_input, steps_input], out)
            gr.Examples([
                [prompt, neg_prompt, 999, 20], 
                ["underwater world coral reef, colorful jellyfish, 35mm, cinematic lighting, shallow depth of field,  ultra quality, masterpiece, realistic", neg_prompt, 89, 20],
                ["a photo realistic happy white poodle dog ​​playing in the grass, extremely detailed, high res, 8k, masterpiece, dynamic angle", neg_prompt, 1569, 15],
                ["Astronaut on Mars watching sunset, best quality, cinematic effects,", neg_prompt, 65245, 12],
                ["Black and white street photography of a rainy night in New York, reflections on wet pavement", neg_prompt, 48199, 10]
            ], [positive_input, neg_input, seed_input, steps_input])
    
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



.. .. raw:: html

..    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>

