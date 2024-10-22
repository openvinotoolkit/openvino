PyTorch Deployment via "torch.compile"
======================================



The ``torch.compile`` feature enables you to use OpenVINO for PyTorch-native applications.
It speeds up PyTorch code by JIT-compiling it into optimized kernels.
By default, Torch code runs in eager-mode, but with the use of ``torch.compile`` it goes through the following steps:

1. **Graph acquisition** - the model is rewritten as blocks of subgraphs that are either:

   * compiled by TorchDynamo and "flattened",
   * falling back to the eager-mode, due to unsupported Python constructs (like control-flow code).

2. **Graph lowering** - all PyTorch operations are decomposed into their constituent kernels specific to the chosen backend.
3. **Graph compilation** - the kernels call their corresponding low-level device-specific operations.



How to Use
####################


.. tab-set::

   .. tab-item:: Image Generation

      .. tab-set::

         .. tab-item:: Stable-Diffusion-2

            .. code-block:: py
               :force:

               import torch
               from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

               model_id = "stabilityai/stable-diffusion-2-1"

               # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
               pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
               pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

               + pipe.text_encoder = torch.compile(pipe.text_encoder, backend="openvino") #Optional
               + pipe.unet = torch.compile(pipe.unet, backend=“openvino”)
               + pipe.vae.decode = torch.compile(pipe.vae.decode, backend=“openvino”) #Optional

               prompt = "a photo of an astronaut riding a horse on mars"
               image = pipe(prompt).images[0]

               image.save("astronaut_rides_horse.png")


         .. tab-item:: Stable-Diffusion-3

            .. code-block:: py

               import torch
               from diffusers import StableDiffusion3Pipeline

               pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float32)

               + pipe.transformer = torch.compile(pipe.transformer, backend="openvino")

               image = pipe(
                   "A cat holding a sign that says hello world",
                   negative_prompt="",
                   num_inference_steps=28,
                   guidance_scale=7.0,
               ).images[0]

               image.save('out.png')

         .. tab-item:: Stable-Diffusion-XL

            .. code-block:: py

               import torch
               from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler

               unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
               pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
               pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

               + pipe.text_encoder = torch.compile(pipe.text_encoder, backend="openvino") #Optional
               + pipe.unet = torch.compile(pipe.unet, backend="openvino")
               + pipe.vae.decode = torch.compile(pipe.vae.decode, backend="openvino") #Optional

               prompt = "a close-up picture of an old man standing in the rain"
               image = pipe(prompt, num_inference_steps=5, guidance_scale=8.0).images[0]
               image.save("result.png")

   .. tab-item:: Text Generation

      .. tab-set::

         .. tab-item:: Llama-3.2-1B

            .. code-block:: py

               import torch
               from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

               model_name_or_path = "meta-llama/Llama-3.2-1B-Instruct"
               tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float32)
               model = AutoModelForCausalLM.from_pretrained(
                   model_name_or_path,
                   trust_remote_code=True,
                   device_map='cpu',
                   torch_dtype=torch.float32
               )

               prompt = "Tell me about AI"

               + model.forward = torch.compile(model.forward, backend="openvino", options={'aot_autograd': True})

               pipe = pipeline(
                   "text-generation",
                   model=model,
                   tokenizer=tokenizer,
                   max_new_tokens=64
               )
               result = pipe(prompt)
               print(result[0]['generated_text'])


         .. tab-item:: Llama-2-7B-GPTQ

            .. code-block:: py

               import torch
               from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

               model_name_or_path = "TheBloke/Llama-2-7B-GPTQ"
               tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.float32)
               model = AutoModelForCausalLM.from_pretrained(
                   model_name_or_path,
                   trust_remote_code=True,
                   device_map='cpu',
                   torch_dtype=torch.float32
               )

               prompt = "Tell me about AI"

               + model.forward = torch.compile(model.forward, backend="openvino", options={'aot_autograd': True})

               pipe = pipeline(
                   "text-generation",
                   model=model,
                   tokenizer=tokenizer,
                   max_new_tokens=64
               )
               result = pipe(prompt)
               print(result[0]['generated_text'])


         .. tab-item:: Chatglm-4-GPTQ

            .. code-block:: py

               import torch
               from transformers import AutoModelForCausalLM, AutoTokenizer

               query = "tell me about AI“

               tokenizer = AutoTokenizer.from_pretrained("mcavus/glm-4v-9b-gptq-4bit-dynamo", trust_remote_code=True)
               inputs = tokenizer.apply_chat_template([{"role": "user", "content": query}],
                                                      add_generation_prompt=True,
                                                      tokenize=True,
                                                      return_tensors="pt",
                                                      return_dict=True
                                                      )
               model = AutoModelForCausalLM.from_pretrained(
                   "mcavus/glm-4v-9b-gptq-4bit-dynamo",
                   torch_dtype=torch.float32,
                   low_cpu_mem_usage=True,
                   trust_remote_code=True
               )

               + model.transformer.encoder.forward = torch.compile(model.transformer.encoder.forward, backend="openvino", options={"aot_autograd":True})

               gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
               with torch.no_grad():
                   outputs = model.generate(**inputs, **gen_kwargs)
                   outputs = outputs[:, inputs['input_ids'].shape[1]:]
                   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
















To use ``torch.compile``, you need to define the ``openvino`` backend in your PyTorch application.
This way Torch FX subgraphs will be directly converted to OpenVINO representation without
any additional PyTorch-based tracing/scripting.
This approach works only for the **package distributed via pip**, as it is now configured with
`torch_dynamo_backends entrypoint <https://pytorch.org/docs/stable/torch.compiler_custom_backends.html#registering-custom-backends>`__.

.. code-block:: python

   ...
   model = torch.compile(model, backend='openvino')
   ...

For OpenVINO installed via channels other than pip, such as conda, and versions older than
2024.1, an additional import statement is needed:

.. code-block:: python

   import openvino.torch

   ...
   model = torch.compile(model, backend='openvino')
   ...



.. image:: ../assets/images/torch_compile_backend_openvino.svg
   :alt: torch.compile execution diagram
   :width: 992px
   :height: 720px
   :scale: 60%
   :align: center

Options
++++++++++++++++++++

It is possible to use additional arguments for ``torch.compile`` to set the backend device,
enable model caching, set the cache directory etc. You can use a dictionary of the available options:

* ``device`` - enables selecting a specific hardware device to run the application.
  By default, the OpenVINO backend for ``torch.compile`` runs PyTorch applications
  on CPU. If you set this variable to ``GPU.0``, for example, the application will
  use the integrated graphics processor instead.
* ``aot_autograd`` - enables aot_autograd graph capture. The aot_autograd graph capture
  is needed to enable dynamic shapes or to finetune a model. For models with dynamic
  shapes, it is recommended to set this option to ``True``. By default, aot_autograd
  is set to ``False``.
* ``model_caching`` - enables saving the optimized model files to a hard drive,
  after the first application run. This makes them available for the following
  application executions, reducing the first-inference latency. By default, this
  variable is set to ``False``. Set it to ``True`` to enable caching.
* ``cache_dir`` - enables defining a custom directory for the model files (if
  ``model_caching`` is set to ``True``). By default, the OpenVINO IR is saved
  in the cache sub-directory, created in the application's root directory.
* ``decompositions`` - enables defining additional operator decompositions. By
  default, this is an empty list. For example, to add a decomposition for
  an operator ``my_op``, add ``'decompositions': [torch.ops.aten.my_op.default]``
  to the options.
* ``disabled_ops`` - enables specifying operators that can be disabled from
  openvino execution and make it fall back to native PyTorch runtime. For
  example, to disable an operator ``my_op`` from OpenVINO execution, add
  ``'disabled_ops': [torch.ops.aten.my_op.default]`` to the options. By
  default, this is an empty list.
* ``config`` - enables passing any OpenVINO configuration option as a dictionary
  to this variable. For details on the various options, refer to the
  :ref:`OpenVINO Advanced Features <openvino-advanced-features>`.

See the example below for details:

.. code-block:: python

   model = torch.compile(model, backend="openvino", options = {"device" : "CPU", "model_caching" : True, "cache_dir": "./model_cache"})

You can also set OpenVINO specific configuration options by adding them as a dictionary under ``config`` key in ``options``:

.. code-block:: python

   opts = {"device" : "CPU", "config" : {"PERFORMANCE_HINT" : "LATENCY"}}
   model = torch.compile(model, backend="openvino", options=opts)


Windows support
+++++++++++++++++++++

PyTorch supports ``torch.compile`` officially on Windows from version 2.3.0 onwards.

For PyTorch versions below 2.3.0, the ``torch.compile`` feature is not supported on Windows
officially. However, it can be accessed by running the following instructions:

1. Install the PyTorch nightly wheel file - `2.1.0.dev20230713 <https://download.pytorch.org/whl/nightly/cpu/torch-2.1.0.dev20230713%2Bcpu-cp38-cp38-win_amd64.whl>`__ ,
2. Update the file at ``<python_env_root>/Lib/site-packages/torch/_dynamo/eval_frames.py``
3. Find the function called ``check_if_dynamo_supported()``:

   .. code-block:: console

      def check_if_dynamo_supported():
          if sys.platform == "win32":
              raise RuntimeError("Windows not yet supported for torch.compile")
          if sys.version_info >= (3, 11):
              raise RuntimeError("Python 3.11+ not yet supported for torch.compile")

4. Put in comments the first two lines in this function, so it looks like this:

   .. code-block:: console

      def check_if_dynamo_supported():
       #if sys.platform == "win32":
       #    raise RuntimeError("Windows not yet supported for torch.compile")
       if sys.version_info >= (3, 11):
           `raise RuntimeError("Python 3.11+ not yet supported for torch.compile")

Support for PyTorch 2 export quantization (Preview)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PyTorch 2 export quantization is supported by OpenVINO backend in ``torch.compile``. To be able
to access this feature, follow the steps provided in
`PyTorch 2 Export Post Training Quantization with X86 Backend through Inductor <https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_x86_inductor.html>`__
and update the provided sample as explained below.

1. If you are using PyTorch version 2.3.0 or later, disable constant folding in quantization to
   be able to benefit from the optimization in the OpenVINO backend. This can be done by passing
   ``fold_quantize=False`` parameter into the ``convert_pt2e`` function. To do so, change this
   line:

   .. code-block:: python

      converted_model = convert_pt2e(prepared_model)

   to the following:

   .. code-block:: python

      converted_model = convert_pt2e(prepared_model, fold_quantize=False)

2. Set ``torch.compile`` backend as OpenVINO and execute the model.

   Update this line below:

   .. code-block:: python

      optimized_model = torch.compile(converted_model)

   As below:

   .. code-block:: python

      optimized_model = torch.compile(converted_model, backend="openvino")

TorchServe Integration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

TorchServe is a performant, flexible, and easy to use tool for serving PyTorch models in production. For more information on the details of TorchServe,
you can refer to `TorchServe github repository. <https://github.com/pytorch/serve>`__. With OpenVINO ``torch.compile`` integration into TorchServe you can serve
PyTorch models in production and accelerate them with OpenVINO on various Intel hardware. Detailed instructions on how to use OpenVINO with TorchServe are
available in `TorchServe examples. <https://github.com/pytorch/serve/tree/master/examples/pt2/torch_compile_openvino>`__

Support for Automatic1111 Stable Diffusion WebUI
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Automatic1111 Stable Diffusion WebUI is an open-source repository that hosts a browser-based interface for the Stable Diffusion
based image generation. It allows users to create realistic and creative images from text prompts.
Stable Diffusion WebUI is supported on Intel CPUs, Intel integrated GPUs, and Intel discrete GPUs by leveraging OpenVINO
``torch.compile`` capability. Detailed instructions are available in
`Stable Diffusion WebUI repository. <https://github.com/openvinotoolkit/stable-diffusion-webui/wiki/Installation-on-Intel-Silicon>`__


Architecture
#################

The ``torch.compile`` feature is part of PyTorch 2.0, and is based on:

* **TorchDynamo** - a Python-level JIT that hooks into the frame evaluation API in CPython,
  (PEP 523) to dynamically modify Python bytecode right before it is executed (PyTorch operators
  that cannot be extracted to FX graph are executed in the native Python environment).
  It maintains the eager-mode capabilities using
  `Guards <https://pytorch.org/docs/stable/torch.compiler_guards_overview.html>`__ to ensure the
  generated graphs are valid.

* **AOTAutograd** - generates the backward graph corresponding to the forward graph captured by TorchDynamo.
* **PrimTorch** - decomposes complicated PyTorch operations into simpler and more elementary ops.
* **TorchInductor** - a deep learning compiler that generates fast code for multiple accelerators and backends.


When the PyTorch module is wrapped with ``torch.compile``, TorchDynamo traces the module and
rewrites Python bytecode to extract sequences of PyTorch operations into an FX Graph,
which can be optimized by the OpenVINO backend. The Torch FX graphs are first converted to
inlined FX graphs and the graph partitioning module traverses inlined FX graph to identify
operators supported by OpenVINO.

All the supported operators are clustered into OpenVINO submodules, converted to the OpenVINO
graph using OpenVINO's PyTorch decoder, and executed in an optimized manner using OpenVINO runtime.
All unsupported operators fall back to the native PyTorch runtime on CPU. If the subgraph
fails during OpenVINO conversion, the subgraph falls back to PyTorch's default inductor backend.



Additional Resources
############################

* `PyTorch 2.0 documentation <https://pytorch.org/docs/stable/index.html>`_

