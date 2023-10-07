# Optimize and Deploy Generative AI Models {#gen_ai_guide}

@sphinxdirective

Generative AI is an innovative technique that creates new data, such as text, images, video, or audio, using neural networks. OpenVINO accelerates Generative AI use cases as they mostly rely on model inference, allowing for faster development and better performance. When it comes to generative models, OpenVINO supports:

* Conversion, optimization and inference for Text, Image and Audio generative models (for example, llama2, MPT, OPT, Stable Diffusion, Stable Diffusion XL, etc.) 
* Int8 weights compression for text generation models 
* Reduced storage format (fp16 precision for non-compressed models and int8 for compressed models) 
* Inference on CPU and GPU platforms, including Intel integrated and discrete ARC GPUs 
 

There are two main paths how OpenVINO could be used to run Generative AI use cases:

* As a backend for Hugging Face frameworks (transformers, diffusers) with use of `Optimum-Intel <https://huggingface.co/docs/optimum/intel/inference>`__ extension 
* With use of OpenVINO native APIs (Python and C++) with use of custom pipeline code 

 
In both cases, OpenVINO runtime and tools will be used, difference is mostly in API that you prefer and footprint of final solution. Use of native APIs allows using generative models in C++ applications, having minimal runtime dependencies and minimizing application footprint. Approach with OpenVINO Native APIs will require implementation of glue code (generation loop, text tokenization or scheduler functions) which is hidden as implementation detail internally within Hugging Face libraries. 

We recommend starting with use of Hugging Face frameworks, initially experimenting with different models and scenarios, ding your fit and then convert to OpenVINO native APIs if there is a demand.  

Within Optimum-Intel product there are interfaces that allow to optimize (compress weights) model with use of `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__ and export model to OpenVINO model format to be used in native API applications. 

The table below summarizes differences of both Hugging Face and Native approaches. 

.. list-table::
   :widths: 20 25 55
   :header-rows: 1

   * -  
     - Hugging Face through OpenVINO
     - OpenVINO Native API
   * - Model support
     - Wide set of Models
     - Wide set of Models
   * - APIs
     - Python (Hugging Face API)
     - Python, C++ (OpenVINO API)
   * - Model Format
     - Source Framework / OpenVINO
     - OpenVINO
   * - Inference code
     - Hugging Face based
     - Custom inference pipelines
   * - Additional dependencies
     - As much as Hugging Face has
     - Ligthweight (e.g. numpy, boost, etc.)
   * - Application footprint
     - Large
     - Small
   * - Pre/post-processing and glue code
     - Available at Hugging Face OOB
     - OpenVINO samples and notebooks
   * - Performance
     - Good
     - Best


Running Generative AI models using Hugging Face Optimium-Intel 
##############################################################

Prerequisites
+++++++++++++++++++++++++++

* Create a Python environment
* Install Optimium-Intel running:

.. code-block:: console

    pip install optimum[openvino,nncf]


To start using OpenVINO as a backend for Hugging Face you should change original Hugging Face code in two places. 

.. code-block:: diff

    -from transformers import AutoModelForCausalLM
    +from optimum.intel. import OVModelForCausalLM

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    -model = AutoModelForCausalLM.from_pretrained(model_id)
    +model = OVModelForCausalLM.from_pretrained(model_id, export=True)


After that, you can call ``.save_pretrained()`` method to save model to the folder in the OpenVINO Intermediate Representation and use it further.

.. code-block:: python

    model.save_pretrained(model_dir)


Alternatively, you can download and convert model using CLI interface: ``optimum-cli export openvino --model meta-llama/Llama-2-7b-chat-hf llama_openvino``
In this case, you can load the converted model in OpenVINO representation directly from the disk:

.. code-block:: python

    model_id = "llama_openvino"
    model = OVModelForCausalLM.from_pretrained(model_id)


By default, inference will run on CPU. To select a different inference device, for example GPU, add `device="GPU"` to the `.from_pretrained()` call. To switch to a different device after the model has been loaded, use the `.to()` method. The device naming convention is the same as in OpenVINO native API:

.. code-block:: python

    model.to("GPU")


Optimum-Intel API also provides out-of-the-box model optimization through weights compression using NNCF which substantially reduces the model footdpring and inference latency:

.. code-block:: python

    model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)


Wieght compression is applied by default to models larger than 1 billion parameters and also available for CLI interface as ``--int8`` option.

Working with models tuned with LoRA
++++++++++++++++++++++++++++++++++++

Low-ranking Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream task or custom data. But it requires some extra steps to be done for efficient deployment using the Hugging Face API. Namely, the trained adapters should be fused into the baseline model to avoid extra computation. This is how it can be done for LLMs:

.. code-block:: python

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    lora_adaptor = "./lora_adaptor"

    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
    model.merge_and_unload()
    model.get_base_model().save_pretrained("fused_lora_model")


Now the model can be converted to OpenVINO using Optimum-Intel Python API or CLI interfaces mentioned above.

Running Generative AI models using native OpenVINO APIs 
########################################################

To run Generative AI models using native OpenVINO APIs you need to follow regular **Сonvert -> Optimize -> Deploy** path with few simplifications. 

To convert model from Hugging Face you can use Optimum-Intel export feature that allows to export model in OpenVINO format without invoking conversion API and tools directly, as it is shown above. In this case, the conversion process is a bit more simplified. You can still use regular conversion path if model comes from outside of Hugging Face ecosystem, i.e., in source framework format (PyTorch, etc.) 

Model optimization could be performed within Hugging Face or directly using NNCF as described :doc:`here <weight_compression>`.

Inference code that uses native API cannot benefit from Hugging Face pipelines, hence you need to write your own or take it from the examples available. Below are few examples of popular Generative AI scenarios:

* In case of text Large Language Models (LLMs), this includes tokenization, inference and token selection loop, and de-tokenization. In case if token selection involves beam search, this also needs to be written.  
* For Image generation models you need make a pipeline that includes several model: inference for source (e.g. text) encoder models, inference loop for diffusion process and inference for decoding part. Scheduler code is also required. 

To write such pipelines, we recommend following examples that we have provided as a part of OpenVINO: 

* `llama2.openvino <https://github.com/OpenVINO-dev-contest/llama2.openvino>`__
* `LLM optimization by custom operation embedding for OpenVINO <https://github.com/luo-cheng2021/ov.cpu.llm.experimental>`__
* `C++ Implementation of Stable Diffusion <https://github.com/yangsu2022/OV_SD_CPP>`__


Additional Resources
############################

* `Optimum-Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`_
* `Optimum-Intel GitHub <https://github.com/huggingface/optimum-intel>`_
* :doc:`Weight Compression <weight_compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`_

@endsphinxdirective