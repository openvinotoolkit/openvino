Create an LLM-powered Chatbot using OpenVINO
============================================

In the rapidly evolving world of artificial intelligence (AI), chatbots
have emerged as powerful tools for businesses to enhance customer
interactions and streamline operations. Large Language Models (LLMs) are
artificial intelligence systems that can understand and generate human
language. They use deep learning algorithms and massive amounts of data
to learn the nuances of language and produce coherent and relevant
responses. While a decent intent-based chatbot can answer basic,
one-touch inquiries like order management, FAQs, and policy questions,
LLM chatbots can tackle more complex, multi-touch questions. LLM enables
chatbots to provide support in a conversational manner, similar to how
humans do, through contextual memory. Leveraging the capabilities of
Language Models, chatbots are becoming increasingly intelligent, capable
of understanding and responding to human language with remarkable
accuracy.

Previously, we already discussed how to build an instruction-following
pipeline using OpenVINO and Optimum Intel, please check out `Dolly
example <../240-dolly-2-instruction-following>`__ for reference. In this
tutorial, we consider how to use the power of OpenVINO for running Large
Language Models for chat. We will use a pre-trained model from the
`Hugging Face
Transformers <https://huggingface.co/docs/transformers/index>`__
library. To simplify the user experience, the `Hugging Face Optimum
Intel <https://huggingface.co/docs/optimum/intel/index>`__ library is
used to convert the models to OpenVINO™ IR format.

The tutorial consists of the following steps:

-  Install prerequisites
-  Download and convert the model from a public source using the
   `OpenVINO integration with Hugging Face
   Optimum <https://huggingface.co/blog/openvino>`__.
-  Compress model weights to INT8 precision using
   `NNCF <https://github.com/openvinotoolkit/nncf>`__
-  Create a chat inference pipeline
-  Run chat pipeline

**Table of contents:**

- `Prerequisites <#prerequisites>`__
- `Select model for inference <#select-model-for-inference>`__
- `Instantiate Model using Optimum Intel <#instantiate-model-using-optimum-intel>`__
- `Compress model weights <#compress-model-weights>`__
- `Weights Compression using Optimum Intel <#weights-compression-using-optimum-intel>`__
- `Weights Compression using NNCF <#weights-compression-using-nncf->`__
- `Select device for inference and model variant <#select-device-for-inference-and-model-variant->`__
- `Run Chatbot <#run-chatbot>`__

Prerequisites 
-------------------------------------------------------

Install required dependencies

.. code:: ipython3

    %pip install -q --extra-index-url https://download.pytorch.org/whl/cpu\
    "git+https://github.com/huggingface/optimum-intel.git"\
    "nncf>=2.6.0"\
    "gradio"\
    "onnx" "onnxruntime" "einops" "transformers>=4.31.0"\
    "openvino==2023.2.0.dev20230922"

Select model for inference 
--------------------------------------------------------------------

The tutorial supports different models, you can select one from the
provided options to compare the quality of open source LLM solutions.
>\ **Note**: conversion of some models can require additional actions
from user side and at least 64GB RAM for conversion.

The available options are:

-  **red-pajama-3b-chat** - A 2.8B parameter pre-trained language model
   based on GPT-NEOX architecture. It was developed by Together Computer
   and leaders from the open-source AI community. The model is
   fine-tuned on OASST1 and Dolly2 datasets to enhance chatting ability.
   More details about model can be found in `HuggingFace model
   card <https://huggingface.co/togethercomputer/RedPajama-INCITE-Chat-3B-v1>`__.
-  **llama-2-7b-chat** - LLama 2 is the second generation of LLama
   models developed by Meta. Llama 2 is a collection of pre-trained and
   fine-tuned generative text models ranging in scale from 7 billion to
   70 billion parameters. llama-2-7b-chat is 7 billions parameters
   version of LLama 2 finetuned and optimized for dialogue use case.
   More details about model can be found in the
   `paper <https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/>`__,
   `repository <https://github.com/facebookresearch/llama>`__ and
   `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__
   >\ **Note**: run model with demo, you will need to accept license
   agreement. >You must be a registered user in 🤗 Hugging Face Hub.
   Please visit `HuggingFace model
   card <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>`__,
   carefully read terms of usage and click accept button. You will need
   to use an access token for the code below to run. For more
   information on access tokens, refer to `this section of the
   documentation <https://huggingface.co/docs/hub/security-tokens>`__.
   >You can login on Hugging Face Hub in notebook environment, using
   following code:

.. code:: python

       ## login to huggingfacehub to get access to pretrained model 
       from huggingface_hub import notebook_login, whoami

       try:
           whoami()
           print('Authorization token already provided')
       except OSError:
           notebook_login()

-  **mpt-7b-chat** - MPT-7B is part of the family of
   MosaicPretrainedTransformer (MPT) models, which use a modified
   transformer architecture optimized for efficient training and
   inference. These architectural changes include performance-optimized
   layer implementations and the elimination of context length limits by
   replacing positional embeddings with Attention with Linear Biases
   (`ALiBi <https://arxiv.org/abs/2108.12409>`__). Thanks to these
   modifications, MPT models can be trained with high throughput
   efficiency and stable convergence. MPT-7B-chat is a chatbot-like
   model for dialogue generation. It was built by finetuning MPT-7B on
   the
   `ShareGPT-Vicuna <https://huggingface.co/datasets/jeffwan/sharegpt_vicuna>`__,
   `HC3 <https://huggingface.co/datasets/Hello-SimpleAI/HC3>`__,
   `Alpaca <https://huggingface.co/datasets/tatsu-lab/alpaca>`__,
   `HH-RLHF <https://huggingface.co/datasets/Anthropic/hh-rlhf>`__, and
   `Evol-Instruct <https://huggingface.co/datasets/victor123/evol_instruct_70k>`__
   datasets. More details about the model can be found in `blog
   post <https://www.mosaicml.com/blog/mpt-7b>`__,
   `repository <https://github.com/mosaicml/llm-foundry/>`__ and
   `HuggingFace model
   card <https://huggingface.co/mosaicml/mpt-7b-chat>`__.

.. code:: ipython3

    from config import SUPPORTED_MODELS
    import ipywidgets as widgets

.. code:: ipython3

    model_ids = list(SUPPORTED_MODELS)
    
    model_id = widgets.Dropdown(
        options=model_ids,
        value=model_ids[0],
        description='Model:',
        disabled=False,
    )
    
    model_id




.. parsed-literal::

    Dropdown(description='Model:', options=('red-pajama-3b-chat', 'llama-2-chat-7b', 'mpt-7b-chat'), value='red-pa…



.. code:: ipython3

    model_configuration = SUPPORTED_MODELS[model_id.value]
    print(f"Selected model {model_id.value}")


.. parsed-literal::

    Selected model red-pajama-3b-chat


Instantiate Model using Optimum Intel 
-------------------------------------------------------------------------------

Optimum Intel can be used to load optimized models from the `Hugging
Face Hub <https://huggingface.co/docs/optimum/intel/hf.co/models>`__ and
create pipelines to run an inference with OpenVINO Runtime using Hugging
Face APIs. The Optimum Inference models are API compatible with Hugging
Face Transformers models. This means we just need to replace
``AutoModelForXxx`` class with the corresponding ``OVModelForXxx``
class.

Below is an example of the RedPajama model

.. code:: diff

   -from transformers import AutoModelForCausalLM
   +from optimum.intel.openvino import OVModelForCausalLM
   from transformers import AutoTokenizer, pipeline

   model_id = "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
   -model = AutoModelForCausalLM.from_pretrained(model_id)
   +model = OVModelForCausalLM.from_pretrained(model_id, export=True)

Model class initialization starts with calling ``from_pretrained``
method. When downloading and converting Transformers model, the
parameter ``export=True`` should be added. We can save the converted
model for the next usage with the ``save_pretrained`` method. Tokenizer
class and pipelines API are compatible with Optimum models.

To optimize the generation process and use memory more efficiently, the
``use_cache=True`` option is enabled. Since the output side is
auto-regressive, an output token hidden state remains the same once
computed for every further generation step. Therefore, recomputing it
every time you want to generate a new token seems wasteful. With the
cache, the model saves the hidden state once it has been computed. The
model only computes the one for the most recently generated output token
at each time step, re-using the saved ones for hidden tokens. This
reduces the generation complexity from O(n^3) to O(n^2) for a
transformer model. More details about how it works can be found in this
`article <https://scale.com/blog/pytorch-improvements#Text%20Translation>`__.
With this option, the model gets the previous step’s hidden states
(cached attention keys and values) as input and additionally provides
hidden states for the current step as output. It means for all next
iterations, it is enough to provide only a new token obtained from the
previous step and cached key values to get the next token prediction.

In our case, MPT model currently is not covered by Optimum Intel, we
will convert it manually and create wrapper compatible with Optimum
Intel.

Compress model weights 
----------------------------------------------------------------

The Weights Compression algorithm is aimed at compressing the weights of
the models and can be used to optimize the model footprint and
performance of large models where the size of weights is relatively
larger than the size of activations, for example, Large Language Models
(LLM).

Weights Compression using Optimum Intel 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To enable weights compression via NNCF for models supported by Optimum
Intel ``OVQuantizer`` class should be used instantiated by PyTorch model
using ``from_pretrained`` method.
``OVQuantizer.quantize(save_directory=save_dir, weights_only=True)``
enables weights compression and model conversion to OpenVINO
Intermediate Representation format. We will consider how to do it on
RedPajama and LLAMA examples.

   **Note**: This tutorial involves conversion model for both FP16 and
   INT8 weights compression scenarios. It maybe memory and
   time-consuming in first run. You can manually disable FP16 conversion
   using CONVERT_FP16 variable below, CONVERT_INT8 variable can be used
   for disabling conversion model with weights compression respectively.

.. code:: ipython3

    CONVERT_FP16 = True
    CONVERT_INT8 = True

.. code:: ipython3

    from pathlib import Path
    from optimum.intel import OVQuantizer
    from transformers import AutoModelForCausalLM
    from optimum.intel.openvino import OVModelForCausalLM
    import logging
    import nncf
    import gc
    
    nncf.set_log_level(logging.ERROR)
    
    compressed_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    model_dir = Path(model_id.value) / "FP16"
    pt_model_id = model_configuration["model_id"]
    
    if "mpt" not in model_id.value:
        if CONVERT_INT8 and not compressed_model_dir.exists():
            pt_model = AutoModelForCausalLM.from_pretrained(pt_model_id)
            quantizer = OVQuantizer.from_pretrained(pt_model)
            quantizer.quantize(save_directory=compressed_model_dir, weights_only=True)
            del quantizer
            del pt_model
            gc.collect()
    
        if CONVERT_FP16 and not model_dir.exists():
            ov_model = OVModelForCausalLM.from_pretrained(pt_model_id, export=True, compile=False)
            ov_model.half()
            ov_model.save_pretrained(model_dir)
            del ov_model
    gc.collect();


.. parsed-literal::

    INFO:nncf:NNCF initialized successfully. Supported frameworks detected: torch, tensorflow, onnx, openvino


.. parsed-literal::

    No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'
    2023-09-19 19:06:00.934297: I tensorflow/core/util/port.cc:110] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2023-09-19 19:06:00.971948: I tensorflow/core/platform/cpu_feature_guard.cc:182] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2023-09-19 19:06:01.591238: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
    /home/ea/work/ov_venv/lib/python3.8/site-packages/transformers/deepspeed.py:23: FutureWarning: transformers.deepspeed module is deprecated and will be removed in a future version. Please import deepspeed modules directly from transformers.integrations
      warnings.warn(


Weights Compression using NNCF 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You also can perform weights compression for PyTorch models using NNCF
directly. ``nncf.compress_weights`` function accept PyTorch model
instance and compress its weights for Linear and Embedding layers. We
will consider this variant based on MPT model.

To begin compression, we should define model conversion first.

.. code:: ipython3

    from functools import wraps
    import torch
    from transformers import AutoModelForCausalLM
    from nncf import compress_weights
    import openvino as ov
    from typing import Optional, Union, Dict, Tuple, List
    
    def flattenize_inputs(inputs):
        """
        Helper function for making nested inputs flattens
        """
        flatten_inputs = []
        for input_data in inputs:
            if input_data is None:
                continue
            if isinstance(input_data, (list, tuple)):
                flatten_inputs.extend(flattenize_inputs(input_data))
            else:
                flatten_inputs.append(input_data)
        return flatten_inputs
    
    def cleanup_torchscript_cache():
        """
        Helper for removing cached model representation
        """
        torch._C._jit_clear_class_registry()
        torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
        torch.jit._state._clear_class_state()
    
    def convert_mpt(pt_model:torch.nn.Module, model_path:Path):
        """
        MPT model conversion function
        
        Params:
          pt_model: PyTorch model
          model_path: path for saving model
        Returns:
          None
        """
        ov_out_path = Path(model_path) / "openvino_model.xml"
        pt_model.config.save_pretrained(ov_out_path.parent)
        pt_model.config.use_cache = True
        outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long), attention_mask=torch.ones((1, 10), dtype=torch.long))
        inputs = ["input_ids"]
        outputs = ["logits"]
    
        dynamic_shapes = {"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
        for idx in range(len(outs.past_key_values)):
            inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
            dynamic_shapes[inputs[-1]] = {2: "past_sequence + sequence"}
            dynamic_shapes[inputs[-2]] = {3: "past_sequence + sequence"}
            outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])
                
        inputs.append("attention_mask")
        dummy_inputs = {"input_ids": torch.ones((1,2), dtype=torch.long), "past_key_values": outs.past_key_values, "attention_mask": torch.ones((1,12), dtype=torch.long)}
        pt_model.config.torchscript = True
        orig_forward = pt_model.forward
        @wraps(orig_forward)
        def ts_patched_forward(input_ids: torch.Tensor, past_key_values: Tuple[Tuple[torch.Tensor]], attention_mask: torch.Tensor):
            pkv_list = list(past_key_values)
            outs = orig_forward(input_ids=input_ids, past_key_values=pkv_list, attention_mask=attention_mask)
            return (outs.logits, tuple(outs.past_key_values))
        pt_model.forward = ts_patched_forward
        ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
        pt_model.forward = orig_forward
        for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
            input_node = m_input.get_node()
            if input_node.element_type == ov.Type.dynamic:
                m_input.get_node().set_element_type(ov.Type.f32)
            shape = list(input_data.shape)
            if inp_name in dynamic_shapes:
                for k in dynamic_shapes[inp_name]:
                    shape[k] = -1
            input_node.set_partial_shape(ov.PartialShape(shape))
            m_input.get_tensor().set_names({inp_name})
            
        for out, out_name in zip(ov_model.outputs, outputs):
            out.get_tensor().set_names({out_name})     
    
        ov_model.validate_nodes_and_infer_types()
        ov.save_model(ov_model, ov_out_path)
        del ov_model
        cleanup_torchscript_cache()
        del pt_model

Now, we know how to convert model to OpenVINO format, we can save
floating point and compressed model variants

.. code:: ipython3

    compressed_model_dir = Path(model_id.value) / "INT8_compressed_weights"
    model_dir = Path(model_id.value) / "FP16"
    
    if "mpt" in model_id.value and (not compressed_model_dir.exists() or not model_dir.exists()):
        model = AutoModelForCausalLM.from_pretrained(model_configuration["model_id"], torch_dtype=torch.float32, trust_remote_code=True)
        if CONVERT_FP16 and not model_dir.exists():
            convert_mpt(model, model_dir)
        if CONVERT_INT8 and not compressed_model_dir.exists():
            compressed_model = compress_weights(model)
            convert_mpt(compressed_model, compressed_model_dir)
    
    gc.collect();

.. code:: ipython3

    fp16_weights = model_dir / "openvino_model.bin"
    int8_weights = compressed_model_dir / "openvino_model.bin"
    
    if fp16_weights.exists():
        print(f'Size of FP16 model in MB is {fp16_weights.stat().st_size / 1024 / 1024}')
    if int8_weights.exists():
        print(f'Size of model with INT8 compressed weights in MB is {int8_weights.stat().st_size / 1024 / 1024}')
    if int8_weights.exists() and fp16_weights.exists():
        print(f"Model compression rate: {fp16_weights.stat().st_size / int8_weights.stat().st_size:.3f}")


.. parsed-literal::

    Size of FP16 model in MB is 5299.166286468506
    Size of model with INT8 compressed weights in MB is 2659.578887939453
    Model compression rate: 1.992


Select device for inference and model variant 
---------------------------------------------------------------------------------------

.. code:: ipython3

    core = ov.Core()
    device = widgets.Dropdown(
        options=core.available_devices + ["AUTO"],
        value='CPU',
        description='Device:',
        disabled=False,
    )




.. parsed-literal::

    VBox(children=(Dropdown(description='Device:', options=('CPU', 'GPU', 'AUTO'), value='CPU'), Checkbox(value=Tr…



.. code:: ipython3

    int8_compressed_weights = widgets.Checkbox(
        value=True,
        description='Use compressed weights',
        disabled=False
    )
    
    widgets.VBox([device, int8_compressed_weights])

The cell below create ``OVMPTModel`` model wrapper based on
``OVModelForCausalLM`` model.

.. code:: ipython3

    from transformers import AutoConfig
    import torch
    
    from optimum.intel.openvino import OVModelForCausalLM
    from optimum.utils import NormalizedTextConfig, NormalizedConfigManager
    from transformers.modeling_outputs import CausalLMOutputWithPast
    import numpy as np
    from pathlib import Path
    
        
    class OVMPTModel(OVModelForCausalLM):
        """
        Optimum intel compatible model wrapper for MPT
        """
        def __init__(
            self,
            model: "Model",
            config: "PretrainedConfig" = None,
            device: str = "CPU",
            dynamic_shapes: bool = True,
            ov_config: Optional[Dict[str, str]] = None,
            model_save_dir: Optional[Union[str, Path]] = None,
            **kwargs,
        ):
            NormalizedConfigManager._conf["mpt"] = NormalizedTextConfig.with_args(num_layers="n_layers", num_attention_heads="n_heads")
            super().__init__(model, config, device, dynamic_shapes, ov_config, model_save_dir, **kwargs)
    
        def _reshape(
            self,
            model: "Model",
            *args,
            **kwargs
        ):
            shapes = {}
            for inputs in model.inputs:
                shapes[inputs] = inputs.get_partial_shape()
                if shapes[inputs].rank.get_length() in [2, 3]:
                    shapes[inputs][1] = -1
                else:
                    if ".key" in inputs.get_any_name():
                        shapes[inputs][3] = -1
                    else:
                        shapes[inputs][2] = -1
                    
            model.reshape(shapes)
            return model
    
        def forward(
            self,
            input_ids: torch.LongTensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            **kwargs,
        ) -> CausalLMOutputWithPast:
            self.compile()
    
            if self.use_cache and past_key_values is not None:
                input_ids = input_ids[:, -1:]
    
            inputs = {}
            if past_key_values is not None:
                # Flatten the past_key_values
                past_key_values = tuple(
                    past_key_value for pkv_per_layer in past_key_values for past_key_value in pkv_per_layer
                )
                # Add the past_key_values to the decoder inputs
                inputs = dict(zip(self.key_value_input_names, past_key_values))
    
            # Create empty past_key_values for decoder_with_past first generation step
            elif self.use_cache:
                shape_input_ids = input_ids.shape
                num_attention_heads = (
                    self.normalized_config.num_attention_heads if self.config.model_type == "bloom" else 1
                )
                for input_name in self.key_value_input_names:
                    model_inputs = self.model.input(input_name)
                    shape = model_inputs.get_partial_shape()
                    shape[0] = shape_input_ids[0] * num_attention_heads
                    if shape[2].is_dynamic:
                        shape[2] = 0
                    if shape[1].is_dynamic:
                        shape[1] = 0
                    if shape.rank.get_length() == 4 and shape[3].is_dynamic:
                        shape[3] = 0
                    inputs[input_name] = ov.Tensor(model_inputs.get_element_type(), shape.get_shape())
    
            inputs["input_ids"] = np.array(input_ids)
    
            # Add the attention_mask inputs when needed
            if "attention_mask" in self.input_names and attention_mask is not None:
                inputs["attention_mask"] = np.array(attention_mask)
    
            # Run inference
            self.request.start_async(inputs, shared_memory=True)
            self.request.wait()
    
            logits = torch.from_numpy(self.request.get_tensor("logits").data).to(self.device)
    
            if self.use_cache:
                # Tuple of length equal to : number of layer * number of past_key_value per decoder layer (2 corresponds to the self-attention layer)
                past_key_values = tuple(self.request.get_tensor(key).data for key in self.key_value_output_names)
                # Tuple of tuple of length `n_layers`, with each tuple of length equal to 2 (k/v of self-attention)
                past_key_values = tuple(
                    past_key_values[i : i + self.num_pkv] for i in range(0, len(past_key_values), self.num_pkv)
                )
            else:
                past_key_values = None
    
            return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

The cell below demonstrates how to instantiate model based on selected
variant of model weights and inference device

.. code:: ipython3

    from pathlib import Path
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer
    
    model_dir = Path(model_id.value) / ("FP16" if not int8_compressed_weights.value else "INT8_compressed_weights")
    model_name = model_configuration["model_id"]
    
    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    model_class = OVModelForCausalLM if "mpt" not in model_id.value else OVMPTModel
    ov_model = model_class.from_pretrained(model_dir, device=device.value, ov_config=ov_config, config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True), trust_remote_code=True)


.. parsed-literal::

    The argument `trust_remote_code` is to be used along with export=True. It will be ignored.
    Compiling the model...


.. code:: ipython3

    tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
    test_string = "2 + 2 ="
    input_tokens = tok(test_string, return_tensors="pt", **tokenizer_kwargs)
    answer = ov_model.generate(**input_tokens, max_new_tokens=2)
    print(tok.batch_decode(answer)[0])


.. parsed-literal::

    Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
    /home/ea/work/ov_venv/lib/python3.8/site-packages/optimum/intel/openvino/modeling_decoder.py:364: FutureWarning: `shared_memory` is deprecated and will be removed in 2024.0. Value of `shared_memory` is going to override `share_inputs` value. Please use only `share_inputs` explicitly.
      self.request.start_async(inputs, shared_memory=True)


.. parsed-literal::

    2 + 2 = 4.


Run Chatbot 
-----------------------------------------------------

Now, when model created, we can setup Chatbot interface using
`Gradio <https://www.gradio.app/>`__. The diagram below illustrates how
the chatbot pipeline works

.. figure:: https://user-images.githubusercontent.com/29454499/255523209-d9336491-c7ba-4dc1-98f0-07f23743ce89.png
   :alt: generation pipeline

   generation pipeline

As can be seen, the pipeline very similar to instruction-following with
only changes that previous conversation history additionally passed as
input with next user question for getting wider input context. On the
first iteration, the user provided instructions joined to conversation
history (if exists) converted to token ids using a tokenizer, then
prepared input provided to the model. The model generates probabilities
for all tokens in logits format The way the next token will be selected
over predicted probabilities is driven by the selected decoding
methodology. You can find more information about the most popular
decoding methods in this
`blog <https://huggingface.co/blog/how-to-generate>`__. The result
generation updates conversation history for next conversation step. it
makes stronger connection of next question with previously provided and
allows user to make clarifications regarding previously provided
answers.

| There are several parameters that can control text generation quality:
  \* ``Temperature`` is a parameter used to control the level of
  creativity in AI-generated text. By adjusting the ``temperature``, you
  can influence the AI model’s probability distribution, making the text
  more focused or diverse.
| Consider the following example: The AI model has to complete the
  sentence “The cat is \____.” with the following token probabilities:

::

   playing: 0.5  
   sleeping: 0.25  
   eating: 0.15  
   driving: 0.05  
   flying: 0.05  

   - **Low temperature** (e.g., 0.2): The AI model becomes more focused and deterministic, choosing tokens with the highest probability, such as "playing."  
   - **Medium temperature** (e.g., 1.0): The AI model maintains a balance between creativity and focus, selecting tokens based on their probabilities without significant bias, such as "playing," "sleeping," or "eating."  
   - **High temperature** (e.g., 2.0): The AI model becomes more adventurous, increasing the chances of selecting less likely tokens, such as "driving" and "flying."

-  ``Top-p``, also known as nucleus sampling, is a parameter used to
   control the range of tokens considered by the AI model based on their
   cumulative probability. By adjusting the ``top-p`` value, you can
   influence the AI model’s token selection, making it more focused or
   diverse. Using the same example with the cat, consider the following
   top_p settings:

   -  **Low top_p** (e.g., 0.5): The AI model considers only tokens with
      the highest cumulative probability, such as “playing.”
   -  **Medium top_p** (e.g., 0.8): The AI model considers tokens with a
      higher cumulative probability, such as “playing,” “sleeping,” and
      “eating.”
   -  **High top_p** (e.g., 1.0): The AI model considers all tokens,
      including those with lower probabilities, such as “driving” and
      “flying.”

-  ``Top-k`` is an another popular sampling strategy. In comparison with
   Top-P, which chooses from the smallest possible set of words whose
   cumulative probability exceeds the probability P, in Top-K sampling K
   most likely next words are filtered and the probability mass is
   redistributed among only those K next words. In our example with cat,
   if k=3, then only “playing”, “sleeping” and “eating” will be taken
   into account as possible next word.
-  ``Repetition Penalty`` This parameter can help penalize tokens based
   on how frequently they occur in the text, including the input prompt.
   A token that has already appeared five times is penalized more
   heavily than a token that has appeared only one time. A value of 1
   means that there is no penalty and values larger than 1 discourage
   repeated tokens.

.. code:: ipython3

    from threading import Event, Thread
    from uuid import uuid4
    
    import gradio as gr
    import torch
    from transformers import (
        AutoTokenizer,
        StoppingCriteria,
        StoppingCriteriaList,
        TextIteratorStreamer,
    )
    
    
    model_name = model_configuration["model_id"]
    history_template = model_configuration["history_template"]
    current_message_template = model_configuration["current_message_template"]
    start_message = model_configuration["start_message"]
    stop_tokens = model_configuration.get("stop_tokens")
    tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})
    
    
    
    max_new_tokens = 256
    
    class StopOnTokens(StoppingCriteria):
        def __init__(self, token_ids):
            self.token_ids = token_ids
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_id in self.token_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False
    
    if stop_tokens is not None:
        if isinstance(stop_tokens[0], str):
            stop_tokens = tok.convert_tokens_to_ids(stop_tokens)
            
        stop_tokens = [StopOnTokens(stop_tokens)]
    
    def default_partial_text_processor(partial_text:str, new_text:str):
        """
        helper for updating partially generated answer, used by de
        
        Params:
          partial_text: text buffer for storing previosly generated text
          new_text: text update for the current step
        Returns:
          updated text string
        
        """
        partial_text += new_text
        return partial_text
    
    text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)
        
    def convert_history_to_text(history:List[Tuple[str, str]]):
        """
        function for conversion history stored as list pairs of user and assistant messages to string according to model expected conversation template
        Params:
          history: dialogue history
        Returns:
          history in text format
        """
        text = start_message + "".join(
            [
                "".join(
                    [
                        history_template.format(user=item[0], assistant=item[1])
                    ]
                )
                for item in history[:-1]
            ]
        )
        text += "".join(
            [
                "".join(
                    [
                        current_message_template.format(user=history[-1][0], assistant=history[-1][1])
                    ]
                )
            ]
        )
        return text
    
    
    
    def user(message, history):
        """
        callback function for updating user messages in interface on submit button click
        
        Params:
          message: current message
          history: conversation history
        Returns:
          None
        """
        # Append the user's message to the conversation history
        return "", history + [[message, ""]]
    
    
    def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
        """
        callback function for running chatbot on submit button click
        
        Params:
          history: conversation history
          temperature:  parameter for control the level of creativity in AI-generated text. 
                        By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
          top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
          top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
          repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
          conversation_id: unique conversation identifier.
        
        """
    
        # Construct the input message string for the model by concatenating the current system message and conversation history
        messages = convert_history_to_text(history)
    
        # Tokenize the messages string
        input_ids = tok(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
        if input_ids.shape[1] > 2000:
            history = [history[-1]]
            messages = convert_history_to_text(history)
            input_ids = tok(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
        streamer = TextIteratorStreamer(tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0.0,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            streamer=streamer,
        )
        if stop_tokens is not None:
            generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)
    
        stream_complete = Event()
    
        def generate_and_signal_complete():
            """
            genration function for single thread
            """
            global start_time
            ov_model.generate(**generate_kwargs)
            stream_complete.set()
    
        t1 = Thread(target=generate_and_signal_complete)
        t1.start()
    
        # Initialize an empty string to store the generated text
        partial_text = ""
        for new_text in streamer:
            partial_text = text_processor(partial_text, new_text)
            history[-1][1] = partial_text
            yield history
    
    
    def get_uuid():
        """
        universal unique identifier for thread
        """
        return str(uuid4())
    
    
    with gr.Blocks(
        theme=gr.themes.Soft(),
        css=".disclaimer {font-variant-caps: all-small-caps;}",
    ) as demo:
        conversation_id = gr.State(get_uuid)
        gr.Markdown(
            f"""<h1><center>OpenVINO {model_id.value} Chatbot</center></h1>"""
        )
        chatbot = gr.Chatbot(height=500)
        with gr.Row():
            with gr.Column():
                msg = gr.Textbox(
                    label="Chat Message Box",
                    placeholder="Chat Message Box",
                    show_label=False,
                    container=False
                )
            with gr.Column():
                with gr.Row():
                    submit = gr.Button("Submit")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
        with gr.Row():
            with gr.Accordion("Advanced Options:", open=False):
                with gr.Row():
                    with gr.Column():
                        with gr.Row():
                            temperature = gr.Slider(
                                label="Temperature",
                                value=0.1,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                interactive=True,
                                info="Higher values produce more diverse outputs",
                            )
                    with gr.Column():
                        with gr.Row():
                            top_p = gr.Slider(
                                label="Top-p (nucleus sampling)",
                                value=1.0,
                                minimum=0.0,
                                maximum=1,
                                step=0.01,
                                interactive=True,
                                info=(
                                    "Sample from the smallest possible set of tokens whose cumulative probability "
                                    "exceeds top_p. Set to 1 to disable and sample from all tokens."
                                ),
                            )
                    with gr.Column():
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k",
                                value=50,
                                minimum=0.0,
                                maximum=200,
                                step=1,
                                interactive=True,
                                info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                            )
                    with gr.Column():
                        with gr.Row():
                            repetition_penalty = gr.Slider(
                                label="Repetition Penalty",
                                value=1.1,
                                minimum=1.0,
                                maximum=2.0,
                                step=0.1,
                                interactive=True,
                                info="Penalize repetition — 1.0 to disable.",
                            )
        gr.Examples([
            ["Hello there! How are you doing?"],
            ["What is OpenVINO?"],
            ["Who are you?"],
            ["Can you explain to me briefly what is Python programming language?"],
            ["Explain the plot of Cinderella in a sentence."],
            ["What are some common mistakes to avoid when writing code?"],
            ["Write a 100-word blog post on “Benefits of Artificial Intelligence and OpenVINO“"]
        ], 
            inputs=msg, 
            label="Click on any example and press the 'Submit' button"
        )
    
        submit_event = msg.submit(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        submit_click_event = submit.click(
            fn=user,
            inputs=[msg, chatbot],
            outputs=[msg, chatbot],
            queue=False,
        ).then(
            fn=bot,
            inputs=[
                chatbot,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                conversation_id,
            ],
            outputs=chatbot,
            queue=True,
        )
        stop.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[submit_event, submit_click_event],
            queue=False,
        )
        clear.click(lambda: None, None, chatbot, queue=False)
    
    demo.queue(max_size=2)
    # if you are launching remotely, specify server_name and server_port
    #  demo.launch(server_name='your server name', server_port='server port in int')
    # if you have any issue to launch on your platform, you can pass share=True to launch method:
    # demo.launch(share=True)
    # it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
    demo.launch()


.. parsed-literal::

    Running on local URL:  http://127.0.0.1:7860
    
    To create a public link, set `share=True` in `launch()`.



.. .. raw:: html

..    <div><iframe src="http://127.0.0.1:7860/" width="100%" height="500" allow="autoplay; camera; microphone; clipboard-read; clipboard-write;" frameborder="0" allowfullscreen></iframe></div>


.. code:: ipython3

    # please run this cell for stopping gradio interface
    demo.close()


.. parsed-literal::

    Closing server running on port: 7860

