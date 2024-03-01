.. {#gen_ai_guide}

Enabling OpenVINO Runtime Optimizations
=====================================================

OpenVINO runtime provides a set of optimizations for more efficient LLM inference. This includes **Dynamic quantization** of activations of 4/8-bit quantized MatMuls and **KV-cache quantization**.

* **Dynamic quantization** enables quantization of activations of MatMul operations that have 4 or 8-bit quantized weights (see :doc:`LLM Weight Compression <weight_compression>`).
  It improves inference latency and throughput of LLMs, though it may cause insignificant deviation in generation accuracy.  Quantization is performed in a
  group-wise manner, with configurable group size. It means that values in a group share quantization parameters. Larger group sizes lead to faster inference but lower accuracy. Recommended group size values are: ``32``, ``64``, or ``128``. To enable Dynamic quantization, use the corresponding
  inference property as follows:


  .. code-block:: python

      model = OVModelForCausalLM.from_pretrained(
          model_path,
          ov_config={"DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", "PERFORMANCE_HINT": "LATENCY"}
      )



* **KV-cache quantization** allows lowering the precision of Key and Value cache in LLMs. This helps reduce memory consumption during inference, improving latency and throughput. KV-cache can be quantized into the following precisions:
  ``u8``, ``bf16``, ``f16``.  If ``u8`` is used, KV-cache quantization is also applied in a group-wise manner. Thus, it can use ``DYNAMIC_QUANTIZATION_GROUP_SIZE`` value if defined.
  Otherwise, the group size ``32`` is used by default. KV-cache quantization can be enabled as follows:


  .. code-block:: python

      model = OVModelForCausalLM.from_pretrained(
          model_path,
          ov_config={"KV_CACHE_PRECISION": "u8", "DYNAMIC_QUANTIZATION_GROUP_SIZE": "32", "PERFORMANCE_HINT": "LATENCY"}
      )


.. note::

  Currently, both Dynamic quantization and KV-cache quantization are available for CPU device.


Working with Models Tuned with LoRA
#########################################

Low-rank Adaptation (LoRA) is a popular method to tune Generative AI models to a downstream task
or custom data. However, it requires some extra steps to be done for efficient deployment using
the Hugging Face API. Namely, the trained adapters should be fused into the baseline model to
avoid extra computation. This is how it can be done for LLMs:

.. code-block:: python

    model_id = "meta-llama/Llama-2-7b-chat-hf"
    lora_adaptor = "./lora_adaptor"

    model = AutoModelForCausalLM.from_pretrained(model_id, use_cache=True)
    model = PeftModelForCausalLM.from_pretrained(model, lora_adaptor)
    model.merge_and_unload()
    model.get_base_model().save_pretrained("fused_lora_model")


Now the model can be converted to OpenVINO using Optimum Intel Python API or CLI interfaces mentioned above.


Additional Resources
#####################

* `Optimum Intel documentation <https://huggingface.co/docs/optimum/intel/inference>`__
* :doc:`LLM Weight Compression <weight_compression>`
* `Neural Network Compression Framework <https://github.com/openvinotoolkit/nncf>`__
* `GenAI Pipeline Repository <https://github.com/openvinotoolkit/openvino.genai>`__
* `OpenVINO Tokenizers <https://github.com/openvinotoolkit/openvino_contrib/tree/master/modules/custom_operations/user_ie_extensions/tokenizer/python>`__
* :doc:`Stateful Models Low-Level Details <openvino_docs_OV_UG_stateful_models_intro>`
* :doc:`Working with Textual Data <openvino_docs_OV_UG_string_tensors>`
