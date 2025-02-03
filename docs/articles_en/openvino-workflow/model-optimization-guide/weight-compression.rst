LLM Weight Compression
=========================

.. toctree::
   :maxdepth: 1
   :hidden:

   weight-compression/4-bit-weight-quantization
   weight-compression/microscaling-quantization



Weight compression enhances the efficiency of models by reducing their memory footprint,
a crucial factor for Large Language Models (LLMs). It is especially effective for networks with high memory requirements.

Unlike full model quantization, where both weights and activations are quantized, it
only targets weights, keeping activations as floating-point numbers. This means preserving most
of the model's accuracy while improving its
speed and reducing its size. The reduction in size is especially noticeable with larger models.
For instance the 8 billion parameter Llama 3 model can be reduced
from about 16.1 GB to 4.8 GB using 4-bit weight quantization on top of a bfloat16 model.

.. note::

   With smaller language models (i.e. less than 1B parameters), low-bit weight
   compression may result in more accuracy reduction than with larger models.

LLMs and other generative AI models that require
extensive memory to store the weights during inference can benefit
from weight compression as it:

* enables inference of exceptionally large models that cannot be accommodated in the
  device memory;
* reduces storage and memory overhead, making models more lightweight and less resource
  intensive for deployment;
* improves inference speed by reducing the latency of memory access when computing the
  operations with weights, for example, Linear layers. The weights are smaller and thus
  faster to load from memory;
* unlike full static quantization, does not require sample data to calibrate the range of
  activation values.

Currently, `NNCF <https://github.com/openvinotoolkit/nncf>`__
provides weight quantization to 8 and 4-bit integer data types as a compression
method primarily designed to optimize LLMs.


Compression Methods (8-bit vs. 4-bit)
#####################################

For models that come from `Hugging Face <https://huggingface.co/models>`__ and are supported
by Optimum, it is recommended to use the **Optimum Intel API**, which employs NNCF weight
compression capabilities to optimize various large Transformer models.

The NNCF ``nncf.compress_weights()`` API, with most of its options, is exposed in the
``.from_pretrained()`` method of Optimum Intel classes. Optimum also has several datasets
for data-aware quantization available out-of-the-box.

You can use the examples below to perform data-free 8-bit or 4-bit weight quantization.
Before you start, make sure Optimum Intel is installed in your environment
by running the following command:

.. code-block:: python

   pip install optimum[openvino]

**8-bit weight quantization** offers a good balance between reducing the size and lowering the
accuracy of a model. It usually results in significant improvements for Transformer-based models
and guarantees good model performance for a vast majority of supported CPU and GPU platforms.
By default, weights are compressed asymmetrically to "INT8_ASYM" mode.

.. tab-set::

   .. tab-item:: Compression with Optimum-Intel
      :sync: optimum

      Load a pre-trained Hugging Face model, compress it to INT8_ASYM, using the
      Optimum Intel API, and then execute inference with a text phrase:

      Simply use the optimum-cli command line tool:

      .. code-block:: console

         optimum-cli export openvino --model microsoft/Phi-3.5-mini-instruct --weight-format int8 ov_phi-3.5-mini-instruct

      You can also use the code sample to the same effect:

      .. code-block:: python

         from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
         from transformers import AutoTokenizer, pipeline

         # Load and compress a model from Hugging Face.
         model_id = "microsoft/Phi-3.5-mini-instruct"
         model = OVModelForCausalLM.from_pretrained(
             model_id,
             export=True,
             quantization_config=OVWeightQuantizationConfig(bits=8)
         )

         # Inference
         tokenizer = AutoTokenizer.from_pretrained(model_id)
         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
         phrase = "The weather is"
         results = pipe(phrase)
         print(results)

      For more details, refer to the article on how to
      :doc:`infer LLMs using Optimum Intel <../../openvino-workflow-generative/inference-with-optimum-intel>`.

   .. tab-item:: Compression with NNCF
      :sync: nncf

      Load a pre-trained Hugging Face model, using the Optimum Intel API,
      compress it to INT8_ASYM, using NNCF, and then execute inference with a text phrase:

      .. code-block:: python

        from nncf import compress_weights, CompressWeightsMode
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer, pipeline

        # Load a model and compress it with NNCF.
        model_id = "microsoft/Phi-3.5-mini-instruct"
        model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False)
        model.model = compress_weights(model.model, mode=CompressWeightsMode.INT8_ASYM)

        # Inference
        model.compile()
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        phrase = "The weather is"
        results = pipe(phrase)
        print(results)


Here is an example of code using NNCF to perform asymmetrical 8-bit weight quantization of
a model in the OpenVINO IR format:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]


**4-bit weight quantization** is actually a mixed-precision compression,
primarily INT4 and a backup asymmetric INT8 precisions. It produces a smaller model,
offering lower inference latency but potentially noticeable accuracy degradation,
depending on the model.

.. tab-set::

   .. tab-item:: Compression with Optimum-Intel
      :sync: optimum

      Load a pre-trained Hugging Face model, compress it to INT4, using the
      Optimum Intel API, and then execute inference with a text phrase:

      Simply use the optimum-cli command line tool:

      .. code-block:: console

         optimum-cli export openvino --model microsoft/Phi-3.5-mini-instruct --weight-format int4 --awq --scale-estimation --dataset wikitext2 --group-size 64 --ratio 1.0 ov_phi-3.5-mini-instruct

      You can also use the code sample to the same effect:

      .. code-block:: python

         from optimum.intel.openvino import OVModelForCausalLM, OVWeightQuantizationConfig
         from transformers import AutoTokenizer, pipeline

         # Load and compress a model from Hugging Face.
         model_id = "microsoft/Phi-3.5-mini-instruct"
         model = OVModelForCausalLM.from_pretrained(
             model_id,
             export=True,
             quantization_config=OVWeightQuantizationConfig(
                 bits=4,
                 quant_method="awq",
                 scale_estimation=True,
                 dataset="wikitext2",
                 group_size=64,
                 ratio=1.0
             )
         )

         # Inference
         tokenizer = AutoTokenizer.from_pretrained(model_id)
         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
         phrase = "The weather is"
         results = pipe(phrase)
         print(results)

   .. tab-item:: Compression with NNCF
      :sync: nncf

      Load a pre-trained Hugging Face model, using the Optimum Intel API,
      compress it to INT4 using NNCF, and then execute inference with a text phrase:

      .. code-block:: python

         from nncf import compress_weights, CompressWeightsMode
         from optimum.intel.openvino import OVModelForCausalLM
         from transformers import AutoTokenizer, pipeline

         # Load a model and compress it with NNCF.
         model_id = "microsoft/Phi-3.5-mini-instruct"
         model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False)
         model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM)

         # Inference
         model.compile()
         tokenizer = AutoTokenizer.from_pretrained(model_id)
         pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
         phrase = "The weather is"
         results = pipe(phrase)
         print(results)


      For more details, refer to the article on how to
      :doc:`infer LLMs using Optimum Intel <../../../openvino-workflow-generative/inference-with-optimum-intel>`.


Refer to the article about
:doc:`4-bit weight quantization <./weight-compression/4-bit-weight-quantization>`
for more details.

Once the model has been optimized, it is ready for compilation and inference. The model can
also be :ref:`saved into a compressed format <save_pretrained>`, resulting in a
smaller binary file.

The table below summarizes the benefits and trade-offs for each compression type in terms of
memory reduction, speed gain, and accuracy loss.

.. list-table::
   :widths: 25 20 20 20
   :header-rows: 1

   * -
     - Memory Reduction
     - Latency Improvement
     - Accuracy Loss
   * - INT8 Asymmetric
     - Low
     - Medium
     - Low
   * - INT4 Symmetric
     - High
     - High
     - High
   * - INT4 Asymmetric
     - High
     - Medium
     - Medium


**Use synthetic data for LLM weight compression**

It is possible to generate a synthetic dataset using the `nncf.data.generate_text_data` method for
data-aware weight compression. The method takes a language model (e.g. from `optimum.intel.openvino`)
and a tokenizer (e.g. from `transformers`) as input and returns the list of strings generated by the model.
Note that dataset generation takes time and depends on various conditions, like the model size,
requested dataset length or environment setup. Also, since the dataset is generated by the model output,
it does not guarantee significant accuracy improvement after compression. This method is recommended
only when a better dataset is not available. Refer to the
`example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_synthetic_data>`__
for details of the usage.

.. code-block:: python

   from nncf import Dataset
   from nncf.data import generate_text_data
   from functools import partial

   from transformers import AutoTokenizer, AutoModelForCausalLM

   # Example: Generating synthetic dataset
   tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
   hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, export=True, load_in_8bit=False
   )

   # Synthetic-based compression
   synthetic_dataset = nncf.data.generate_text_data(hf_model, tokenizer, dataset_size=100)
   quantization_dataset = nncf.Dataset(
       synthetic_dataset,
       transform_fn # See the example in NNCF repo to learn how to make transform_fn.
   )

   model = compress_weights(
       model,
       mode=CompressWeightsMode.INT4_ASYM,
       group_size=64,
       ratio=1.0,
       dataset=quantization_dataset,
       awq=True,
       scale_estimation=True
   )  # The model is openvino.Model.

For data-aware weight compression refer to the following
`example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama>`__.

.. note::

   Some methods can be stacked on top of one another to achieve a better
   accuracy-performance trade-off after weight quantization. For example, the **Scale Estimation**
   method can be applied along with **AWQ** and mixed-precision quantization (the ``ratio`` parameter).


Exporting and Loading Compressed Models
########################################

Once a model has been compressed with NNCF or Optimum Intel,
it can be saved and exported to use in a future session or in a
deployment environment. The compression process takes a while,
so it is preferable to compress the model once, save it, and then
load the compressed model later for faster time to first inference.

.. code-block:: python
   :name: save_pretrained

   # Save compressed model for faster loading later
   model.save_pretrained("Phi-3.5-mini-instruct-int4-sym-ov")
   tokenizer.save_pretrained("Phi-3.5-mini-instruct-int4-sym-ov")

   # Load a saved model
   model = OVModelForCausalLM.from_pretrained("Phi-3.5-mini-instruct-int4-sym-ov")
   tokenizer = AutoTokenizer.from_pretrained("Phi-3.5-mini-instruct-int4-sym-ov")

.. tip::

   Models optimized with with NNCF or Optimum Intel can be used with
   :doc:`OpenVINO GenAI <../../openvino-workflow-generative/inference-with-genai>`.


Auto-tuning of Weight Compression Parameters
############################################

To find the optimal weight compression parameters for a particular model, refer to the
`example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__ ,
where weight compression parameters are being searched from the subset of values.
To speed up the search, a self-designed validation pipeline called
`WhoWhatBench <https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark>`__
is used. The pipeline can quickly evaluate the changes in the accuracy of the optimized
model compared to the baseline.


Compression Metrics Examples
############################

Below you will find examples of text-generation Language Models with different
optimization settings in a data-free setup, where no dataset is used at the optimization step.
The Perplexity metric is a measurement of response accuracy, where a higher complexity
score indicates a lower accuracy. It is measured on the
`Lambada OpenAI dataset <https://github.com/openai/gpt-2/issues/131#issuecomment-497136199>`__.

.. dropdown:: Perplexity\* in data-free optimization

   .. list-table::
      :widths: 40 55 25 25
      :header-rows: 1

      * - Model
        - Optimization
        - Perplexity\*
        - Model Size (Gb)
      * - databricks/dolly-v2-3b
        - FP32
        - 5.01
        - 10.3
      * - databricks/dolly-v2-3b
        - INT8_ASYM
        - 5.07
        - 2.6
      * - databricks/dolly-v2-3b
        - INT4_ASYM,group_size=32,ratio=0.5
        - 5.28
        - 2.2
      * - facebook/opt-6.7b
        - FP32
        - 4.25
        - 24.8
      * - facebook/opt-6.7b
        - INT8_ASYM
        - 4.27
        - 6.2
      * - facebook/opt-6.7b
        - INT4_ASYM,group_size=64,ratio=0.8
        - 4.32
        - 4.1
      * - meta-llama/Llama-2-7b-chat-hf
        - FP32
        - 3.28
        - 25.1
      * - meta-llama/Llama-2-7b-chat-hf
        - INT8_ASYM
        - 3.29
        - 6.3
      * - meta-llama/Llama-2-7b-chat-hf
        - INT4_ASYM,group_size=128,ratio=0.8
        - 3.41
        - 4.0
      * - togethercomputer/RedPajama-INCITE-7B-Instruct
        - FP32
        - 4.15
        - 25.6
      * - togethercomputer/RedPajama-INCITE-7B-Instruct
        - INT8_ASYM
        - 4.17
        - 6.4
      * - togethercomputer/RedPajama-INCITE-7B-Instruct
        - INT4_ASYM,group_size=128,ratio=1.0
        - 4.17
        - 3.6
      * - meta-llama/Llama-2-13b-chat-hf
        - FP32
        - 2.92
        - 48.5
      * - meta-llama/Llama-2-13b-chat-hf
        - INT8_ASYM
        - 2.91
        - 12.1
      * - meta-llama/Llama-2-13b-chat-hf
        - INT4_SYM,group_size=64,ratio=0.8
        - 2.98
        - 8.0


.. dropdown:: Perplexity\* in data-aware optimization

   The following table shows accuracy metric in a data-aware 4-bit weight quantization
   setup measured on the `Wikitext dataset <https://arxiv.org/pdf/1609.07843.pdf>`__.

   .. list-table::
      :widths: 40 55 25 25
      :header-rows: 1

      * - Model
        - Optimization
        - Word perplexity\*
        - Model Size (Gb)
      * - meta-llama/llama-7b-chat-hf
        - FP32
        - 11.57
        - 12.61
      * - meta-llama/llama-7b-chat-hf
        - INT4_SYM,group_size=128,ratio=1.0,awq=True
        - 12.34
        - 2.6
      * - stabilityai_stablelm-3b-4e1t
        - FP32
        - 10.17
        - 10.41
      * - stabilityai_stablelm-3b-4e1t
        - INT4_SYM,group_size=64,ratio=1.0,awq=True
        - 10.89
        - 2.6
      * - HuggingFaceH4/zephyr-7b-beta
        - FP32
        - 9.82
        - 13.99
      * - HuggingFaceH4/zephyr-7b-beta
        - INT4_SYM,group_size=128,ratio=1.0
        - 10.32
        - 2.6


\*Perplexity metric in both tables was measured without the Dynamic Quantization feature
enabled in the OpenVINO runtime.


Additional Resources
####################

- `Data-aware Weight Compression Example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama>`__
- `Tune Weight Compression Parameters Example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__
- `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
  : Repository containing example pipelines that implement image and text generation
  tasks. It also provides a tool to benchmark LLMs.
- `WhoWhatBench <https://github.com/openvinotoolkit/openvino.genai/tree/master/tools/who_what_benchmark>`__
- `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__
- :doc:`Post-training Quantization <quantizing-models-post-training>`
- :doc:`Training-time Optimization <compressing-models-during-training>`
- `OCP Microscaling Formats (MX) Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
