LLM Weight Compression
=========================

.. toctree::
   :maxdepth: 1
   :hidden:

   weight-compression/microscaling-quantization


Weight compression is a technique for enhancing the efficiency of models,
especially those with large memory requirements. This method reduces the model's
memory footprint, a crucial factor for Large Language Models (LLMs).

Unlike full model quantization, where weights and activations are quantized,
weight compression in `Neural Network Compression Framework (NNCF) <https://github.com/openvinotoolkit/nncf>`__
only targets the model's weights. This approach allows the activations to remain as
floating-point numbers, preserving most of the model's accuracy while improving its
speed and reducing its size.

The reduction in size is especially noticeable with larger models,
for instance the 7 billion parameter Llama 2 model can be reduced
from about 25GB to 4GB using 4-bit weight compression. With smaller models (i.e. less
than 1B parameters), weight compression may result in more accuracy reduction than
with larger models.

LLMs and other models that require
extensive memory to store the weights during inference can benefit
from weight compression as it:

* enables inference of exceptionally large models that cannot be accommodated in the
  device memory;

* reduces storage and memory overhead, making models more lightweight and less resource
  intensive for deployment;

* improves inference speed by reducing the latency of memory access when computing the
  operations with weights, for example, Linear layers. The weights are smaller and thus
  faster to load from memory;

* unlike quantization, does not require sample data to calibrate the range of
  activation values.

Currently, `NNCF <https://github.com/openvinotoolkit/nncf>`__
provides weight quantization to 8 and 4-bit integer data types as a compression
method primarily designed to optimize LLMs.



Compress Model Weights
######################

**8-bit weight quantization** method offers a balance between model size reduction and
maintaining accuracy, which usually leads to significant performance improvements for
Transformer-based models. Models with 8-bit compressed weights are performant on the
vast majority of supported CPU and GPU platforms. By default, weights are compressed
asymmetrically to "INT8_ASYM" mode.


The code snippet below shows how to do asymmetrical 8-bit quantization of the model weights
represented in OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_8bit]


Now, the model is ready for compilation and inference.
It can be also saved into a compressed format, resulting in a smaller binary file.

**4-bit weight quantization** method stands for an INT4-INT8 mixed-precision weight quantization,
where INT4 is considered as the primary precision and asymmetric INT8 is the backup one.
It usually results in a smaller model size and lower inference latency, although the accuracy
degradation could be higher, depending on the model.

The code snippet below shows how to do 4-bit quantization of the model weights represented
in OpenVINO IR using NNCF:

.. tab-set::

   .. tab-item:: OpenVINO
      :sync: openvino

      .. doxygensnippet:: docs/optimization_guide/nncf/code/weight_compression_openvino.py
         :language: python
         :fragment: [compression_4bit]


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



The INT4 method has several parameters that can provide different performance-accuracy
trade-offs after optimization:

* ``mode`` - there are two optimization modes: symmetric and asymmetric.

  **Symmetric Compression** - ``INT4_SYM``

  INT4 Symmetric mode involves quantizing weights to a signed 4-bit integer
  symmetrically without zero point. This mode is faster than the INT8_ASYM, making
  it ideal for situations where **speed and size reduction are prioritized over accuracy**.

  .. code-block:: python

    from nncf import compress_weights
    from nncf import CompressWeightsMode

    compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM)

  **Asymmetric Compression** - ``INT4_ASYM``

  INT4 Asymmetric mode also uses an unsigned 4-bit integer but quantizes weights
  asymmetrically with a non-fixed zero point. This mode slightly compromises speed in
  favor of better accuracy compared to the symmetric mode. This mode is useful when
  **minimal accuracy loss is crucial**, but a faster performance than INT8 is still desired.

  .. code-block:: python

    from nncf import compress_weights
    from nncf import CompressWeightsMode

    compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_ASYM)

* ``group_size`` controls the size of the group of weights that share the same
  quantization parameters. Shared quantization parameters help to speed up the
  calculation of activation values as they are dequantized and quantized between
  layers. However, they can reduce accuracy. The following group sizes are
  recommended: ``128``, ``64``, ``32`` (``128`` is default value).

  `Smaller Group Size`: Leads to a more accurate model but increases the model's
  footprint and reduces inference speed.

  `Larger Group Size`: Results in faster inference and a smaller model, but might
  compromise accuracy.

* ``ratio`` controls the ratio between the layers compressed to the precision defined
  by ``mode`` and the rest of the layers that will be kept in the ``backup_mode`` in the optimized model.
  Ratio is a decimal between 0 and 1. For example, 0.8 means that 80% of layers will be
  compressed to the precision defined by ``mode``, while the rest will be compressed to
  ``backup_mode`` precision. The default value for ratio is 1.

  `Higher Ratio (more layers set to mode precision)`: Reduces the model size and increase inference speed but
  might lead to higher accuracy degradation.

  `Lower Ratio (more layers set to backup_mode precision)`: Maintains better accuracy but results in a larger model size
  and potentially slower inference.

  In this example, 90% of the model's layers are quantized to INT4 asymmetrically with
  a group size of 64:

  .. code-block:: python

    from nncf import compress_weights, CompressWeightsMode

    # Example: Compressing weights with INT4_ASYM mode, group size of 64, and 90% INT4 ratio
    compressed_model = compress_weights(
      model,
      mode=CompressWeightsMode.INT4_ASYM,
      group_size=64,
      ratio=0.9,
    )

* ``scale_estimation`` - boolean parameter that enables more accurate estimation of
  quantization scales. Especially helpful when the weights of all layers are quantized to
  4 bits. Requires dataset.

* ``awq`` - boolean parameter that enables the AWQ method for more accurate INT4 weight
  quantization. Especially helpful when the weights of all the layers are quantized to
  4 bits. The method can sometimes result in reduced accuracy when used with
  Dynamic Quantization of activations. Requires dataset.

* ``gptq`` - boolean parameter that enables the GPTQ method for more accurate INT4 weight
  quantization. Requires dataset.

* ``dataset`` - calibration dataset for data-aware weight compression. It is required
  for some compression options, for example, ``scale_estimation``, ``gptq`` or ``awq``. Some types
  of ``sensitivity_metric`` can use data for precision selection.

* ``sensitivity_metric`` - controls the metric to estimate the sensitivity of compressing
  layers in the bit-width selection algorithm. Some of the metrics require dataset to be
  provided. The following types are supported:

  * ``nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR`` - data-free metric computed as
    the inverted 8-bit quantization noise. Weights with highest value of this metric can
    be accurately quantized channel-wise to 8-bit. The idea is to leave these weights in
    8 bit, and quantize the rest of layers to 4-bit group-wise. Since group-wise is more
    accurate than per-channel, accuracy should not degrade.

  * ``nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION`` - requires dataset. The average
    Hessian trace of weights with respect to the layer-wise quantization error multiplied
    by L2 norm of 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE`` - requires dataset. The mean
    variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE`` - requires dataset. The maximum
    variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE`` - requires dataset. The mean
    magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.

* ``all_layers`` - boolean parameter that enables INT4 weight quantization of all
  Fully-Connected and Embedding layers, including the first and last layers in the model.

* ``lora_correction`` - boolean parameter that enables the LoRA Correction Algorithm
  to further improve the accuracy of INT4 compressed models on top of other
  algorithms - AWQ and Scale Estimation.

* ``backup_mode`` - defines a backup precision for mixed-precision weight compression.
  There are three modes: INT8_ASYM, INT8_SYM, and NONE, which retains
  the original floating-point precision of the model weights (``INT8_ASYM`` is default value).


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

   # Example: Generating synthetic dataset
   synthetic_data = generate_text_data(model, tokenizer)
   nncf_dataset = nncf.Dataset(synthetic_data, transform_fn)

For data-aware weight compression refer to the following
`example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama>`__.

.. note::

  Some methods can be stacked on top of one another to achieve a better
  accuracy-performance trade-off after weight quantization. For example, the Scale Estimation
  method can be applied along with AWQ and mixed-precision quantization (the ``ratio`` parameter).

The example below shows data-free 4-bit weight quantization
applied on top of OpenVINO IR. Before trying the example, make sure Optimum Intel
is installed in your environment by running the following command:

.. code-block:: python

  pip install optimum[openvino]

The first example loads a pre-trained Hugging Face model using the Optimum Intel API,
compresses it to INT4 using NNCF, and then executes inference with a text phrase.

If the model comes from `Hugging Face <https://huggingface.co/models>`__ and is supported
by Optimum, it may be easier to use the Optimum Intel API to perform weight compression.
The compression type is specified when the model is loaded using the ``load_in_8bit=True``
or ``load_in_4bit=True`` parameter. The second example uses the Weight Compression API
from Optimum Intel instead of NNCF to compress the model to INT8_ASYM.

.. tab-set::

  .. tab-item:: OpenVINO
    :sync: openvino

    .. code-block:: python

      from nncf import compress_weights, CompressWeightsMode
      from optimum.intel.openvino import OVModelForCausalLM
      from transformers import AutoTokenizer, pipeline

      # Load model from Hugging Face
      model_id = "HuggingFaceH4/zephyr-7b-beta"
      model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=False, compile=False)

      # Compress to INT4 Symmetric
      model.model = compress_weights(model.model, mode=CompressWeightsMode.INT4_SYM)

      # Inference
      model.compile()
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
      phrase = "The weather is"
      results = pipe(phrase)
      print(results)

  .. tab-item:: Optimum-Intel

    .. code-block:: python

      from optimum.intel.openvino import OVModelForCausalLM
      from transformers import AutoTokenizer, pipeline

      # Load and compress model from Hugging Face
      model_id = "HuggingFaceH4/zephyr-7b-beta"
      model = OVModelForCausalLM.from_pretrained(model_id, export=True, load_in_8bit=True)

      # Inference
      tokenizer = AutoTokenizer.from_pretrained(model_id)
      pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
      phrase = "The weather is"
      results = pipe(phrase)
      print(results)


Exporting and Loading Compressed Models
########################################

Once a model has been compressed with NNCF or Optimum Intel,
it can be saved and exported to use in a future session or in a
deployment environment. The compression process takes a while,
so it is preferable to compress the model once, save it, and then
load the compressed model later for faster time to first inference.

.. code-block:: python

  # Save compressed model for faster loading later
  model.save_pretrained("zephyr-7b-beta-int4-sym-ov")
  tokenizer.save_pretrained("zephyr-7b-beta-int4-sym-ov")

  # Load a saved model
  model = OVModelForCausalLM.from_pretrained("zephyr-7b-beta-int4-sym-ov")
  tokenizer = AutoTokenizer.from_pretrained("zephyr-7b-beta-int4-sym-ov")

GPTQ Models
############

OpenVINO also supports 4-bit models from Hugging Face
`Transformers <https://github.com/huggingface/transformers>`__ library optimized
with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__. In this case, there is no
need for an additional model optimization step because model conversion will
automatically preserve the INT4 optimization results, allowing model inference to benefit from it.

A compression example using a GPTQ model is shown below.
Make sure to install GPTQ dependencies by running the following command:

.. code-block:: python

  pip install optimum[openvino] auto-gptq

.. code-block:: python

  from optimum.intel.openvino import OVModelForCausalLM
  from transformers import AutoTokenizer, pipeline

  # Load model from Hugging Face already optimized with GPTQ
  model_id = "TheBloke/Llama-2-7B-Chat-GPTQ"
  model = OVModelForCausalLM.from_pretrained(model_id, export=True)

  # Inference
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
  phrase = "The weather is"
  results = pipe(phrase)
  print(results)

An `example of a model <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ>`__
that has been optimized using GPTQ.

Compression Metrics Examples
########################################

The table below shows examples of text-generation Language Models with different
optimization settings in a data-free setup, where no dataset is used at the optimization step.
The Perplexity metric is a measurement of response accuracy, where a higher complexity
score indicates a lower accuracy. It is measured on the
`Lambada OpenAI dataset <https://github.com/openai/gpt-2/issues/131#issuecomment-497136199>`__.

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

Auto-tuning of Weight Compression Parameters
############################################

To find the optimal weight compression parameters for a particular model, refer to the
`example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__ ,
where weight compression parameters are being searched from the subset of values.
To speed up the search, a self-designed validation pipeline called
`WhoWhatBench <https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python/who_what_benchmark>`__
is used. The pipeline can quickly evaluate the changes in the accuracy of the optimized
model compared to the baseline.

Additional Resources
####################

- `Data-aware Weight Compression Example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama>`__
- `Tune Weight Compression Parameters Example <https://github.com/openvinotoolkit/nncf/tree/develop/examples/llm_compression/openvino/tiny_llama_find_hyperparams>`__
- `OpenVINO GenAI Repo <https://github.com/openvinotoolkit/openvino.genai>`__
  : Repository containing example pipelines that implement image and text generation
  tasks. It also provides a tool to benchmark LLMs.
- `WhoWhatBench <https://github.com/openvinotoolkit/openvino.genai/tree/master/llm_bench/python/who_what_benchmark>`__
- `NNCF GitHub <https://github.com/openvinotoolkit/nncf>`__
- :doc:`Post-training Quantization <quantizing-models-post-training>`
- :doc:`Training-time Optimization <compressing-models-during-training>`
- `OCP Microscaling Formats (MX) Specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`__
