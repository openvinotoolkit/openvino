4-bit Weight Quantization
=========================

The 4-bit weight quantization method results in significant reduction in model size and
memory usage, making LLMs more accessible to less performant devices.
It also usually offers lower inference latency, however, depending on specific models,
it may potentially impact the accuracy.

Nevertheless, the INT4 method has several parameters that can provide different performance-accuracy
trade-offs after optimization:

* ``mode`` - there are two optimization modes: symmetric and asymmetric.

  .. tab-set::

     .. tab-item:: Symmetric Compression
        :sync: int4-sym

        INT4 Symmetric mode (``INT4_SYM``) involves quantizing weights to a signed 4-bit integer
        symmetrically without zero point. This mode is faster than the INT8_ASYM, making
        it ideal for situations where **speed and size reduction are prioritized over accuracy**.

        .. code-block:: python

           from nncf import compress_weights
           from nncf import CompressWeightsMode

           compressed_model = compress_weights(model, mode=CompressWeightsMode.INT4_SYM)

     .. tab-item:: Asymmetric Compression
        :sync: int4-asym

        INT4 Asymmetric mode (``INT4_ASYM``) also uses an unsigned 4-bit integer but quantizes weights
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

  | **Higher Ratio (more layers set to mode precision)**:
  | Reduces the model size and increase inference speed but
    might lead to higher accuracy degradation.

  | **Lower Ratio (more layers set to backup_mode precision)**:
  | Maintains better accuracy but results in a larger model size
    and potentially slower inference.

  In the example below, 90% of the model's layers are quantized to INT4 asymmetrically with
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

* ``scale_estimation`` - a boolean parameter that enables more accurate estimation of
  quantization scales. Especially helpful when the weights of all layers are quantized to
  4 bits. Requires dataset.

* ``awq`` - a boolean parameter that enables the AWQ method for more accurate INT4 weight
  quantization. Especially helpful when the weights of all the layers are quantized to
  4 bits. The method can sometimes result in reduced accuracy when used with
  Dynamic Quantization of activations. Requires dataset.

* ``gptq`` - a boolean parameter that enables the GPTQ method for more accurate INT4 weight
  quantization. Requires dataset.

* ``dataset`` - a calibration dataset for data-aware weight compression. It is required
  for some compression options, for example, ``scale_estimation``, ``gptq`` or ``awq``. Some types
  of ``sensitivity_metric`` can use data for precision selection.

* ``sensitivity_metric`` - controls the metric to estimate the sensitivity of compressing
  layers in the bit-width selection algorithm. Some of the metrics require dataset to be
  provided. The following types are supported:

  * ``nncf.SensitivityMetric.WEIGHT_QUANTIZATION_ERROR`` - a data-free metric computed as
    the inverted 8-bit quantization noise. Weights with highest value of this metric can
    be accurately quantized channel-wise to 8-bit. The idea is to leave these weights in
    8 bit, and quantize the rest of layers to 4-bit group-wise. Since group-wise is more
    accurate than per-channel, accuracy should not degrade.

  * ``nncf.SensitivityMetric.HESSIAN_INPUT_ACTIVATION`` - requires a dataset. The average
    Hessian trace of weights with respect to the layer-wise quantization error multiplied
    by L2 norm of 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_VARIANCE`` - requires a dataset. The mean
    variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MAX_ACTIVATION_VARIANCE`` - requires a dataset. The maximum
    variance of the layers' inputs multiplied by inverted 8-bit quantization noise.

  * ``nncf.SensitivityMetric.MEAN_ACTIVATION_MAGNITUDE`` - requires a dataset. The mean
    magnitude of the layers' inputs multiplied by inverted 8-bit quantization noise.

* ``all_layers`` - a boolean parameter that enables INT4 weight quantization of all
  Fully-Connected and Embedding layers, including the first and last layers in the model.

* ``lora_correction`` - a boolean parameter that enables the LoRA Correction Algorithm
  to further improve the accuracy of INT4 compressed models on top of other
  algorithms - AWQ and Scale Estimation.

* ``backup_mode`` - defines a backup precision for mixed-precision weight compression.
  There are three modes: INT8_ASYM, INT8_SYM, and NONE, which retains
  the original floating-point precision of the model weights (``INT8_ASYM`` is default value).



.. tip::

   NNCF enables you to stack the supported optimization methods. For example, AWQ,
   Scale Estimation and GPTQ methods may be enabled all together to achieve better accuracy.

4-bit Weight Quantization with GPTQ
###################################

You can use models from Hugging Face
`Transformers <https://github.com/huggingface/transformers>`__ library, which are quantized
with `GPTQ <https://github.com/PanQiWei/AutoGPTQ>`__ algorithm. Such models do not require
additional optimization step because the conversion will automatically preserve
the INT4 optimization results, and model inference will eventually benefit from it.

See the `example of a model <https://huggingface.co/TheBloke/Llama-2-7B-Chat-GPTQ>`__
that has been optimized with GPTQ.

You can also refer to the code sample below which shows how to load a 4-bit
GPTQ model and run inference.

.. dropdown:: Using a GPTQ model.

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
