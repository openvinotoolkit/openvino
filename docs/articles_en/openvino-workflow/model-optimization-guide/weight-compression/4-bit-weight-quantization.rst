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

* ``ignored_scope`` - excludes selected layers from compression to preserve accuracy. See
  :ref:`Exclude layers from compression <wc_ignored_scope>`.

* ``scoped_configs`` - assigns custom compression settings (``mode`` and ``group_size``) to
  selected layers. See
  :ref:`Set per-layer compression settings <wc_scoped_configs>`.

.. tip::

   NNCF enables you to stack the supported optimization methods. For example, AWQ,
   Scale Estimation and GPTQ methods may be enabled all together to achieve better accuracy.

.. _wc_ignored_scope:

Exclude layers from compression (ignored_scope)
###############################################

The ``ignored_scope`` parameter excludes selected layers from compression, for example, the
last layer of the model. Excluded layers keep their original floating-point precision.

Layers are selected with ``nncf.Scope`` by the following rules:

* ``names`` - a list of node names.
* ``patterns`` - a list of regular expressions matched against node names.
* ``types`` - a list of operation types. The types follow the framework of the model, for
  example, ``MatMul`` or ``Convolution`` for OpenVINO models.
* ``subgraphs`` - a list of ``nncf.Subgraph`` objects. All nodes along all simple paths in the
  graph from the input to the output nodes are selected.

By default, each rule must match at least one node of the model, otherwise an error is raised.
Set ``validate=False`` to skip this check.

.. note::

   ``nncf.Scope`` is available starting with NNCF 3.5.0. In earlier versions, use
   ``nncf.IgnoredScope``, which accepts the same arguments.

Examples:

.. code-block:: python

   import nncf

   # Exclude by node name
   names = ["layer_1", "layer_2", "layer_3"]
   compressed_model = nncf.compress_weights(
     model,
     mode=nncf.CompressWeightsMode.INT4_SYM,
     ignored_scope=nncf.Scope(names=names),
   )

   # Exclude by operation type
   types = ["Convolution"]
   compressed_model = nncf.compress_weights(
     model,
     mode=nncf.CompressWeightsMode.INT4_SYM,
     ignored_scope=nncf.Scope(types=types),
   )

   # Exclude by regular expression
   regex = [".*layer_.*"]
   compressed_model = nncf.compress_weights(
     model,
     mode=nncf.CompressWeightsMode.INT4_SYM,
     ignored_scope=nncf.Scope(patterns=regex),
   )

   # Exclude by subgraph
   subgraph = nncf.Subgraph(inputs=["layer_1", "layer_2"], outputs=["layer_3"])
   compressed_model = nncf.compress_weights(
     model,
     mode=nncf.CompressWeightsMode.INT4_SYM,
     ignored_scope=nncf.Scope(subgraphs=[subgraph]),
   )

.. _wc_scoped_configs:

Set per-layer compression settings (scoped_configs)
###################################################

.. note::

   ``scoped_configs`` is available starting with NNCF 3.5.0.

The ``scoped_configs`` parameter is a list of ``nncf.ScopedConfig`` objects. Each of them
assigns a compression ``mode`` and ``group_size`` to selected layers, while the rest of the model
is compressed with the main settings. If ``group_size`` is not set, the default group size of
the given ``mode`` is used. The value ``-1`` means per-channel quantization.

A scoped config selects layers by the same rules as ``ignored_scope``: ``names``, ``patterns``,
``types`` and ``subgraphs``, and also accepts the ``validate`` argument.

In the example below, a Mixture-of-Experts model is compressed to INT4 with the main settings,
which covers the experts, while the attention and router layers are kept in 8 bits to preserve
accuracy:

.. code-block:: python

   import nncf

   compressed_model = nncf.compress_weights(
     model,
     # Main settings: applied to all layers that are not matched by a scoped config,
     # including the experts
     mode=nncf.CompressWeightsMode.INT4_SYM,
     group_size=64,
     ratio=1.0,
     scoped_configs=[
       # Keep the attention and router layers in INT8 per-channel to preserve accuracy
       nncf.ScopedConfig(
         patterns=[".*self_attn.*", ".*router.*"],
         mode=nncf.CompressWeightsMode.INT8_ASYM,
         group_size=-1,
       ),
       # Compress the expert down projections to asymmetric INT4 with a larger group size
       nncf.ScopedConfig(
         patterns=[".*mlp.down_proj.*"],
         mode=nncf.CompressWeightsMode.INT4_ASYM,
         group_size=128,
       ),
     ],
   )

The compression settings of a layer are selected as follows:

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Layer matched by
     - Result
   * - One or more scoped configs
     - Compressed with the settings of the scoped config. If several scoped configs match the
       layer, the last one in the list is applied.
   * - ``ignored_scope`` and a scoped config
     - Compressed with the settings of the scoped config. Scoped configs take precedence over
       ``ignored_scope``.
   * - ``ignored_scope`` only
     - Not compressed.
   * - Neither
     - Compressed with the main settings: ``mode``, ``group_size``, ``ratio``, ``backup_mode``
       and ``all_layers``. The ``ratio`` is counted only over these layers.

Limitations:

* Codebook compression modes are not supported by scoped configs. Passing one raises an error.
* If the ``mode`` of a scoped config is not supported for the data type of a layer's weights,
  the scoped config is not applied to that layer, and NNCF logs a warning.

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
