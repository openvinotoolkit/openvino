Managing iGPU Memory Allocation for OpenVINO Inference
========================================================

This guide explains how to reduce and manage memory pressure when running OpenVINO
inference on Intel integrated GPUs (iGPUs), especially in edge deployments and
OpenVINO Model Server (OVMS) text-generation deployments.

Most model-compression, KV-cache, and GPU plugin recommendations in this guide also
apply to discrete GPU deployments. OVMS-specific settings are labelled separately.
Platform-memory guidance is specific to iGPU systems.

Unlike discrete GPUs, iGPUs use host system memory rather than dedicated VRAM. In
practice, reducing iGPU memory pressure usually means reducing the model and runtime
memory footprint, configuring OpenVINO and OVMS to avoid avoidable allocation
failures, and, on supported platforms, adjusting OS or platform graphics-memory
settings.

When to Use This Guide
#######################

Use this guide when an OpenVINO deployment targets ``GPU`` on an integrated GPU and
you see one or more of these symptoms:

* Out-of-memory errors during model compilation or inference.
* Model loading failures for large LLMs, long-context workloads, high-resolution
  vision models, or high-concurrency serving.
* GPU allocation errors where one buffer exceeds the device's maximum
  single-allocation size.
* OVMS text generation that terminates or slows down as the KV cache grows.
* Edge deployments where the iGPU, CPU, OVMS process, operating system, and other
  services all compete for the same host RAM.

For LLM serving, the main memory consumers are model weights, runtime buffers, and
KV cache. KV cache grows with input context length, generated tokens, and request
concurrency.

1. Reduce Model Memory Footprint First
########################################

Reducing the model footprint is usually the most reliable fix because it lowers
pressure on every layer of the stack: model load, graph compilation, runtime
execution, and OVMS serving.

Weight Compression
+++++++++++++++++++

Use OpenVINO-supported compression during model export or preparation:

* INT4 weight compression is usually the recommended first option for LLM
  deployments where model size is the primary memory pressure. It is often the only
  practical option for loading high-accuracy LLMs on memory-constrained hardware.
* INT8 weight compression reduces weight storage compared with higher-precision
  weights and is useful for smaller LLMs or systems where enough RAM can be
  allocated.
* FP16 conversion can reduce model size compared with FP32 with lower accuracy risk
  for many workloads. FP16 or higher precision may be more common for non-LLM use
  cases such as audio, embeddings, and vision models.

For OpenVINO GenAI workflows, export and apply weight compression, for example
``optimum-cli export openvino --weight-format int4`` or ``--weight-format int8``.
For OpenVINO/NNCF workflows, use ``nncf.compress_weights()`` with an appropriate
``CompressWeightsMode``, such as ``INT4_SYM``, ``INT4_ASYM``, or ``INT8_ASYM``. See
`Generative Model Preparation <https://docs.openvino.ai/2026/openvino-workflow-generative/genai-model-preparation.html>`__,
`LLM Weight Compression <https://docs.openvino.ai/2026/openvino-workflow/model-optimization-guide/weight-compression.html>`__,
and the `NNCF documentation <https://openvinotoolkit.github.io/nncf/>`__.

Do not rely on a fixed model-size threshold such as "only for models over N billion
parameters." The decision should be based on measured model load behavior, context
length, expected concurrency, and available memory on the target system.

2. Manage KV-Cache and Request Scheduling Memory
##################################################

For LLMs, KV-cache memory is driven mainly by context length, generated tokens,
precision, and concurrent requests. These drivers apply whether inference is run
through OpenVINO Runtime, OpenVINO GenAI, OVMS, or another serving layer.

KV-Cache Precision for LLMs
+++++++++++++++++++++++++++

KV-cache quantization lowers the precision of Key and Value cache tensors during
LLM inference and can reduce memory use. For the GPU plugin, supported KV-cache
precision options include ``u8``, ``u4``, and ``f16``.

Important GPU behavior:

* For GPU KV-cache quantization, OpenVINO ignores ``DYNAMIC_QUANTIZATION_GROUP_SIZE``
  because that option applies to activation dynamic quantization, not KV-cache
  quantization.
* When KV-cache quantization is enabled on GPU, ``get_state()`` and ``set_state()``
  APIs are not supported.
* When ``KV_CACHE_PRECISION`` is not set, OpenVINO selects the effective KV-cache
  precision based on the model and device; there is no single version-neutral
  ``u8`` or ``u4`` default for every GPU configuration.
* To avoid KV-cache quantization, set ``KV_CACHE_PRECISION`` to ``f16``.
* INT4 KV-cache compression is especially relevant when KV-cache size is
  significant, such as long input prompts above 32K tokens.

Example OpenVINO runtime configuration with an explicit override:

.. code-block:: python

   ov_config = {
       "KV_CACHE_PRECISION": "u8",
   }

Leaving the property unset uses OpenVINO's automatic selection. Setting it
explicitly overrides that selection. If OpenVINO would select ``u4``, overriding it
with ``u8`` increases the KV-cache data precision and memory use relative to
``u4``. Validate accuracy for the target workload when choosing a lower precision
manually.

OpenVINO GenAI Controls
+++++++++++++++++++++++

For OpenVINO GenAI applications:

* Control input context length before generation. When truncation is acceptable
  for the application, OpenVINO GenAI tokenization supports the ``max_length``
  option. See `Tokenization <https://openvinotoolkit.github.io/openvino.genai/docs/guides/tokenization/>`__.
* Limit generated tokens with ``GenerationConfig.max_new_tokens``. See
  `Inference with OpenVINO GenAI <https://docs.openvino.ai/2026/openvino-workflow-generative/inference-with-genai.html>`__.
* Pass ``KV_CACHE_PRECISION`` in the pipeline configuration only when an explicit
  override of OpenVINO's automatic selection is needed.
* For ``ContinuousBatchingPipeline``, limit the number of scheduled sequences with
  ``SchedulerConfig.max_num_seqs``. See
  `SchedulerConfig <https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.SchedulerConfig.html>`__.

OVMS Text-Generation Settings
+++++++++++++++++++++++++++++

The following controls are specific to OVMS text-generation deployments.

``cache_size``
---------------

For OVMS LLM continuous batching, ``cache_size`` is the KV-cache memory size in GB.

* ``cache_size = 0`` or unset means dynamic allocation.
* Dynamic allocation grows as batched requests require more context storage.
* Dynamic allocation can avoid fixed-cache limits, but in some cases it can consume
  all available RAM.
* A fixed ``cache_size`` should be chosen based on available memory, model size,
  expected concurrency, and expected context length.
* If fixed ``cache_size`` is too small, long or concurrent requests may be
  rejected, fail, or terminate before normal stopping criteria are reached.

Start by observing normal-load behavior in OVMS logs. OVMS logs report cache usage,
and current releases also report current KV-cache allocation alongside usage
metrics.

Example:

.. code-block:: sh

   ovms --model_path /models/llm \
     --model_name my-llm \
     --task text_generation \
     --target_device GPU \
     --cache_size 10

For a static cache, choose a value that leaves headroom rather than targeting 100%
cache use.

Prefix Caching
----------------

Prefix caching can improve performance when OVMS requests repeat the same prompt
prefix, such as chat applications that resend conversation history. It avoids
reevaluating the same prefix tokens and can reduce repeated compute work.

Do not tune prefix caching as a primary memory-reduction setting. Its memory
impact is workload-dependent and usually small compared with model weights,
KV-cache precision, cache size, context length, and concurrency.

For OVMS hybrid-attention or linear-attention deployments with prefix caching,
``--cache_interval_multiplier`` controls the interval for linear-attention cache
checkpoints. Larger values can reduce memory use for long contexts but make
checkpoint reuse coarser; smaller values can improve reuse for shorter inputs at
higher memory cost. See the
`OVMS long-context demo <https://github.com/openvinotoolkit/model_server/tree/main/demos/continuous_batching/long_context>`__.

Concurrency and Token Scheduling
----------------------------------

For OVMS, tune these alongside ``cache_size``:

* ``--max_num_seqs``: maximum number of sequences processed together. Higher
  values can increase concurrent KV-cache pressure.
* ``--max_num_batched_tokens``: maximum number of tokens batched together in one
  iteration. The default is already low, so reducing it is usually not the first
  optimization for long prompts. For low-concurrency or single-client
  long-context workloads, increasing this value can improve prompt processing and
  first-token latency. For high-concurrency workloads, lower or bounded values
  can still be useful to keep per-iteration work and scheduling behavior
  predictable.
* ``--rest_workers``: can limit or increase the number of concurrent REST requests
  processed by OVMS.

For memory-constrained systems, avoid setting concurrency parameters higher than
the platform can sustain. Outside OVMS, apply the same principle by limiting
concurrent requests and token budget in the serving layer.

OVMS KV-Cache Precision
--------------------------

OVMS exposes KV-cache precision through the ``--kv_cache_precision``
text-generation parameter. This controls the same OpenVINO KV-cache precision
setting shown earlier, but uses the OVMS command-line interface instead of an
OpenVINO runtime configuration dictionary.

When ``--kv_cache_precision`` is unset, its value is empty and OVMS leaves
precision selection to the model and OpenVINO. OVMS does not apply a separate
fixed KV-cache precision default. Setting the parameter explicitly overrides the
automatic selection.

Example:

.. code-block:: sh

   ovms --model_path /models/llm \
     --model_name my-llm \
     --task text_generation \
     --target_device GPU \
     --kv_cache_precision u8

Example with explicit cache sizing:

.. code-block:: sh

   ovms --model_path /models/llm \
     --model_name my-llm \
     --task text_generation \
     --target_device GPU \
     --cache_size 10 \
     --kv_cache_precision u8

OVMS also accepts OpenVINO plugin configuration through ``--plugin_config``, which
is useful for GPU runtime properties that are unrelated to text-generation
parameters. See
`OpenVINO runtime settings in the OVMS LLM reference <https://docs.openvino.ai/2026/model-server/ovms_docs_llm_reference.html#openvino-runtime-settings>`__.

3. Check GPU Allocation Limits and Runtime Properties
#######################################################

Query Device Memory Properties
++++++++++++++++++++++++++++++

OpenVINO exposes read-only GPU memory properties:

* ``GPU_DEVICE_TOTAL_MEM_SIZE``: for iGPU, this reports host memory size.
* ``GPU_DEVICE_MAX_ALLOC_MEM_SIZE``: maximum size of a single memory object
  allocation.

Python example:

.. code-block:: python

   import openvino as ov

   core = ov.Core()
   total_mem = core.get_property("GPU", "GPU_DEVICE_TOTAL_MEM_SIZE")
   max_alloc = core.get_property("GPU", "GPU_DEVICE_MAX_ALLOC_MEM_SIZE")

   print(f"Total memory: {total_mem / (1024**3):.2f} GB")
   print(f"Max single allocation: {max_alloc / (1024**3):.2f} GB")

C++ example:

.. code-block:: cpp

   #include <openvino/openvino.hpp>
   #include <openvino/runtime/intel_gpu/properties.hpp>

   #include <iomanip>
   #include <iostream>

   int main() {
       ov::Core core;

       auto total_mem = core.get_property("GPU", ov::intel_gpu::device_total_mem_size);
       auto max_alloc = core.get_property("GPU", ov::intel_gpu::device_max_alloc_mem_size);

       std::cout << "Total memory: "
                 << std::fixed << std::setprecision(2)
                 << (total_mem / (1024.0 * 1024.0 * 1024.0))
                 << " GB\n";

       std::cout << "Max single allocation: "
                 << std::fixed << std::setprecision(2)
                 << (max_alloc / (1024.0 * 1024.0 * 1024.0))
                 << " GB\n";

       return 0;
   }

``GPU_ENABLE_LARGE_ALLOCATIONS``
++++++++++++++++++++++++++++++++

OpenVINO's GPU plugin checks the device maximum allocation size. On pre-Xe2
platforms, if a model requires a single allocation larger than the device limit,
``GPU_ENABLE_LARGE_ALLOCATIONS`` can bypass that check and switch addressing mode
to allow allocations larger than 4 GB.

Use this only when the failure is caused by a single allocation exceeding the max
allocation size. This can lower performance because it changes the GPU addressing
mode. On Xe2 and newer platforms, this option is enabled by default. Product
families with Xe2 or newer graphics include select Intel® Core™ Ultra Series 2
processors, Intel® Arc™ B-Series Graphics, and Intel® Core™ Ultra Series 3
processors.

Python example:

.. code-block:: python

   import openvino as ov

   core = ov.Core()
   compiled_model = core.compile_model(
       model,
       "GPU",
       {"GPU_ENABLE_LARGE_ALLOCATIONS": "YES"},
   )

C++ example:

.. code-block:: cpp

   #include <openvino/openvino.hpp>
   #include <openvino/runtime/intel_gpu/properties.hpp>

   ov::Core core;
   auto compiled_model = core.compile_model(
       model,
       "GPU",
       ov::intel_gpu::hint::enable_large_allocations(true)
   );

OVMS example:

.. code-block:: sh

   ovms --model_path /models/my-model \
     --target_device GPU \
     --plugin_config '{"GPU_ENABLE_LARGE_ALLOCATIONS": "YES"}'

Other GPU Plugin Properties
+++++++++++++++++++++++++++

When a GPU plugin property is used as an environment variable, add the ``OV_``
prefix. For example, use ``OV_GPU_ENABLE_SDPA_OPTIMIZATION``, not
``GPU_ENABLE_SDPA_OPTIMIZATION``.

.. list-table::
   :widths: 30 15 55
   :header-rows: 1

   * - Property
     - Default
     - Use
   * - ``GPU_ENABLE_SDPA_OPTIMIZATION``
     - ``true``
     - Keeps SDPA optimized where supported. This is normally expected to remain
       enabled, especially for transformer memory behavior.
   * - ``GPU_ENABLE_KERNELS_REUSE``
     - ``false``
     - Can reduce memory footprint by reusing kernels, but applies only to
       single-stream scenarios and can increase host execution-time overhead.
   * - ``GPU_IMPLS_CACHE_CAPACITY``
     - ``300`` entries
     - Controls the capacity of the LRU implementations cache that is created for
       each program object for dynamic models.
   * - ``GPU_DEVICE_TOTAL_MEM_SIZE``
     - Read-only
     - For iGPU, reports host memory size.
   * - ``GPU_DEVICE_MAX_ALLOC_MEM_SIZE``
     - Read-only
     - Query the maximum single allocation size.

4. Platform and System Memory Considerations
##############################################

iGPU inference uses system memory, so leave enough free RAM for the OS, OVMS or
the application process, model weights, runtime buffers, and KV cache.

Recommendations:

* Size RAM based on the actual model, context length, and concurrency target.
* Avoid unnecessary processes on constrained edge systems.
* Use the fastest supported system memory configuration for the platform, because
  iGPU performance depends heavily on shared memory bandwidth.
* Keep GPU drivers current.

For integrated Intel® Arc™ GPUs on Windows 10 or 11 systems with more than 10 GB
of system memory, OpenVINO's GPU OOM guidance notes that shared memory allocation
can be increased in Intel Graphics Software under
``Graphics > General > Shared GPU Memory Override``, if the setting is available
and not already at its upper limit.

BIOS options such as DVMT, graphics memory reservation, Above 4G Decoding, and
Re-Size BAR are platform-specific. Check them when the target system exposes them
or the platform vendor documents them, but do not treat fixed DVMT values or those
firmware options as OpenVINO requirements.

Quick Reference Checklist
###########################

* For Gen AI model memory pressure, start with INT4 weight compression unless
  accuracy, model size, or RAM headroom makes INT8 a better fit.
* For LLMs, set ``KV_CACHE_PRECISION`` to ``u4`` to reduce KV-cache memory use,
  especially for long contexts above 32K tokens.
* For OpenVINO GenAI, control prompt length, set ``max_new_tokens``, and, for
  ``ContinuousBatchingPipeline``, bound concurrent scheduled sequences with
  ``SchedulerConfig.max_num_seqs``.
* For OVMS deployments, choose ``cache_size`` deliberately and tune
  ``--max_num_seqs``, ``--max_num_batched_tokens``, and request concurrency based
  on available memory and observed logs.
* For OVMS hybrid-attention deployments with prefix caching, tune
  ``--cache_interval_multiplier`` only when memory budget and prefix-cache
  granularity need to be balanced.
* Query ``GPU_DEVICE_MAX_ALLOC_MEM_SIZE`` and ``GPU_DEVICE_TOTAL_MEM_SIZE`` to
  understand the platform limits.
* On pre-Xe2 platforms, enable ``GPU_ENABLE_LARGE_ALLOCATIONS`` only when a
  single allocation exceeds the device max allocation size.
* Keep platform drivers current and check platform-specific graphics memory
  settings only where the target system exposes them.
