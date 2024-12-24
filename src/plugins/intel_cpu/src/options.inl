// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Namespace, property name, default value, [validator], description

// OV_CONFIG_RELEASE_OPTION(ov, cache_dir, "", "Directory where model cache can be stored. Caching is disabled if empty") // ???
OV_CONFIG_RELEASE_OPTION(ov::hint, inference_precision, ov::element::f32,
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::undefined); },
    "Model floating-point inference precision. Supported values: f32, bf16, f16, undefined")
OV_CONFIG_RELEASE_OPTION(ov::hint, performance_mode, ov::hint::PerformanceMode::LATENCY,
    "High-level hint that defines target model inference mode. It may impact number of streams, auto batching, etc")
OV_CONFIG_RELEASE_OPTION(ov::hint, execution_mode, ov::hint::ExecutionMode::PERFORMANCE,
     "High-level hint that defines the most important metric for the model. Performance mode allows unsafe optimizations that may reduce the model accuracy")

OV_CONFIG_RELEASE_OPTION(ov, num_streams, 1, "Defines number of streams to be used for inference")
OV_CONFIG_RELEASE_OPTION(ov, inference_num_threads, 0, "Defines maximum number of threads that can be used for inference tasks")
OV_CONFIG_RELEASE_OPTION(ov::hint, num_requests, 0, "Provides number of requests populated by the application") // TODO: Do we need validator?
OV_CONFIG_RELEASE_OPTION(ov::internal, exclusive_async_requests, false, "")

OV_CONFIG_RELEASE_OPTION(ov::hint, enable_cpu_pinning, false, "Controls if CPU threads are pinned to the cores or not")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_hyper_threading, false, "Defined if hyper threading is used during inference")
OV_CONFIG_RELEASE_OPTION(ov::hint, scheduling_core_type, ov::hint::SchedulingCoreType::ANY_CORE, "Defines CPU core type which can be used during inference")

OV_CONFIG_RELEASE_OPTION(ov::hint, model_distribution_policy, {},
    [](std::set<ov::hint::ModelDistributionPolicy> val) {
        for (auto& row : val) {
            if ((row != ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL))
                return false;
        }
        return true;
    },
    "Defines model distribution policy for inference with multiple sockets/devices. Supported values: TENSOR_PARALLEL")


OV_CONFIG_RELEASE_OPTION(ov::hint, dynamic_quantization_group_size, 0,
    "Defines group size for dynamic quantization optimization. Supported values: [0, UINT64_MAX], where 0 - disabled DQ, UINT64_MAX - per-tensor DQ")
OV_CONFIG_RELEASE_OPTION(ov::hint, kv_cache_precision, ov::element::f32,
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::f16, ov::element::bf16, ov::element::u8); },
    "Specifies precision for kv cache compression. Supported values: f32, bf16, f16, u8")

OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, cpu_runtime_cache_capacity, 0,
    [](int val) { return val >= 0; },
    "Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per stream. Supported values: [0, INT32_MAX]")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, sparse_weights_decompression_rate, 1.0f,
    [](float val) { return val >= 0.f && val <= 1.f; },
    "Defines threshold for sparse weights decompression feature activation (1.0 means the feature is disabled). Supported values: [0.0f, 1.0f]")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, denormals_optimization, nullptr,
    "DefineS whether to perform denormals optimization (enables FTZ and DAZ)")

OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, lp_transforms_mode, false, "Defines if Low Precision Trasformations (LPT) should be enabled")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, snippets_mode, SnippetsMode::ENABLE,
    [](ov::intel_cpu::SnippetsMode val) { return one_of(val, ov::intel_cpu::SnippetsMode::ENABLE, ov::intel_cpu::SnippetsMode::DISABLE, ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK); },
    "Defines Snippets code generation pipeline mode. Supported values: ov::intel_cpu::SnippetsMode::ENABLE/DISABLE/IGNORE_CALLBACK")

OV_CONFIG_RELEASE_OPTION(ov, enable_profiling, false, "Enable profiling for the plugin")
OV_CONFIG_RELEASE_OPTION(ov::log, level, ov::log::Level::NO, "Defines Log level")
OV_CONFIG_RELEASE_OPTION(ov::device, id, "", "ID of the current device")

OV_CONFIG_RELEASE_OPTION(ov, cache_encryption_callbacks, EncryptionCallbacks{}, "Callbacks used to encrypt/decrypt the model")
OV_CONFIG_RELEASE_OPTION(ov::internal, caching_with_mmap, true, "Defines if caching with mmap should be enabled")

#if defined(OV_CPU_WITH_ACL)
    OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, acl_fast_math, false, "Defines if ACL fast-math mode should be enabled")
#endif

OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, verbose, "0", "Enables logging for debugging purposes.")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, exec_graph_path, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, average_counters, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_dir, "cpu_dump", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_format, BlobDumpFormat::TEXT, "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_node_exec_id, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_node_ports, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_node_type, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, blob_dump_node_name, "", "")
OV_CONFIG_DEBUG_OPTION(ov::intel_cpu, summary_perf, "", "")