// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Namespace, property name, default value, description, [validator]

OV_CONFIG_RELEASE_OPTION(ov::hint, inference_precision, ov::element::f32,
    "Model floating-point inference precision. Supported values: f32, bf16, f16, dynamic",
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::bf16, ov::element::f16, ov::element::dynamic); })
OV_CONFIG_RELEASE_OPTION(ov::hint, performance_mode, ov::hint::PerformanceMode::LATENCY,
    "High-level hint that defines target model inference mode. It may impact number of streams, auto batching, etc")
OV_CONFIG_RELEASE_OPTION(ov::hint, execution_mode, ov::hint::ExecutionMode::PERFORMANCE,
    "High-level hint that defines the most important metric for the model. Performance mode allows unsafe optimizations that may reduce the model accuracy")
OV_CONFIG_RELEASE_OPTION(ov, num_streams, 1, "Defines number of streams to be used for inference. Supported values: ov::streams::NUMA, ov::streams::AUTO or [0, INT32_MAX]",
    [](ov::streams::Num val) {
        return (one_of(val, ov::streams::NUMA, ov::streams::AUTO) || val >= 0);
    })
OV_CONFIG_RELEASE_OPTION(ov, inference_num_threads, 0, "Defines maximum number of threads that can be used for inference tasks")
OV_CONFIG_RELEASE_OPTION(ov::hint, num_requests, 0, "Provides number of requests populated by the application")
OV_CONFIG_RELEASE_OPTION(ov::internal, exclusive_async_requests, false, "")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_cpu_pinning, false, "Controls if CPU threads are pinned to the cores or not")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_cpu_reservation, false, "Reserve cpus which will not be used by other plugin or compiled model")
OV_CONFIG_RELEASE_OPTION(ov::hint, enable_hyper_threading, false, "Defined if hyper threading is used during inference")
OV_CONFIG_RELEASE_OPTION(ov::hint, scheduling_core_type, ov::hint::SchedulingCoreType::ANY_CORE, "Defines CPU core type which can be used during inference")
OV_CONFIG_RELEASE_OPTION(ov::hint, model_distribution_policy, {},
    "Defines model distribution policy for inference with multiple sockets/devices. Supported values: TENSOR_PARALLEL",
    [](std::set<ov::hint::ModelDistributionPolicy> val) {
        for (auto& row : val) {
            if ((row != ov::hint::ModelDistributionPolicy::TENSOR_PARALLEL))
                return false;
        }
        return true;
    })
OV_CONFIG_RELEASE_OPTION(ov::hint, kv_cache_precision, ov::element::f32,
    "Specifies precision for kv cache compression. Supported values: f32, bf16, f16, u8",
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::f16, ov::element::bf16, ov::element::u8); })
OV_CONFIG_RELEASE_OPTION(ov, key_cache_precision, ov::element::f32,
    "Specifies precision for key cache compression. Supported values: f32, bf16, f16, u8",
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::f16, ov::element::bf16, ov::element::u8); })
OV_CONFIG_RELEASE_OPTION(ov, value_cache_precision, ov::element::f32,
    "Specifies precision for value cache compression. Supported values: f32, bf16, f16, u8, u4",
    [](ov::element::Type val) { return one_of(val, ov::element::f32, ov::element::f16, ov::element::bf16, ov::element::u8, ov::element::u4); })
OV_CONFIG_RELEASE_OPTION(ov::hint, dynamic_quantization_group_size, 0,
    "Defines group size for dynamic quantization optimization. Supported values: [0, UINT64_MAX], where 0 - disabled DQ, UINT64_MAX - per-tensor DQ")
OV_CONFIG_RELEASE_OPTION(ov, key_cache_group_size, 0,
    "Defines group size for key cache compression. Supported values: [0, UINT64_MAX], where 0 - disabled compression, UINT64_MAX - per-tensor compression")
OV_CONFIG_RELEASE_OPTION(ov, value_cache_group_size, 0,
    "Defines group size for key cache compression. Supported values: [0, UINT64_MAX], where 0 - disabled compression, UINT64_MAX - per-tensor compression")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, key_cache_quant_mode, CacheQuantMode::AUTO,
    "Defines quantization mode for key cache. Supported values: AUTO, BY_CHANNEL, BY_HIDDEN")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, sparse_weights_decompression_rate, 1.0f,
    "Defines threshold for sparse weights decompression feature activation (1.0 means the feature is disabled). Supported values: [0.0f, 1.0f]",
    [](float val) { return val >= 0.f && val <= 1.f; })
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, denormals_optimization, DenormalsOptimization::Mode::DEFAULT,
    "Defines whether to perform denormals optimization (enables FTZ and DAZ). Supported values: ov::intel_cpu::DenormalsOptimization::Mode::DEFAULT/ON/OFF")
OV_CONFIG_RELEASE_OPTION(ov::internal, enable_lp_transformations, false, "Defines if Low Precision Trasformations (LPT) should be enabled")
OV_CONFIG_RELEASE_OPTION(ov, enable_profiling, false, "Enable profiling for the plugin")
OV_CONFIG_RELEASE_OPTION(ov::log, level, ov::log::Level::NO, "Defines Log level")
OV_CONFIG_RELEASE_OPTION(ov::device, id, "", "ID of the current device. Supports only empty device id",
    [](std::string val) { return val.empty(); })

OV_CONFIG_RELEASE_OPTION(ov, cache_encryption_callbacks, EncryptionCallbacks{}, "Callbacks used to encrypt/decrypt the model")
OV_CONFIG_RELEASE_OPTION(ov::internal, caching_with_mmap, true, "Defines if caching with mmap should be enabled")
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, cpu_runtime_cache_capacity, 0,
    "Defines how many records can be stored in the CPU runtime parameters cache per CPU runtime parameter type per stream. Supported values: [0, INT32_MAX]",
    [](int val) { return val >= 0; })
OV_CONFIG_RELEASE_OPTION(ov::intel_cpu, snippets_mode, SnippetsMode::ENABLE,
    "Defines Snippets code generation pipeline mode. Supported values: ov::intel_cpu::SnippetsMode::ENABLE/DISABLE/IGNORE_CALLBACK",

[](ov::intel_cpu::SnippetsMode val) { return one_of(val, ov::intel_cpu::SnippetsMode::ENABLE, ov::intel_cpu::SnippetsMode::DISABLE, ov::intel_cpu::SnippetsMode::IGNORE_CALLBACK); })
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_cpu, acl_fast_math, false, "Defines if ACL fast-math mode should be enabled")
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_cpu, model_type, ov::intel_cpu::ModelType::UNKNOWN,
    "Defines model type hint. Supported values: ov::intel_cpu::SnippetsMode::UNKNOWN/CNN/LLM")

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