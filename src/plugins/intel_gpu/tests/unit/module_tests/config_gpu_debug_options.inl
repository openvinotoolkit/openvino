// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef GPU_DEBUG_CONFIG
OV_CONFIG_OPTION(ov::intel_gpu, verbose, false, "Enable")
OV_CONFIG_OPTION(ov::intel_gpu, help, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_usm, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_onednn_post_ops, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_profiling_data, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_graphs, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_sources, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_tensors, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_memory_pool, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, dump_iterations, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, host_time_profiling, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, max_kernels_per_batch, 8, "")
OV_CONFIG_OPTION(ov::intel_gpu, impls_cache_capacity, 0, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_async_compilation, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_shape_agnostic_impls, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_runtime_buffer_fusing, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_memory_reuse, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_post_ops_fusions, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, disable_horizontal_fc_fusion, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, use_usm_host, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, enable_kv_cache_compression, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, asym_dynamic_quantization, false, "")
OV_CONFIG_OPTION(ov::intel_gpu, mem_prealloc_options, "", "")
OV_CONFIG_OPTION(ov::intel_gpu, load_dump_raw_binary, "", "")

#endif
