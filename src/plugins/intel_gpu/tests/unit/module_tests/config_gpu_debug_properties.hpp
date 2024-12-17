// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

#ifdef GPU_DEBUG_CONFIG

namespace ov {
namespace intel_gpu {

static constexpr Property<bool, ov::PropertyMutability::RW> verbose{"VERBOSE"};
static constexpr Property<bool, ov::PropertyMutability::RW> help{"HELP"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_usm{"DISABLE_USM"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_onednn_post_ops{"DISABLE_ONEDNN_POST_OPS"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_profiling_data{"DUMP_PROFILING_DATA"};
// static constexpr Property<std::string, ov::PropertyMutability::RW> dump_graphs{"DUMP_GRAPHS"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_sources{"DUMP_SOURCES"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_tensors{"DUMP_TENSORS"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_memory_pool{"DUMP_MEMORY_POOL"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_iterations{"DUMP_ITERATIONS"};
static constexpr Property<bool, ov::PropertyMutability::RW> host_time_profiling{"HOST_TIME_PROFILING"};
// static constexpr Property<size_t, ov::PropertyMutability::RW> max_kernels_per_batch{"MAX_KERNELS_PER_BATCH"};
static constexpr Property<size_t, ov::PropertyMutability::RW> impls_cache_capacity{"IMPLS_CACHE_CAPACITY"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_async_compilation{"DISABLE_ASYNC_COMPILATION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_shape_agnostic_impls{"DISABLE_SHAPE_AGNOSTIC_IMPLS"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_runtime_buffer_fusing{"DISABLE_RUNTIME_BUFFER_FUSING"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_memory_reuse{"DISABLE_MEMORY_REUSE"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_post_ops_fusions{"DISABLE_POST_OPS_FUSIONS"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_horizontal_fc_fusion{"DISABLE_HORIZONTAL_FC_FUSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> use_usm_host{"USE_USM_HOST"};
static constexpr Property<bool, ov::PropertyMutability::RW> enable_kv_cache_compression{"ENABLE_KV_CACHE_COMPRESSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> asym_dynamic_quantization{"ASYM_DYNAMIC_QUANTIZATION"};
static constexpr Property<std::string, ov::PropertyMutability::RW> mem_prealloc_options{"MEM_PREALLOC_OPTIONS"};
static constexpr Property<std::string, ov::PropertyMutability::RW> load_dump_raw_binary{"LOAD_DUMP_RAW_BINARY"};

}  // namespace intel_gpu
}  // namespace ov

#endif  // GPU_DEBUG_CONFIG
