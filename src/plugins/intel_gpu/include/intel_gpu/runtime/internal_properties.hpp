// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include "intel_gpu/primitives/implementation_desc.hpp"
namespace ov {
namespace intel_gpu {

/**
 * @brief Read-only property to get GPU driver version
 */
static constexpr Property<std::string, PropertyMutability::RO> driver_version{"GPU_DRIVER_VERSION"};

/**
 * @brief Read-only property to get GPU driver version
 */
static constexpr Property<std::string, PropertyMutability::RO> device_id{"GPU_DEVICE_ID"};

enum class QueueTypes : int16_t {
    in_order,
    out_of_order
};

inline std::ostream& operator<<(std::ostream& os, const QueueTypes& val) {
    switch (val) {
        case QueueTypes::in_order: os << "in-order"; break;
        case QueueTypes::out_of_order: os << "out-of-order"; break;
        default: os << "unknown";
    }

    return os;
}

enum class DumpFormat : uint8_t {
    binary = 0,
    text = 1,
    text_raw = 2,
};

inline std::ostream& operator<<(std::ostream& os, const DumpFormat& val) {
    switch (val) {
        case DumpFormat::binary: os << "binary"; break;
        case DumpFormat::text: os << "text"; break;
        case DumpFormat::text_raw: os << "text_raw"; break;
        default: os << "unknown";
    }

    return os;
}

enum class DumpTensors : uint8_t {
    all = 0,
    in = 1,
    out = 2,
};

inline std::ostream& operator<<(std::ostream& os, const DumpTensors& val) {
    switch (val) {
        case DumpTensors::all: os << "all"; break;
        case DumpTensors::in: os << "in"; break;
        case DumpTensors::out: os << "out"; break;
        default: os << "unknown";
    }

    return os;
}

/**
 * @brief Defines queue type that must be used for model execution
 */
static constexpr Property<QueueTypes, PropertyMutability::RW> queue_type{"GPU_QUEUE_TYPE"};

static constexpr Property<bool, PropertyMutability::RW> enable_memory_pool{"GPU_ENABLE_MEMORY_POOL"};
static constexpr Property<bool, PropertyMutability::RW> optimize_data{"GPU_OPTIMIZE_DATA"};
static constexpr Property<bool, PropertyMutability::RW> allow_static_input_reorder{"GPU_ALLOW_STATIC_INPUT_REORDER"};
static constexpr Property<bool, PropertyMutability::RW> partial_build_program{"GPU_PARTIAL_BUILD"};
static constexpr Property<bool, PropertyMutability::RW> allow_new_shape_infer{"GPU_ALLOW_NEW_SHAPE_INFER"};
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> custom_outputs{"GPU_CUSTOM_OUTPUTS"};
static constexpr Property<ImplForcingMap, PropertyMutability::RW> force_implementations{"GPU_FORCE_IMPLEMENTATIONS"};
static constexpr Property<std::string, PropertyMutability::RW> config_file{"CONFIG_FILE"};
static constexpr Property<bool, PropertyMutability::RW> enable_lp_transformations{"LP_TRANSFORMS_MODE"};
static constexpr Property<float, PropertyMutability::RW> buffers_preallocation_ratio{"GPU_BUFFERS_PREALLOCATION_RATIO"};
static constexpr Property<size_t, PropertyMutability::RW> max_kernels_per_batch{"GPU_MAX_KERNELS_PER_BATCH"};
static constexpr Property<bool, PropertyMutability::RW> use_onednn{"USE_ONEDNN"};

static constexpr Property<bool, ov::PropertyMutability::RW> help{"HELP"};
static constexpr Property<size_t, ov::PropertyMutability::RW> verbose{"VERBOSE"};
static constexpr Property<std::string, ov::PropertyMutability::RW> log_to_file{"GPU_LOG_TO_FILE"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_usm{"GPU_DISABLE_USM"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_onednn_post_ops_opt{"DISABLE_ONEDNN_POST_OPS_OPT"};
static constexpr Property<std::string, PropertyMutability::RW> dump_graphs_path{"GPU_DUMP_GRAPHS_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_profiling_data_path{"GPU_DUMP_PROFILING_DATA_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_sources_path{"GPU_DUMP_SOURCES_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_tensors_path{"GPU_DUMP_TENSORS_PATH"};
static constexpr Property<DumpTensors, ov::PropertyMutability::RW> dump_tensors{"DUMP_TENSORS"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_layers{"GPU_DUMP_LAYERS"};
static constexpr Property<DumpFormat, ov::PropertyMutability::RW> dump_tensors_format{"DUMP_TENSORS_FORMAT"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_memory_pool_path{"GPU_DUMP_MEMORY_POOL_PATH"};
static constexpr Property<int32_t, ov::PropertyMutability::RW> dump_batch_limit{"GPU_DUMP_BATCH_LIMIT"};
static constexpr Property<std::set<int64_t>, ov::PropertyMutability::RW> dump_iterations{"GPU_DUMP_ITERATIONS"};
static constexpr Property<bool, ov::PropertyMutability::RW> host_time_profiling{"HOST_TIME_PROFILING"};
static constexpr Property<size_t, ov::PropertyMutability::RW> impls_cache_capacity{"IMPLS_CACHE_CAPACITY"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_async_compilation{"DISABLE_ASYNC_COMPILATION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_runtime_buffer_fusing{"DISABLE_RUNTIME_BUFFER_FUSING"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_memory_reuse{"DISABLE_MEMORY_REUSE"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_post_ops_fusions{"DISABLE_POST_OPS_FUSIONS"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_horizontal_fc_fusion{"DISABLE_HORIZONTAL_FC_FUSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_fc_swiglu_fusion{"DISABLE_FC_SWIGLU_FUSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_fake_alignment{"DISABLE_FAKE_ALIGNMENT"};
static constexpr Property<bool, ov::PropertyMutability::RW> use_usm_host{"USE_USM_HOST"};
static constexpr Property<bool, ov::PropertyMutability::RW> asym_dynamic_quantization{"ASYM_DYNAMIC_QUANTIZATION"};
static constexpr Property<std::string, ov::PropertyMutability::RW> mem_prealloc_options{"MEM_PREALLOC_OPTIONS"};
static constexpr Property<std::vector<std::string>, ov::PropertyMutability::RW> load_dump_raw_binary{"LOAD_DUMP_RAW_BINARY"};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::QueueTypes;
}  // namespace cldnn
