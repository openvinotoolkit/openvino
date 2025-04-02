// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/shape_predictor.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"

#include "intel_gpu/primitives/implementation_desc.hpp"
namespace ov::intel_gpu {

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

inline std::istream& operator>>(std::istream& is, QueueTypes& val) {
    std::string str;
    is >> str;
    if (str == "in-order") {
        val = QueueTypes::in_order;
    } else if (str == "out-of-order") {
        val = QueueTypes::out_of_order;
    } else {
        OPENVINO_THROW("Unsupported QueueTypes value: ", str);
    }
    return is;
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

inline std::istream& operator>>(std::istream& is, DumpFormat& val) {
    std::string str;
    is >> str;
    if (str == "binary") {
        val = DumpFormat::binary;
    } else if (str == "text") {
        val = DumpFormat::text;
    } else if (str == "text_raw") {
        val = DumpFormat::text_raw;
    } else {
        OPENVINO_THROW("Unsupported DumpFormat value: ", str);
    }
    return is;
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

inline std::istream& operator>>(std::istream& is, DumpTensors& val) {
    std::string str;
    is >> str;
    if (str == "all") {
        val = DumpTensors::all;
    } else if (str == "in") {
        val = DumpTensors::in;
    } else if (str == "out") {
        val = DumpTensors::out;
    } else {
        OPENVINO_THROW("Unsupported DumpTensors value: ", str);
    }
    return is;
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
static constexpr Property<float, PropertyMutability::RW> buffers_preallocation_ratio{"GPU_BUFFERS_PREALLOCATION_RATIO"};
static constexpr Property<size_t, PropertyMutability::RW> max_kernels_per_batch{"GPU_MAX_KERNELS_PER_BATCH"};
static constexpr Property<bool, PropertyMutability::RW> use_onednn{"GPU_USE_ONEDNN"};
static constexpr Property<bool, PropertyMutability::RW> use_cm{"GPU_USE_CM"};

static constexpr Property<bool, ov::PropertyMutability::RW> help{"HELP"};
static constexpr Property<size_t, ov::PropertyMutability::RW> verbose{"VERBOSE"};
static constexpr Property<bool, ov::PropertyMutability::RW> verbose_color{"VERBOSE_COLOR"};
static constexpr Property<std::string, ov::PropertyMutability::RW> debug_config{"GPU_DEBUG_CONFIG"};
static constexpr Property<std::string, ov::PropertyMutability::RW> log_to_file{"GPU_LOG_TO_FILE"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_usm{"GPU_DISABLE_USM"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_onednn_post_ops_opt{"GPU_DISABLE_ONEDNN_POST_OPS_OPT"};
static constexpr Property<std::string, PropertyMutability::RW> dump_graphs_path{"GPU_DUMP_GRAPHS_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_profiling_data_path{"GPU_DUMP_PROFILING_DATA_PATH"};
static constexpr Property<bool, ov::PropertyMutability::RW> dump_profiling_data_per_iter{"GPU_DUMP_PROFILING_DATA_PER_ITER"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_sources_path{"GPU_DUMP_SOURCES_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_tensors_path{"GPU_DUMP_TENSORS_PATH"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dry_run_path{"GPU_DRY_RUN_PATH"};
static constexpr Property<DumpTensors, ov::PropertyMutability::RW> dump_tensors{"GPU_DUMP_TENSORS"};
static constexpr Property<std::vector<std::string>, ov::PropertyMutability::RW> dump_layer_names{"GPU_DUMP_LAYER_NAMES"};
static constexpr Property<DumpFormat, ov::PropertyMutability::RW> dump_tensors_format{"GPU_DUMP_TENSORS_FORMAT"};
static constexpr Property<std::string, ov::PropertyMutability::RW> dump_memory_pool_path{"GPU_DUMP_MEMORY_POOL_PATH"};
static constexpr Property<bool, ov::PropertyMutability::RW> dump_memory_pool{"GPU_DUMP_MEMORY_POOL"};
static constexpr Property<int32_t, ov::PropertyMutability::RW> dump_batch_limit{"GPU_DUMP_BATCH_LIMIT"};
static constexpr Property<std::set<int64_t>, ov::PropertyMutability::RW> dump_iterations{"GPU_DUMP_ITERATIONS"};
static constexpr Property<size_t, ov::PropertyMutability::RW> host_time_profiling{"GPU_HOST_TIME_PROFILING"};
static constexpr Property<size_t, ov::PropertyMutability::RW> impls_cache_capacity{"GPU_IMPLS_CACHE_CAPACITY"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_async_compilation{"GPU_DISABLE_ASYNC_COMPILATION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_runtime_buffer_fusing{"GPU_DISABLE_RUNTIME_BUFFER_FUSING"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_memory_reuse{"GPU_DISABLE_MEMORY_REUSE"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_post_ops_fusions{"GPU_DISABLE_POST_OPS_FUSIONS"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_horizontal_fc_fusion{"GPU_DISABLE_HORIZONTAL_FC_FUSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_fc_swiglu_fusion{"GPU_DISABLE_FC_SWIGLU_FUSION"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_fake_alignment{"GPU_DISABLE_FAKE_ALIGNMENT"};
static constexpr Property<bool, ov::PropertyMutability::RW> disable_runtime_skip_reorder{"GPU_DISABLE_RUNTIME_SKIP_REORDER"};
static constexpr Property<size_t, ov::PropertyMutability::RW> usm_policy{"GPU_USM_POLICY"};
static constexpr Property<bool, ov::PropertyMutability::RW> asym_dynamic_quantization{"GPU_ASYM_DYNAMIC_QUANTIZATION"};
static constexpr Property<ShapePredictor::Settings, ov::PropertyMutability::RW> shape_predictor_settings{"GPU_SHAPE_PREDICTOR_SETTINGS"};
static constexpr Property<std::vector<std::string>, ov::PropertyMutability::RW> load_dump_raw_binary{"GPU_LOAD_DUMP_RAW_BINARY"};
static constexpr Property<std::vector<std::string>, ov::PropertyMutability::RW> start_after_processes{"GPU_START_AFTER_PROCESSES"};

}  // namespace ov::intel_gpu

namespace cldnn {
using ov::intel_gpu::QueueTypes;
}  // namespace cldnn
