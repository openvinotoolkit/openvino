// Copyright (C) 2018-2022 Intel Corporation
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

/**
 * @brief Defines queue type that must be used for model execution
 */
static constexpr Property<QueueTypes, PropertyMutability::RW> queue_type{"GPU_QUEUE_TYPE"};

static constexpr Property<bool, PropertyMutability::RW> enable_memory_pool{"GPU_ENABLE_MEMORY_POOL"};
static constexpr Property<bool, PropertyMutability::RW> optimize_data{"GPU_OPTIMIZE_DATA"};
static constexpr Property<bool, PropertyMutability::RW> allow_static_input_reorder{"GPU_ALLOW_STATIC_INPUT_REORDER"};
static constexpr Property<bool, PropertyMutability::RW> partial_build_program{"GPU_PARTIAL_BUILD"};
static constexpr Property<bool, PropertyMutability::RW> allow_new_shape_infer{"GPU_ALLOW_NEW_SHAPE_INFER"};
static constexpr Property<std::string, PropertyMutability::RW> dump_graphs{"GPU_DUMP_GRAPHS"};
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> custom_outputs{"GPU_CUSTOM_OUTPUTS"};

/// @brief Tuning mode.
enum class TuningMode {
    /// @brief Tuning is disabled.
    tuning_disabled,

    /// @brief Tuning using the cached data (no on-line tuning for non-existing data).
    tuning_use_cache,

    /// @brief Tuning using the cached data if exist, tune and update cache otherwise.
    tuning_tune_and_cache,

    /// @brief Tuning using the cached data and update tasks.
    /// @details Performs updating tasks like removal of invalid caches, promoting to new format, etc.
    /// No tuning for non-existing data.
    tuning_use_and_update,

    /// @brief Retune the cache data even if it exists.
    tuning_retune_and_cache
};

struct TuningConfig {
    TuningMode mode;
    std::string cache_file_path;

    TuningConfig() : mode(TuningMode::tuning_disabled), cache_file_path("") {}
};

inline std::ostream& operator<<(std::ostream& os, const TuningConfig& val) {
    os << val.cache_file_path;
    return os;
}

static constexpr Property<TuningConfig, PropertyMutability::RW> tuning_config{"GPU_TUNING_CONFIG"};

static constexpr Property<ImplForcingMap, PropertyMutability::RW> force_implementations{"GPU_FORCE_IMPLEMENTATIONS"};
static constexpr Property<std::string, PropertyMutability::RW> config_file{"CONFIG_FILE"};
static constexpr Property<bool, PropertyMutability::RW> enable_lp_transformations{"LP_TRANSFORMS_MODE"};
static constexpr Property<bool, PropertyMutability::RW> enable_dynamic_batch{"DYN_BATCH_ENABLED"};
static constexpr Property<size_t, PropertyMutability::RW> max_dynamic_batch{"DYN_BATCH_LIMIT"};
static constexpr Property<bool, PropertyMutability::RW> exclusive_async_requests{"EXCLUSIVE_ASYNC_REQUESTS"};
static constexpr Property<bool, PropertyMutability::RW> nv12_two_inputs{"GPU_NV12_TWO_INPUTS"};

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::QueueTypes;
}  // namespace cldnn
