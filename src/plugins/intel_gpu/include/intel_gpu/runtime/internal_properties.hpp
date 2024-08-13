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

/**
 * @brief Defines queue type that must be used for model execution
 */
static constexpr Property<QueueTypes, PropertyMutability::RW> queue_type{"GPU_QUEUE_TYPE"};

static constexpr Property<bool, PropertyMutability::RW> enable_memory_pool{"GPU_ENABLE_MEMORY_POOL"};
static constexpr Property<bool, PropertyMutability::RW> optimize_data{"GPU_OPTIMIZE_DATA"};
static constexpr Property<bool, PropertyMutability::RW> allow_static_input_reorder{"GPU_ALLOW_STATIC_INPUT_REORDER"};
static constexpr Property<bool, PropertyMutability::RW> partial_build_program{"GPU_PARTIAL_BUILD"};
static constexpr Property<bool, PropertyMutability::RW> allow_new_shape_infer{"GPU_ALLOW_NEW_SHAPE_INFER"};
static constexpr Property<bool, PropertyMutability::RW> use_only_static_kernels_for_dynamic_shape{"GPU_USE_ONLY_STATIC_KERNELS_FOR_DYNAMIC_SHAPE"};
static constexpr Property<std::string, PropertyMutability::RW> dump_graphs{"GPU_DUMP_GRAPHS"};
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> custom_outputs{"GPU_CUSTOM_OUTPUTS"};
static constexpr Property<ImplForcingMap, PropertyMutability::RW> force_implementations{"GPU_FORCE_IMPLEMENTATIONS"};
static constexpr Property<std::string, PropertyMutability::RW> config_file{"CONFIG_FILE"};
static constexpr Property<bool, PropertyMutability::RW> enable_lp_transformations{"LP_TRANSFORMS_MODE"};
static constexpr Property<size_t, PropertyMutability::RW> max_dynamic_batch{"DYN_BATCH_LIMIT"};
static constexpr Property<bool, PropertyMutability::RW> nv12_two_inputs{"GPU_NV12_TWO_INPUTS"};
static constexpr Property<float, PropertyMutability::RW> buffers_preallocation_ratio{"GPU_BUFFERS_PREALLOCATION_RATIO"};
static constexpr Property<size_t, PropertyMutability::RW> max_kernels_per_batch{"GPU_MAX_KERNELS_PER_BATCH"};
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
static constexpr Property<std::wstring, PropertyMutability::RW> weights_path{"GPU_WEIGHTS_PATH"};
#else
static constexpr Property<std::string, PropertyMutability::RW> weights_path{"GPU_WEIGHTS_PATH"};
#endif

}  // namespace intel_gpu
}  // namespace ov

namespace cldnn {
using ov::intel_gpu::QueueTypes;
}  // namespace cldnn
