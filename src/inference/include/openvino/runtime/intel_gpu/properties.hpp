// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for GPU plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_gpu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_gpu {

/**
 * @brief Read-only property which defines size of memory in bytes available for the device. For iGPU it returns host
 * memory size, for dGPU - dedicated gpu memory size
 */
static constexpr Property<uint64_t, PropertyMutability::RO> device_total_mem_size{"GPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief Read-only property to get microarchitecture identifier in major.minor.revision format
 */
static constexpr Property<std::string, PropertyMutability::RO> uarch_version{"GPU_UARCH_VERSION"};

/**
 * @brief Read-only property to get count of execution units for current GPU
 */
static constexpr Property<int32_t, PropertyMutability::RO> execution_units_count{"GPU_EXECUTION_UNITS_COUNT"};

/**
 * @brief Read-only property to get statistics of GPU memory allocated by engine for each allocation type
 * It contains information about current memory usage
 */
static constexpr Property<std::map<std::string, uint64_t>, PropertyMutability::RO> memory_statistics{
    "GPU_MEMORY_STATISTICS"};

/**
 * @brief Read-only property to get maximum batch size which does not cause performance degradation due to memory swap
 * impact.
 */
static constexpr Property<uint32_t, PropertyMutability::RO> max_batch_size{"GPU_MAX_BATCH_SIZE"};

/**
 * @brief Possible return value for OPTIMIZATION_CAPABILITIES metric
 *  - "HW_MATMUL" - Defines if device has hardware block for matrix multiplication
 */
static constexpr Property<bool, PropertyMutability::RO> hw_matmul{"GPU_HW_MATMUL"};

/**
 * @brief Turning on this key enables to unroll recurrent layers such as TensorIterator or Loop with fixed iteration
 * count. This key is turned on by default. Turning this key on will achieve better inference performance for loops with
 * not too many iteration counts (less than 16, as a rule of thumb). Turning this key off will achieve better
 * performance for both graph loading time and inference time with many iteration counts (greater than 16). Note that
 * turning this key on will increase the graph loading time in proportion to the iteration counts.
 * Thus, this key should be turned off if graph loading time is considered to be most important target to optimize.*/
static constexpr Property<bool> enable_loop_unrolling{"GPU_ENABLE_LOOP_UNROLLING"};

namespace hint {
/**
 * @brief This key instructs the GPU plugin to use throttle hints the OpenCL queue throttle hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf,
 * chapter 9.19. This option should be used with an unsigned integer value (1 is lowest energy consumption)
 * 0 means no throttle hint is set and default queue created.
 */
static constexpr Property<uint32_t> queue_throttle{"GPU_QUEUE_THROTTLE"};

/**
 * @brief This key instructs the GPU plugin to use the OpenCL queue priority hint
 * as defined in https://www.khronos.org/registry/OpenCL/specs/opencl-2.1-extensions.pdf
 * this option should be used with an unsigned integer value (1 is lowest priority)
 * 0 means no priority hint is set and default queue is created.
 */
static constexpr Property<uint32_t> queue_priority{"GPU_QUEUE_PRIORITY"};

/**
 * @brief Enum to define possible host task priorities
 */
enum class HostTaskPriority {
    LOW = 0,
    HIGH = 1,
    ANY = 2,
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const HostTaskPriority& queue_priority) {
    switch (queue_priority) {
    case HostTaskPriority::LOW:
        return os << "LOW";
    case HostTaskPriority::HIGH:
        return os << "HIGH";
    case HostTaskPriority::ANY:
        return os << "ANY";
    default:
        throw ov::Exception{"Unsupported host task priority"};
    }
}

inline std::istream& operator>>(std::istream& is, HostTaskPriority& queue_priority) {
    std::string str;
    is >> str;
    if (str == "LOW") {
        queue_priority = HostTaskPriority::LOW;
    } else if (str == "HIGH") {
        queue_priority = HostTaskPriority::HIGH;
    } else if (str == "ANY") {
        queue_priority = HostTaskPriority::ANY;
    } else {
        throw ov::Exception{"Unsupported host task priority: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief This key instructs the GPU plugin which cpu core type of TBB affinity used in load network.
 * This option has 3 types of levels: HIGH, LOW, and ANY. It is only affected on Hybrid CPUs.
 */
static constexpr Property<HostTaskPriority> host_task_priority{"GPU_HOST_TASK_PRIORITY"};
}  // namespace hint

namespace memory_type {
/**
 * @brief These keys instruct the GPU plugin to use surface/buffer memory type.
 */
static constexpr auto surface = "GPU_SURFACE";
static constexpr auto buffer = "GPU_BUFFER";
}  // namespace memory_type
}  // namespace intel_gpu
}  // namespace ov
