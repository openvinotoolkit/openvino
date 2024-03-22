//
// Copyright (C) 2022 Intel Corporation.
// SPDX-License-Identifier: Apache 2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>

namespace ov {
namespace intel_npu {

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of already allocated NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_alloc_mem_size{"NPU_DEVICE_ALLOC_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of available NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get VPU driver version (for both discrete/integrated VPU devices)
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> driver_version{"NPU_DRIVER_VERSION"};

/**
 * @brief Read-only property to get the name of used backend
 */
static constexpr ov::Property<std::string, ov::PropertyMutability::RO> backend_name{"NPU_BACKEND_NAME"};

/**
 * @brief Defines the options corresponding to the legacy set of values.
 */
enum class LegacyPriority {
    LOW = 0,     //!<  Low priority
    MEDIUM = 1,  //!<  Medium priority
    HIGH = 2     //!<  High priority
};

inline std::ostream& operator<<(std::ostream& os, const LegacyPriority& priority) {
    switch (priority) {
    case LegacyPriority::LOW:
        return os << "MODEL_PRIORITY_LOW";
    case LegacyPriority::MEDIUM:
        return os << "MODEL_PRIORITY_MED";
    case LegacyPriority::HIGH:
        return os << "MODEL_PRIORITY_HIGH";
    default:
        OPENVINO_THROW("Unsupported model priority value");
    }
}

inline std::istream& operator>>(std::istream& is, LegacyPriority& priority) {
    std::string str;
    is >> str;
    if (str == "MODEL_PRIORITY_LOW") {
        priority = LegacyPriority::LOW;
    } else if (str == "MODEL_PRIORITY_MED") {
        priority = LegacyPriority::MEDIUM;
    } else if (str == "MODEL_PRIORITY_HIGH") {
        priority = LegacyPriority::HIGH;
    } else {
        OPENVINO_THROW("Unsupported model priority: ", str);
    }
    return is;
}

/**
 * @brief Due to driver compatibility constraints, the set of model priority values corresponding to the OpenVINO legacy
 * API is being maintained here.
 * @details The OpenVINO API has made changes with regard to the values used to describe model priorities (e.g.
 * "MODEL_PRIORITY_MED" -> "MEDIUM"). The NPU plugin can't yet discard this since the newer values may not be
 * recognized by older drivers.
 */
static constexpr ov::Property<LegacyPriority, ov::PropertyMutability::RO> legacy_model_priority{"MODEL_PRIORITY"};

}  // namespace intel_npu
}  // namespace ov
