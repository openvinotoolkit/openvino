// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/runtime/properties.hpp>
#include <string>

namespace ov {

/**
 * @brief Namespace with Intel AUTO specific properties
 */
namespace intel_auto {

/**
 * @brief Enum to define auto cpu usage
 */
enum class AutoCpuUsage {
    NO_INFERENCE = 0,   // AUTO cannot offload any inference workload to CPU
    ACCERLATE_FIL = 1,  // AUTO only can use CPU as accelerator to start infer beginning frames when target device is
                        // loading network. AUTO cannot use CPU as target device
    ACCERLATE_FIL_ONE_FRAM = 2,  // AUTO only can use CPU as accelerator to infer one frame when target device is
                                 // loading network. AUTO cannot use CPU as target device.
    FULL_STRENGTH = 3,           // AUTO can use CPU as target device or accelerator.(default)
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const AutoCpuUsage& autoCpuUsage) {
    switch (autoCpuUsage) {
    case AutoCpuUsage::NO_INFERENCE:
        return os << "AUTO_CPU_NO_INFERENCE";
    case AutoCpuUsage::ACCERLATE_FIL:
        return os << "AUTO_CPU_ACCERLATE_FIL";
    case AutoCpuUsage::ACCERLATE_FIL_ONE_FRAM:
        return os << "AUTO_CPU_ACCERLATE_FIL_ONE_FRAM";
    case AutoCpuUsage::FULL_STRENGTH:
        return os << "AUTO_CPU_FULL_STRENGTH";
    default:
        throw ov::Exception{"Unsupported AutoCpuUsage"};
    }
};

inline std::istream& operator>>(std::istream& is, AutoCpuUsage& autoCpuUsage) {
    std::string str;
    is >> str;
    if (str == "AUTO_CPU_NO_INFERENCE") {
        autoCpuUsage = AutoCpuUsage::NO_INFERENCE;
    } else if (str == "AUTO_CPU_ACCERLATE_FIL") {
        autoCpuUsage = AutoCpuUsage::ACCERLATE_FIL;
    } else if (str == "AUTO_CPU_ACCERLATE_FIL_ONE_FRAM") {
        autoCpuUsage = AutoCpuUsage::ACCERLATE_FIL_ONE_FRAM;
    } else if (str == "AUTO_CPU_FULL_STRENGTH") {
        autoCpuUsage = AutoCpuUsage::FULL_STRENGTH;
    } else {
        throw ov::Exception{"Unsupported AutoCpuUsage: " + str};
    }
    return is;
};
/** @endcond */

/**
 * @brief auto_cpu_usage setting that should be one of NO_INFERENCE, ACCERLATE_FIL, ACCERLATE_FIL_ONE_FRAM,
 * FULL_STRENGTH
 */
static constexpr Property<AutoCpuUsage> auto_cpu_usage{"AUTO_CPU_USAGE"};

}  // namespace intel_auto
}  // namespace ov