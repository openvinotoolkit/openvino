// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for definition of cpu processor type for cpu inference
 *
 * @file openvino/runtime/intel_cpu/processor_type.hpp
 */
#pragma once

#include "openvino/runtime/common.hpp"
namespace ov {

namespace intel_cpu {

/**
 * @enum       ProcessorType
 * @brief      This enum contains defination of processor type used for CPU inference.
 */
enum class ProcessorType {
    UNDEFINED = -1,     //!<  Default setting. All processors can be used on one socket platform. And only processors of
                        //!<  physical cores can be used on two socket platform.
    ALL_CORE = 1,       //!<  All processors can be used. If hyper threading is enabled, both processor of one
                        //!<  performance-core can be used.
    PHY_CORE_ONLY = 2,  //!<  Only processors of physical cores can be used. If hyper threading is enabled, only one
                        //!<  processor of one performance-core can be used.
    P_CORE_ONLY = 3,    //!<  Only processors of performance-cores can be used. If hyper threading is enabled, both
                        //!<  processor of one performance-core can be used.
    E_CORE_ONLY = 4,    //!<  Only processors of efficient-cores can be used.
    PHY_P_CORE_ONLY = 5,  //!<  Only processors of physical performance-cores can be used. If hyper threading is
                          //!<  enabled, only one processor of one performance-core can be used.
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ProcessorType& processor_type) {
    switch (processor_type) {
    case ProcessorType::UNDEFINED:
        return os << "CPU_UNDEFINED";
    case ProcessorType::ALL_CORE:
        return os << "CPU_ALL_CORE";
    case ProcessorType::PHY_CORE_ONLY:
        return os << "CPU_PHY_CORE_ONLY";
    case ProcessorType::P_CORE_ONLY:
        return os << "CPU_P_CORE_ONLY";
    case ProcessorType::E_CORE_ONLY:
        return os << "CPU_E_CORE_ONLY";
    case ProcessorType::PHY_P_CORE_ONLY:
        return os << "CPU_PHY_P_CORE_ONLY";
    default:
        throw ov::Exception{"Unsupported processor type!"};
    }
}

inline std::istream& operator>>(std::istream& is, ProcessorType& processor_type) {
    std::string str;
    is >> str;
    if (str == "CPU_UNDEFINED") {
        processor_type = ProcessorType::UNDEFINED;
    } else if (str == "CPU_ALL_CORE") {
        processor_type = ProcessorType::ALL_CORE;
    } else if (str == "CPU_PHY_CORE_ONLY") {
        processor_type = ProcessorType::PHY_CORE_ONLY;
    } else if (str == "CPU_P_CORE_ONLY") {
        processor_type = ProcessorType::P_CORE_ONLY;
    } else if (str == "CPU_E_CORE_ONLY") {
        processor_type = ProcessorType::E_CORE_ONLY;
    } else if (str == "CPU_PHY_P_CORE_ONLY") {
        processor_type = ProcessorType::PHY_P_CORE_ONLY;
    } else {
        throw ov::Exception{"Unsupported processor type: " + str};
    }
    return is;
}
/** @endcond */

}  // namespace intel_cpu

}  // namespace ov
