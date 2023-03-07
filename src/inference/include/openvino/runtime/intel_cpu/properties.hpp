
// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for CPU device
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_cpu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_cpu_prop_cpp_api Intel CPU specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel CPU specific properties.
 */

/**
 * @brief Namespace with Intel CPU specific properties
 */
namespace intel_cpu {

/**
 * @brief This property define whether to perform denormals optimization.
 * @ingroup ov_runtime_cpu_prop_cpp_api
 *
 * Computation with denormals is very time consuming. FTZ(Flushing denormals to zero) and DAZ(Denormals as zero)
 * could significantly improve the performance, but it does not comply with IEEE standard. In most cases, this behavior
 * has little impact on model accuracy. Users could enable this optimization if no or acceptable accuracy drop is seen.
 * The following code enables denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(true)); // enable denormals optimization
 * @endcode
 *
 * The following code disables denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(false)); // disable denormals optimization
 * @endcode
 */
static constexpr Property<bool> denormals_optimization{"CPU_DENORMALS_OPTIMIZATION"};

static constexpr Property<float> sparse_weights_decompression_rate{"SPARSE_WEIGHTS_DECOMPRESSION_RATE"};

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

static constexpr Property<ProcessorType> processor_type{"CPU_PROCESSOR_TYPE"};
}  // namespace intel_cpu
}  // namespace ov
