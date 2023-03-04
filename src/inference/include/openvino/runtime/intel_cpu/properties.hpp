
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
#include "openvino/runtime/system_conf.hpp"

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
using ov::ProcessorType;

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ProcessorType& cpu_processor_type) {
    switch (cpu_processor_type) {
    case ProcessorType::UNDEFINED:
        return os << "UNDEFINED";
    case ProcessorType::ALL:
        return os << "ALL";
    case ProcessorType::PHY_CORE_ONLY:
        return os << "PHY_CORE_ONLY";
    case ProcessorType::P_CORE_ONLY:
        return os << "P_CORE_ONLY";
    case ProcessorType::E_CORE_ONLY:
        return os << "E_CORE_ONLY";
    case ProcessorType::PHY_P_CORE_ONLY:
        return os << "PHY_P_CORE_ONLY";
    default:
        throw ov::Exception{"Unsupported processor type!"};
    }
}

inline std::istream& operator>>(std::istream& is, ProcessorType& cpu_processor_type) {
    std::string str;
    is >> str;
    if (str == "UNDEFINED") {
        cpu_processor_type = ProcessorType::UNDEFINED;
    } else if (str == "ALL") {
        cpu_processor_type = ProcessorType::ALL;
    } else if (str == "PHY_CORE_ONLY") {
        cpu_processor_type = ProcessorType::PHY_CORE_ONLY;
    } else if (str == "P_CORE_ONLY") {
        cpu_processor_type = ProcessorType::P_CORE_ONLY;
    } else if (str == "E_CORE_ONLY") {
        cpu_processor_type = ProcessorType::E_CORE_ONLY;
    } else if (str == "PHY_P_CORE_ONLY") {
        cpu_processor_type = ProcessorType::PHY_P_CORE_ONLY;
    } else {
        throw ov::Exception{"Unsupported processor type: " + str};
    }
    return is;
}
/** @endcond */

static constexpr Property<ProcessorType> processor_type{"PROCESSOR_TYPE"};

}  // namespace intel_cpu
}  // namespace ov
