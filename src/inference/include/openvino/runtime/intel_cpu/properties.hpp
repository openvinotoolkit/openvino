
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
 * @enum       SchedulingCoreType
 * @brief      This enum contains defination of core type can be used for CPU inference.
 */
enum class SchedulingCoreType {
    ALL = 1,         //!<  All processors can be used.
    PCORE_ONLY = 2,  //!<  Only processors of performance-cores can be used.
    ECORE_ONLY = 3,  //!<  Only processors of efficient-cores can be used.
};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const SchedulingCoreType& core_type) {
    switch (core_type) {
    case SchedulingCoreType::ALL:
        return os << "CPU_ALL";
    case SchedulingCoreType::PCORE_ONLY:
        return os << "CPU_PCORE_ONLY";
    case SchedulingCoreType::ECORE_ONLY:
        return os << "CPU_ECORE_ONLY";
    default:
        throw ov::Exception{"Unsupported core type!"};
    }
}

inline std::istream& operator>>(std::istream& is, SchedulingCoreType& core_type) {
    std::string str;
    is >> str;
    if (str == "CPU_ALL") {
        core_type = SchedulingCoreType::ALL;
    } else if (str == "CPU_PCORE_ONLY") {
        core_type = SchedulingCoreType::PCORE_ONLY;
    } else if (str == "CPU_ECORE_ONLY") {
        core_type = SchedulingCoreType::ECORE_ONLY;
    } else {
        throw ov::Exception{"Unsupported core type: " + str};
    }
    return is;
}
/** @endcond */

/**
 * @brief This property define core type can be used for CPU inference.
 * @ingroup ov_runtime_cpu_prop_cpp_api
 *
 * Developer can use this property to select specific CPU cores for CPU inference. Please refer SchedulingCoreType for
 * all definition of core type.
 *
 * The following code is example to only use efficient-cores for inference on hybrid CPU. If user sets this
 * configuration on a platform with only performance-cores, CPU inference will still run on the performance-cores.
 *
 * @code
 * ie.set_property(ov::intel_cpu::scheduling_core_type(ov::intel_cpu::SchedulingCoreType::ECORE_ONLY));
 * @endcode
 */
static constexpr Property<SchedulingCoreType> scheduling_core_type{"CPU_SCHEDULING_CORE_TYPE"};

}  // namespace intel_cpu
}  // namespace ov

