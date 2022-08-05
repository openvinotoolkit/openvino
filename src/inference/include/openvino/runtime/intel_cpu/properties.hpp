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
 * could significantly improve the performance while it does not comply with IEEE standard. In most cases this behavior
 * has little impact for model accuracy. Users could enable this optimization if see no or acceptable accuracy drop.
 * The following code enables denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(true)); // enable denormals optimization
 * @endcode
 *
 * While following code disable denormals optimization
 *
 * @code
 * ie.set_property(ov::denormals_optimization(false)); // disable denormals optimization
 * @endcode
 */
static constexpr Property<bool> denormals_optimization{"CPU_DENORMALS_OPTIMIZATION"};

}  // namespace intel_cpu
}  // namespace ov
