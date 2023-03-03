// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <vector>

#include "openvino/runtime/intel_cpu/properties.hpp"

namespace ov {
namespace intel_cpu {

/**
 * @brief      Limit available CPU resource in processors type table according to processor type property
 * @param[in]  input_type input value of processor type property.
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     updated proc_type_table which removed unmatched processors
 */
std::vector<std::vector<int>> apply_processor_type(const ProcessorType input_type,
                                                   const std::vector<std::vector<int>> proc_type_table);

}  // namespace intel_cpu
}  // namespace ov