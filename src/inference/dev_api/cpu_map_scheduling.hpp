// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CPU map scheduling
 * @file ie_icore.hpp
 */

#pragma once

#include <vector>

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @brief      Limit available CPU resource in processors type table according to scheduling core type property
 * @param[in]  input_type input value of core type property.
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     updated proc_type_table which removed unmatched processors
 */
std::vector<std::vector<int>> apply_scheduling_core_type(const ov::hint::SchedulingCoreType input_type,
                                                         const std::vector<std::vector<int>> proc_type_table);

}  // namespace ov