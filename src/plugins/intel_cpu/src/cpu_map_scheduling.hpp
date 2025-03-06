// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CPU map scheduling
 * @file cpu_map_scheduling.hpp
 */

#pragma once

#include <vector>

#include "config.h"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov::intel_cpu {

/**
 * @brief      Limit available CPU resource in processors type table according to scheduling core type property
 * @param[in]  input_type input value of core type property.
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     updated proc_type_table which removed unmatched processors
 */
std::vector<std::vector<int>> apply_scheduling_core_type(ov::hint::SchedulingCoreType& input_type,
                                                         const std::vector<std::vector<int>>& proc_type_table);

/**
 * @brief      Limit available CPU resource in processors type table according to hyper threading property
 * @param[in]  input_ht_hint indicate value of property enable_hyper_threading.
 * @param[in]  input_ht_changed indicate if value is set by user.
 * @param[in]  input_pm_hint indicate value of property performance_mode.
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     updated proc_type_table which removed unmatched processors
 */
std::vector<std::vector<int>> apply_hyper_threading(bool& input_ht_hint,
                                                    const bool input_ht_changed,
                                                    const std::string& input_pm_hint,
                                                    const std::vector<std::vector<int>>& proc_type_table);

/**
 * @brief      Check enableCpuPinning in different platform
 * @param[in]  cpu_pinning the property enableCpuPinning set by user.
 * @param[in]  cpu_pinning_changed the property changedCpuPinning which value depends on whether user sets
 * enableCpuPinning. true: user sets property enableCpuPinning. false: user does not set property enableCpuPinning.
 * @param[in]  cpu_reservation the property enableCpuReservation set by user. False by default
 * @param[in]  streams_info_table indicate streams detail of this model
 * @return     whether pinning threads to cpu cores
 */
bool check_cpu_pinning(const bool cpu_pinning,
                       const bool cpu_pinning_changed,
                       const bool cpu_reservation,
                       const std::vector<std::vector<int>>& streams_info_table);

}  // namespace ov::intel_cpu
