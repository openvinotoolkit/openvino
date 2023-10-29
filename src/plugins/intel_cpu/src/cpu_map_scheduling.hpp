// Copyright (C) 2018-2023 Intel Corporation
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

namespace ov {
namespace intel_cpu {

/**
 * @enum       SubStreamsMode
 * @brief      This enum contains definition of each sub streams mode, indicating the main stream situation.
 */
enum class SubStreamsMode {
    SUB_STREAMS_NULL,        //!< Do not create sub streams
    SUB_STREAMS_FOR_SOCKET,  //!< Create sub streams for multiple sockets in main stream
};

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
                                                    const std::string input_pm_hint,
                                                    const std::vector<std::vector<int>>& proc_type_table);

/**
 * @brief      whether pinning cpu cores according to enableCpuPinning property
 * @param[in]  input_type indicate value of property enableCpuPinning.
 * @param[in]  input_changed indicate if value is set by user.
 * @param[in]  num_streams number of streams
 * @param[in]  latency_threading_mode is the scope of candidate processors per stream for latency hint
 * @param[in]  proc_type_table candidate processors available at this time
 * @return     whether pinning threads to cpu cores
 */
bool get_cpu_pinning(bool& input_value,
                     const bool input_changed,
                     const int num_streams,
                     const Config::LatencyThreadingMode latency_threading_mode,
                     const std::vector<std::vector<int>>& proc_type_table);

}  // namespace intel_cpu
}  // namespace ov
