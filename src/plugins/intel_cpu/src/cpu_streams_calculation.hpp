// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <vector>

namespace ov {
namespace intel_cpu {
/**
 * @brief      Generate streams information table according to processors type table
 * @param[in]  input_streams is target streams set by user via NUM_STREAMS or hints.
 *               - input "0" mean function generate the optimal number of streams
 *               - LATENCY hint equals 1 stream.
 * @param[in]  input_threads is max threads set by user via INFERNECE_NUM_THREADS.
 *               - input "0" mean function can use all resource in proc_type_table
 *               - When user limit max threads, streams in output cannot be more than max threads
 * @param[in]  model_prefer_threads is preferred threads per stream based on model generated in previous function
 *               - input "0" mean function generate the optimal threads per stream based on platform
 * @param[in]  proc_type_table candidate processors available at this time
 *               - candidate processors have benn updated based on properties like "Ecore only" in previous function
 * @return     summary table of streams info will be used by StreamsExecutor
 */
std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const int input_threads,
                                                     const int model_prefer_threads,
                                                     const std::vector<std::vector<int>> proc_type_table);
}  // namespace intel_cpu
}  // namespace ov
