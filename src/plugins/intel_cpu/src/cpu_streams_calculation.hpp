// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file cpu_streams_calculation.hpp
 * @brief A header file for CPU streams calulation implementation.
 */

#pragma once

#include <memory>
#include <vector>

#include "config.h"
#include "graph.h"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_cpu {
/**
 * @brief      Generate streams information table according to processors type table.
 * @param[in]  input_streams is the targeted number of streams set by user via ov::num_streams or the default value.
 * @param[in]  input_streams_changed indicates if streams is set by user via ov::num_streams.
 * @param[in]  input_threads is the max number of threads set by user via ov::inference_num_threads or the default
 * value.
 *               - input "0" indicates that the function can use all resource in proc_type_table.
 *               - If user limits the max number of threads, the final number of streams output cannot exceed the max
 * number of threads.
 * @param[in]  input_infer_requests is max number of infer requests set by user via ov::hint::num_requests.
 *               - input "0" indicates that the function can use all resource in proc_type_table.
 *               - If user limits the max number of infer requests, the final number of streams output cannot exceed the
 * max number of infer requests.
 * @param[in]  model_prefer_threads is preferred number of threads per stream based on the model generated in previous
 * function.
 *               - input "0" indicates that the function generates the optimal number of threads per stream based on
 * processors type information.
 * @param[in]  input_perf_hint is performance hint set by user via ov::hint::performance_mode or the default value.
 * @param[in]  latencyThreadingMode is the scope of candidate processors per stream for latency hint
 *               - user can select all processors per numa node, per socket, or per platform.
 * @param[in]  proc_type_table is currently available candidate processors.
 *               - candidate processors have benn updated based on user input hints like ov::hint::scheduling_core_type
 * in previous function.
 * @return     streams information table which will be used by StreamsExecutor.
 */
std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const bool input_streams_changed,
                                                     const int input_threads,
                                                     const int input_infer_requests,
                                                     const int model_prefer_threads,
                                                     const std::string input_perf_hint,
                                                     const Config::LatencyThreadingMode latencyThreadingMode,
                                                     const std::vector<std::vector<int>>& proc_type_table);
/**
 * @brief      Get model_prefer_threads
 * @param[in]  num_streams is target streams set by user via NUM_STREAMS or hints.
 *               - input "0" mean function generate the optimal number of streams
 *               - LATENCY hint equals 1 stream.
 * @param[in]  proc_type_table candidate processors available at this time
 *               - candidate processors have benn updated based on properties like "Ecore only" in previous function
 * @param[in]  ngraphFunc ngraph function
 * @param[in]  config intel cpu configuration
 * @return     model_prefer_threads "0" means generating the optimal threads per stream based on platform
 */
int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ngraph::Function>& ngraphFunc,
                             Config& config);

/**
 * @brief      Generate streams information according to processors type table
 * @param[in]  streams number of streams
 * @param[in]  ngraphFunc graph handle
 * @param[in]  config intel cpu configuration
 * @param[in]  proc_type_table candidate processors available at current platform
 * @param[in]  preferred_nthreads_per_stream is initial preferred number of threads per stream
 * @return     candidate processors have benn updated based on user input hints like ov::hint::scheduling_core_type and
 * ov::hint::enable_hyper_threading
 */
std::vector<std::vector<int>> generate_stream_info(const int streams,
                                                   const std::shared_ptr<ngraph::Function>& ngraphFunc,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream = -1);

struct StreamCfg {
    int num_streams;               // Number of streams
    int num_threads;               // Number of threads
    int big_core_streams;          // Number of streams in Performance-core(big core)
    int small_core_streams;        // Number of streams in Efficient-core(small core)
    int threads_per_stream_big;    // Threads per stream in big cores
    int threads_per_stream_small;  // Threads per stream in small cores
    int small_core_offset;
};

/**
 * @brief      Get information about number of streams, threads and pinning threads on different processors
 * @param[in]  streams number of streams
 * @param[in]  ngraphFunc graph handle
 * @param[in]  config intel cpu configuration
 */
void get_num_streams(const int streams,
                     const std::shared_ptr<ngraph::Function>& ngraphFunc,
                     Config& config);

/**
 * @brief      Get default number of streams in certain latency threading mode
 * @param[in]  latency_threading_mode is the scope of candidate processors per stream for latency hint
 * @return     number of streams
 */
int get_default_latency_streams(Config::LatencyThreadingMode latency_threading_mode);

}  // namespace intel_cpu
}  // namespace ov
