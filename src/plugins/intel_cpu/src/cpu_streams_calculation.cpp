// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <transformations/utils/utils.hpp>

#include "graph.h"
#include "ie_system_conf.h"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "performance_heuristics.hpp"
#include "threading/ie_cpu_streams_info.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const int input_threads,
                                                     const int input_infer_requests,
                                                     const int model_prefer_threads,
                                                     const std::vector<std::vector<int>> proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE);
    std::vector<std::vector<int>> streams_info_table;

    if (1 == input_streams) {
        stream_info[NUMBER_OF_STREAMS] = 1;
        int limit_threads = (input_threads == 0) ? model_prefer_threads : input_threads;
        if (proc_type_table[0][ALL_PROC] == proc_type_table[0][EFFICIENT_CORE_PROC]) {
            stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
            stream_info[THREADS_PER_STREAM] = (input_threads == 0)
                                                  ? proc_type_table[0][EFFICIENT_CORE_PROC]
                                                  : std::min(proc_type_table[0][EFFICIENT_CORE_PROC], limit_threads);
            streams_info_table.push_back(stream_info);
        } else if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                   ((limit_threads == 0) || (limit_threads > proc_type_table[0][MAIN_CORE_PROC]))) {
            stream_info[PROC_TYPE] = ALL_PROC;
            int n_threads = std::accumulate(proc_type_table[0].begin() + MAIN_CORE_PROC,
                                            proc_type_table[0].begin() + HYPER_THREADING_PROC,
                                            0);
            stream_info[THREADS_PER_STREAM] = (limit_threads == 0) ? n_threads : std::min(n_threads, limit_threads);
            streams_info_table.push_back(stream_info);
            stream_info[NUMBER_OF_STREAMS] = 0;
            n_threads = stream_info[THREADS_PER_STREAM];
            for (int n = MAIN_CORE_PROC; n < HYPER_THREADING_PROC; n++) {
                if (0 != proc_type_table[0][n]) {
                    stream_info[PROC_TYPE] = n;
                    if (n_threads <= proc_type_table[0][n]) {
                        stream_info[THREADS_PER_STREAM] = n_threads;
                        streams_info_table.push_back(stream_info);
                        break;
                    } else {
                        stream_info[THREADS_PER_STREAM] = proc_type_table[0][n];
                        streams_info_table.push_back(stream_info);
                        n_threads -= proc_type_table[0][n];
                    }
                }
            }
        } else {
            stream_info[PROC_TYPE] = MAIN_CORE_PROC;
            stream_info[THREADS_PER_STREAM] = (limit_threads == 0)
                                                  ? proc_type_table[0][MAIN_CORE_PROC]
                                                  : std::min(proc_type_table[0][MAIN_CORE_PROC], limit_threads);
            streams_info_table.push_back(stream_info);
        }
        return streams_info_table;

    } else {
        int n_streams = 0;
        int n_threads = 0;
        int n_threads_per_stream = 0;
        int base_type = MAIN_CORE_PROC;

        if (proc_type_table.size() == 1) {
            n_threads = (0 == input_threads) ? proc_type_table[0][ALL_PROC]
                                             : std::min(proc_type_table[0][ALL_PROC], input_threads);
        } else {
            n_threads = (0 == input_threads) ? proc_type_table[0][MAIN_CORE_PROC]
                                             : std::min(proc_type_table[0][MAIN_CORE_PROC], input_threads);
        }

        if (0 != input_streams) {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            n_streams = (input_infer_requests > 0) ? std::min(input_streams, input_infer_requests) : input_streams;
            if (n_streams >= n_threads) {
                n_streams = n_threads;
                n_threads_per_stream = 1;
            } else {
                n_threads_per_stream = std::min(std::max(1, n_threads / n_streams), proc_type_table[0][base_type]);
                if (proc_type_table.size() == 1) {
                    if ((n_threads_per_stream > proc_type_table[0][base_type]) &&
                        (n_threads_per_stream < proc_type_table[0][base_type] * 2)) {
                        n_threads_per_stream = proc_type_table[0][base_type];
                    } else if (n_threads_per_stream < proc_type_table[0][base_type]) {
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][base_type] /
                            ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                    }
                }
            }
        } else {
            base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            if (0 == model_prefer_threads) {
                int n_proc = std::min(n_threads, proc_type_table[0][base_type]);
                if (0 == n_proc % 4) {
                    n_threads_per_stream = 4;
                } else if (0 == n_proc % 5) {
                    n_threads_per_stream = 5;
                } else if (0 == n_proc % 3) {
                    n_threads_per_stream = 3;
                } else if (proc_type_table.size() == 1) {
                    n_threads_per_stream = n_proc;
                } else {
                    n_threads_per_stream = (n_proc > 16) ? 4 : std::max(1, static_cast<int>(n_proc / 4));
                }
                n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
                    n_streams = input_infer_requests;
                    n_threads_per_stream =
                        std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
                } else {
                    while (n_streams < n_threads_per_stream) {
                        if (1 == n_threads_per_stream) {
                            break;
                        } else {
                            n_threads_per_stream = static_cast<int>((n_threads_per_stream * 2 - 1) / 2);
                            n_threads_per_stream = static_cast<int>(
                                proc_type_table[0][base_type] /
                                ((proc_type_table[0][base_type] + n_threads_per_stream - 1) / n_threads_per_stream));
                            n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                        }
                    }
                }
            } else {
                n_streams = ((n_threads + model_prefer_threads - 1) / model_prefer_threads);
                n_streams = (input_infer_requests > 0) ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = std::min(static_cast<int>(n_threads / n_streams), proc_type_table[0][base_type]);
            }
        }

        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;

        if (proc_type_table.size() == 1) {
            while (1) {
                for (int n = MAIN_CORE_PROC; n < PROC_TYPE_TABLE_SIZE; n++) {
                    if (0 != proc_type_table[0][n]) {
                        stream_info[PROC_TYPE] = n;
                        stream_info[NUMBER_OF_STREAMS] =
                            static_cast<int>(proc_type_table[0][n] / stream_info[THREADS_PER_STREAM]);
                        if (n_streams <= stream_info[NUMBER_OF_STREAMS]) {
                            stream_info[NUMBER_OF_STREAMS] = n_streams;
                            streams_info_table.push_back(stream_info);
                            return streams_info_table;
                        } else {
                            streams_info_table.push_back(stream_info);
                            n_streams -= stream_info[NUMBER_OF_STREAMS];
                        }
                    }
                }
                if (1 == stream_info[THREADS_PER_STREAM]) {
                    return streams_info_table;
                } else {
                    stream_info[THREADS_PER_STREAM] -= 1;
                    std::vector<std::vector<int>>().swap(streams_info_table);
                }
            }
        } else {
            stream_info[NUMBER_OF_STREAMS] = n_streams;
            stream_info[PROC_TYPE] = MAIN_CORE_PROC;
            stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
            streams_info_table.push_back(stream_info);
            return streams_info_table;
        }
    }
}

int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ngraph::Function>& ngraphFunc,
                             const InferenceEngine::IStreamsExecutor::Config streamExecutorConfig) {
    const int sockets = static_cast<int>(getAvailableNUMANodes().size());
    auto model_prefer = 0;
    // latency
    if (num_streams <= sockets && num_streams > 0) {
        if (streamExecutorConfig._threadBindingType == IStreamsExecutor::ThreadBindingType::HYBRID_AWARE) {
            bool fp_intesive = !ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(ngraphFunc);
            const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
            const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
            // by default the latency case uses (faster) Big cores only, depending on the compute ratio
            model_prefer = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] /
                                                                 (fp_intesive ? fp32_threshold : int8_threshold))
                               ? proc_type_table[0][MAIN_CORE_PROC]
                               : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    } else { // throughput
        const auto isa = dnnl::get_effective_cpu_isa();
        float isaSpecificThreshold = 1.0f;
        switch (isa) {
        case dnnl::cpu_isa::sse41:
            isaSpecificThreshold = 0.5f;
            break;
        case dnnl::cpu_isa::avx2:
        case dnnl::cpu_isa::avx512_core:
            isaSpecificThreshold = 1.0f;
            break;
        case dnnl::cpu_isa::avx512_core_vnni:
        case dnnl::cpu_isa::avx2_vnni:
            isaSpecificThreshold = 2.0f;
            break;
        case dnnl::cpu_isa::avx512_core_amx:
            isaSpecificThreshold = 4.0f;
            break;
        default:
            isaSpecificThreshold = 1.0f;
        }
        // the more "capable" the CPU in general, the more streams we may want to keep to keep it utilized
        const float memThresholdAssumeLimitedForISA = ov::MemBandwidthPressure::LIMITED / isaSpecificThreshold;
        const float L2_cache_size = dnnl::utils::get_cache_size(2 /*level*/, true /*per core */);
        ov::MemBandwidthPressure networkToleranceForLowCache =
            ov::MemBandwidthPressureTolerance(ngraphFunc, L2_cache_size, memThresholdAssumeLimitedForISA);
        model_prefer = IStreamsExecutor::Config::StreamMode::DEFAULT;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                model_prefer = 1;
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            model_prefer = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            model_prefer = 2;
        }
        if (model_prefer == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 && sockets == 1) {
            model_prefer = 2;
        }
    }

    return model_prefer;
}

StreamCfg parse_streams_table(std::vector<std::vector<int>> streams_table) {
    StreamCfg streams_info = {0};
    for (int i = 0; i < streams_table.size(); i++) {
        if (streams_table[i][PROC_TYPE] == ALL_PROC) {
            streams_info.num_streams = streams_table[i][NUMBER_OF_STREAMS];
            streams_info.num_threads = streams_table[i][THREADS_PER_STREAM];
        } else if (streams_table[i][PROC_TYPE] == MAIN_CORE_PROC) {
            streams_info.big_core_streams = streams_table[i][NUMBER_OF_STREAMS];
            streams_info.threads_per_stream_big = streams_table[i][THREADS_PER_STREAM];
        } else if (streams_table[i][PROC_TYPE] == EFFICIENT_CORE_PROC) {
            streams_info.small_core_streams = streams_table[i][NUMBER_OF_STREAMS];
            streams_info.threads_per_stream_small = streams_table[i][THREADS_PER_STREAM];
        } else if (streams_table[i][PROC_TYPE] == HYPER_THREADING_PROC) {
            streams_info.big_core_logic_streams = streams_table[i][NUMBER_OF_STREAMS];
            streams_info.threads_per_stream_big = streams_table[i][THREADS_PER_STREAM];
        }
    }
    streams_info.num_streams =
        streams_info.num_streams == 0
            ? streams_info.big_core_streams + streams_info.small_core_streams + streams_info.big_core_logic_streams
            : streams_info.num_streams;
    streams_info.num_threads = streams_info.num_threads == 0
                                   ? ((streams_info.big_core_streams + streams_info.big_core_logic_streams) *
                                          streams_info.threads_per_stream_big +
                                      streams_info.small_core_streams * streams_info.threads_per_stream_small)
                                   : streams_info.num_threads;
    return streams_info;
}

std::pair<std::string, StreamCfg> get_num_streams(
    const int streams,
    const int infer_requests,
    const std::shared_ptr<ngraph::Function>& ngraphFunc,
    const InferenceEngine::IStreamsExecutor::Config streamExecutorConfig) {
    const std::vector<std::vector<int>> proc_type_table = get_num_available_cpu_cores();
    const int model_prefer = get_model_prefer_threads(streams, proc_type_table, ngraphFunc, streamExecutorConfig);
    const std::vector<std::vector<int>> stream_info_table =
        get_streams_info_table(streams, streamExecutorConfig._threads, infer_requests, model_prefer, proc_type_table);
    StreamCfg streams_info = parse_streams_table(stream_info_table);

    DEBUG_LOG(
        "[ p_e_core_info ] streams (threads): ",
        streams_info.num_streams,
        "(",
        streams_info.num_threads,
        ") -- PCore: ",
        streams_info.big_core_streams,
        "(",
        streams_info.threads_per_stream_big,
        ") ",
        streams_info.big_core_logic_streams,
        "(",
        streams_info.threads_per_stream_big,
        ")  ECore: ",
        streams_info.small_core_streams,
        "(",
        streams_info.threads_per_stream_small,
        ")");

    return std::pair<std::string, StreamCfg>(std::to_string(streams_info.num_streams), streams_info);
}
}  // namespace intel_cpu
}  // namespace ov
