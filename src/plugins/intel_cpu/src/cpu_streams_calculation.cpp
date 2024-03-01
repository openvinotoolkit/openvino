// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include "cpu_map_scheduling.hpp"
#include "graph.h"
#include "openvino/runtime/performance_heuristics.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "transformations/utils.hpp"
#include "transformations/utils/utils.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <unordered_set>

using namespace ov;
using namespace ov::threading;

#define INIT_VAL -100

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const bool input_streams_changed,
                                                     const int input_threads,
                                                     const int input_infer_requests,
                                                     const int model_prefer_threads,
                                                     const int input_current_socket_id,
                                                     const std::string input_perf_hint,
                                                     const Config::LatencyThreadingMode latencyThreadingMode,
                                                     const std::vector<std::vector<int>>& proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE, INIT_VAL);
    std::vector<std::vector<int>> streams_info_table;
    std::vector<std::vector<int>> proc_socket_table;

    int n_streams = 0;
    int n_threads = 0;
    int n_threads_per_stream = 0;
    int current_socket_id = -1;

    auto update_ids_method = [&](const std::vector<int>& one_proc_info) {
        stream_info[STREAM_NUMA_NODE_ID] = one_proc_info[PROC_NUMA_NODE_ID];
        stream_info[STREAM_SOCKET_ID] = one_proc_info[PROC_SOCKET_ID];
    };

    auto update_mix_stream_info = [&](const std::vector<int>& one_proc_info,
                                      const std::vector<std::vector<int>>& one_proc_table,
                                      const int& target_proc) {
        stream_info[PROC_TYPE] = ALL_PROC;
        stream_info[NUMBER_OF_STREAMS] = 1;
        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
        update_ids_method(one_proc_info);
        streams_info_table.push_back(stream_info);
        stream_info[NUMBER_OF_STREAMS] = 0;
        int total_threads = stream_info[THREADS_PER_STREAM];
        int socket_id = stream_info[STREAM_SOCKET_ID];
        int node_start = one_proc_table.size() == 1 ? 0 : 1;
        int node_end = one_proc_table.size() == 1 ? 1 : one_proc_table.size();
        // When n_mode is 3, the following loop only selects CPUs on socket with the same id as current_socket_id.
        // When n_mode is 2, the following loop only selects CPUs on sockets with id different from current_socket_id.
        // When n_mode is 1, the following loop selects CPUs on all sockets.
        for (int n_mode = current_socket_id < 0 ? 1 : 3; (n_mode > 0) && (total_threads > 0); n_mode--) {
            for (int n = MAIN_CORE_PROC; (n <= HYPER_THREADING_PROC) && (total_threads > 0); n++) {
                for (int index = node_start; (index < node_end) && (total_threads > 0); index++) {
                    if (((n_mode == 1) && ((socket_id < 0) || (socket_id == one_proc_table[index][PROC_SOCKET_ID]))) ||
                        ((n_mode == 2) && (current_socket_id != one_proc_table[index][PROC_SOCKET_ID]) &&
                         ((socket_id < 0) || (socket_id == one_proc_table[index][PROC_SOCKET_ID]))) ||
                        ((n_mode == 3) && (current_socket_id == one_proc_table[index][PROC_SOCKET_ID]) &&
                         ((socket_id < 0) || (socket_id == one_proc_table[index][PROC_SOCKET_ID])))) {
                        if ((0 != one_proc_table[index][n]) && ((ALL_PROC == target_proc) || (n == target_proc))) {
                            stream_info[PROC_TYPE] = n;
                            stream_info[STREAM_NUMA_NODE_ID] = one_proc_table[index][PROC_NUMA_NODE_ID];
                            stream_info[STREAM_SOCKET_ID] = one_proc_table[index][PROC_SOCKET_ID];
                            if (total_threads <= one_proc_table[index][n]) {
                                stream_info[THREADS_PER_STREAM] = total_threads;
                                streams_info_table.push_back(stream_info);
                                total_threads -= stream_info[THREADS_PER_STREAM];
                                return;
                            } else {
                                stream_info[THREADS_PER_STREAM] = one_proc_table[index][n];
                                streams_info_table.push_back(stream_info);
                                total_threads -= stream_info[THREADS_PER_STREAM];
                            }
                        }
                    }
                }
            }
        }
    };

    auto update_streams_per_node = [&](const int& proc_type, const std::vector<int>& one_proc_info) {
        if ((one_proc_info[PROC_NUMA_NODE_ID] < 0) && (stream_info[NUMBER_OF_STREAMS] == 1)) {
            update_mix_stream_info(one_proc_info, proc_type_table, proc_type);
        } else {
            if (0 != one_proc_info[proc_type]) {
                if (n_threads_per_stream == -1) {
                    stream_info[THREADS_PER_STREAM] = (proc_type == EFFICIENT_CORE_PROC) ? 2 : 1;
                }
                stream_info[PROC_TYPE] = proc_type;
                update_ids_method(one_proc_info);
                stream_info[NUMBER_OF_STREAMS] =
                    static_cast<int>(one_proc_info[proc_type] / stream_info[THREADS_PER_STREAM]);
                if (n_streams < stream_info[NUMBER_OF_STREAMS]) {
                    stream_info[NUMBER_OF_STREAMS] = n_streams;
                }
                if (stream_info[NUMBER_OF_STREAMS] > 0) {
                    streams_info_table.push_back(stream_info);
                    n_streams -= stream_info[NUMBER_OF_STREAMS];
                }
            }
        }
    };

    auto check_threads_per_stream = [&]() {
        int count = 0;
        while (1) {
            for (int n_type = MAIN_CORE_PROC; n_type <= HYPER_THREADING_PROC; n_type++) {
                count += static_cast<int>(proc_type_table[0][n_type] / n_threads_per_stream);
            }
            if (count >= n_streams) {
                return;
            } else {
                count = 0;
                if (n_threads_per_stream > 1) {
                    n_threads_per_stream--;
                } else {
                    n_streams = n_threads;
                    return;
                }
            }
        }
    };

    if (proc_type_table.size() == 1) {
        proc_socket_table.push_back(proc_type_table[0]);
    } else {
        std::unordered_set<int> socket_id_list(proc_type_table.size());
        for (size_t i = 1; i < proc_type_table.size(); i++) {
            if (!socket_id_list.count(proc_type_table[i][PROC_SOCKET_ID])) {
                proc_socket_table.push_back(proc_type_table[i]);
                socket_id_list.insert(proc_type_table[i][PROC_SOCKET_ID]);
            } else {
                for (auto& row : proc_socket_table) {
                    if (row[PROC_SOCKET_ID] == proc_type_table[i][PROC_SOCKET_ID]) {
                        for (int n = 0; n <= HYPER_THREADING_PROC; n++) {
                            row[n] += proc_type_table[i][n];
                        }
                        if (row[PROC_NUMA_NODE_ID] != proc_type_table[i][PROC_NUMA_NODE_ID]) {
                            row[PROC_NUMA_NODE_ID] = -1;
                        }
                    }
                }
            }
        }
    }

    if (((input_streams_changed == false) &&
         (input_perf_hint == ov::util::to_string(ov::hint::PerformanceMode::LATENCY))) ||
        ((input_streams_changed == true) && (input_streams == 1))) {
        n_streams = 1;
        stream_info[NUMBER_OF_STREAMS] = n_streams;
        if (input_threads > 0) {
            n_threads_per_stream = std::min(input_threads, proc_type_table[0][ALL_PROC]);
            if (proc_type_table.size() == 1) {
                if ((n_threads_per_stream > proc_type_table[0][MAIN_CORE_PROC]) &&
                    (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                    stream_info[PROC_TYPE] = ALL_PROC;
                }
            } else {
                current_socket_id = input_current_socket_id == -1 ? get_current_socket_id() : input_current_socket_id;
            }
        } else if ((((input_streams_changed == false) || ((input_streams_changed == true) && (input_streams == 1))) &&
                    (latencyThreadingMode == Config::LatencyThreadingMode::PER_PLATFORM)) ||
                   (proc_type_table.size() == 1)) {
            if ((proc_type_table.size() == 1) && (model_prefer_threads > 0)) {
                if ((model_prefer_threads == proc_type_table[0][MAIN_CORE_PROC]) &&
                    (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                    stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                    n_threads_per_stream =
                        proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
                    stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
                    update_ids_method(proc_type_table[0]);
                } else if (proc_type_table[0][MAIN_CORE_PROC] == 0) {
                    stream_info[PROC_TYPE] = EFFICIENT_CORE_PROC;
                    n_threads_per_stream = proc_type_table[0][EFFICIENT_CORE_PROC];
                    stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
                    update_ids_method(proc_type_table[0]);
                } else {
                    stream_info[PROC_TYPE] = ALL_PROC;
                    n_threads_per_stream = proc_type_table[0][ALL_PROC];
                }
            } else {
                n_threads_per_stream = proc_type_table[0][ALL_PROC];
            }
        } else if (((input_streams_changed == false) || ((input_streams_changed == true) && (input_streams == 1))) &&
                   (latencyThreadingMode == Config::LatencyThreadingMode::PER_SOCKET)) {
            current_socket_id = input_current_socket_id == -1 ? get_current_socket_id() : input_current_socket_id;
            for (auto& row : proc_socket_table) {
                if (row[PROC_SOCKET_ID] == current_socket_id) {
                    n_threads_per_stream = std::max(n_threads_per_stream, row[ALL_PROC]);
                }
            }
        } else {
            current_socket_id = input_current_socket_id == -1 ? get_current_socket_id() : input_current_socket_id;
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                if (proc_type_table[i][PROC_SOCKET_ID] == current_socket_id) {
                    n_threads_per_stream = std::max(n_threads_per_stream, proc_type_table[i][ALL_PROC]);
                }
            }
        }
    } else {
        n_threads =
            input_threads > 0 ? std::min(proc_type_table[0][ALL_PROC], input_threads) : proc_type_table[0][ALL_PROC];
        if ((input_streams_changed == true) && (input_streams > 0)) {
            n_streams = input_infer_requests > 0 ? std::min(input_infer_requests, input_streams) : input_streams;
            if (n_streams >= n_threads) {
                n_streams = n_threads;
                n_threads_per_stream = 1;
            } else {
                n_threads_per_stream =
                    std::min(static_cast<int>(n_threads / n_streams),
                             proc_type_table[0][MAIN_CORE_PROC] == 0 ? proc_type_table[0][EFFICIENT_CORE_PROC]
                                                                     : proc_type_table[0][MAIN_CORE_PROC]);
                check_threads_per_stream();
            }
        } else {
            int base_type = (proc_type_table[0][MAIN_CORE_PROC] == 0) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
            if (0 == model_prefer_threads) {
                int n_proc = 0;

                if (proc_type_table.size() == 1) {
                    n_proc = std::min(n_threads, proc_type_table[0][base_type]);
                } else {
                    for (size_t i = 1; i < proc_type_table.size(); i++) {
                        n_proc = std::max(n_proc, proc_type_table[i][base_type]);
                    }
                    n_proc = std::min(n_threads, n_proc);
                }

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
                    if (proc_type_table.size() == 1) {
                        n_threads_per_stream = std::min(static_cast<int>(n_threads / n_streams), n_proc);
                    } else {
                        n_threads_per_stream = static_cast<int>(n_threads / n_streams);
                    }
                } else {
                    while ((n_streams * 2 <= n_threads_per_stream) && (n_threads_per_stream > 1)) {
                        n_threads_per_stream = static_cast<int>(n_threads_per_stream / 2);
                        n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                    }
                }
            } else if ((1 == model_prefer_threads) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
                       (proc_type_table[0][MAIN_CORE_PROC] > 0) && (n_threads > proc_type_table[0][MAIN_CORE_PROC])) {
                n_streams = (n_threads >= proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC])
                                ? static_cast<int>(n_threads - proc_type_table[0][EFFICIENT_CORE_PROC] / 2)
                                : static_cast<int>(proc_type_table[0][MAIN_CORE_PROC] +
                                                   (n_threads - proc_type_table[0][MAIN_CORE_PROC]) / 2);
                n_streams = input_infer_requests > 0 ? std::min(n_streams, input_infer_requests) : n_streams;
                n_threads_per_stream = -1;
            } else {
                n_streams = ((n_threads + model_prefer_threads - 1) / model_prefer_threads);
                if ((input_infer_requests > 0) && (n_streams > input_infer_requests)) {
                    n_streams = input_infer_requests;
                    n_threads_per_stream = static_cast<int>(n_threads / n_streams);
                    check_threads_per_stream();
                } else {
                    n_threads_per_stream =
                        model_prefer_threads > 0 ? model_prefer_threads : static_cast<int>(n_threads / n_streams);
                }
            }
        }
    }

    int total_streams = n_streams;

    if (stream_info[PROC_TYPE] == INIT_VAL) {
        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;

        for (int n_type = MAIN_CORE_PROC; (n_type <= HYPER_THREADING_PROC) && (n_streams > 0); n_type++) {
            if (proc_type_table.size() == 1) {
                if (proc_type_table[0][n_type] >= stream_info[THREADS_PER_STREAM]) {
                    update_streams_per_node(n_type, proc_type_table[0]);
                }
            } else {
                for (size_t n_node = 1; (n_node < proc_type_table.size()) && (n_streams > 0); n_node++) {
                    if ((proc_type_table[n_node][n_type] >= stream_info[THREADS_PER_STREAM]) &&
                        ((current_socket_id < 0) || (proc_type_table[n_node][PROC_SOCKET_ID] == current_socket_id))) {
                        update_streams_per_node(n_type, proc_type_table[n_node]);
                    }
                }
            }
        }

        if (total_streams == n_streams) {
            if (proc_type_table.size() == 1) {
                if (proc_type_table[0][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                    update_mix_stream_info(proc_type_table[0], proc_type_table, ALL_PROC);
                    n_streams--;
                }
            } else {
                for (size_t n_node = 0; (n_node < proc_socket_table.size()) && (n_streams > 0); n_node++) {
                    if ((proc_socket_table[n_node][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) &&
                        ((current_socket_id < 0) || (proc_socket_table[n_node][PROC_SOCKET_ID] == current_socket_id))) {
                        update_mix_stream_info(proc_socket_table[n_node], proc_type_table, ALL_PROC);
                        n_streams--;
                    }
                }
            }
        }

        if (total_streams == n_streams) {
            update_mix_stream_info(proc_type_table[0], proc_type_table, ALL_PROC);
            n_streams--;
        }

        if (n_streams > 0) {
            std::vector<std::vector<int>> remain_proc_type_table(proc_type_table);
            size_t stream_table_size = streams_info_table.size();

            for (size_t i = 0; i < stream_table_size; i++) {
                if ((streams_info_table[i][STREAM_NUMA_NODE_ID] >= 0) &&
                    (streams_info_table[i][STREAM_SOCKET_ID] >= 0)) {
                    for (auto& row : remain_proc_type_table) {
                        if ((streams_info_table[i][STREAM_NUMA_NODE_ID] == row[PROC_NUMA_NODE_ID]) &&
                            (streams_info_table[i][STREAM_SOCKET_ID] == row[PROC_SOCKET_ID])) {
                            row[streams_info_table[i][PROC_TYPE]] -= (streams_info_table[i][NUMBER_OF_STREAMS] == 0
                                                                          ? 1
                                                                          : streams_info_table[i][NUMBER_OF_STREAMS]) *
                                                                     streams_info_table[i][THREADS_PER_STREAM];
                        }
                    }
                }
            }

            while (n_streams > 0) {
                update_mix_stream_info(proc_type_table[0], remain_proc_type_table, ALL_PROC);

                if (stream_table_size == streams_info_table.size()) {
                    break;
                }
                n_streams--;
                int numa_node_id = streams_info_table[stream_table_size + 1][STREAM_NUMA_NODE_ID];
                int socket_id = streams_info_table[stream_table_size + 1][STREAM_SOCKET_ID];
                for (size_t i = stream_table_size + 1; i < streams_info_table.size(); i++) {
                    numa_node_id = numa_node_id == streams_info_table[i][STREAM_NUMA_NODE_ID] ? numa_node_id : -1;
                    socket_id = socket_id == streams_info_table[i][STREAM_SOCKET_ID] ? socket_id : -1;
                    for (auto& row : remain_proc_type_table) {
                        if ((streams_info_table[i][STREAM_NUMA_NODE_ID] == row[PROC_NUMA_NODE_ID]) &&
                            (streams_info_table[i][STREAM_SOCKET_ID] == row[PROC_SOCKET_ID])) {
                            row[streams_info_table[i][PROC_TYPE]] -= (streams_info_table[i][NUMBER_OF_STREAMS] == 0
                                                                          ? 1
                                                                          : streams_info_table[i][NUMBER_OF_STREAMS]) *
                                                                     streams_info_table[i][THREADS_PER_STREAM];
                        }
                    }
                }
                streams_info_table[stream_table_size][STREAM_NUMA_NODE_ID] = numa_node_id;
                streams_info_table[stream_table_size][STREAM_SOCKET_ID] = socket_id;
                stream_table_size = streams_info_table.size();
            }
        }
    } else {
        if (stream_info[PROC_TYPE] == ALL_PROC) {
            update_mix_stream_info(proc_type_table[0], proc_type_table, ALL_PROC);
        } else if (stream_info[PROC_TYPE] == MAIN_CORE_PROC) {
            if (stream_info[THREADS_PER_STREAM] == proc_socket_table[0][MAIN_CORE_PROC]) {
                streams_info_table.push_back(stream_info);
            } else {
                stream_info[PROC_TYPE] = ALL_PROC;
                streams_info_table.push_back(stream_info);
                stream_info[NUMBER_OF_STREAMS] = 0;
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                stream_info[THREADS_PER_STREAM] = proc_socket_table[0][MAIN_CORE_PROC];
                streams_info_table.push_back(stream_info);
                stream_info[PROC_TYPE] = HYPER_THREADING_PROC;
                stream_info[THREADS_PER_STREAM] = proc_socket_table[0][HYPER_THREADING_PROC];
                streams_info_table.push_back(stream_info);
            }
        } else {
            streams_info_table.push_back(stream_info);
        }
    }

    return streams_info_table;
}

int get_model_prefer_threads(const int num_streams,
                             const std::vector<std::vector<int>> proc_type_table,
                             const std::shared_ptr<ov::Model>& model,
                             Config& config) {
    const int sockets = get_default_latency_streams(config.latencyThreadingMode);
    auto model_prefer = 0;
    if (-1 == config.modelPreferThreads) {
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
            ov::mem_bandwidth_pressure_tolerance(model, L2_cache_size, memThresholdAssumeLimitedForISA);

#if ((defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__linux__))
        config.modelPreferThreads = 4;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if (networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) {
                config.modelPreferThreads = 8;
            }
        } else if ((networkToleranceForLowCache.max_mem_tolerance < ov::MemBandwidthPressure::LIMITED) &&
                   ((networkToleranceForLowCache.ratio_mem_limited_deconvs > ov::MemBandwidthPressure::LIMITED) ||
                    (networkToleranceForLowCache.ratio_mem_limited_gemms > ov::MemBandwidthPressure::LIMITED))) {
            config.modelPreferThreads = 8;
        }
#elif((defined(OPENVINO_ARCH_ARM) || defined(OPENVINO_ARCH_ARM64)) && defined(__APPLE__))
        config.modelPreferThreads = 1;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                config.modelPreferThreads = 4;
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.ratio_mem_limited_deconvs > ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_compute_convs < ov::MemBandwidthPressure::ALL) {
            config.modelPreferThreads = 4;
        } else if (networkToleranceForLowCache.ratio_mem_limited_deconvs <= ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_mem_limited_convs <= ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_compute_convs > ov::MemBandwidthPressure::LIMITED) {
            config.modelPreferThreads = 2;
        }
#else
        config.modelPreferThreads = 0;
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
                config.modelPreferThreads = 1;
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
            config.modelPreferThreads = 2;
        }
        if (config.modelPreferThreads == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 && sockets == 1) {
            config.modelPreferThreads = 2;
        }
#endif
    }

    // latency
    if (num_streams <= sockets && num_streams > 0) {
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0) {
#ifdef __APPLE__
            if ((proc_type_table.size() == 1) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0)) {
                model_prefer = proc_type_table[0][MAIN_CORE_PROC] > proc_type_table[0][EFFICIENT_CORE_PROC]
                                   ? proc_type_table[0][MAIN_CORE_PROC]
                                   : proc_type_table[0][ALL_PROC];
            }
#else
            bool llm_related = has_matmul_with_compressed_weights(model);
            bool int8_intensive = ov::op::util::has_op_with_type<ov::op::v0::FakeQuantize>(model) || llm_related;
            const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
            const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
            // By default the latency case uses (faster) Big cores only, depending on the compute ratio
            // But on MTL detected by ov::get_number_of_blocked_cores(), use Big and Little cores together in Big cores
            // only cases except LLM.
            model_prefer = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] /
                                                                 (int8_intensive ? int8_threshold : fp32_threshold))
                               ? ((!llm_related && ov::get_number_of_blocked_cores())
                                      ? proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC]
                                      : proc_type_table[0][MAIN_CORE_PROC])
                               : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
#endif
        }
    } else {  // throughput
        model_prefer = config.modelPreferThreads;
    }

    return model_prefer;
}

std::vector<std::vector<int>> generate_stream_info(const int streams,
                                                   const int input_current_socket_id,
                                                   const std::shared_ptr<ov::Model>& model,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream) {
    int model_prefer_threads = preferred_nthreads_per_stream;
    proc_type_table = apply_scheduling_core_type(config.schedulingCoreType, proc_type_table);

    proc_type_table = apply_hyper_threading(config.enableHyperThreading,
                                            config.changedHyperThreading,
                                            ov::util::to_string(config.hintPerfMode),
                                            proc_type_table);

    if (-1 == preferred_nthreads_per_stream) {
        model_prefer_threads = get_model_prefer_threads(streams, proc_type_table, model, config);
    }

    auto streams_info_table = get_streams_info_table(config.streams,
                                                     config.streamsChanged,
                                                     config.threads,
                                                     config.hintNumRequests,
                                                     model_prefer_threads,
                                                     input_current_socket_id,
                                                     ov::util::to_string(config.hintPerfMode),
                                                     config.latencyThreadingMode,
                                                     proc_type_table);

    auto cpu_reservation =
        get_cpu_pinning(config.enableCpuPinning, config.changedCpuPinning, proc_type_table, streams_info_table);

    config.streamExecutorConfig = IStreamsExecutor::Config{"CPUStreamsExecutor",
                                                           config.streams,
                                                           config.threadsPerStream,
                                                           config.threadBindingType,
                                                           1,
                                                           0,
                                                           config.threads,
                                                           IStreamsExecutor::Config::PreferredCoreType::ANY,
                                                           streams_info_table,
                                                           cpu_reservation};

    return proc_type_table;
}

void get_num_streams(const int streams, const std::shared_ptr<ov::Model>& model, Config& config) {
    std::vector<std::vector<int>> proc_type_table = get_proc_type_table();

    generate_stream_info(streams, -1, model, config, proc_type_table);
}

int get_default_latency_streams(Config::LatencyThreadingMode latency_threading_mode) {
    if (latency_threading_mode == Config::LatencyThreadingMode::PER_NUMA_NODE) {
        return get_num_sockets();
    } else if (latency_threading_mode == Config::LatencyThreadingMode::PER_SOCKET) {
        return get_num_numa_nodes();
    } else {
        return 1;
    }
}

}  // namespace intel_cpu
}  // namespace ov
