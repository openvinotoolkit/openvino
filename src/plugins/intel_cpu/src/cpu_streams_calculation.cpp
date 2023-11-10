// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <transformations/utils/utils.hpp>
#include <unordered_set>

#include "cpu_map_scheduling.hpp"
#include "graph.h"
#include "ie_system_conf.h"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"
#include "performance_heuristics.hpp"

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
                                                     const std::string input_perf_hint,
                                                     const Config::LatencyThreadingMode latencyThreadingMode,
                                                     const std::vector<std::vector<int>>& proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE, INIT_VAL);
    std::vector<std::vector<int>> streams_info_table;
    std::vector<std::vector<int>> proc_socket_table;

    int n_streams = 0;
    int n_threads = 0;
    int n_threads_per_stream = 0;

    auto update_ids_method = [&](const std::vector<int>& one_proc_info) {
        stream_info[STREAM_NUMA_NODE_ID] = one_proc_info[PROC_NUMA_NODE_ID];
        stream_info[STREAM_SOCKET_ID] = one_proc_info[PROC_SOCKET_ID];
    };

    auto update_mix_stream_info = [&](const std::vector<int>& one_proc_info,
                                      const std::vector<std::vector<int>>& one_proc_table) {
        stream_info[PROC_TYPE] = ALL_PROC;
        stream_info[NUMBER_OF_STREAMS] = 1;
        stream_info[THREADS_PER_STREAM] = n_threads_per_stream;
        update_ids_method(one_proc_info);
        streams_info_table.push_back(stream_info);
        stream_info[NUMBER_OF_STREAMS] = 0;
        int total_threads = stream_info[THREADS_PER_STREAM];
        int numa_node_id = stream_info[STREAM_NUMA_NODE_ID];
        int socket_id = stream_info[STREAM_SOCKET_ID];
        int node_start = one_proc_table.size() == 1 ? 0 : 1;
        int node_end = one_proc_table.size() == 1 ? 1 : one_proc_table.size();
        for (int n = MAIN_CORE_PROC; n <= HYPER_THREADING_PROC; n++) {
            for (int index = node_start; index < node_end; index++) {
                if (((numa_node_id < 0) || (numa_node_id == one_proc_table[index][PROC_NUMA_NODE_ID])) &&
                    ((socket_id < 0) || (socket_id == one_proc_table[index][PROC_SOCKET_ID]))) {
                    if (0 != one_proc_table[index][n]) {
                        stream_info[PROC_TYPE] = n;
                        if (total_threads <= one_proc_table[index][n]) {
                            stream_info[THREADS_PER_STREAM] = total_threads;
                            stream_info[STREAM_NUMA_NODE_ID] = one_proc_table[index][PROC_NUMA_NODE_ID];
                            stream_info[STREAM_SOCKET_ID] = one_proc_table[index][PROC_SOCKET_ID];
                            streams_info_table.push_back(stream_info);
                            return;
                        } else {
                            stream_info[THREADS_PER_STREAM] = one_proc_table[index][n];
                            stream_info[STREAM_NUMA_NODE_ID] = one_proc_table[index][PROC_NUMA_NODE_ID];
                            stream_info[STREAM_SOCKET_ID] = one_proc_table[index][PROC_SOCKET_ID];
                            streams_info_table.push_back(stream_info);
                            total_threads -= one_proc_table[index][n];
                        }
                    }
                }
            }
        }
    };

    auto update_streams_per_node = [&](const int& proc_type, const std::vector<int>& one_proc_info) {
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

    if (((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY)) &&
         ((latencyThreadingMode == Config::LatencyThreadingMode::PER_PLATFORM) || (proc_type_table.size() == 1))) ||
        ((input_streams_changed == true) && (input_streams == 1))) {
        n_streams = 1;
        if ((proc_type_table.size() == 1) && (input_threads == 0) && (model_prefer_threads > 0)) {
            stream_info[NUMBER_OF_STREAMS] = n_streams;
            if ((model_prefer_threads == proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = MAIN_CORE_PROC;
                n_threads_per_stream = proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][HYPER_THREADING_PROC];
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
            n_threads_per_stream = input_threads > 0 ? std::min(input_threads, proc_type_table[0][ALL_PROC])
                                                     : proc_type_table[0][ALL_PROC];
            if ((proc_type_table.size() == 1) && (n_threads_per_stream > proc_type_table[0][MAIN_CORE_PROC]) &&
                (proc_type_table[0][MAIN_CORE_PROC] > 0)) {
                stream_info[PROC_TYPE] = ALL_PROC;
            }
        }
    } else if ((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY)) &&
               (latencyThreadingMode == Config::LatencyThreadingMode::PER_SOCKET)) {
        for (auto& row : proc_socket_table) {
            n_threads_per_stream = std::max(n_threads_per_stream, row[ALL_PROC]);
        }
        n_threads_per_stream = input_threads > 0 ? std::min(input_threads, n_threads_per_stream) : n_threads_per_stream;
        for (auto& row : proc_socket_table) {
            if (n_threads_per_stream <= row[ALL_PROC]) {
                n_streams++;
            }
        }
        n_streams = input_threads > 0 ? static_cast<int>(input_threads / n_threads_per_stream) : n_streams;
        n_streams = input_infer_requests > 0 ? std::min(input_infer_requests, n_streams) : n_streams;
    } else if ((input_streams_changed == false) && (input_perf_hint == CONFIG_VALUE(LATENCY)) &&
               (latencyThreadingMode == Config::LatencyThreadingMode::PER_NUMA_NODE)) {
        if (proc_type_table.size() == 1) {
            n_streams = 1;
            n_threads_per_stream = input_threads > 0 ? std::min(input_threads, proc_type_table[0][ALL_PROC])
                                                     : proc_type_table[0][ALL_PROC];
        } else {
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                n_threads_per_stream = std::max(n_threads_per_stream, proc_type_table[i][ALL_PROC]);
            }
            n_threads_per_stream =
                input_threads > 0 ? std::min(input_threads, n_threads_per_stream) : n_threads_per_stream;
            for (size_t i = 1; i < proc_type_table.size(); i++) {
                if (n_threads_per_stream <= proc_type_table[i][ALL_PROC]) {
                    n_streams++;
                }
            }
            n_streams = input_threads > 0 ? static_cast<int>(input_threads / n_threads_per_stream) : n_streams;
            n_streams = input_infer_requests > 0 ? std::min(input_infer_requests, n_streams) : n_streams;
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
                    if (proc_type_table[n_node][n_type] >= stream_info[THREADS_PER_STREAM]) {
                        update_streams_per_node(n_type, proc_type_table[n_node]);
                    }
                }
            }
        }

        if (total_streams == n_streams) {
            if (proc_type_table.size() == 1) {
                if (proc_type_table[0][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                    update_mix_stream_info(proc_type_table[0], proc_type_table);
                    n_streams--;
                }
            } else {
                for (size_t n_node = 1; (n_node < proc_type_table.size()) && (n_streams > 0); n_node++) {
                    if (proc_type_table[n_node][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                        update_mix_stream_info(proc_type_table[n_node], proc_type_table);
                        n_streams--;
                    }
                }
            }
            for (size_t n_node = 0; (n_node < proc_socket_table.size()) && (n_streams > 0); n_node++) {
                if (proc_socket_table[n_node][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                    update_mix_stream_info(proc_socket_table[n_node], proc_type_table);
                    n_streams--;
                }
            }
        }

        if (total_streams == n_streams) {
            for (size_t n_node = 0; (n_node < proc_socket_table.size()) && (n_streams > 0); n_node++) {
                if (proc_socket_table[n_node][ALL_PROC] >= stream_info[THREADS_PER_STREAM]) {
                    update_mix_stream_info(proc_socket_table[n_node], proc_type_table);
                    n_streams--;
                }
            }
        }

        if (total_streams == n_streams) {
            update_mix_stream_info(proc_type_table[0], proc_type_table);
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
                update_mix_stream_info(proc_type_table[0], remain_proc_type_table);

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
            update_mix_stream_info(proc_socket_table[0], proc_type_table);
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
            ov::MemBandwidthPressureTolerance(model, L2_cache_size, memThresholdAssumeLimitedForISA);
#if defined(__arm__) || defined(__aarch64__)
        config.modelPreferThreads = 1;
#else
        config.modelPreferThreads = ov::threading::IStreamsExecutor::Config::StreamMode::DEFAULT;
#endif
        if (networkToleranceForLowCache.max_mem_tolerance == ov::MemBandwidthPressure::UNKNOWN) {
            if ((networkToleranceForLowCache.ratio_compute_convs == ov::MemBandwidthPressure::ALL) ||
                (networkToleranceForLowCache.ratio_compute_deconvs == ov::MemBandwidthPressure::ALL)) {
                // all relevant layers (convs, etc) are compute-limited, the most aggressive val for #streams
#if defined(__arm__) || defined(__aarch64__)
                config.modelPreferThreads = 4;
#else
                config.modelPreferThreads = 1;
#endif
            }  // otherwise (no recognized layers) falling back to the default value
        } else if (networkToleranceForLowCache.max_mem_tolerance > memThresholdAssumeLimitedForISA) {
            // network is below the ISA-specific threshold
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.max_mem_tolerance > ov::MemBandwidthPressure::LIMITED) {
            // network is below general threshold
#if defined(__arm__) || defined(__aarch64__)
            config.modelPreferThreads = 1;
        } else if (networkToleranceForLowCache.ratio_mem_limited_deconvs > ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_compute_convs < ov::MemBandwidthPressure::ALL) {
            config.modelPreferThreads = 4;
        } else if (networkToleranceForLowCache.ratio_mem_limited_deconvs <= ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_mem_limited_convs <= ov::MemBandwidthPressure::LIMITED &&
                   networkToleranceForLowCache.ratio_compute_convs > ov::MemBandwidthPressure::LIMITED) {
            config.modelPreferThreads = 2;
#else
            config.modelPreferThreads = 2;
#endif
        }
        if (config.modelPreferThreads == 1 && proc_type_table[0][EFFICIENT_CORE_PROC] == 0 && sockets == 1) {
            config.modelPreferThreads = 2;
        }
    }

    // latency
    if (num_streams <= sockets && num_streams > 0) {
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && proc_type_table[0][MAIN_CORE_PROC] > 0) {
#ifdef __APPLE__
            if ((proc_type_table.size() == 1) && (proc_type_table[0][EFFICIENT_CORE_PROC] > 0)) {
                model_prefer = proc_type_table[0][ALL_PROC];
            }
#else
            bool fp_intesive = !ov::op::util::has_op_with_type<ov::op::v0::FakeQuantize>(model);
            const int int8_threshold = 4;  // ~relative efficiency of the VNNI-intensive code for Big vs Little cores;
            const int fp32_threshold = 2;  // ~relative efficiency of the AVX2 fp32 code for Big vs Little cores;
            // by default the latency case uses (faster) Big cores only, depending on the compute ratio
            model_prefer = proc_type_table[0][MAIN_CORE_PROC] > (proc_type_table[0][EFFICIENT_CORE_PROC] /
                                                                 (fp_intesive ? fp32_threshold : int8_threshold))
                               ? proc_type_table[0][MAIN_CORE_PROC]
                               : proc_type_table[0][MAIN_CORE_PROC] + proc_type_table[0][EFFICIENT_CORE_PROC];
#endif
        }
    } else {  // throughput
        model_prefer = config.modelPreferThreads;
    }

    return model_prefer;
}

std::vector<std::vector<int>> generate_stream_info(const int streams,
                                                   const std::shared_ptr<ov::Model>& model,
                                                   Config& config,
                                                   std::vector<std::vector<int>>& proc_type_table,
                                                   int preferred_nthreads_per_stream) {
    int model_prefer_threads = preferred_nthreads_per_stream;
    IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;
    proc_type_table = apply_scheduling_core_type(config.schedulingCoreType, proc_type_table);

    proc_type_table = apply_hyper_threading(config.enableHyperThreading,
                                            config.changedHyperThreading,
                                            config.perfHintsConfig.ovPerfHint,
                                            proc_type_table);
    executor_config._cpu_reservation = get_cpu_pinning(config.enableCpuPinning,
                                                       config.changedCpuPinning,
                                                       streams,
                                                       config.latencyThreadingMode,
                                                       proc_type_table);
    if (-1 == preferred_nthreads_per_stream) {
        model_prefer_threads = get_model_prefer_threads(streams, proc_type_table, model, config);
    }

    executor_config._streams_info_table = get_streams_info_table(executor_config._streams,
                                                                 executor_config._streams_changed,
                                                                 executor_config._threads,
                                                                 config.perfHintsConfig.ovPerfHintNumRequests,
                                                                 model_prefer_threads,
                                                                 config.perfHintsConfig.ovPerfHint,
                                                                 config.latencyThreadingMode,
                                                                 proc_type_table);
    return proc_type_table;
}

void get_num_streams(const int streams, const std::shared_ptr<ov::Model>& model, Config& config) {
    IStreamsExecutor::Config& executor_config = config.streamExecutorConfig;
    std::vector<std::vector<int>> proc_type_table = get_proc_type_table();

    generate_stream_info(streams, model, config, proc_type_table);

    executor_config = IStreamsExecutor::Config::reserve_cpu_threads(executor_config);
    executor_config._threadsPerStream = executor_config._streams_info_table[0][THREADS_PER_STREAM];
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
