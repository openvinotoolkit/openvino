// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_streams_calculation.hpp"

#include <algorithm>
#include <cstdio>
#include <iostream>
#include <numeric>

#include "ie_system_conf.h"
#include "threading/ie_cpu_streams_info.hpp"

using namespace InferenceEngine;

namespace ov {
namespace intel_cpu {

std::vector<std::vector<int>> get_streams_info_table(const int input_streams,
                                                     const int input_threads,
                                                     const int model_prefer_threads,
                                                     const std::vector<std::vector<int>> proc_type_table) {
    std::vector<int> stream_info(CPU_STREAMS_TABLE_SIZE);
    std::vector<std::vector<int>> streams_info_table;

    if (1 == input_streams) {
        stream_info[NUMBER_OF_STREAMS] = 1;
        int limit_threads = (input_threads == 0) ? model_prefer_threads : input_threads;
        if ((proc_type_table[0][EFFICIENT_CORE_PROC] > 0) &&
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

        if (proc_type_table.size() == 1) {
            n_threads = (0 == input_threads) ? proc_type_table[0][ALL_PROC]
                                             : std::min(proc_type_table[0][ALL_PROC], input_threads);
        } else {
            n_threads = (0 == input_threads) ? proc_type_table[0][MAIN_CORE_PROC]
                                             : std::min(proc_type_table[0][MAIN_CORE_PROC], input_threads);
        }

        if (0 != input_streams) {
            if (input_streams >= n_threads) {
                n_streams = n_threads;
                n_threads_per_stream = 1;
            } else {
                n_streams = input_streams;
                n_threads_per_stream =
                    std::min(std::max(1, n_threads / input_streams), proc_type_table[0][MAIN_CORE_PROC]);
                if (proc_type_table.size() == 1) {
                    if ((n_threads_per_stream > proc_type_table[0][MAIN_CORE_PROC]) &&
                        (n_threads_per_stream < proc_type_table[0][MAIN_CORE_PROC] * 2)) {
                        n_threads_per_stream = proc_type_table[0][MAIN_CORE_PROC];
                    } else if (n_threads_per_stream < proc_type_table[0][MAIN_CORE_PROC]) {
                        n_threads_per_stream = int(
                            proc_type_table[0][MAIN_CORE_PROC] /
                            ((proc_type_table[0][MAIN_CORE_PROC] + n_threads_per_stream - 1) / n_threads_per_stream));
                    }
                }
            }
        } else {
            if (0 == model_prefer_threads) {
                int n_proc = std::min(n_threads, proc_type_table[0][MAIN_CORE_PROC]);
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

                while (n_streams < n_threads_per_stream) {
                    if (1 == n_threads_per_stream) {
                        break;
                    } else {
                        n_threads_per_stream = static_cast<int>((n_threads_per_stream * 2 - 1) / 2);
                        n_threads_per_stream = static_cast<int>(
                            proc_type_table[0][MAIN_CORE_PROC] /
                            ((proc_type_table[0][MAIN_CORE_PROC] + n_threads_per_stream - 1) / n_threads_per_stream));
                        n_streams = static_cast<int>(n_threads / n_threads_per_stream);
                    }
                }
            } else {
                n_streams = ((n_threads + model_prefer_threads - 1) / model_prefer_threads);
                n_threads_per_stream = static_cast<int>(n_threads / n_streams);
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
}  // namespace intel_cpu
}  // namespace ov