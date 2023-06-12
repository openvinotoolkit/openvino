// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "openvino/runtime/threading/istreams_executor.hpp"
#include "threading/ie_cpu_streams_info.hpp"

using namespace InferenceEngine;

namespace ov {
namespace threading {

void get_cur_stream_info(const int stream_id,
                         const bool cpu_reservation,
                         const std::vector<std::vector<int>> proc_type_table,
                         const std::vector<std::vector<int>> streams_info_table,
                         const std::vector<int> stream_numa_node_ids,
                         StreamCreateType& stream_type,
                         int& concurrency,
                         int& core_type,
                         int& numa_node_id) {
    int stream_total = 0;
    size_t stream_info_id = 0;
    for (size_t i = 0; i < streams_info_table.size(); i++) {
        stream_total =
            i > 0 ? stream_total + streams_info_table[i][NUMBER_OF_STREAMS] : streams_info_table[i][NUMBER_OF_STREAMS];
        if (stream_id < stream_total) {
            stream_info_id = i;
            break;
        }
    }
    concurrency = streams_info_table[stream_info_id][THREADS_PER_STREAM];
    core_type = streams_info_table[stream_info_id][PROC_TYPE];
    numa_node_id = stream_numa_node_ids.size() > 0 ? stream_numa_node_ids[stream_id] : 0;

    if (cpu_reservation) {
        stream_type = STREAM_WITH_OBSERVE;
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
#if defined(_WIN32) || defined(__APPLE__)
            stream_type = STREAM_WITH_CORE_TYPE;
#endif
        }
    } else {
        stream_type = STREAM_WITHOUT_PARAM;
        if (proc_type_table[0][EFFICIENT_CORE_PROC] > 0 && core_type != ALL_PROC) {
            stream_type = STREAM_WITH_CORE_TYPE;
        } else if (proc_type_table.size() > 1 && numa_node_id >= 0) {
            stream_type = STREAM_WITH_NUMA_ID;
        }
    }
}

}  // namespace threading
}  // namespace ov
