// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <vector>

#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {
namespace threading {

void get_cur_stream_info(const int stream_id,
                         const bool cpu_reservation,
                         const std::vector<std::vector<int>> proc_type_table,
                         const std::vector<std::vector<int>> streams_info_table,
                         StreamCreateType& stream_type,
                         int& concurrency,
                         int& core_type,
                         int& numa_node_id) {
    int stream_total = 0;
    size_t stream_info_id = 0;
    bool cpu_reserve = cpu_reservation;
    for (size_t i = 0; i < streams_info_table.size(); i++) {
        stream_total += streams_info_table[i][NUMBER_OF_STREAMS];
        if (stream_id < stream_total) {
            stream_info_id = i;
            break;
        }
    }
    concurrency = streams_info_table[stream_info_id][THREADS_PER_STREAM];
    core_type = streams_info_table[stream_info_id][PROC_TYPE];
    numa_node_id = streams_info_table[stream_info_id][STREAM_NUMA_NODE_ID];

#if defined(_WIN32) || defined(__APPLE__)
    cpu_reserve = false;
#endif
    if (cpu_reserve) {
        stream_type = STREAM_WITH_OBSERVE;
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
