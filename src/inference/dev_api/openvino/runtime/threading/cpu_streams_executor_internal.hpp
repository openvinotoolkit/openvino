// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file openvino/runtime/threading/cpu_streams_executor_internal.hpp
 * @brief A header file for OpenVINO Streams-based Executor Interface
 */

#pragma once

#include <string>
#include <vector>

namespace ov {
namespace threading {

enum StreamCreateType {
    STREAM_WITHOUT_PARAM = 0,  // new task_arena with no parameters, no threads binding
    STREAM_WITH_CORE_TYPE,     // new task_arena with core type, threads binding with core type
    STREAM_WITH_NUMA_ID,       // new task_arena with numa node id, threads binding with numa node id
    STREAM_WITH_OBSERVE        // new task_arena with no parameters, threads binding with observe
};

/**
 * @brief      Get current stream information
 * @param[in]  stream_id stream id
 * @param[in]  cpu_pinning whether to bind threads to cpus
 * @param[in]  org_proc_type_table available processors in the platform
 * @param[in]  streams_info_table streams information table
 * @param[out]  stream_type stream create type
 * @param[out]  concurrency the number of threads created at the same time
 * @param[out]  core_type core type
 * @param[out]  numa_node_id numa node id
 * @param[out]  max_threads_per_core the max number of threads per cpu core
 */
void get_cur_stream_info(const int stream_id,
                         const bool cpu_pinning,
                         const std::vector<std::vector<int>> org_proc_type_table,
                         const std::vector<std::vector<int>> streams_info_table,
                         StreamCreateType& stream_type,
                         int& concurrency,
                         int& core_type,
                         int& numa_node_id,
                         int& max_threads_per_core);

/**
 * @brief      Reserve cpu resource by streams info
 * @param[in]  _streams_info_table streams info table
 * @param[in]  _numa_nodes number of numa nodes
 * @param[out]  _cpu_mapping_table CPU mapping table for each processor
 * @param[out]  _proc_type_table summary table of number of processors per type
 * @param[out] _stream_processors processors grouped in stream which is used in core binding in cpu streams executor
 * @param[in]  _cpu_status set cpu status
 * @return
 */
void reserve_cpu_by_streams_info(const std::vector<std::vector<int>> _streams_info_table,
                                 const int _numa_nodes,
                                 std::vector<std::vector<int>>& _cpu_mapping_table,
                                 std::vector<std::vector<int>>& _proc_type_table,
                                 std::vector<std::vector<int>>& _stream_processors,
                                 const int _cpu_status);

/**
 * @brief      Update proc_type_table
 * @param[in]  _cpu_mapping_table CPU mapping table for each processor
 * @param[in]  _numa_nodes total number for nodes in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @return
 */
void update_proc_type_table(const std::vector<std::vector<int>> _cpu_mapping_table,
                            const int _numa_nodes,
                            std::vector<std::vector<int>>& _proc_type_table);

}  // namespace threading
}  // namespace ov