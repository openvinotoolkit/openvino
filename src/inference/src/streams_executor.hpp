// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides a set minimal required Streams Executor API.
 * @file streams_executor.hpp
 */
#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"

namespace ov {

class CPU {
public:
    CPU();
    ~CPU(){};
    int _processors = 0;
    int _numa_nodes = 0;
    int _cores = 0;
    std::vector<std::vector<int>> _org_proc_type_table;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
    std::mutex _cpu_mutex;
    std::mutex _plugin_mutex;
    int _socket_idx = 0;
    int _num_threads = 0;
};

CPU& cpu_info();

void reserve_cpu_by_streams_info(const std::vector<std::vector<int>> _streams_info_table,
                                 const std::vector<std::vector<int>> _cpu_mapping_table,
                                 const std::vector<std::vector<int>> _proc_type_table,
                                 const int _numa_nodes,
                                 std::vector<std::vector<int>>& _stream_processors,
                                 std::vector<int>& _stream_numa_node_ids,
                                 const int _cpu_status);

#ifdef __linux__
/**
 * @brief      Parse CPU cache infomation on Linux
 * @param[in]  _system_info_table system information for this platform.
 * @param[out]  _processors total number for processors in system.
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_cache_info_linux(const std::vector<std::vector<std::string>> _system_info_table,
                            int& _processors,
                            int& _sockets,
                            int& _cores,
                            std::vector<std::vector<int>>& _proc_type_table,
                            std::vector<std::vector<int>>& _cpu_mapping_table);

/**
 * @brief      Parse CPU frequency infomation on Linux
 * @param[in]  _system_info_table system information for this platform.
 * @param[out]  _processors total number for processors in system.
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_freq_info_linux(const std::vector<std::vector<std::string>> _system_info_table,
                           int& _processors,
                           int& _sockets,
                           int& _cores,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table);
#endif

#if defined(_WIN32)
/**
 * @brief      Parse processors infomation on Windows
 * @param[in]  base_ptr buffer object pointer of Windows system infomation
 * @param[in]  len buffer object length of Windows system infomation
 * @param[out] _processors total number for processors in system.
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_processor_info_win(const char* base_ptr,
                              const unsigned long len,
                              int& _processors,
                              int& _sockets,
                              int& _cores,
                              std::vector<std::vector<int>>& _proc_type_table,
                              std::vector<std::vector<int>>& _cpu_mapping_table);
#endif

#if defined(__APPLE__)
/**
 * @brief      Parse processors infomation on Linux
 * @param[in]  _processors total number for processors in system.
 * @param[out] _numa_nodes total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @return
 */
int parse_processor_info_macos(int& _processors,
                               int& _numa_nodes,
                               int& _cores,
                               std::vector<std::vector<int>>& _proc_type_table);
#endif

}  // namespace ov
