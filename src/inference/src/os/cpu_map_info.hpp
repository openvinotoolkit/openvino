// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides a set of CPU map and parser functions.
 * @file cpu_map_info.hpp
 */
#pragma once

#include <mutex>
#include <string>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/util/log.hpp"

namespace ov {

class CPU {
public:
    CPU();
    ~CPU(){};
    void cpu_debug() {
#ifdef ENABLE_OPENVINO_DEBUG
        OPENVINO_DEBUG("[ threading ] cpu_mapping_table:");
        for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
            OPENVINO_DEBUG(_cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID] , " ",
                           _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID], " ",
                           _cpu_mapping_table[i][CPU_MAP_SOCKET_ID], " ", _cpu_mapping_table[i][CPU_MAP_CORE_ID],
                           " ", _cpu_mapping_table[i][CPU_MAP_CORE_TYPE], " ",
                           _cpu_mapping_table[i][CPU_MAP_GROUP_ID], " ",
                           _cpu_mapping_table[i][CPU_MAP_USED_FLAG]);
        }
        OPENVINO_DEBUG("[ threading ] org_proc_type_table:");
        for (size_t i = 0; i < _proc_type_table.size(); i++) {
            OPENVINO_DEBUG(_proc_type_table[i][ALL_PROC], " ", _proc_type_table[i][MAIN_CORE_PROC], " ",
                           _proc_type_table[i][EFFICIENT_CORE_PROC], " ",
                           _proc_type_table[i][HYPER_THREADING_PROC], " ", _proc_type_table[i][PROC_NUMA_NODE_ID],
                           " ", _proc_type_table[i][PROC_SOCKET_ID]);
        }
#endif
    }
    int _processors = 0;
    int _numa_nodes = 0;
    int _sockets = 0;
    int _cores = 0;
    int _blocked_cores = 0;
    std::vector<std::vector<int>> _org_proc_type_table;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
    std::map<int, int> _socketid_mapping_table;
    std::map<int, int> _numaid_mapping_table;
    std::mutex _cpu_mutex;
    int _socket_idx = 0;

};

CPU& cpu_info();

#ifdef __linux__
/**
 * @brief      Parse nodes information to update _sockets, proc_type_table and cpu_mapping_table on Linux
 * @param[in]  node_info_table nodes information for this platform.
 * @param[in]  _numa_nodes total number for nodes in system
 * @param[out] _sockets total number for sockets in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_node_info_linux(const std::vector<std::string> node_info_table,
                           const int& _numa_nodes,
                           int& _sockets,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table);

/**
 * @brief      Parse CPU cache infomation on Linux
 * @param[in]  system_info_table cpus information for this platform.
 * @param[in]  node_info_table nodes information for this platform.
 * @param[out] _processors total number for processors in system.
 * @param[out] _numa_nodes total number for nodes in system
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_cache_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                            const std::vector<std::string> node_info_table,
                            int& _processors,
                            int& _numa_nodes,
                            int& _sockets,
                            int& _cores,
                            std::vector<std::vector<int>>& _proc_type_table,
                            std::vector<std::vector<int>>& _cpu_mapping_table);

/**
 * @brief      Parse CPU frequency infomation on Linux
 * @param[in]  system_info_table cpus information for this platform.
 * @param[in]  node_info_table nodes information for this platform.
 * @param[out] _processors total number for processors in system.
 * @param[out] _numa_nodes total number for nodes in system
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_freq_info_linux(const std::vector<std::vector<std::string>> system_info_table,
                           const std::vector<std::string> node_info_table,
                           int& _processors,
                           int& _numa_nodes,
                           int& _sockets,
                           int& _cores,
                           std::vector<std::vector<int>>& _proc_type_table,
                           std::vector<std::vector<int>>& _cpu_mapping_table);

/**
 * @brief      update proc_type_table and cpu_mapping_table for vaild processors.
 * @param[in]  phy_core_list CPU cores id list for physical core of Intel Performance-cores.
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void update_valid_processor_linux(const std::vector<int> phy_core_list,
                                  int& _sockets,
                                  int& _cores,
                                  std::vector<std::vector<int>>& _proc_type_table,
                                  std::vector<std::vector<int>>& _cpu_mapping_table);

/**
 * @brief      Get cpu_mapping_table from the number of processors, cores and numa nodes
 * @param[in]  _processors total number for processors in system.
 * @param[in]  _numa_nodes total number for numa nodes in system
 * @param[in]  _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void get_cpu_mapping_from_cores(const int _processors,
                                const int _numa_nodes,
                                const int _cores,
                                std::vector<std::vector<int>>& _proc_type_table,
                                std::vector<std::vector<int>>& _cpu_mapping_table);

#endif

#if defined(_WIN32)
/**
 * @brief      Parse processors infomation on Windows
 * @param[in]  base_ptr buffer object pointer of Windows system infomation
 * @param[in]  len buffer object length of Windows system infomation
 * @param[out] _processors total number for processors in system.
 * @param[out] _numa_nodes total number for nodes in system
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _blocked_cores total number for blocked processors in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_processor_info_win(const char* base_ptr,
                              const unsigned long len,
                              int& _processors,
                              int& _numa_nodes,
                              int& _sockets,
                              int& _cores,
                              int& _blocked_cores,
                              std::vector<std::vector<int>>& _proc_type_table,
                              std::vector<std::vector<int>>& _cpu_mapping_table);
#endif

#if defined(__APPLE__)
/**
 * @brief      Parse processors infomation on MacOS
 * @param[in]  system_info_table cpus information for this platform.
 * @param[out] _processors total number for processors in system.
 * @param[out] _numa_nodes total number for sockets in system
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 */
void parse_processor_info_macos(const std::vector<std::pair<std::string, uint64_t>>& system_info_table,
                                int& _processors,
                                int& _numa_nodes,
                                int& _sockets,
                                int& _cores,
                                std::vector<std::vector<int>>& _proc_type_table);
#endif

}  // namespace ov
