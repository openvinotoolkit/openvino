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

namespace ov {

struct CPU {
    int _processors = 0;
    int _sockets = 0;
    int _cores = 0;
    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;
    std::mutex _cpu_mutex;
    std::mutex _task_mutex;
    int _plugin_status = PLUGIN_USED_START;
    int _socket_idx = 0;

    CPU() {
        init_cpu(*this);
    }
    void init_cpu(CPU& cpu);
};

#ifdef __linux__
/**
 * @brief      Parse processors infomation on Linux
 * @param[in]  _processors total number for processors in system.
 * @param[in]  _system_info_table system information for this platform.
 * @param[out] _sockets total number for sockets in system
 * @param[out] _cores total number for physical CPU cores in system
 * @param[out] _proc_type_table summary table of number of processors per type
 * @param[out] _cpu_mapping_table CPU mapping table for each processor
 * @return
 */
void parse_processor_info_linux(const int _processors,
                                const std::vector<std::vector<std::string>> _system_info_table,
                                int& _sockets,
                                int& _cores,
                                std::vector<std::vector<int>>& _proc_type_table,
                                std::vector<std::vector<int>>& _cpu_mapping_table);
#endif

#if (defined(_WIN32) || defined(_WIN64))
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

}  // namespace ov
