// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sched.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "ie_common.h"
#include "ie_system_conf.h"
#include "streams_executor.hpp"
#include "threading/ie_parallel_custom_arena.hpp"

namespace ov {

void parse_processor_info_linux(const int _processors,
                                const std::vector<std::vector<std::string>> system_info_table,
                                int& _sockets,
                                int& _cores,
                                std::vector<std::vector<int>>& _proc_type_table,
                                std::vector<std::vector<int>>& _cpu_mapping_table) {
    int n_group = 0;

    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    auto UpdateProcMapping = [&](const int nproc) {
        if (-1 == _cpu_mapping_table[nproc][CPU_MAP_CORE_ID]) {
            int core_1 = 0;
            int core_2 = 0;
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str = "";

            if (((endpos = system_info_table[nproc][0].find(',', pos)) != std::string::npos) ||
                ((endpos = system_info_table[nproc][0].find('-', pos)) != std::string::npos)) {
                sub_str = system_info_table[nproc][0].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][0].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_2][CPU_MAP_PROCESSOR_ID] = core_2;

                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_ID] = _cores;

                /**
                 * Processor 0 need to handle system interception on Linux. So use second processor as physical core
                 * and first processor as logic core
                 */
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[core_2][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;

                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;
                _cpu_mapping_table[core_2][CPU_MAP_GROUP_ID] = n_group;

                _cores++;
                n_group++;

                _proc_type_table[0][ALL_PROC] += 2;
                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;

            } else if ((endpos = system_info_table[nproc][1].find('-', pos)) != std::string::npos) {
                sub_str = system_info_table[nproc][1].substr(pos, endpos);
                core_1 = std::stoi(sub_str);
                sub_str = system_info_table[nproc][1].substr(endpos + 1);
                core_2 = std::stoi(sub_str);

                for (int m = core_1; m <= core_2; m++) {
                    _cpu_mapping_table[m][CPU_MAP_PROCESSOR_ID] = m;
                    _cpu_mapping_table[m][CPU_MAP_CORE_ID] = _cores;
                    _cpu_mapping_table[m][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    _cpu_mapping_table[m][CPU_MAP_GROUP_ID] = n_group;

                    _cores++;

                    _proc_type_table[0][ALL_PROC]++;
                    _proc_type_table[0][EFFICIENT_CORE_PROC]++;
                }

                n_group++;

            } else {
                core_1 = std::stoi(system_info_table[nproc][0]);

                _cpu_mapping_table[core_1][CPU_MAP_PROCESSOR_ID] = core_1;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[core_1][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[core_1][CPU_MAP_GROUP_ID] = n_group;

                _cores++;
                n_group++;

                _proc_type_table[0][ALL_PROC]++;
                _proc_type_table[0][MAIN_CORE_PROC]++;
            }
        }
        return;
    };

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (int n = 0; n < _processors; n++) {
        if (-1 == _cpu_mapping_table[n][CPU_MAP_SOCKET_ID]) {
            std::string::size_type pos = 0;
            std::string::size_type endpos = 0;
            std::string sub_str;

            int core_1;
            int core_2;

            if (0 == _sockets) {
                _proc_type_table.push_back(line_value_0);
            } else {
                _proc_type_table.push_back(_proc_type_table[0]);
                _proc_type_table[0] = line_value_0;
            }

            while (1) {
                if ((endpos = system_info_table[n][2].find('-', pos)) != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos, endpos);
                    core_1 = std::stoi(sub_str);
                    sub_str = system_info_table[n][2].substr(endpos + 1);
                    core_2 = std::stoi(sub_str);

                    for (int m = core_1; m <= core_2; m++) {
                        _cpu_mapping_table[m][CPU_MAP_SOCKET_ID] = _sockets;
                        UpdateProcMapping(m);
                    }

                } else if (pos != std::string::npos) {
                    sub_str = system_info_table[n][2].substr(pos);
                    core_1 = std::stoi(sub_str);
                    _cpu_mapping_table[core_1][CPU_MAP_SOCKET_ID] = _sockets;
                    UpdateProcMapping(core_1);
                    endpos = pos;
                }

                if ((pos = system_info_table[n][2].find(',', endpos)) != std::string::npos) {
                    pos++;
                } else {
                    break;
                }
            }
            _sockets++;
        }
    }
    if (_sockets > 1) {
        _proc_type_table.push_back(_proc_type_table[0]);
        _proc_type_table[0] = line_value_0;

        for (int m = 1; m <= _sockets; m++) {
            for (int n = 0; n < PROC_TYPE_TABLE_SIZE; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
        }
    }
};

}  // namespace InferenceEngine
