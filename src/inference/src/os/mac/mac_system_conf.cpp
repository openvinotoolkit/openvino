// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <sys/sysctl.h>

#include <memory>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

namespace ov {

CPU::CPU() {
    uint64_t output = 0;
    size_t size = sizeof(output);
    std::vector<std::string> ctl_name_list = {"hw.ncpu",
                                              "hw.physicalcpu",
                                              "hw.optional.arm64",
                                              "hw.perflevel0.physicalcpu",
                                              "hw.perflevel1.physicalcpu"};
    std::vector<std::pair<std::string, uint64_t>> system_info_table;

    for (auto& row : ctl_name_list) {
        if (sysctlbyname(row.c_str(), &output, &size, NULL, 0) >= 0) {
            system_info_table.push_back(std::make_pair(row, output));
        }
    }

    parse_processor_info_macos(system_info_table, _processors, _numa_nodes, _sockets, _cores, _proc_type_table);
    _org_proc_type_table = _proc_type_table;

    cpu_debug();
}

void parse_processor_info_macos(const std::vector<std::pair<std::string, uint64_t>>& system_info_table,
                                int& _processors,
                                int& _numa_nodes,
                                int& _sockets,
                                int& _cores,
                                std::vector<std::vector<int>>& _proc_type_table) {
    _processors = 0;
    _numa_nodes = 0;
    _sockets = 0;
    _cores = 0;

    auto it = std::find_if(system_info_table.begin(),
                           system_info_table.end(),
                           [&](const std::pair<std::string, uint64_t>& item) {
                               return item.first == "hw.ncpu";
                           });

    if (it == system_info_table.end()) {
        OPENVINO_THROW("Unable to get number of cpus from macOS!");
    } else {
        _processors = static_cast<int>(it->second);
    }

    it = std::find_if(system_info_table.begin(),
                      system_info_table.end(),
                      [&](const std::pair<std::string, uint64_t>& item) {
                          return item.first == "hw.physicalcpu";
                      });

    if (it == system_info_table.end()) {
        _cores = _processors;
    } else {
        _cores = static_cast<int>(it->second);
    }

    _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));

    _numa_nodes = 1;
    _sockets = 1;

    _proc_type_table[0][ALL_PROC] = _processors;
    _proc_type_table[0][MAIN_CORE_PROC] = _cores;
    _proc_type_table[0][HYPER_THREADING_PROC] = _processors - _cores;

    it = std::find_if(system_info_table.begin(),
                      system_info_table.end(),
                      [&](const std::pair<std::string, uint64_t>& item) {
                          return item.first == "hw.optional.arm64";
                      });

    if (it != system_info_table.end()) {
        it = std::find_if(system_info_table.begin(),
                          system_info_table.end(),
                          [&](const std::pair<std::string, uint64_t>& item) {
                              return item.first == "hw.perflevel0.physicalcpu";
                          });

        if (it != system_info_table.end()) {
            _proc_type_table[0][MAIN_CORE_PROC] = it->second;

            it = std::find_if(system_info_table.begin(),
                              system_info_table.end(),
                              [&](const std::pair<std::string, uint64_t>& item) {
                                  return item.first == "hw.perflevel1.physicalcpu";
                              });

            if (it != system_info_table.end()) {
                _proc_type_table[0][EFFICIENT_CORE_PROC] = it->second;
            }
        } else {
            _proc_type_table[0][EFFICIENT_CORE_PROC] = _cores / 2;
            _proc_type_table[0][MAIN_CORE_PROC] = _cores - _proc_type_table[0][EFFICIENT_CORE_PROC];
        }
    }
}

}  // namespace ov
