// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <sys/sysctl.h>

#include <memory>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

namespace ov {

CPU::CPU() {
    parse_processor_info_macos(_processors, _numa_nodes, _sockets, _cores, _proc_type_table);
    _org_proc_type_table = _proc_type_table;
}

int parse_processor_info_macos(int& _processors,
                               int& _numa_nodes,
                               int& _sockets,
                               int& _cores,
                               std::vector<std::vector<int>>& _proc_type_table) {
    uint64_t output = 0;
    size_t size = sizeof(output);

    _processors = 0;
    _numa_nodes = 0;
    _cores = 0;

    if (sysctlbyname("hw.ncpu", &output, &size, NULL, 0) < 0) {
        return -1;
    } else {
        _processors = static_cast<int>(output);
    }

    if (sysctlbyname("hw.physicalcpu", &output, &size, NULL, 0) < 0) {
        _processors = 0;
        return -1;
    } else {
        _cores = static_cast<int>(output);
    }

    _numa_nodes = 1;
    _sockets = 1;
    
    if (sysctlbyname("hw.optional.arm64", &output, &size, NULL, 0) < 0) {
        _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
        _proc_type_table[0][ALL_PROC] = _processors;
        _proc_type_table[0][MAIN_CORE_PROC] = _cores;
        _proc_type_table[0][HYPER_THREADING_PROC] = _processors - _cores;
        _proc_type_table[0][PROC_NUMA_NODE_ID] = 0;
        _proc_type_table[0][PROC_SOCKET_ID] = 0;
    } else {
        if (sysctlbyname("hw.perflevel0.physicalcpu", &output, &size, NULL, 0) < 0) {
            _processors = 0;
            _cores = 0;
            _numa_nodes = 0;
            _sockets = 0;
            return -1;
        } else {
            _proc_type_table.resize(1, std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
            _proc_type_table[0][ALL_PROC] = _processors;
            _proc_type_table[0][MAIN_CORE_PROC] = output;
            _proc_type_table[0][PROC_NUMA_NODE_ID] = 0;
            _proc_type_table[0][PROC_SOCKET_ID] = 0;
        }

        if (sysctlbyname("hw.perflevel1.physicalcpu", &output, &size, NULL, 0) < 0) {
            return 0;
        } else {
            _proc_type_table[0][EFFICIENT_CORE_PROC] = output;
        }
    }

    return 0;
}

}  // namespace ov
