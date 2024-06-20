// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>

#include <memory>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/runtime/system_conf.hpp"
#include "os/cpu_map_info.hpp"

namespace ov {

CPU::CPU() {
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &len) || GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
        return;
    }

    std::unique_ptr<char[]> base_shared_ptr(new char[len]);
    char* base_ptr = base_shared_ptr.get();
    if (!GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)base_ptr, &len)) {
        return;
    }

    parse_processor_info_win(base_ptr,
                             len,
                             _processors,
                             _numa_nodes,
                             _sockets,
                             _cores,
                             _blocked_cores,
                             _proc_type_table,
                             _cpu_mapping_table);
    _org_proc_type_table = _proc_type_table;

    // ensure that get_org_numa_id and get_org_socket_id can return the correct value
    for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
        auto numa_id = _cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID];
        auto socket_id = _cpu_mapping_table[i][CPU_MAP_SOCKET_ID];
        if (_numaid_mapping_table.find(numa_id) == _numaid_mapping_table.end()) {
            _numaid_mapping_table.insert({numa_id, numa_id});
        }
        if (_socketid_mapping_table.find(socket_id) == _socketid_mapping_table.end()) {
            _socketid_mapping_table.insert({socket_id, socket_id});
        }
    }

    cpu_debug();
}

void parse_processor_info_win(const char* base_ptr,
                              const unsigned long len,
                              int& _processors,
                              int& _numa_nodes,
                              int& _sockets,
                              int& _cores,
                              int& _blocked_cores,
                              std::vector<std::vector<int>>& _proc_type_table,
                              std::vector<std::vector<int>>& _cpu_mapping_table) {
    std::vector<int> list;
    std::vector<int> proc_info;

    std::vector<int> proc_init_line({0, 0, 0, 0, -1, -1});
    std::vector<int> cpu_init_line(CPU_MAP_TABLE_SIZE, -1);

    char* info_ptr = (char*)base_ptr;
    int list_len = 0;
    int base_proc = 0;
    int group = 0;

    int group_start = 0;
    int group_end = 0;
    int group_id = 0;
    int group_type = 0;

    int num_package = 0;

    _processors = 0;
    _sockets = 0;
    _cores = 0;
    _blocked_cores = 0;

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = NULL;

    auto MaskToList = [&](const KAFFINITY mask_input) {
        KAFFINITY mask = mask_input;
        int cnt = 0;

        list.clear();
        list_len = 0;
        while (mask != 0) {
            if (0x1 == (mask & 0x1)) {
                list.push_back(cnt);
                list_len++;
            }
            cnt++;
            mask >>= 1;
        }
        return;
    };

    _proc_type_table.push_back(proc_init_line);

    for (; info_ptr < base_ptr + len; info_ptr += (DWORD)info->Size) {
        info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_ptr;

        if (info->Relationship == RelationProcessorPackage) {
            MaskToList(info->Processor.GroupMask->Mask);
            if (num_package > 0) {
                _sockets++;
                _proc_type_table.push_back(_proc_type_table[0]);
                _proc_type_table[0] = proc_init_line;
            }
            num_package++;
        } else if (info->Relationship == RelationProcessorCore) {
            MaskToList(info->Processor.GroupMask->Mask);

            if (0 == list[0]) {
                base_proc = _processors;
            }

            if (2 == list_len) {
                proc_info = cpu_init_line;
                proc_info[CPU_MAP_PROCESSOR_ID] = list[0] + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = _sockets;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                proc_info[CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                proc_info[CPU_MAP_GROUP_ID] = group;
                _cpu_mapping_table.push_back(proc_info);

                proc_info = cpu_init_line;
                proc_info[CPU_MAP_PROCESSOR_ID] = list[1] + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = _sockets;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                proc_info[CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                proc_info[CPU_MAP_GROUP_ID] = group;
                _cpu_mapping_table.push_back(proc_info);

                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;
                group++;

            } else {
                proc_info = cpu_init_line;
                proc_info[CPU_MAP_PROCESSOR_ID] = list[0] + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = _sockets;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                if ((_processors > group_start) && (_processors <= group_end)) {
                    proc_info[CPU_MAP_CORE_TYPE] = group_type;
                    proc_info[CPU_MAP_GROUP_ID] = group_id;
                    if (group_id == CPU_BLOCKED) {
                        proc_info[CPU_MAP_USED_FLAG] = CPU_BLOCKED;
                        _blocked_cores++;
                    } else {
                        _proc_type_table[0][group_type]++;
                    }
                }
                _cpu_mapping_table.push_back(proc_info);
            }
            _proc_type_table[0][ALL_PROC] += list_len;
            _processors += list_len;
            _cores++;
        } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 2)) {
            MaskToList(info->Cache.GroupMask.Mask);

            if (4 == list_len) {
                if (_processors <= list[list_len - 1] + base_proc) {
                    group_start = list[0];
                    group_end = list[list_len - 1];
                    group_id = group++;
                    group_type = EFFICIENT_CORE_PROC;
                }
                for (int m = 0; m < _processors - list[0]; m++) {
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_GROUP_ID] = group_id;
                    _proc_type_table[0][EFFICIENT_CORE_PROC]++;
                }
            } else if ((2 == list_len) && (-1 == _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE])) {
                if (_processors <= list[list_len - 1] + base_proc) {
                    group_start = list[0];
                    group_end = list[list_len - 1];
                    if (_proc_type_table[0][EFFICIENT_CORE_PROC] > 0) {
                        group_id = CPU_BLOCKED;
                        group_type = EFFICIENT_CORE_PROC;
                        _blocked_cores++;
                    } else {
                        group_id = group++;
                        group_type = MAIN_CORE_PROC;
                    }
                }
                for (int m = 0; m < _processors - list[0]; m++) {
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_CORE_TYPE] = group_type;
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_GROUP_ID] = group_id;
                    if (group_id == CPU_BLOCKED) {
                        _cpu_mapping_table[list[m] + base_proc][CPU_MAP_USED_FLAG] = CPU_BLOCKED;
                    } else {
                        _cpu_mapping_table[list[m] + base_proc][CPU_MAP_USED_FLAG] = NOT_USED;
                        _proc_type_table[0][MAIN_CORE_PROC]++;
                    }
                }

            } else if (1 == list_len) {
                if ((_cpu_mapping_table.size() > list[0]) &&
                    (_cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE] == -1)) {
                    group_id = group++;
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_GROUP_ID] = group_id;
                    _proc_type_table[0][MAIN_CORE_PROC]++;
                }
            }
        }
    }
    _sockets++;
    _processors -= _blocked_cores;
    _cores -= _blocked_cores;
    _proc_type_table[0][ALL_PROC] -= _blocked_cores;
    if (_sockets > 1) {
        _proc_type_table.push_back(_proc_type_table[0]);
        _proc_type_table[0] = proc_init_line;

        for (int m = 1; m <= _sockets; m++) {
            for (int n = 0; n <= HYPER_THREADING_PROC; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
            _proc_type_table[m][PROC_SOCKET_ID] = m - 1;
            _proc_type_table[m][PROC_NUMA_NODE_ID] = m - 1;
        }
    } else {
        _proc_type_table[0][PROC_SOCKET_ID] = 0;
        _proc_type_table[0][PROC_NUMA_NODE_ID] = 0;
    }
    _numa_nodes = _sockets;
}

int get_number_of_cpu_cores(bool bigCoresOnly) {
    const int fallback_val = parallel_get_max_threads();
    DWORD sz = 0;
    // querying the size of the resulting structure, passing the nullptr for the buffer
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &sz) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return fallback_val;

    std::unique_ptr<uint8_t[]> ptr(new uint8_t[sz]);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
                                          reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get()),
                                          &sz))
        return fallback_val;

    int phys_cores = 0;
    size_t offset = 0;
    do {
        offset += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get() + offset)->Size;
        phys_cores++;
    } while (offset < sz);

#if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        phys_cores = custom::info::default_concurrency(
            custom::task_arena::constraints{}.set_core_type(core_types.back()).set_max_threads_per_core(1));
    }
#endif
    return phys_cores;
}

#if !(OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> get_available_numa_nodes() {
    return {-1};
}
#endif

}  // namespace ov
