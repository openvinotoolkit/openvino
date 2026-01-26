// Copyright (C) 2018-2026 Intel Corporation
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
    std::vector<int> numa_list;
    std::unordered_set<int> l3_set;

    const std::vector<int> proc_init_line({0, 0, 0, 0, 0, 0, 0});
    const std::vector<int> cpu_init_line(CPU_MAP_TABLE_SIZE, -1);

    constexpr int initial_core_type = -1;
    constexpr int initial_numa_mask = -1;
    constexpr int group_with_1_core = 1;
    constexpr int group_with_2_cores = 2;
    constexpr int group_with_4_cores = 4;

    char* info_ptr = (char*)base_ptr;
    int list_len = 0;
    int base_proc = 0;
    int base_proc_socket = 0;
    int group = 0;
    int proc_group = 0;
    int group_start = 0;
    int group_end = 0;
    int group_id = 0;
    int group_type = initial_core_type;
    int num_package = 0;
    int cur_numa_mask = initial_numa_mask;
    int max_group_cnt = 0;

    _processors = 0;
    _sockets = 0;
    _numa_nodes = 0;
    _cores = 0;
    _blocked_cores = 0;

    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = nullptr;

    auto MaskToList = [&](const KAFFINITY mask_input) {
        list.clear();
        list_len = 0;
        for (int cnt = 0; mask_input >> cnt; ++cnt) {
            if ((mask_input >> cnt) & 0x1) {
                list.push_back(cnt);
                ++list_len;
                if ((mask_input >> cnt) == 0x1) {
                    break;
                }
            }
        }
    };

    auto add_new_numa_node = [&](const int cur_proc) {
        _proc_type_table.push_back(proc_init_line);
        ++_numa_nodes;
        int next_idx = cur_proc + 1;
        if (next_idx < static_cast<int>(_cpu_mapping_table.size())) {
            if (group != static_cast<int>(numa_list.size()) - 1 &&
                numa_list[group + 1] == _cpu_mapping_table[next_idx][CPU_MAP_NUMA_NODE_ID]) {
                ++group;
            }
            _proc_type_table[_numa_nodes][PROC_NUMA_NODE_ID] = _numa_nodes;
            _proc_type_table[_numa_nodes][PROC_SOCKET_ID] = _cpu_mapping_table[next_idx][CPU_MAP_SOCKET_ID];
        }
    };

    for (; info_ptr < base_ptr + len; info_ptr += (DWORD)info->Size) {
        info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_ptr;
        // Use references for repeated vector access
        if (info->Relationship == RelationProcessorPackage) {
            MaskToList(info->Processor.GroupMask->Mask);
            if (num_package > 0) {
                ++_sockets;
                cur_numa_mask = initial_numa_mask;
                if (_processors < 64) {
                    l3_set.clear();
                } else {
                    base_proc_socket = _processors;
                }
            }
            ++num_package;
        } else if (info->Relationship == RelationProcessorCore) {
            MaskToList(info->Processor.GroupMask->Mask);
            if (list.empty())
                continue;
            if (list[0] == 0 || proc_group != info->Processor.GroupMask->Group) {
                base_proc = _processors - base_proc_socket - list[0];
                proc_group = info->Processor.GroupMask->Group;
            }
            if (list_len == group_with_2_cores) {
                std::vector<int> proc_info(cpu_init_line);
                proc_info[CPU_MAP_PROCESSOR_ID] = list[0] + base_proc_socket + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = info->Processor.GroupMask->Group;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                proc_info[CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                proc_info[CPU_MAP_GROUP_ID] = group;
                _cpu_mapping_table.push_back(proc_info);

                proc_info = cpu_init_line;
                proc_info[CPU_MAP_PROCESSOR_ID] = list[1] + base_proc_socket + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = info->Processor.GroupMask->Group;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                proc_info[CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                proc_info[CPU_MAP_GROUP_ID] = group;
                _cpu_mapping_table.push_back(proc_info);
                ++group;
            } else {
                std::vector<int> proc_info(cpu_init_line);
                proc_info[CPU_MAP_PROCESSOR_ID] = list[0] + base_proc_socket + base_proc;
                proc_info[CPU_MAP_NUMA_NODE_ID] = info->Processor.GroupMask->Group;
                proc_info[CPU_MAP_SOCKET_ID] = _sockets;
                proc_info[CPU_MAP_CORE_ID] = _cores;
                if ((_processors > group_start) && (_processors <= group_end)) {
                    proc_info[CPU_MAP_CORE_TYPE] = group_type;
                    proc_info[CPU_MAP_GROUP_ID] =
                        (group_type == MAIN_CORE_PROC && group_end - group_start != 1) ? group++ : group_id;
                    if (group_id == CPU_BLOCKED) {
                        proc_info[CPU_MAP_USED_FLAG] = CPU_BLOCKED;
                        ++_blocked_cores;
                    }
                }
                _cpu_mapping_table.push_back(proc_info);
            }
            _processors += list_len;
            ++_cores;
        } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 2)) {
            MaskToList(info->Cache.GroupMask.Mask);
            if (list_len == group_with_1_core) {
                int idx = list[0] + base_proc_socket + base_proc;
                if (_cpu_mapping_table.size() > static_cast<size_t>(idx) &&
                    _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] == initial_core_type) {
                    _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    _cpu_mapping_table[idx][CPU_MAP_GROUP_ID] = group++;
                }
            } else if (_cpu_mapping_table[list[0] + base_proc_socket + base_proc][CPU_MAP_CORE_TYPE] ==
                       initial_core_type) {
                if (l3_set.empty() || l3_set.count(list[0]) ||
                    list[0] + base_proc_socket + base_proc > static_cast<int>(l3_set.size())) {
                    if (_processors <= list[list_len - 1] + base_proc_socket + base_proc) {
                        group_start = list[0] + base_proc_socket + base_proc;
                        group_end = list[list_len - 1] + base_proc_socket + base_proc;
                        group_type = (list_len == group_with_4_cores) ? EFFICIENT_CORE_PROC : MAIN_CORE_PROC;
                    } else {
                        group_type = (group_type == initial_core_type) ? MAIN_CORE_PROC : group_type;
                    }
                    group_id = group++;
                    for (int m = 0; m < _processors - base_proc_socket - base_proc - list[0]; ++m) {
                        int idx = list[m] + base_proc_socket + base_proc;
                        if (_cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] == initial_core_type) {
                            _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = group_type;
                            _cpu_mapping_table[idx][CPU_MAP_GROUP_ID] = group_id;
                        }
                    }
                } else {
                    if (_processors <= list[list_len - 1] + base_proc_socket + base_proc) {
                        group_start = list[0];
                        group_end = list[list_len - 1];
                        group_id = (list_len == group_with_2_cores) ? CPU_BLOCKED : group++;
                        group_type = LP_EFFICIENT_CORE_PROC;
                    }
                    for (int m = 0; m < _processors - base_proc_socket - base_proc - list[0]; ++m) {
                        int idx = list[m] + base_proc_socket + base_proc;
                        _cpu_mapping_table[idx][CPU_MAP_CORE_TYPE] = group_type;
                        _cpu_mapping_table[idx][CPU_MAP_GROUP_ID] = group_id;
                        if (group_id == CPU_BLOCKED) {
                            _cpu_mapping_table[idx][CPU_MAP_USED_FLAG] = CPU_BLOCKED;
                            ++_blocked_cores;
                        } else {
                            _cpu_mapping_table[idx][CPU_MAP_USED_FLAG] = NOT_USED;
                        }
                    }
                }
            }
        } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 3)) {
            MaskToList(info->Cache.GroupMask.Mask);
            l3_set.insert(list.begin(), list.end());
        } else if (info->Relationship == RelationNumaNode) {
            numa_list.push_back(info->NumaNode.GroupMask.Group);
        } else if (info->Relationship == RelationGroup) {
            max_group_cnt = info->Group.MaximumGroupCount;
        }
    }

    _proc_type_table.push_back(proc_init_line);
    group_id = 0;
    group = _cpu_mapping_table[0][CPU_MAP_NUMA_NODE_ID];

    for (int n = 0; n < _processors; n++) {
        _cpu_mapping_table[n][CPU_MAP_NUMA_NODE_ID] = _numa_nodes;
        if (_cpu_mapping_table[n][CPU_MAP_USED_FLAG] == NOT_USED) {
            _proc_type_table[_numa_nodes][_cpu_mapping_table[n][CPU_MAP_CORE_TYPE]]++;
            _proc_type_table[_numa_nodes][ALL_PROC]++;
        }
        if (n != _processors - 1) {
            if (_cpu_mapping_table[n][CPU_MAP_SOCKET_ID] != _cpu_mapping_table[n + 1][CPU_MAP_SOCKET_ID]) {
                add_new_numa_node(n);
                continue;
            }
            if (_cpu_mapping_table[_processors - 1][CPU_MAP_NUMA_NODE_ID] <= max_group_cnt &&
                group < numa_list.size() - 1 &&
                _cpu_mapping_table[n + 1][CPU_MAP_NUMA_NODE_ID] == numa_list[group + 1] &&
                _cpu_mapping_table[n + 1][CPU_MAP_NUMA_NODE_ID] != numa_list[group]) {
                add_new_numa_node(n);
                continue;
            }
        }
    }
    _processors -= _blocked_cores;
    _cores -= _blocked_cores;

    if (_proc_type_table.size() > 1) {
        _proc_type_table.emplace(_proc_type_table.begin(), proc_init_line);

        for (int m = 1; m <= _proc_type_table.size() - 1; m++) {
            for (int n = 0; n <= HYPER_THREADING_PROC; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
            _proc_type_table[0][PROC_SOCKET_ID] =
                _proc_type_table[0][PROC_SOCKET_ID] == _proc_type_table[m][PROC_SOCKET_ID]
                    ? _proc_type_table[m][PROC_SOCKET_ID]
                    : -1;
            _proc_type_table[0][PROC_NUMA_NODE_ID] =
                _proc_type_table[0][PROC_NUMA_NODE_ID] == _proc_type_table[m][PROC_NUMA_NODE_ID]
                    ? _proc_type_table[m][PROC_NUMA_NODE_ID]
                    : -1;
        }
    } else {
        _proc_type_table[0][PROC_SOCKET_ID] = 0;
        _proc_type_table[0][PROC_NUMA_NODE_ID] = 0;
    }
    ++_sockets;
    ++_numa_nodes;
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

#if OV_THREAD_USE_TBB
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        phys_cores = custom::info::default_concurrency(
            custom::task_arena::constraints{}.set_core_type(core_types.back()).set_max_threads_per_core(1));
    }
#endif
    return phys_cores;
}

#if !OV_THREAD_USE_TBB
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> get_available_numa_nodes() {
    return {-1};
}
#endif

}  // namespace ov
