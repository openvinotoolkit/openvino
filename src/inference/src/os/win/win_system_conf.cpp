// Copyright (C) 2018-2023 Intel Corporation
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
#include "streams_executor.hpp"

namespace ov {

struct CPU {
    int _processors = 0;
    int _sockets = 0;
    int _cores = 0;

    std::vector<std::vector<int>> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;

    CPU() {
        DWORD len = 0;
        if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &len) ||
            GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return;
        }

        std::shared_ptr<char> base_shared_ptr(new char[len]);
        char* base_ptr = base_shared_ptr.get();
        if (!GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)base_ptr, &len)) {
            return;
        }

        _processors = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);

        parse_processor_info_win(base_ptr, len, _processors, _sockets, _cores, _proc_type_table, _cpu_mapping_table);
    }
};
static CPU cpu;

void parse_processor_info_win(const char* base_ptr,
                              const unsigned long len,
                              const int _processors,
                              int& _sockets,
                              int& _cores,
                              std::vector<std::vector<int>>& _proc_type_table,
                              std::vector<std::vector<int>>& _cpu_mapping_table) {
    _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));

    std::vector<int> list;

    char* info_ptr = (char*)base_ptr;
    int list_len = 0;
    int base_proc = 0;
    int proc_count = 0;
    int group = 0;
    _sockets = -1;

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

    std::vector<int> line_value_0(PROC_TYPE_TABLE_SIZE, 0);

    for (; info_ptr < base_ptr + len; info_ptr += (DWORD)info->Size) {
        info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_ptr;

        if (info->Relationship == RelationProcessorPackage) {
            _sockets++;
            MaskToList(info->Processor.GroupMask->Mask);
            if (0 == _sockets) {
                _proc_type_table.push_back(line_value_0);
            } else {
                _proc_type_table.push_back(_proc_type_table[0]);
                _proc_type_table[0] = line_value_0;
            }
        } else if (info->Relationship == RelationProcessorCore) {
            MaskToList(info->Processor.GroupMask->Mask);

            if (proc_count >= _processors) {
                break;
            }

            if (0 == list[0]) {
                base_proc = proc_count;
            }

            if (2 == list_len) {
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_PROCESSOR_ID] = list[0] + base_proc;
                _cpu_mapping_table[list[1] + base_proc][CPU_MAP_PROCESSOR_ID] = list[1] + base_proc;

                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_SOCKET_ID] = _sockets;
                _cpu_mapping_table[list[1] + base_proc][CPU_MAP_SOCKET_ID] = _sockets;

                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_ID] = _cores;
                _cpu_mapping_table[list[1] + base_proc][CPU_MAP_CORE_ID] = _cores;

                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE] = HYPER_THREADING_PROC;
                _cpu_mapping_table[list[1] + base_proc][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;

                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_GROUP_ID] = group;
                _cpu_mapping_table[list[1] + base_proc][CPU_MAP_GROUP_ID] = group;

                _proc_type_table[0][MAIN_CORE_PROC]++;
                _proc_type_table[0][HYPER_THREADING_PROC]++;
                group++;

            } else {
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_PROCESSOR_ID] = list[0] + base_proc;
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_SOCKET_ID] = _sockets;
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_ID] = _cores;
            }
            _proc_type_table[0][ALL_PROC] += list_len;
            proc_count += list_len;
            _cores++;

        } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 2)) {
            MaskToList(info->Cache.GroupMask.Mask);

            if (4 == list_len) {
                for (int m = 0; m < list_len; m++) {
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                    _cpu_mapping_table[list[m] + base_proc][CPU_MAP_GROUP_ID] = group;
                    _proc_type_table[0][EFFICIENT_CORE_PROC]++;
                }
                group++;

            } else if (1 == list_len) {
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                _cpu_mapping_table[list[0] + base_proc][CPU_MAP_GROUP_ID] = group;
                _proc_type_table[0][MAIN_CORE_PROC]++;
                group++;
            }
        }
    }
    _sockets++;
    if (_sockets > 1) {
        _proc_type_table.push_back(_proc_type_table[0]);
        _proc_type_table[0] = line_value_0;

        for (int m = 1; m <= _sockets; m++) {
            for (int n = 0; n < PROC_TYPE_TABLE_SIZE; n++) {
                _proc_type_table[0][n] += _proc_type_table[m][n];
            }
        }
    }
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
