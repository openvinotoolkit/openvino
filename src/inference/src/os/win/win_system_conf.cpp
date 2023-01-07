// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <windows.h>

#include <iostream>
#include <memory>
#include <vector>

#include "ie_system_conf.h"
#include "threading/ie_parallel_custom_arena.hpp"

namespace InferenceEngine {

struct CPU {
    int _processors = 0;
    int _sockets = 0;
    int _cores = 0;

    std::vector<int> _proc_type_table;
    std::vector<std::vector<int>> _cpu_mapping_table;

    CPU() {
        DWORD len = 0;
        if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &len) ||
            GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return;
        }

        char* base_ptr = new char[len];
        if (!GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)base_ptr, &len)) {
            return;
        }

        _proc_type_table.resize(EFFICIENT_CORE_PROC + 1, 0);
        _processors = GetMaximumProcessorCount(ALL_PROCESSOR_GROUPS);
        _cpu_mapping_table.resize(_processors, std::vector<int>(CPU_MAP_USED_FLAG + 1, -1));

        std::vector<int> list;

        char* info_ptr = base_ptr;
        int list_len = 0;
        int base_proc = 0;
        int mask_len = 0;
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

        for (; info_ptr < base_ptr + len; info_ptr += (DWORD)info->Size) {
            info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_ptr;

            if (info->Relationship == RelationProcessorPackage) {
                _sockets++;
                MaskToList(info->Processor.GroupMask->Mask);
                mask_len = list_len;

            } else if (info->Relationship == RelationProcessorCore) {
                MaskToList(info->Processor.GroupMask->Mask);

                if (_proc_type_table[ALL_PROC] >= _processors) {
                    break;
                }

                if (0 == list[0]) {
                    base_proc = _proc_type_table[ALL_PROC];
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

                    _proc_type_table[MAIN_CORE_PROC]++;
                    _proc_type_table[HYPER_THREADING_PROC]++;
                    group++;

                } else {
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_PROCESSOR_ID] = list[0] + base_proc;
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_SOCKET_ID] = _sockets;
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_ID] = _cores;
                }
                _proc_type_table[ALL_PROC] += list_len;
                _cores++;

            } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 2)) {
                MaskToList(info->Cache.GroupMask.Mask);

                if (4 == list_len) {
                    for (int m = 0; m < list_len; m++) {
                        _cpu_mapping_table[list[m] + base_proc][CPU_MAP_CORE_TYPE] = EFFICIENT_CORE_PROC;
                        _cpu_mapping_table[list[m] + base_proc][CPU_MAP_GROUP_ID] = group;
                        _proc_type_table[EFFICIENT_CORE_PROC]++;
                    }
                    group++;

                } else if (1 == list_len) {
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
                    _cpu_mapping_table[list[0] + base_proc][CPU_MAP_GROUP_ID] = group;
                    _proc_type_table[MAIN_CORE_PROC]++;
                    group++;
                }
            }
        }
        delete[] base_ptr;
    }
};
static CPU cpu;

int getNumberOfCPUCores(bool bigCoresOnly) {
    const int fallback_val = parallel_get_max_threads();
    DWORD sz = 0;
    // querying the size of the resulting structure, passing the nullptr for the buffer
    if (GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &sz) ||
        GetLastError() != ERROR_INSUFFICIENT_BUFFER)
        return fallback_val;

    std::unique_ptr<uint8_t[]> ptr(new uint8_t[sz]);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get()), &sz))
        return fallback_val;

    int phys_cores = 0;
    size_t offset = 0;
    do {
        offset += reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(ptr.get() + offset)->Size;
        phys_cores++;
    } while (offset < sz);

    #if (IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        phys_cores = custom::info::default_concurrency(custom::task_arena::constraints{}
                                                               .set_core_type(core_types.back())
                                                               .set_max_threads_per_core(1));
    }
    #endif
    return phys_cores;
}

#if !(IE_THREAD == IE_THREAD_TBB || IE_THREAD == IE_THREAD_TBB_AUTO)
// OMP/SEQ threading on the Windows doesn't support NUMA
std::vector<int> getAvailableNUMANodes() { return {-1}; }
#endif

}  // namespace InferenceEngine
