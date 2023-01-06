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
    
    int _node = 0;
    int _p_cores = 0;
    int _e_cores = 0;
    int _phy_cores = 0;
    int _proc = 0;
 

    CPU() {
        /**
         * New method to get CPU infomation and CPU map. Below is the structure of CPU map and two sample.
         *  1. Four processors of two Pcore
         *  2. Four processors of four Ecores shared L2 cache
         *
         *  Proc ID : Socket ID | HW Core ID | Phy Core of Pcores | Logic Core of Pcores | ID of Ecore Group | Used
         *  (index)
         *     0         1            1                0                      1                   0             0
         *     1         1            1                1                      0                   0             0
         *     2         1            2                0                      2                   0             0
         *     3         1            2                2                      0                   0             0
         *     4         1            3                0                      0                   1             0
         *     5         1            4                0                      0                   1             0
         *     6         1            5                0                      0                   1             0
         *     7         1            6                0                      0                   1             0
         */
        _cpu_mapping.resize(GetActiveProcessorCount(ALL_PROCESSOR_GROUPS), std::vector<int>(CPU_MAP_USED_PROC + 1, 0));

        DWORD len = 0;
        if (GetLogicalProcessorInformationEx(RelationAll, nullptr, &len) ||
            GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
            return;
        }

        char* base_ptr = new char[len];
        if (!GetLogicalProcessorInformationEx(RelationAll, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)base_ptr, &len)) {
            return;
        }

        char* info_ptr = base_ptr;
        int list[64] = {0};
        int list_len = 0;
        int ecore_group = 0;
        int base_proc = 0;
        int mask_len = 0;

        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX info = NULL;

        auto MaskToList = [&](const KAFFINITY mask_input) {
            KAFFINITY mask = mask_input;
            KAFFINITY i = 1;

            list_len = 0;

            while (mask != 0) {
                list[list_len] = int(log2(mask));
                mask = mask - (i << list[list_len]);
                list_len++;
            }

            return;
        };

        for (; info_ptr < base_ptr + len; info_ptr += (DWORD)info->Size) {
            info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)info_ptr;

            if (info->Relationship == RelationProcessorPackage) {
                base_proc = _proc;
                _node++;

                MaskToList(info->Processor.GroupMask->Mask);

                mask_len = list_len;

            } else if (info->Relationship == RelationProcessorCore) {
                _phy_cores++;

                MaskToList(info->Processor.GroupMask->Mask);

                if (2 == list_len) {
                    _p_cores++;

                    _cpu_mapping[list[0] + base_proc][CPU_MAP_SOCKET] = _node;
                    _cpu_mapping[list[1] + base_proc][CPU_MAP_SOCKET] = _node;

                    _cpu_mapping[list[0] + base_proc][CPU_MAP_CORE] = _phy_cores;
                    _cpu_mapping[list[1] + base_proc][CPU_MAP_CORE] = _phy_cores;

                    _cpu_mapping[list[0] + base_proc][CPU_MAP_PHY_CORE] = _p_cores;
                    _cpu_mapping[list[1] + base_proc][CPU_MAP_LOG_CORE] = _p_cores;

                } else {
                    _cpu_mapping[list[0] + base_proc][CPU_MAP_SOCKET] = _node;

                    _cpu_mapping[list[0] + base_proc][CPU_MAP_CORE] = _phy_cores;
                }

                _proc += list_len;

                if (list[0] + 1 == mask_len) {
                    base_proc = _proc;
                }

            } else if ((info->Relationship == RelationCache) && (info->Cache.Level == 2)) {
                MaskToList(info->Cache.GroupMask.Mask);

                if (4 == list_len) {
                    ecore_group++;
                    for (int m = 0; m < list_len; m++) {
                        _e_cores++;
                        _cpu_mapping[list[m] + base_proc][CPU_MAP_SMALL_CORE] = ecore_group;
                    }
                } else if (1 == list_len) {
                    _p_cores++;
                    _cpu_mapping[list[0] + base_proc][CPU_MAP_PHY_CORE] = _p_cores;
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
