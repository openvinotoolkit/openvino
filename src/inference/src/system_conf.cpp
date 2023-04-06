// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/system_conf.hpp"

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

#include "dev/threading/parallel_custom_arena.hpp"
#include "ie_common.h"
#include "openvino/core/visibility.hpp"
#include "streams_executor.hpp"

#define XBYAK_NO_OP_NAMES
#define XBYAK_UNDEF_JNL
#include <xbyak/xbyak_util.h>

namespace ov {

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)

// note: MSVC 2022 (17.4) is not able to compile the next line for ARM and ARM64
// so, we disable this code since for non-x86 platforms it returns 'false' anyway

static Xbyak::util::Cpu& get_cpu_info() {
    static Xbyak::util::Cpu cpu;
    return cpu;
}

bool with_cpu_x86_sse42() {
    return get_cpu_info().has(Xbyak::util::Cpu::tSSE42);
}

bool with_cpu_x86_avx() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX);
}

bool with_cpu_x86_avx2() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX2);
}

bool with_cpu_x86_avx512f() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX512F);
}

bool with_cpu_x86_avx512_core() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX512F | Xbyak::util::Cpu::tAVX512DQ | Xbyak::util::Cpu::tAVX512BW);
}

bool with_cpu_x86_avx512_core_vnni() {
    return with_cpu_x86_avx512_core() && get_cpu_info().has(Xbyak::util::Cpu::tAVX512_VNNI);
}

bool with_cpu_x86_bfloat16() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX512_BF16);
}

bool with_cpu_x86_avx512_core_amx_int8() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAMX_INT8);
}

bool with_cpu_x86_avx512_core_amx_bf16() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAMX_BF16);
}

bool with_cpu_x86_avx512_core_amx() {
    return with_cpu_x86_avx512_core_amx_int8() || with_cpu_x86_avx512_core_amx_bf16();
}

#else  // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

bool with_cpu_x86_sse42() {
    return false;
}
bool with_cpu_x86_avx() {
    return false;
}
bool with_cpu_x86_avx2() {
    return false;
}
bool with_cpu_x86_avx512f() {
    return false;
}
bool with_cpu_x86_avx512_core() {
    return false;
}
bool with_cpu_x86_avx512_core_vnni() {
    return false;
}
bool with_cpu_x86_bfloat16() {
    return false;
}
bool with_cpu_x86_avx512_core_amx_int8() {
    return false;
}
bool with_cpu_x86_avx512_core_amx_bf16() {
    return false;
}
bool with_cpu_x86_avx512_core_amx() {
    return false;
}

#endif  // OPENVINO_ARCH_X86 || OPENVINO_ARCH_X86_64

bool check_open_mp_env_vars(bool include_omp_num_threads) {
    for (auto&& var : {"GOMP_CPU_AFFINITY",
                       "GOMP_DEBUG"
                       "GOMP_RTEMS_THREAD_POOLS",
                       "GOMP_SPINCOUNT"
                       "GOMP_STACKSIZE"
                       "KMP_AFFINITY"
                       "KMP_NUM_THREADS"
                       "MIC_KMP_AFFINITY",
                       "MIC_OMP_NUM_THREADS"
                       "MIC_OMP_PROC_BIND"
                       "MKL_DOMAIN_NUM_THREADS"
                       "MKL_DYNAMIC"
                       "MKL_NUM_THREADS",
                       "OMP_CANCELLATION"
                       "OMP_DEFAULT_DEVICE"
                       "OMP_DISPLAY_ENV"
                       "OMP_DYNAMIC",
                       "OMP_MAX_ACTIVE_LEVELS"
                       "OMP_MAX_TASK_PRIORITY"
                       "OMP_NESTED",
                       "OMP_NUM_THREADS"
                       "OMP_PLACES"
                       "OMP_PROC_BIND"
                       "OMP_SCHEDULE"
                       "OMP_STACKSIZE",
                       "OMP_THREAD_LIMIT"
                       "OMP_WAIT_POLICY"
                       "PHI_KMP_AFFINITY",
                       "PHI_KMP_PLACE_THREADS"
                       "PHI_OMP_NUM_THREADS"}) {
        if (getenv(var)) {
            if (0 != strcmp(var, "OMP_NUM_THREADS") || include_omp_num_threads)
                return true;
        }
    }
    return false;
}

#if defined(__APPLE__) || defined(__EMSCRIPTEN__)
// for Linux and Windows the getNumberOfCPUCores (that accounts only for physical cores) implementation is OS-specific
// (see cpp files in corresponding folders), for __APPLE__ it is default :
int get_number_of_cpu_cores(bool) {
    return parallel_get_max_threads();
}
#    if !((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes() {
    return {-1};
}
#    endif
int get_number_of_logical_cpu_cores(bool) {
    return parallel_get_max_threads();
}
std::vector<std::vector<int>> get_num_available_cpu_cores() {
    return {{-1}};
}
bool is_cpu_map_available() {
    return false;
}
std::vector<int> reserve_available_cpus(const ColumnOfProcessorTypeTable core_type,
                                        const int num_cpus,
                                        const int seek_status,
                                        const int reset_status,
                                        const bool reserve_logic_core) {
    return {};
}
std::vector<int> get_logic_cores(const std::vector<int> cpu_ids) {
    return {};
}
void set_cpu_used(std::vector<int>& cpu_ids, int used) {}

#else

static CPU cpu;

#    ifndef _WIN32
int get_number_of_cpu_cores(bool bigCoresOnly) {
    unsigned numberOfProcessors = cpu._processors;
    unsigned totalNumberOfCpuCores = cpu._cores;
    IE_ASSERT(totalNumberOfCpuCores != 0);
    cpu_set_t usedCoreSet, currentCoreSet, currentCpuSet;
    CPU_ZERO(&currentCpuSet);
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet);

    for (unsigned processorId = 0u; processorId < numberOfProcessors; processorId++) {
        if (CPU_ISSET(processorId, &currentCpuSet)) {
            unsigned coreId = processorId % totalNumberOfCpuCores;
            if (!CPU_ISSET(coreId, &usedCoreSet)) {
                CPU_SET(coreId, &usedCoreSet);
                CPU_SET(processorId, &currentCoreSet);
            }
        }
    }
    int phys_cores = CPU_COUNT(&currentCoreSet);
#        if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        phys_cores = custom::info::default_concurrency(
            custom::task_arena::constraints{}.set_core_type(core_types.back()).set_max_threads_per_core(1));
    }
#        endif
    return phys_cores;
}

#        if !((OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes() {
    std::vector<int> nodes((0 == cpu._sockets) ? 1 : cpu._sockets);
    std::iota(std::begin(nodes), std::end(nodes), 0);
    return nodes;
}
#        endif
#    endif

std::vector<std::vector<int>> get_num_available_cpu_cores() {
    return cpu._proc_type_table;
}

bool is_cpu_map_available() {
    return cpu._proc_type_table.size() > 0 && cpu._num_threads == cpu._proc_type_table[0][ALL_PROC];
}

std::vector<int> reserve_available_cpus(const ColumnOfProcessorTypeTable core_type,
                                        const int num_cpus,
                                        const int seek_status,
                                        const int reset_status,
                                        const bool reserve_logic_core) {
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    std::vector<int> cpu_ids;
    int socket = -1;
    if (reset_status >= PLUGIN_USED_START && cpu._sockets > 1) {
        socket = cpu._socket_idx;
        cpu._socket_idx = (cpu._socket_idx + 1) % cpu._sockets;
    }
    if (core_type < PROC_TYPE_TABLE_SIZE && core_type >= ALL_PROC) {
        for (int i = 0; i < cpu._processors; i++) {
            if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == core_type &&
                cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] == seek_status &&
                (socket < 0 || (socket >= 0 && cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] == socket))) {
                cpu_ids.push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
            }
            if (static_cast<int>(cpu_ids.size()) == num_cpus) {
                break;
            }
        }
        if (reserve_logic_core) {
            auto logic_ids = get_logic_cores(cpu_ids);
            cpu_ids.insert(cpu_ids.end(), logic_ids.begin(), logic_ids.end());
        }
        set_cpu_used(cpu_ids, reset_status);
    } else {
        IE_THROW() << "Wrong value for core_type " << core_type;
    }
    return cpu_ids;
}

std::vector<int> get_logic_cores(const std::vector<int> cpu_ids) {
    std::vector<int> logic_cores;
    if (cpu._proc_type_table[0][HYPER_THREADING_PROC] > 0) {
        int cpu_size = static_cast<int>(cpu_ids.size());
        for (int i = 0; i < cpu._processors; i++) {
            for (int j = 0; j < cpu_size; j++) {
                if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_ID] == cpu._cpu_mapping_table[cpu_ids[j]][CPU_MAP_CORE_ID] &&
                    cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == HYPER_THREADING_PROC) {
                    logic_cores.push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                }
            }
            if (cpu_ids.size() == logic_cores.size()) {
                break;
            }
        }
    }

    return logic_cores;
}

void set_cpu_used(std::vector<int>& cpu_ids, int used) {
    const auto cpu_size = static_cast<int>(cpu_ids.size());
    for (int i = 0; i < cpu_size; i++) {
        if (cpu_ids[i] < cpu._processors) {
            cpu._cpu_mapping_table[cpu_ids[i]][CPU_MAP_USED_FLAG] = used;
        }
    }
    // update _proc_type_table
    if (used == NOT_USED || used >= PLUGIN_USED_START) {
        std::vector<int> all_table;
        int start = cpu._sockets > 1 ? 1 : 0;
        if (is_cpu_map_available()) {
            cpu._proc_type_table.assign(cpu._proc_type_table.size(), std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
            all_table.resize(PROC_TYPE_TABLE_SIZE, 0);
            for (int i = 0; i < cpu._processors; i++) {
                if (cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] < PLUGIN_USED_START) {
                    cpu._proc_type_table[cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] + start]
                                        [cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
                    cpu._proc_type_table[cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] + start][ALL_PROC]++;
                    all_table[cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
                    all_table[ALL_PROC]++;
                }
            }
            if (cpu._sockets > 1) {
                cpu._proc_type_table[0] = all_table;
            }
        }
    }
}

int get_number_of_logical_cpu_cores(bool bigCoresOnly) {
    int logical_cores = parallel_get_max_threads();
#    if (OV_THREAD == OV_THREAD_TBB || OV_THREAD == OV_THREAD_TBB_AUTO)
    auto core_types = custom::info::core_types();
    if (bigCoresOnly && core_types.size() > 1) /*Hybrid CPU*/ {
        logical_cores = custom::info::default_concurrency(
            custom::task_arena::constraints{}.set_core_type(core_types.back()).set_max_threads_per_core(-1));
    }
#    endif
    return logical_cores;
}
#endif

#if ((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes() {
    return custom::info::numa_nodes();
}
// this is impl only with the TBB
std::vector<int> get_available_cores_types() {
    return custom::info::core_types();
}
#else
// as the core types support exists only with the TBB, the fallback is same for any other threading API
std::vector<int> get_available_cores_types() {
    return {-1};
}
#endif

}  // namespace ov
