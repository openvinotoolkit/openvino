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
#include "threading/ie_cpu_streams_info.hpp"

#ifdef __APPLE__
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    define XBYAK_NO_OP_NAMES
#    define XBYAK_UNDEF_JNL
#    include <xbyak/xbyak_util.h>
#endif

using namespace InferenceEngine;

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
                       "GOMP_DEBUG",
                       "GOMP_RTEMS_THREAD_POOLS",
                       "GOMP_SPINCOUNT",
                       "GOMP_STACKSIZE",
                       "KMP_AFFINITY",
                       "KMP_NUM_THREADS",
                       "MIC_KMP_AFFINITY",
                       "MIC_OMP_NUM_THREADS",
                       "MIC_OMP_PROC_BIND",
                       "MKL_DOMAIN_NUM_THREADS",
                       "MKL_DYNAMIC",
                       "MKL_NUM_THREADS",
                       "OMP_CANCELLATION",
                       "OMP_DEFAULT_DEVICE",
                       "OMP_DISPLAY_ENV",
                       "OMP_DYNAMIC",
                       "OMP_MAX_ACTIVE_LEVELS",
                       "OMP_MAX_TASK_PRIORITY",
                       "OMP_NESTED",
                       "OMP_NUM_THREADS",
                       "OMP_PLACES",
                       "OMP_PROC_BIND",
                       "OMP_SCHEDULE",
                       "OMP_STACKSIZE",
                       "OMP_THREAD_LIMIT",
                       "OMP_WAIT_POLICY",
                       "PHI_KMP_AFFINITY",
                       "PHI_KMP_PLACE_THREADS",
                       "PHI_OMP_NUM_THREADS"}) {
        if (getenv(var)) {
            if (0 != strcmp(var, "OMP_NUM_THREADS") || include_omp_num_threads)
                return true;
        }
    }
    return false;
}

CPU& cpu_info() {
    static CPU cpu;
    return cpu;
}

#if defined(__EMSCRIPTEN__)
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
std::vector<std::vector<int>> get_proc_type_table() {
    return {{-1}};
}
bool is_cpu_map_available() {
    return false;
}
int get_num_numa_nodes() {
    return -1;
}
std::vector<std::vector<int>> reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table) {
    return {{-1}};
}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

#elif defined(__APPLE__)
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

bool is_cpu_map_available() {
    CPU& cpu = cpu_info();
    return cpu._proc_type_table.size() > 0;
}

std::vector<std::vector<int>> get_proc_type_table() {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    return cpu._proc_type_table;
}

int get_num_numa_nodes() {
    return cpu_info()._numa_nodes;
}
std::vector<std::vector<int>> reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table) {
    return {{-1}};
}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

#else

#    ifndef _WIN32
int get_number_of_cpu_cores(bool bigCoresOnly) {
    CPU& cpu = cpu_info();
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
    CPU& cpu = cpu_info();
    std::vector<int> nodes((0 == cpu._numa_nodes) ? 1 : cpu._numa_nodes);
    std::iota(std::begin(nodes), std::end(nodes), 0);
    return nodes;
}
#        endif
#    endif

std::vector<std::vector<int>> get_proc_type_table() {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    return cpu._proc_type_table;
}

bool is_cpu_map_available() {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    return cpu._proc_type_table.size() > 0 && cpu._num_threads == cpu._proc_type_table[0][ALL_PROC];
}

int get_num_numa_nodes() {
    return cpu_info()._numa_nodes;
}

std::vector<std::vector<int>> reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table) {
    std::vector<int> cpu_ids;
    int info_table_size = static_cast<int>(streams_info_table.size());
    std::vector<std::vector<int>> stream_ids;
    std::vector<std::vector<std::vector<int>>> res_stream_ids;
    CPU& cpu = cpu_info();
    stream_ids.assign(info_table_size, std::vector<int>());
    res_stream_ids.assign(info_table_size, std::vector<std::vector<int>>());

    for (int i = 0; i < cpu._processors; i++) {
        for (int j = 0; j < info_table_size; j++) {
            if (static_cast<int>(res_stream_ids[j].size()) < streams_info_table[j][NUMBER_OF_STREAMS]) {
                std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
                if (cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] == streams_info_table[j][PROC_TYPE] &&
                    cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] == NOT_USED) {
                    stream_ids[j].push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                    cpu_ids.push_back(cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                }
                if (static_cast<int>(stream_ids[j].size()) == streams_info_table[j][THREADS_PER_STREAM]) {
                    std::vector<int> stream_group(stream_ids[j].begin(), stream_ids[j].end());
                    res_stream_ids[j].push_back(stream_group);
                    stream_ids[j].clear();
                }
            }
        }
    }
    auto flatten_stream_ids =
        std::accumulate(res_stream_ids.begin(),
                        res_stream_ids.end(),
                        decltype(res_stream_ids)::value_type{},
                        [](std::vector<std::vector<int>>& pre, std::vector<std::vector<int>>& cur) {
                            pre.insert(pre.end(), cur.begin(), cur.end());
                            return pre;
                        });

    return flatten_stream_ids;
}

void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    const auto cpu_size = static_cast<int>(cpu_ids.size());
    for (int i = 0; i < cpu_size; i++) {
        if (cpu_ids[i] < cpu._processors) {
            cpu._cpu_mapping_table[cpu_ids[i]][CPU_MAP_USED_FLAG] = used;
        }
    }
    // update _proc_type_table
    if (used == NOT_USED || used >= PLUGIN_USED_START) {
        std::vector<int> all_table;
        int start = cpu._numa_nodes > 1 ? 1 : 0;
        if (cpu._proc_type_table.size() > 0 && cpu._num_threads == cpu._proc_type_table[0][ALL_PROC]) {
            cpu._proc_type_table.assign(cpu._proc_type_table.size(), std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
            all_table.resize(PROC_TYPE_TABLE_SIZE, 0);
            for (int i = 0; i < cpu._processors; i++) {
                if (cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] < PLUGIN_USED_START &&
                    cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] >= 0 &&
                    cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] >= ALL_PROC) {
                    cpu._proc_type_table[cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] + start]
                                        [cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
                    cpu._proc_type_table[cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] + start][ALL_PROC]++;
                    all_table[cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
                    all_table[ALL_PROC]++;
                }
            }
            if (cpu._numa_nodes > 1) {
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
