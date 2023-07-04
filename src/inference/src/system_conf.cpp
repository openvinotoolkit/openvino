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
#include "openvino/core/except.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/runtime/threading/cpu_streams_info.hpp"
#include "openvino/util/log.hpp"
#include "os/cpu_map_info.hpp"

#ifdef __APPLE__
#    include <sys/sysctl.h>
#    include <sys/types.h>
#endif

#if defined(OPENVINO_ARCH_X86) || defined(OPENVINO_ARCH_X86_64)
#    define XBYAK_NO_OP_NAMES
#    define XBYAK_UNDEF_JNL
#    include <xbyak/xbyak_util.h>
#endif

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
std::vector<std::vector<int>> get_org_proc_type_table() {
    return {{-1}};
}
bool is_cpu_map_available() {
    return false;
}
int get_num_numa_nodes() {
    return -1;
}
void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            std::vector<int>& stream_numa_node_ids,
                            const int cpu_status) {}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

int get_socket_by_numa_node(int numa_node_id) {
    return -1;
};

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

std::vector<std::vector<int>> get_org_proc_type_table() {
    CPU& cpu = cpu_info();
    return cpu._org_proc_type_table;
}

int get_num_numa_nodes() {
    return cpu_info()._numa_nodes;
}
void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            std::vector<int>& stream_numa_node_ids,
                            const int cpu_status) {}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

int get_socket_by_numa_node(int numa_node_id) {
    return -1;
};

#else

#    ifndef _WIN32
int get_number_of_cpu_cores(bool bigCoresOnly) {
    CPU& cpu = cpu_info();
    unsigned numberOfProcessors = cpu._processors;
    unsigned totalNumberOfCpuCores = cpu._cores;
    OPENVINO_ASSERT(totalNumberOfCpuCores != 0, "Total number of cpu cores can not be 0.");
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

std::vector<std::vector<int>> get_org_proc_type_table() {
    CPU& cpu = cpu_info();
    return cpu._org_proc_type_table;
}

bool is_cpu_map_available() {
    CPU& cpu = cpu_info();
    return cpu._cpu_mapping_table.size() > 0;
}

int get_num_numa_nodes() {
    return cpu_info()._numa_nodes;
}

void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            std::vector<int>& stream_numa_node_ids,
                            const int cpu_status) {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._plugin_mutex};

    auto proc_type_table = get_proc_type_table();

    reserve_cpu_by_streams_info(streams_info_table,
                                cpu._cpu_mapping_table,
                                proc_type_table,
                                cpu._numa_nodes,
                                stream_processors,
                                cpu_status);

    OPENVINO_DEBUG << "[ threading ] cpu_mapping_table:";
    for (size_t i = 0; i < cpu._cpu_mapping_table.size(); i++) {
        OPENVINO_DEBUG << cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_CORE_ID] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_GROUP_ID] << " "
                       << cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG];
    }
    OPENVINO_DEBUG << "[ threading ] proc_type_table:";
    for (size_t i = 0; i < cpu._proc_type_table.size(); i++) {
        OPENVINO_DEBUG << cpu._proc_type_table[i][ALL_PROC] << " " << cpu._proc_type_table[i][MAIN_CORE_PROC] << " "
                       << cpu._proc_type_table[i][EFFICIENT_CORE_PROC] << " "
                       << cpu._proc_type_table[i][HYPER_THREADING_PROC] << " "
                       << cpu._proc_type_table[i][PROC_NUMA_NODE_ID] << " " << cpu._proc_type_table[i][PROC_SOCKET_ID];
    }
    OPENVINO_DEBUG << "[ threading ] streams_info_table:";
    for (size_t i = 0; i < streams_info_table.size(); i++) {
        OPENVINO_DEBUG << streams_info_table[i][NUMBER_OF_STREAMS] << " " << streams_info_table[i][PROC_TYPE] << " "
                       << streams_info_table[i][THREADS_PER_STREAM] << " " << streams_info_table[i][STREAM_NUMA_NODE_ID]
                       << " " << streams_info_table[i][STREAM_SOCKET_ID];
    }
}

void reserve_cpu_by_streams_info(const std::vector<std::vector<int>> _streams_info_table,
                                 const std::vector<std::vector<int>> _cpu_mapping_table,
                                 const std::vector<std::vector<int>> _proc_type_table,
                                 const int _numa_nodes,
                                 std::vector<std::vector<int>>& _stream_processors,
                                 const int _cpu_status) {
    std::vector<std::vector<int>> streams_info_table = _streams_info_table;
    std::vector<std::vector<std::string>> stream_conditions;
    std::vector<int> stream_pos;
    std::vector<int> stream_num;
    std::vector<int> cpu_ids;
    int num_streams = 0;

    stream_pos.assign(_streams_info_table.size(), 0);
    stream_num.assign(_streams_info_table.size(), 0);
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        stream_pos[i] = num_streams;
        num_streams += _streams_info_table[i][NUMBER_OF_STREAMS];
    }
    _stream_processors.assign(num_streams, std::vector<int>());
    stream_conditions.assign(_streams_info_table.size(), std::vector<std::string>());
    for (size_t i = 0; i < _streams_info_table.size(); i++) {
        std::vector<std::string> proc_types;
        std::vector<std::string> numa_nodes;
        std::vector<std::string> sockets;
        if (_streams_info_table[i][PROC_TYPE] == ALL_PROC) {
            proc_types.push_back(std::to_string(MAIN_CORE_PROC));
            proc_types.push_back(std::to_string(EFFICIENT_CORE_PROC));
            proc_types.push_back(std::to_string(HYPER_THREADING_PROC));
        } else {
            proc_types.push_back(std::to_string(_streams_info_table[i][PROC_TYPE]));
        }
        if (num_streams == 1 && _streams_info_table[0][PROC_TYPE] == MAIN_CORE_PROC &&
            _streams_info_table[0][THREADS_PER_STREAM] > _proc_type_table[0][MAIN_CORE_PROC]) {
            proc_types.push_back(std::to_string(HYPER_THREADING_PROC));
        }
        if (_streams_info_table[i][STREAM_NUMA_NODE_ID] < 0) {
            for (int j = 0; j < _numa_nodes; j++) {
                numa_nodes.push_back(std::to_string(j));
            }
        } else {
            numa_nodes.push_back(std::to_string(_streams_info_table[i][STREAM_NUMA_NODE_ID]));
        }
        if (_streams_info_table[i][STREAM_SOCKET_ID] < 0) {
            for (int j = 0; j < _numa_nodes; j++) {
                sockets.push_back(std::to_string(j));
            }
        } else {
            sockets.push_back(std::to_string(_streams_info_table[i][STREAM_SOCKET_ID]));
        }
        for (auto t : proc_types) {
            for (auto n : numa_nodes) {
                for (auto s : sockets) {
                    stream_conditions[i].push_back(t + n + s);
                }
            }
        }
    }

    for (size_t i = 0; i < _cpu_mapping_table.size(); i++) {
        std::string cpu_string = std::to_string(_cpu_mapping_table[i][CPU_MAP_CORE_TYPE]) +
                                 std::to_string(_cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID]) +
                                 std::to_string(_cpu_mapping_table[i][CPU_MAP_SOCKET_ID]);
        for (size_t j = 0; j < stream_conditions.size(); j++) {
            if (std::find(stream_conditions[j].begin(), stream_conditions[j].end(), cpu_string) !=
                stream_conditions[j].end()) {
                _stream_processors[stream_pos[j]].push_back(_cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID]);
                if (static_cast<int>(_stream_processors[stream_pos[j]].size()) ==
                    _streams_info_table[j][THREADS_PER_STREAM]) {
                    cpu_ids.insert(cpu_ids.end(),
                                   _stream_processors[stream_pos[j]].begin(),
                                   _stream_processors[stream_pos[j]].end());
                    stream_pos[j]++;
                    stream_num[j]++;
                }
                if (stream_num[j] >= _streams_info_table[j][NUMBER_OF_STREAMS]) {
                    stream_conditions[j].clear();
                }
                break;
            }
        }
    }

    if (_cpu_status > NOT_USED) {
        set_cpu_used(cpu_ids, _cpu_status);
    }
}

void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    const auto cpu_size = static_cast<int>(cpu_ids.size());
    if (cpu_size > 0) {
        for (int i = 0; i < cpu_size; i++) {
            if (cpu_ids[i] < cpu._processors) {
                cpu._cpu_mapping_table[cpu_ids[i]][CPU_MAP_USED_FLAG] = used;
            }
        }
        // update _proc_type_table
        std::vector<int> all_table;
        std::map<int, int> numa_node_map;
        if (cpu._numa_nodes > 1) {
            for (int i = 1; i < static_cast<int>(cpu._proc_type_table.size()); i++) {
                numa_node_map.insert(std::pair<int, int>(cpu._proc_type_table[i][PROC_NUMA_NODE_ID], i));
            }
        } else {
            numa_node_map.insert(std::pair<int, int>(cpu._proc_type_table[0][PROC_NUMA_NODE_ID], 0));
        }

        if (cpu._proc_type_table.size() > 0 && cpu._num_threads == cpu._proc_type_table[0][ALL_PROC]) {
            cpu._proc_type_table.assign(cpu._proc_type_table.size(), std::vector<int>(PROC_TYPE_TABLE_SIZE, 0));
            all_table.resize(PROC_TYPE_TABLE_SIZE, 0);
            for (int i = 0; i < cpu._processors; i++) {
                if (cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG] == NOT_USED &&
                    cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] >= 0 &&
                    cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] >= ALL_PROC) {
                    cpu._proc_type_table[numa_node_map.at(cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID])]
                                        [cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE]]++;
                    cpu._proc_type_table[numa_node_map.at(cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID])][ALL_PROC]++;
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

int get_socket_by_numa_node(int numa_node_id) {
    CPU& cpu = cpu_info();
    for (int i = 0; i < cpu._processors; i++) {
        if (cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] == numa_node_id) {
            return cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID];
        }
    }
    return -1;
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
