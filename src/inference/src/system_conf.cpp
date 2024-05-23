// Copyright (C) 2018-2024 Intel Corporation
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

#ifdef __linux__
#    include <sched.h>
#endif

#include "dev/threading/parallel_custom_arena.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/runtime/threading/cpu_streams_executor_internal.hpp"
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

bool with_cpu_x86_avx2_vnni() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX2 | Xbyak::util::Cpu::tAVX_VNNI);
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

bool with_cpu_x86_avx512_core_fp16() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX512_FP16);
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
bool with_cpu_x86_avx2_vnni() {
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
bool with_cpu_x86_avx512_core_fp16() {
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

static std::map<int, std::shared_ptr<ov::CPU>> cpu_map{{-1, std::make_shared<CPU>()}};
static std::mutex cpu_map_mutex;
static int executor_count = 0;
CPU& cpu_info(int executor_id) {
    std::lock_guard<std::mutex> lock{cpu_map_mutex};
    if (cpu_map.find(executor_id) == cpu_map.end()) {
        cpu_map[executor_id] = std::make_shared<CPU>();
    }
    return *cpu_map[executor_id];
}

#if defined(__EMSCRIPTEN__)
// for Linux and Windows the getNumberOfCPUCores (that accounts only for physical cores) implementation is OS-specific
// (see cpp files in corresponding folders), for __APPLE__ it is default :
int get_number_of_cpu_cores(int, bool) {
    return parallel_get_max_threads();
}

#    if !((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes(int executor_id) {
    return {-1};
}
#    endif
int get_number_of_logical_cpu_cores(bool) {
    return parallel_get_max_threads();
}

int get_number_of_blocked_cores(int) {
    return 0;
}

int get_current_socket_id(int) {
    return 0;
}

std::vector<std::vector<int>> get_proc_type_table(int) {
    return {{-1}};
}
std::vector<std::vector<int>> get_org_proc_type_table() {
    return {{-1}};
}
bool is_cpu_map_available(int) {
    return false;
}
int get_num_numa_nodes() {
    return -1;
}
int get_num_sockets(int) {
    return -1;
}
void reserve_available_cpus(int executor_id,
                            const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

int get_socket_by_numa_node(int executor_id, int numa_node_id) {
    return -1;
};

int get_org_socket_id(int socket_id) {
    return -1;
}

int get_org_numa_id(int numa_node_id) {
    return -1;
}

#elif defined(__APPLE__)
// for Linux and Windows the getNumberOfCPUCores (that accounts only for physical cores) implementation is OS-specific
// (see cpp files in corresponding folders), for __APPLE__ it is default :
int get_number_of_cpu_cores(int, bool) {
    return parallel_get_max_threads();
}

#    if !((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes(int executor_id) {
    return {-1};
}
#    endif
int get_number_of_logical_cpu_cores(bool) {
    return parallel_get_max_threads();
}

int get_number_of_blocked_cores(int) {
    CPU& cpu = cpu_info();
    return cpu._blocked_cores;
}

bool is_cpu_map_available(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    return cpu._proc_type_table.size() > 0;
}

int get_current_socket_id(int) {
    return 0;
}

std::vector<std::vector<int>> get_proc_type_table(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    return cpu._proc_type_table;
}

std::vector<std::vector<int>> get_org_proc_type_table() {
    CPU& cpu = cpu_info();
    return cpu._org_proc_type_table;
}

int get_num_numa_nodes(int executor_id) {
    // return cpu._numa_nodes;
    return cpu_info(executor_id)._numa_nodes;
}
int get_num_sockets(int executor_id) {
    // return cpu._sockets;
    return cpu_info(executor_id)._sockets;
}
void reserve_available_cpus(int executor_id,
                            const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {}

void set_cpu_used(int executor_id, const std::vector<int>& cpu_ids, const int used) {}

int get_socket_by_numa_node(int executor_id, int numa_node_id) {
    CPU& cpu = cpu_info(executor_id);
    for (size_t i = 0; i < cpu._proc_type_table.size(); i++) {
        if (cpu._proc_type_table[i][PROC_NUMA_NODE_ID] == numa_node_id) {
            return cpu._proc_type_table[i][PROC_SOCKET_ID];
        }
    }
    return -1;
};

int get_org_socket_id(int socket_id) {
    CPU& cpu = cpu_info();
    auto iter = cpu._socketid_mapping_table.find(socket_id);
    if (iter != cpu._socketid_mapping_table.end()) {
        return iter->second;
    }
    return -1;
}

int get_org_numa_id(int numa_node_id) {
    CPU& cpu = cpu_info();
    auto iter = cpu._numaid_mapping_table.find(numa_node_id);
    if (iter != cpu._numaid_mapping_table.end()) {
        return iter->second;
    }
    return -1;
}

#else

#    ifndef _WIN32
int get_number_of_cpu_cores(int executor_id, bool bigCoresOnly) {
    CPU& cpu = cpu_info(executor_id);
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
std::vector<int> get_available_numa_nodes(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    std::vector<int> nodes((0 == cpu._numa_nodes) ? 1 : cpu._numa_nodes);
    std::iota(std::begin(nodes), std::end(nodes), 0);
    return nodes;
}
#        endif
int get_current_socket_id(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    int cur_processor_id = sched_getcpu();

    for (auto& row : cpu._cpu_mapping_table) {
        if (cur_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            return row[CPU_MAP_SOCKET_ID];
        }
    }

    return 0;
}
#    else
int get_current_socket_id(int) {
    return 0;
}
#    endif

std::vector<std::vector<int>> get_proc_type_table(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    return cpu._proc_type_table;
}

std::vector<std::vector<int>> get_org_proc_type_table(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    return cpu._org_proc_type_table;
}

bool is_cpu_map_available(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    return cpu._cpu_mapping_table.size() > 0;
}

int get_num_numa_nodes(int executor_id) {
    // return cpu._numa_nodes;
    return cpu_info(executor_id)._numa_nodes;
}

int get_num_sockets(int executor_id) {
    // return cpu._sockets;
    return cpu_info(executor_id)._sockets;
}

inline int update_cpuinfo(std::vector<int>& infos, std::vector<int>& cpu_mapping, ColumnOfCPUMappingTable type) {
    bool found = false;
    int maxid = -1;
    for (int j : infos) {
        if (j > maxid) {
            maxid = j;
        }
        if (j == cpu_mapping[type]) {
            found = true;
        }
    }
    if (!found) {
        infos.push_back(cpu_mapping[type]);
    }
    return maxid;
}

int config_available_cpus(int executor_id, std::vector<int>& cpuids) {
    if (cpuids.size() == 0)
        return executor_id;
    if (executor_id == -1) {
        std::lock_guard<std::mutex> lock{cpu_map_mutex};
        executor_id = executor_count++;
    }
    CPU& cpu = cpu_info(executor_id);
    CPU& cpu0 = cpu_info(-1);
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    size_t new_size = cpuids.size();
    auto old_cpu_mapping_table = cpu0._cpu_mapping_table;
    cpu._cpu_mapping_table.resize(new_size, std::vector<int>(CPU_MAP_TABLE_SIZE, -1));
    int index = 0;
    std::vector<int> main_cores;
    std::vector<int> ht_cores;
    std::vector<int> e_cores;
    std::vector<int> numanodes;
    std::vector<int> sockets;
    std::vector<int> processors;
    std::vector<int> cores;
    for (auto& cit : old_cpu_mapping_table) {
        for (int it : cpuids) {
            if (it == cit[CPU_MAP_PROCESSOR_ID]) {
                cpu._cpu_mapping_table[index][CPU_MAP_PROCESSOR_ID] = cit[CPU_MAP_PROCESSOR_ID];
                cpu._cpu_mapping_table[index][CPU_MAP_NUMA_NODE_ID] = cit[CPU_MAP_NUMA_NODE_ID];
                cpu._cpu_mapping_table[index][CPU_MAP_SOCKET_ID] = cit[CPU_MAP_SOCKET_ID];
                cpu._cpu_mapping_table[index][CPU_MAP_CORE_ID] = cit[CPU_MAP_CORE_ID];
                cpu._cpu_mapping_table[index][CPU_MAP_CORE_TYPE] = cit[CPU_MAP_CORE_TYPE];
                cpu._cpu_mapping_table[index][CPU_MAP_GROUP_ID] = cit[CPU_MAP_GROUP_ID];
                cpu._cpu_mapping_table[index][CPU_MAP_USED_FLAG] = cit[CPU_MAP_USED_FLAG];
                if (cpu._cpu_mapping_table[index][CPU_MAP_CORE_TYPE] == MAIN_CORE_PROC)
                    main_cores.push_back(index);
                else if (cpu._cpu_mapping_table[index][CPU_MAP_CORE_TYPE] == HYPER_THREADING_PROC)
                    ht_cores.push_back(index);
                else
                    e_cores.push_back(index);
                update_cpuinfo(numanodes, cit, CPU_MAP_NUMA_NODE_ID);
                update_cpuinfo(sockets, cit, CPU_MAP_SOCKET_ID);
                update_cpuinfo(processors, cit, CPU_MAP_PROCESSOR_ID);
                update_cpuinfo(cores, cit, CPU_MAP_CORE_ID);
                index++;
                break;
            }
        }
    }
    int changed_num = 0;
    for (int i : ht_cores) {
        bool found = false;
        for (int j : main_cores) {
            if (cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] == cpu._cpu_mapping_table[j][CPU_MAP_NUMA_NODE_ID] &&
                cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID] == cpu._cpu_mapping_table[j][CPU_MAP_SOCKET_ID] &&
                cpu._cpu_mapping_table[i][CPU_MAP_CORE_ID] == cpu._cpu_mapping_table[j][CPU_MAP_CORE_ID]) {
                found = true;
                break;
            }
        }
        if (!found) {
            cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE] = MAIN_CORE_PROC;
            changed_num++;
        }
    }

    cpu._numa_nodes = numanodes.size();
    cpu._sockets = sockets.size();
    cpu._processors = processors.size();
    cpu._cores = cores.size();
    cpu._socket_idx = 0;

    cpu._proc_type_table.clear();
    cpu._proc_type_table.assign((cpu._numa_nodes == 1) ? 1 : cpu._numa_nodes + 1,
                                std::vector<int>({0, 0, 0, 0, -1, -1}));
    if (cpu._numa_nodes == 1) {
        for (auto& it : cpu._cpu_mapping_table) {
            cpu._proc_type_table[0][ALL_PROC]++;
            switch (it[CPU_MAP_CORE_TYPE]) {
            case MAIN_CORE_PROC:
                cpu._proc_type_table[0][MAIN_CORE_PROC]++;
                break;
            case HYPER_THREADING_PROC:
                cpu._proc_type_table[0][HYPER_THREADING_PROC]++;
                break;
            case EFFICIENT_CORE_PROC:
                cpu._proc_type_table[0][EFFICIENT_CORE_PROC]++;
                break;
            }
            cpu._proc_type_table[0][PROC_NUMA_NODE_ID] = it[CPU_MAP_NUMA_NODE_ID];
            cpu._proc_type_table[0][PROC_SOCKET_ID] = it[CPU_MAP_SOCKET_ID];
        }
    } else {
        for (auto& it : cpu._cpu_mapping_table) {
            cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][ALL_PROC]++;
            switch (it[CPU_MAP_CORE_TYPE]) {
            case MAIN_CORE_PROC:
                cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][MAIN_CORE_PROC]++;
                break;
            case HYPER_THREADING_PROC:
                cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][HYPER_THREADING_PROC]++;
                break;
            case EFFICIENT_CORE_PROC:
                cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][EFFICIENT_CORE_PROC]++;
                break;
            }
            cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][PROC_NUMA_NODE_ID] = it[CPU_MAP_NUMA_NODE_ID];
            cpu._proc_type_table[it[CPU_MAP_NUMA_NODE_ID] + 1][PROC_SOCKET_ID] = it[CPU_MAP_SOCKET_ID];
        }
    }
    return executor_id;
}

void reserve_available_cpus(int executor_id,
                            const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {
    CPU& cpu = cpu_info(executor_id);
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};

    ov::threading::reserve_cpu_by_streams_info(streams_info_table,
                                               cpu._numa_nodes,
                                               cpu._cpu_mapping_table,
                                               cpu._proc_type_table,
                                               stream_processors,
                                               cpu_status);

    // printf("[ threading ] cpu_mapping_table (%ld):\n", cpu._cpu_mapping_table.size());
    // for (size_t i = 0; i < cpu._cpu_mapping_table.size(); i++) {
    //     printf("%d\t%d\t%d\t%d\t%d\t%d\t%d\n",
    //         cpu._cpu_mapping_table[i][CPU_MAP_PROCESSOR_ID],
    //         cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID],
    //         cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID],
    //         cpu._cpu_mapping_table[i][CPU_MAP_CORE_ID],
    //         cpu._cpu_mapping_table[i][CPU_MAP_CORE_TYPE],
    //         cpu._cpu_mapping_table[i][CPU_MAP_GROUP_ID],
    //         cpu._cpu_mapping_table[i][CPU_MAP_USED_FLAG]);
    // }
    // printf("[ threading ] proc_type_table:\n");
    // for (size_t i = 0; i < cpu._proc_type_table.size(); i++) {
    //     printf("%d\t%d\t%d\t%d\t%d\t%d\n", 
    //     cpu._proc_type_table[i][ALL_PROC],
    //     cpu._proc_type_table[i][MAIN_CORE_PROC],
    //     cpu._proc_type_table[i][EFFICIENT_CORE_PROC],
    //     cpu._proc_type_table[i][HYPER_THREADING_PROC],
    //     cpu._proc_type_table[i][PROC_NUMA_NODE_ID],
    //     cpu._proc_type_table[i][PROC_SOCKET_ID]);
    // }
    // printf("[ threading ] streams_info_table (%ld):", streams_info_table.size());
    // for (size_t i = 0; i < streams_info_table.size(); i++) {
    //     printf("%d\t%d\t%d\t%d\t%d\n", 
    //     streams_info_table[i][NUMBER_OF_STREAMS],
    //     streams_info_table[i][PROC_TYPE],
    //     streams_info_table[i][THREADS_PER_STREAM],
    //     streams_info_table[i][STREAM_NUMA_NODE_ID],
    //     streams_info_table[i][STREAM_SOCKET_ID]);
    // }
    printf("[ threading executor_id=%d] stream_processors (%ld):\n", executor_id, stream_processors.size());
    for (size_t i = 0; i < stream_processors.size(); i++) {
        printf("{ ");
        for (size_t j = 0; j < stream_processors[i].size(); j++) {
            printf("%d ", stream_processors[i][j]);
        }
        printf("}\n");
    }
}

void set_cpu_used(int executor_id, const std::vector<int>& cpu_ids, const int used) {
    CPU& cpu = cpu_info(executor_id);
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};
    const auto cpu_size = static_cast<int>(cpu_ids.size());
    if (cpu_size > 0) {
        for (int i = 0; i < cpu_size; i++) {
            if (cpu_ids[i] < cpu._processors) {
                cpu._cpu_mapping_table[cpu_ids[i]][CPU_MAP_USED_FLAG] = used;
            }
        }
        ov::threading::update_proc_type_table(cpu._cpu_mapping_table, cpu._numa_nodes, cpu._proc_type_table);
    }
}

int get_socket_by_numa_node(int executor_id, int numa_node_id) {
    CPU& cpu = cpu_info(executor_id);
    for (int i = 0; i < cpu._processors; i++) {
        if (cpu._cpu_mapping_table[i][CPU_MAP_NUMA_NODE_ID] == numa_node_id) {
            return cpu._cpu_mapping_table[i][CPU_MAP_SOCKET_ID];
        }
    }
    return -1;
}

int get_number_of_logical_cpu_cores(int, bool bigCoresOnly) {
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

int get_number_of_blocked_cores(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    return cpu._blocked_cores;
}

int get_org_socket_id(int socket_id) {
    CPU& cpu = cpu_info(-1);
    auto iter = cpu._socketid_mapping_table.find(socket_id);
    if (iter != cpu._socketid_mapping_table.end()) {
        return iter->second;
    }
    return -1;
}

int get_org_numa_id(int numa_node_id) {
    CPU& cpu = cpu_info(-1);
    auto iter = cpu._numaid_mapping_table.find(numa_node_id);
    if (iter != cpu._numaid_mapping_table.end()) {
        return iter->second;
    }
    return -1;
}
#endif

#if ((OV_THREAD == OV_THREAD_TBB) || (OV_THREAD == OV_THREAD_TBB_AUTO))
std::vector<int> get_available_numa_nodes(int executor_id) {
    CPU& cpu = cpu_info(executor_id);
    std::vector<int> nodes((0 == cpu._numa_nodes) ? 1 : cpu._numa_nodes);
    std::iota(std::begin(nodes), std::end(nodes), 0);
    return nodes;
    // return custom::info::numa_nodes();
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
