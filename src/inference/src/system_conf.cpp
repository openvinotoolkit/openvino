// Copyright (C) 2018-2025 Intel Corporation
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

#if !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && (defined(__arm__) || defined(__aarch64__))
#    include <asm/hwcap.h> /* Get HWCAP bits from asm/hwcap.h */
#    include <sys/auxv.h>
#    define ARM_COMPUTE_CPU_FEATURE_HWCAP_FPHP    (1 << 9)
#    define ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDHP (1 << 10)
#    define ARM_COMPUTE_CPU_FEATURE_HWCAP_SVE     (1 << 24)
#elif defined(__APPLE__) && defined(__aarch64__)
#    include <sys/sysctl.h>
#    include <sys/types.h>
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

bool with_cpu_x86_avx512_core_amx_fp16() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAMX_FP16);
}

bool with_cpu_x86_avx512_core_amx() {
    return with_cpu_x86_avx512_core_amx_int8() || with_cpu_x86_avx512_core_amx_bf16();
}

bool with_cpu_neon_fp16() {
    return false;
}

bool with_cpu_sve() {
    return false;
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
bool with_cpu_x86_avx512_core_amx_fp16() {
    return false;
}
bool with_cpu_x86_avx512_core_amx() {
    return false;
}
bool with_cpu_neon_fp16() {
#    if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
        !defined(__arm__) && defined(__aarch64__)
    const uint32_t hwcaps = getauxval(AT_HWCAP);
    return hwcaps & (ARM_COMPUTE_CPU_FEATURE_HWCAP_FPHP | ARM_COMPUTE_CPU_FEATURE_HWCAP_ASIMDHP);
#    elif !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
        !defined(__aarch64__) && defined(__arm__)
    return false;
#    elif defined(__aarch64__) && defined(__APPLE__)
    int64_t result(0);
    size_t size = sizeof(result);
    const std::string& cap = "hw.optional.neon_fp16";
    sysctlbyname(cap.c_str(), &result, &size, NULL, 0);
    return result > 0;
#    else
    return false;
#    endif
}
bool with_cpu_sve() {
#    if !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
        !defined(__arm__) && defined(__aarch64__)
    const uint32_t hwcaps = getauxval(AT_HWCAP);
    return hwcaps & ARM_COMPUTE_CPU_FEATURE_HWCAP_SVE;
#    elif !defined(_WIN64) && !defined(BARE_METAL) && !defined(__APPLE__) && !defined(__OpenBSD__) && \
        !defined(__aarch64__) && defined(__arm__)
    return false;
#    elif defined(__aarch64__) && defined(__APPLE__)
    return false;
#    else
    return false;
#    endif
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

int get_number_of_blocked_cores() {
    return 0;
}

int get_current_socket_id() {
    return 0;
}

int get_current_numa_node_id() {
    return 0;
}

std::vector<std::vector<int>> get_proc_type_table() {
    return {{-1}};
}
std::vector<std::vector<int>> get_org_proc_type_table() {
    return {{-1}};
}
int get_num_numa_nodes() {
    return -1;
}
int get_num_sockets() {
    return -1;
}
int get_numa_node_id(int cpu_id) {
    return -1;
}
void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

int get_org_socket_id(int socket_id) {
    return -1;
}

int get_org_numa_id(int numa_node_id) {
    return -1;
}

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

int get_number_of_blocked_cores() {
    CPU& cpu = cpu_info();
    return cpu._blocked_cores;
}

int get_current_socket_id() {
    return 0;
}

int get_current_numa_node_id() {
    return 0;
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
int get_num_sockets() {
    return cpu_info()._sockets;
}
int get_numa_node_id(int cpu_id) {
    return -1;
}
void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {}
void set_cpu_used(const std::vector<int>& cpu_ids, const int used) {}

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
int get_number_of_cpu_cores(bool bigCoresOnly) {
    CPU& cpu = cpu_info();
    unsigned totalNumberOfCpuCores = cpu._cores;
    OPENVINO_ASSERT(totalNumberOfCpuCores != 0, "Total number of cpu cores can not be 0.");

    int phys_cores = totalNumberOfCpuCores;
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

int get_current_socket_id() {
    CPU& cpu = cpu_info();
    int cur_processor_id = sched_getcpu();

    for (auto& row : cpu._cpu_mapping_table) {
        if (cur_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            return row[CPU_MAP_SOCKET_ID];
        }
    }

    return 0;
}

int get_current_numa_node_id() {
    CPU& cpu = cpu_info();
    int cur_processor_id = sched_getcpu();

    for (auto& row : cpu._cpu_mapping_table) {
        if (cur_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            return row[CPU_MAP_NUMA_NODE_ID];
        }
    }

    return 0;
}
#    else

int get_current_socket_id() {
    CPU& cpu = cpu_info();
    int cur_processor_id = GetCurrentProcessorNumber();

    for (auto& row : cpu._cpu_mapping_table) {
        if (cur_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            return row[CPU_MAP_SOCKET_ID];
        }
    }

    return 0;
}

int get_current_numa_node_id() {
    CPU& cpu = cpu_info();
    int cur_processor_id = GetCurrentProcessorNumber();

    for (auto& row : cpu._cpu_mapping_table) {
        if (cur_processor_id == row[CPU_MAP_PROCESSOR_ID]) {
            return row[CPU_MAP_NUMA_NODE_ID];
        }
    }

    return 0;
}
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

int get_num_numa_nodes() {
    return cpu_info()._numa_nodes;
}

int get_num_sockets() {
    return cpu_info()._sockets;
}

int get_numa_node_id(int cpu_id) {
    CPU& cpu = cpu_info();
    return cpu._cpu_mapping_table[cpu_id][CPU_MAP_NUMA_NODE_ID];
}

void reserve_available_cpus(const std::vector<std::vector<int>> streams_info_table,
                            std::vector<std::vector<int>>& stream_processors,
                            const int cpu_status) {
    CPU& cpu = cpu_info();
    std::lock_guard<std::mutex> lock{cpu._cpu_mutex};

    ov::threading::reserve_cpu_by_streams_info(streams_info_table,
                                               cpu._numa_nodes,
                                               cpu._cpu_mapping_table,
                                               cpu._proc_type_table,
                                               stream_processors,
                                               cpu_status);
#    ifdef ENABLE_OPENVINO_DEBUG
    OPENVINO_DEBUG("[ threading ] stream_processors:");
    for (size_t i = 0; i < stream_processors.size(); i++) {
        OPENVINO_DEBUG("{");
        for (size_t j = 0; j < stream_processors[i].size(); j++) {
            OPENVINO_DEBUG(stream_processors[i][j], ",");
        }
        OPENVINO_DEBUG("},");
    }
#    endif
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
        ov::threading::update_proc_type_table(cpu._cpu_mapping_table, cpu._numa_nodes, cpu._proc_type_table);
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

int get_number_of_blocked_cores() {
    CPU& cpu = cpu_info();
    return cpu._blocked_cores;
}

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
