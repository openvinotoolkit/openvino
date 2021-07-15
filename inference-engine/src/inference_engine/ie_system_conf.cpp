// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <cstring>
#include <vector>

#include "threading/ie_parallel_custom_arena.hpp"
#include "ie_system_conf.h"

# define XBYAK_NO_OP_NAMES
# define XBYAK_UNDEF_JNL
# include <xbyak/xbyak_util.h>

namespace InferenceEngine {

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
       return get_cpu_info().has(Xbyak::util::Cpu::tAVX512F |
                                 Xbyak::util::Cpu::tAVX512DQ |
                                 Xbyak::util::Cpu::tAVX512BW);
}

bool with_cpu_x86_bfloat16() {
    return get_cpu_info().has(Xbyak::util::Cpu::tAVX512_BF16);
}

bool checkOpenMpEnvVars(bool includeOMPNumThreads) {
    for (auto&& var : {
        "GOMP_CPU_AFFINITY",
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
        "PHI_OMP_NUM_THREADS"
        }) {
        if (getenv(var)) {
            if (0 != strcmp(var, "OMP_NUM_THREADS") || includeOMPNumThreads)
                return true;
        }
    }
    return false;
}

#if defined(__APPLE__)
// for Linux and Windows the getNumberOfCPUCores (that accounts only for physical cores) implementation is OS-specific
// (see cpp files in corresponding folders), for __APPLE__ it is default :
int getNumberOfCPUCores(bool) { return parallel_get_max_threads();}
#if !((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
std::vector<int> getAvailableNUMANodes() { return {-1}; }
#endif
#endif

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
std::vector<int> getAvailableNUMANodes() {
    return custom::info::numa_nodes();
}
// this is impl only with the TBB
std::vector<int> getAvailableCoresTypes() {
    auto core_types = custom::info::core_types();
    return std::vector<int>(std::begin(core_types), std::end(core_types));
}
#else
// as the core types support exists only with the TBB, the fallback is same for any other threading API
std::vector<int> getAvailableCoresTypes() {
    return {-1};
}
#endif

}  // namespace InferenceEngine
