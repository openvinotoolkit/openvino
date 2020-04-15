// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <cstring>
#include "ie_parallel.hpp"
#include "ie_system_conf.h"
#include <iostream>
#include <vector>

#ifdef ENABLE_MKL_DNN
# define XBYAK_NO_OP_NAMES
# define XBYAK_UNDEF_JNL
# include <xbyak_util.h>
#endif

namespace InferenceEngine {

#ifdef ENABLE_MKL_DNN
static Xbyak::util::Cpu cpu;
#endif

bool with_cpu_x86_sse42() {
#ifdef ENABLE_MKL_DNN
    return cpu.has(Xbyak::util::Cpu::tSSE42);
#else
#if defined(HAVE_SSE)
    return true;
#else
    return false;
#endif
#endif
}

bool with_cpu_x86_avx2() {
#ifdef ENABLE_MKL_DNN
    return cpu.has(Xbyak::util::Cpu::tAVX2);
#else
#if defined(HAVE_AVX2)
    return true;
#else
    return false;
#endif
#endif
}

bool with_cpu_x86_avx512f() {
#ifdef ENABLE_MKL_DNN
    return cpu.has(Xbyak::util::Cpu::tAVX512F);
#else
#if defined(HAVE_AVX512)
    return true;
#else
    return false;
#endif
#endif
}

bool with_cpu_x86_avx512_core() {
#ifdef ENABLE_MKL_DNN
       return cpu.has(Xbyak::util::Cpu::tAVX512F  |
                      Xbyak::util::Cpu::tAVX512DQ |
                      Xbyak::util::Cpu::tAVX512BW);
#else
#if defined(HAVE_AVX512)
    return true;
#else
    return false;
#endif
#endif
}

bool with_cpu_x86_bfloat16() {
#ifdef ENABLE_MKL_DNN
    return cpu.has(Xbyak::util::Cpu::tAVX512_BF16);
#else
    return false;
#endif
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
int getNumberOfCPUCores() { return parallel_get_max_threads();}
#if !((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
std::vector<int> getAvailableNUMANodes() { return {0}; }
#endif
#endif

#if ((IE_THREAD == IE_THREAD_TBB) || (IE_THREAD == IE_THREAD_TBB_AUTO))
std::vector<int> getAvailableNUMANodes() {
    return tbb::info::numa_nodes();
}
#endif

std::exception_ptr& CurrentException() {
     static thread_local std::exception_ptr currentException = nullptr;
    return currentException;
}

}  // namespace InferenceEngine
