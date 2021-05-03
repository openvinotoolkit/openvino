// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cstdlib>
#include <cstring>
#include "threading/ie_parallel_custom_arena.hpp"
#include "ie_system_conf.h"
#include <iostream>
#include <vector>

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
    return custom::info::core_types();
}
#else
// as the core types support exists only with the TBB, the fallback is same for any other threading API
std::vector<int> getAvailableCoresTypes() {
    return {-1};
}
#endif

std::exception_ptr& CurrentException() {
     static thread_local std::exception_ptr currentException = nullptr;
    return currentException;
}

}  // namespace InferenceEngine
