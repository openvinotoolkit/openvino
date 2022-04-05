// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_thread_affinity.hpp"

#include <cerrno>
#include <climits>
#include <tuple>
#include <utility>

#include "ie_system_conf.h"

#if !(defined(__APPLE__) || defined(_WIN32))
#    include <sched.h>
#    include <unistd.h>
#endif

namespace InferenceEngine {
#if !(defined(__APPLE__) || defined(_WIN32))
std::tuple<CpuSet, int> GetProcessMask() {
    for (int ncpus = sizeof(cpu_set_t) / CHAR_BIT; ncpus < 32768 /* reasonable limit of #cores*/; ncpus <<= 1) {
        CpuSet mask{CPU_ALLOC(ncpus)};
        if (nullptr == mask)
            break;
        const size_t size = CPU_ALLOC_SIZE(ncpus);
        CPU_ZERO_S(size, mask.get());
        // the result fits the mask
        if (0 == sched_getaffinity(getpid(), size, mask.get())) {
            return std::make_tuple(std::move(mask), ncpus);
        }
        // other error
        if (errno != EINVAL)
            break;
    }
    return std::make_tuple(nullptr, 0);
}

/* Release the cores affinity mask for the current process */
void ReleaseProcessMask(cpu_set_t* mask) {
    if (nullptr != mask)
        CPU_FREE(mask);
}

bool PinCurrentThreadByMask(int ncores, const CpuSet& procMask) {
    return 0 == sched_setaffinity(0, CPU_ALLOC_SIZE(ncores), procMask.get());
}

bool PinThreadToVacantCore(int thrIdx, int hyperthreads, int ncores, const CpuSet& procMask) {
    if (procMask == nullptr)
        return false;
    const size_t size = CPU_ALLOC_SIZE(ncores);
    const int num_cpus = CPU_COUNT_S(size, procMask.get());
    thrIdx %= num_cpus;  // To limit unique number in [; num_cpus-1] range
    // Place threads with specified step
    int cpu_idx = 0;
    for (int i = 0, offset = 0; i < thrIdx; ++i) {
        cpu_idx += hyperthreads;
        if (cpu_idx >= num_cpus)
            cpu_idx = ++offset;
    }

    // Find index of 'cpu_idx'-th bit that equals to 1
    int mapped_idx = -1;
    while (cpu_idx >= 0) {
        mapped_idx++;
        if (CPU_ISSET_S(mapped_idx, size, procMask.get()))
            --cpu_idx;
    }

    CpuSet targetMask{CPU_ALLOC(ncores)};
    CPU_ZERO_S(size, targetMask.get());
    CPU_SET_S(mapped_idx, size, targetMask.get());
    bool res = PinCurrentThreadByMask(ncores, targetMask);
    return res;
}

bool PinCurrentThreadToSocket(int socket) {
    const int sockets = InferenceEngine::getAvailableNUMANodes().size();
    const int cores = InferenceEngine::getNumberOfCPUCores();
    const int cores_per_socket = cores / sockets;

    int ncpus = 0;
    CpuSet mask;
    std::tie(mask, ncpus) = GetProcessMask();
    CpuSet targetMask{CPU_ALLOC(ncpus)};
    const size_t size = CPU_ALLOC_SIZE(ncpus);
    CPU_ZERO_S(size, targetMask.get());

    for (int core = socket * cores_per_socket; core < (socket + 1) * cores_per_socket; core++) {
        CPU_SET_S(core, size, targetMask.get());
    }
    // respect the user-defined mask for the entire process
    CPU_AND_S(size, targetMask.get(), targetMask.get(), mask.get());
    bool res = false;
    if (CPU_COUNT_S(size, targetMask.get())) {  //  if we have non-zero mask to set
        res = PinCurrentThreadByMask(ncpus, targetMask);
    }
    return res;
}
#else   // no threads pinning/binding on Win/MacOS
std::tuple<CpuSet, int> GetProcessMask() {
    return std::make_tuple(nullptr, 0);
}
void ReleaseProcessMask(cpu_set_t*) {}

bool PinThreadToVacantCore(int thrIdx, int hyperthreads, int ncores, const CpuSet& procMask) {
    return false;
}
bool PinCurrentThreadByMask(int ncores, const CpuSet& procMask) {
    return false;
}
bool PinCurrentThreadToSocket(int socket) {
    return false;
}
#endif  // !(defined(__APPLE__) || defined(_WIN32))
}  //  namespace InferenceEngine
