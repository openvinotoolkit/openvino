// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dev/threading/thread_affinity.hpp"

#include <cerrno>
#include <climits>
#include <tuple>
#include <utility>

#include "openvino/runtime/system_conf.hpp"

#if !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
#    include <sched.h>
#    include <unistd.h>
#endif

namespace ov {
namespace threading {
#if !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
std::tuple<CpuSet, int> get_process_mask() {
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
void release_process_mask(cpu_set_t* mask) {
    if (nullptr != mask)
        CPU_FREE(mask);
}

bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    return 0 == sched_setaffinity(0, CPU_ALLOC_SIZE(ncores), procMask.get());
}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    if (procMask == nullptr)
        return false;
    const size_t size = CPU_ALLOC_SIZE(ncores);
    const int num_cpus = CPU_COUNT_S(size, procMask.get());
    thrIdx %= num_cpus;  // To limit unique number in [; num_cpus-1] range

    int mapped_idx;
    if (cpu_ids.size() > 0) {
        mapped_idx = cpu_ids[thrIdx];
    } else {
        // Place threads with specified step
        int cpu_idx = 0;
        for (int i = 0, offset = 0; i < thrIdx; ++i) {
            cpu_idx += hyperthreads;
            if (cpu_idx >= num_cpus)
                cpu_idx = ++offset;
        }

        // Find index of 'cpu_idx'-th bit that equals to 1
        mapped_idx = -1;
        while (cpu_idx >= 0) {
            mapped_idx++;
            if (CPU_ISSET_S(mapped_idx, size, procMask.get()))
                --cpu_idx;
        }
    }

    CpuSet targetMask{CPU_ALLOC(ncores)};
    CPU_ZERO_S(size, targetMask.get());
    CPU_SET_S(mapped_idx, size, targetMask.get());
    bool res = pin_current_thread_by_mask(ncores, targetMask);
    return res;
}

bool pin_current_thread_to_socket(int socket) {
    auto proc_type_table = get_org_proc_type_table();
    const int sockets = proc_type_table.size() > 1 ? proc_type_table.size() - 1 : 1;
    const int cores = proc_type_table[0][MAIN_CORE_PROC];
    const int cores_per_socket = cores / sockets;

    int ncpus = 0;
    CpuSet mask;
    std::tie(mask, ncpus) = get_process_mask();
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
        res = pin_current_thread_by_mask(ncpus, targetMask);
    }
    return res;
}
#elif defined(_WIN32)
std::tuple<CpuSet, int> get_process_mask() {
    DWORD_PTR pro_mask, sys_mask;
    if (0 != GetProcessAffinityMask(GetCurrentProcess(), &pro_mask, &sys_mask)) {
        CpuSet mask = std::make_unique<cpu_set_t>(pro_mask);
        return std::make_tuple(std::move(mask), 0);
    }
    return std::make_tuple(nullptr, 0);
}
void release_process_mask(cpu_set_t*) {}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    return 0 != SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu_ids[thrIdx]);
}
bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    DWORD_PTR mask = *procMask.get();
    return 0 != SetThreadAffinityMask(GetCurrentThread(), mask);
}
bool pin_current_thread_to_socket(int socket) {
    return false;
}
#else   // no threads pinning/binding on MacOS
std::tuple<CpuSet, int> get_process_mask() {
    return std::make_tuple(nullptr, 0);
}
void release_process_mask(cpu_set_t*) {}

bool pin_thread_to_vacant_core(int thrIdx,
                               int hyperthreads,
                               int ncores,
                               const CpuSet& procMask,
                               const std::vector<int>& cpu_ids) {
    return false;
}
bool pin_current_thread_by_mask(int ncores, const CpuSet& procMask) {
    return false;
}
bool pin_current_thread_to_socket(int socket) {
    return false;
}
#endif  // !(defined(__APPLE__) || defined(__EMSCRIPTEN__) || defined(_WIN32))
}  // namespace threading
}  // namespace ov
