// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "threading/ie_thread_affinity.hpp"

#include "dev/threading/thread_affinity.hpp"

namespace InferenceEngine {

std::tuple<CpuSet, int> GetProcessMask() {
    return ov::threading::get_process_mask();
}

void ReleaseProcessMask(cpu_set_t* mask) {
    ov::threading::release_process_mask(mask);
}

bool PinThreadToVacantCore(int thrIdx,
                           int hyperthreads,
                           int ncores,
                           const CpuSet& procMask,
                           const std::vector<int>& cpu_ids,
                           int cpuIdxOffset) {
    return ov::threading::pin_thread_to_vacant_core(thrIdx, hyperthreads, ncores, procMask, cpu_ids, cpuIdxOffset);
}
bool PinCurrentThreadByMask(int ncores, const CpuSet& procMask) {
    return ov::threading::pin_current_thread_by_mask(ncores, procMask);
}
bool PinCurrentThreadToSocket(int socket) {
    return ov::threading::pin_current_thread_to_socket(socket);
}

}  //  namespace InferenceEngine
