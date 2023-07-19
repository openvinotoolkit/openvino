// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_ie_scheduler.hpp"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include <ie_parallel.hpp>

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

ACLScheduler::ACLScheduler() = default;

unsigned int ACLScheduler::num_threads() const { return parallel_get_num_threads(); }

void ACLScheduler::set_num_threads(unsigned int num_threads) {}

void ACLScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void ACLScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) {
    schedule_common(kernel, hints, window, tensors);
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads) {
    InferenceEngine::parallel_for(workloads.size(), [&] (int wid) {
        ThreadInfo info;
        info.cpu_info    = &cpu_info();
        info.num_threads = parallel_get_num_threads();
        info.thread_id   = wid;
        workloads[wid](info);
    });
}

} // namespace intel_cpu
} // namespace ov