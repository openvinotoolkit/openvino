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
    const Window &     max_window     = window;
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    const unsigned int num_threads    = std::min(num_iterations, static_cast<std::uint32_t>(parallel_get_num_threads()));

    if (!kernel->is_parallelisable() || num_threads == 1) {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run_op(tensors, max_window, info);
    } else {
        const unsigned int                num_windows = num_threads;
        std::vector<IScheduler::Workload> workloads(num_windows);
        for (unsigned int t = 0; t < num_windows; t++) {
            workloads[t] = [t, &hints, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo &info) {
                Window win = max_window.split_window(hints.split_dimension(), t, num_windows);
                win.validate();
                kernel->run_op(tensors, win, info);
            };
        }
        run_workloads(workloads);
    }
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads) {
    InferenceEngine::parallel_for(workloads.size(), [&] (int wid) {
        ThreadInfo info;
        info.cpu_info    = &cpu_info();
        info.num_threads = parallel_get_num_threads();
        info.thread_id   = parallel_get_thread_num();
        workloads[wid](info);
    });
}

} // namespace intel_cpu
} // namespace ov