// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_ie_scheduler.hpp"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "openvino/core/parallel.hpp"

namespace ov::intel_cpu {

using namespace arm_compute;

ACLScheduler::ACLScheduler() = default;

unsigned int ACLScheduler::num_threads() const {
    return parallel_get_num_threads();
}

void ACLScheduler::set_num_threads(unsigned int num_threads) {}

void ACLScheduler::schedule_custom(ICPPKernel* kernel, const Hints& hints, const Window& window, ITensorPack& tensors) {
    const Window& max_window = window;
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
#if OV_THREAD == OV_THREAD_OMP
    // In OpenMP case parallel_get_num_threads() method returns 1 here because it's called outside parallel section
    // This is the reason why this method isn't used to initialize _num_threads
    const auto _num_threads = num_iterations;
#else
    const auto _num_threads = std::min(num_iterations, static_cast<unsigned int>(parallel_get_num_threads()));
#endif
    std::function<void(const Window& window, const ThreadInfo& info)> main_run;
    if (tensors.empty()) {
        main_run = [&](const Window& window, const ThreadInfo& info) {
            kernel->run(window, info);
        };
    } else {
        main_run = [&](const Window& window, const ThreadInfo& info) {
            kernel->run_op(tensors, window, info);
        };
    }

    if (!kernel->is_parallelisable() || _num_threads == 1) {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        main_run(max_window, info);
    } else {
        const auto num_windows = _num_threads;
        const auto hints_split_dimension = hints.split_dimension();

        ov::parallel_for(num_windows, [&](int wid) {
            Window win = max_window.split_window(hints_split_dimension, wid, num_windows);
            win.validate();
            main_run(win, {wid, static_cast<int>(_num_threads), &cpu_info()});
        });
    }
}

void ACLScheduler::schedule(ICPPKernel* kernel, const Hints& hints) {
    ITensorPack tensors;
    schedule_custom(kernel, hints, kernel->window(), tensors);
}

void ACLScheduler::schedule_op(ICPPKernel* kernel, const Hints& hints, const Window& window, ITensorPack& tensors) {
    schedule_custom(kernel, hints, window, tensors);
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload>& workloads) {
    ov::parallel_for(workloads.size(), [&](int wid) {
        workloads[wid]({wid, static_cast<int>(parallel_get_num_threads()), &cpu_info()});
    });
}

}  // namespace ov::intel_cpu
