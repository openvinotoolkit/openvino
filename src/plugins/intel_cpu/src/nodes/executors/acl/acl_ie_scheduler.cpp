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

ACLScheduler::ACLScheduler() {
    arm_compute::lock_guard<arm_compute::Mutex> lock(this->mtx);
    _num_threads = parallel_get_num_threads();
}

unsigned int ACLScheduler::num_threads() const {
    return parallel_get_num_threads();
}

void ACLScheduler::set_num_threads(unsigned int num_threads) {
    arm_compute::lock_guard<arm_compute::Mutex> lock(this->mtx);
    _num_threads = num_threads;
}

void ACLScheduler::custom_schedule(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) {
    arm_compute::lock_guard<arm_compute::Mutex> lock(this->mtx);

    const Window & max_window = window;
    const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
    _num_threads = std::min(num_iterations, static_cast<unsigned int>(parallel_get_num_threads()));

    if (num_iterations == 0) {
        return;
    }

    if (tensors.empty()) {
        main_run = [&](const Window &window, const ThreadInfo &info) {
            kernel->run(window, info);
        };
    } else {
        main_run = [&](const Window &window, const ThreadInfo &info) {
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

        std::vector<Window> win_vec(num_windows);
        for (size_t i = 0; i < win_vec.size(); ++i) {
            win_vec[i] = max_window.split_window(hints_split_dimension, i, num_windows);
            win_vec[i].validate();
        }

        InferenceEngine::parallel_for(num_windows, [&](int wid) {
            main_run(win_vec[wid], {wid, static_cast<int>(_num_threads), &cpu_info()});
        });
    }
}

void ACLScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    custom_schedule(kernel, hints, kernel->window(), tensors);
}

void ACLScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) {
    custom_schedule(kernel, hints, window, tensors);
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads) {
    arm_compute::lock_guard<arm_compute::Mutex> lock(this->mtx);
    InferenceEngine::parallel_for(workloads.size(), [&](int wid) {
        workloads[wid]({wid, static_cast<int>(_num_threads), &cpu_info()});
    });
}

} // namespace intel_cpu
} // namespace ov