// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_ie_scheduler.hpp"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include <ie_parallel.hpp>
#include "openvino/core/except.hpp"

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
    const unsigned int num_iterations = max_window.num_iterations_total();
    _num_threads = std::min(num_iterations, this->num_threads());

    if (num_iterations == 0) {
        return;
    }

    if (!kernel->is_parallelisable() || _num_threads == 1) {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run_op(tensors, max_window, info);
    } else {
        unsigned int num_windows = 0;
        const auto hints_split_dimension = hints.split_dimension();
        switch (hints.strategy()) {
            case arm_compute::IScheduler::StrategyHint::STATIC: {
                num_windows = _num_threads;
            }  break;
            case arm_compute::IScheduler::StrategyHint::DYNAMIC: {
                const unsigned int granule_threshold = (hints.threshold() <= 0) ? _num_threads : hints.threshold();
                num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
            } break;
            default: {
                OPENVINO_ASSERT(!"Unknown hint strategy");
            }
        }

        InferenceEngine::parallel_for(num_windows, [&](int wid) {
            const auto win = max_window.split_window(hints_split_dimension, wid, num_windows);
            const auto final_win = win.collapse(win, 0);
            final_win.validate();
            kernel->run_op(tensors, final_win, {wid, static_cast<int>(_num_threads), &cpu_info()});
        });
    }
}

void ACLScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
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