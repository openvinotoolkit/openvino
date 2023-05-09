// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_parallel.hpp>
#include <ie_common.h>

#include "acl_ie_scheduler.hpp"

namespace ov {
namespace intel_cpu {

ACLScheduler::ACLScheduler() {}
ACLScheduler::~ACLScheduler() {}

void ACLScheduler::set_num_threads(unsigned int num_threads) {}

std::uint32_t ACLScheduler::num_threads() const {
    return parallel_get_max_threads();
}

void ACLScheduler::Schedule(arm_compute::ICPPKernel* kernel,
                            const arm_compute::IScheduler::Hints& hints,
                            const arm_compute::Window& max_window,
                            arm_compute::ITensorPack& tensors) {
    IE_ASSERT(kernel != nullptr);

    auto splitDimension = hints.split_dimension();

    if (splitDimension == arm_compute::IScheduler::split_dimensions_all) {
        splitDimension = (max_window.num_iterations(arm_compute::Window::DimX) > max_window.num_iterations(arm_compute::Window::DimY))
                        ? arm_compute::Window::DimX : arm_compute::Window::DimY;
    }

    const int num_iterations = max_window.num_iterations(splitDimension);
    const int num_threads    = std::min(num_iterations, parallel_get_num_threads());
    if (num_iterations == 0) {
        return;
    }
    if (!kernel->is_parallelisable()) {
        arm_compute::ThreadInfo info;
        info.cpu_info = &cpu_info();
        if (tensors.empty()) {
            kernel->run(max_window, info);
        } else {
            kernel->run_op(tensors, max_window, info);
        }
    } else {
        int num_windows = 0;
        switch (hints.strategy()) {
            case arm_compute::IScheduler::StrategyHint::STATIC: {
                num_windows = num_threads;
            }  break;
            case arm_compute::IScheduler::StrategyHint::DYNAMIC: {
                const int granule_threshold = (hints.threshold() <= 0) ? num_threads : hints.threshold();
                num_windows = num_iterations > granule_threshold ? granule_threshold : num_iterations;
            } break;
            default: {
                IE_ASSERT(!"Unknown strategy");
            }
        }
        InferenceEngine::parallel_for(num_windows, [&] (int workloadId) {
            arm_compute::ThreadInfo   info;
            info.cpu_info       = &cpu_info();
            info.num_threads    = parallel_get_num_threads();
            info.thread_id      = parallel_get_thread_num();
            auto win = max_window.split_window(splitDimension, workloadId, num_windows);
            win.validate();
            if (tensors.empty()) {
                kernel->run(win, info);
            } else {
                kernel->run_op(tensors, win, info);
            }
        });
    }
}

void ACLScheduler::schedule(arm_compute::ICPPKernel* kernel, const arm_compute::IScheduler::Hints& hints) {
    arm_compute::ITensorPack tensors;
    Schedule(kernel, hints, kernel->window(), tensors);
}

void ACLScheduler::schedule_op(arm_compute::ICPPKernel*  kernel,
                               const arm_compute::IScheduler::Hints&  hints,
                               const arm_compute::Window&             window,
                               arm_compute::ITensorPack&              tensors) {
    Schedule(kernel, hints, window, tensors);
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload>& workloads) {
    InferenceEngine::parallel_for(workloads.size(), [&] (int workloadId) {
        arm_compute::ThreadInfo   info;
        info.cpu_info       = &cpu_info();
        info.num_threads    = parallel_get_num_threads();
        info.thread_id      = parallel_get_thread_num();
        workloads[workloadId](info);
    });
}

}  //  namespace intel_cpu
}  //  namespace ov