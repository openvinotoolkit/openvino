// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_ie_scheduler.hpp"

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "arm_compute/core/Error.h"
#include "arm_compute/core/Helpers.h"
#include "arm_compute/core/Utils.h"
#include <ie_parallel.hpp>
#include <ie_common.h>

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

ACLScheduler::ACLScheduler() {}

unsigned int ACLScheduler::num_threads() const {
    return parallel_get_num_threads();
}

void ACLScheduler::set_num_threads(unsigned int num_threads) {}

void ACLScheduler::schedule(ICPPKernel *kernel, const Hints &hints) {
    ITensorPack tensors;
    schedule_common(kernel, hints, kernel->window(), tensors);
}

void ACLScheduler::schedule_op(ICPPKernel *kernel, const Hints &hints, const Window &window, ITensorPack &tensors) {
    ARM_COMPUTE_ERROR_ON_MSG(!kernel, "The child class didn't set the kernel");

    auto splitDimension = hints.split_dimension();

    if (splitDimension == arm_compute::IScheduler::split_dimensions_all) {
        splitDimension = (window.num_iterations(arm_compute::Window::DimX) > window.num_iterations(arm_compute::Window::DimY))
                         ? arm_compute::Window::DimX : arm_compute::Window::DimY;
    }

    const Window &max_window = window;
    const unsigned int num_iterations = max_window.num_iterations(splitDimension);
    const unsigned int num_threads = std::min(num_iterations, static_cast<uint>(parallel_get_num_threads()));

    if (num_iterations == 0) {
        return;
    }

    if (!kernel->is_parallelisable() || num_threads == 1) {
        ThreadInfo info;
        info.cpu_info = &cpu_info();
        kernel->run_op(tensors, max_window, info);
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
        std::vector<IScheduler::Workload> workloads(num_windows);
        for (unsigned int t = 0; t < num_windows; t++) {
            workloads[t] = [t, &splitDimension, &max_window, &num_windows, &kernel, &tensors](const ThreadInfo & info) {
                Window win = max_window.split_window(splitDimension, t, num_windows);
//                std::cout << "\n" + std::to_string(splitDimension) + " " + std::to_string(num_windows) << std::endl;
                win.validate();
                kernel->run_op(tensors, win, info);
            };
        }
        run_workloads(workloads);
    }
}

void ACLScheduler::run_workloads(std::vector<arm_compute::IScheduler::Workload> &workloads) {
    const unsigned int amount_of_work = static_cast<unsigned int>(workloads.size());
    const unsigned int num_threads_to_use = std::min(static_cast<uint>(parallel_get_num_threads()), amount_of_work);
//    std::cout << "thread" + std::to_string(num_threads_to_use) << std::endl;
    std::cout << parallel_get_num_threads() << std::endl;

    if (num_threads_to_use < 1) {
        return;
    }

    InferenceEngine::parallel_for(workloads.size(), [&] (int workloadId) {
        arm_compute::ThreadInfo   info;
        info.cpu_info       = &cpu_info();
        info.num_threads    = num_threads_to_use;
        info.thread_id      = parallel_get_thread_num();
        workloads[workloadId](info);
    });
}

} // namespace intel_cpu
} // namespace ov