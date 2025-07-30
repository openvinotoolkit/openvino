// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_ie_scheduler.hpp"

#include <arm_compute/core/CPP/CPPTypes.h>
#include <arm_compute/core/ITensorPack.h>
#include <arm_compute/runtime/IScheduler.h>

#include <algorithm>
#include <functional>
#include <vector>

#include "arm_compute/core/CPP/ICPPKernel.h"
#include "openvino/core/parallel.hpp"
#include "src/runtime/SchedulerUtils.h"

namespace ov::intel_cpu {

using namespace arm_compute;

ACLScheduler::ACLScheduler() = default;

unsigned int ACLScheduler::num_threads() const {
    return parallel_get_num_threads();
}

void ACLScheduler::set_num_threads(unsigned int num_threads) {}

void ACLScheduler::schedule_custom(ICPPKernel* kernel, const Hints& hints, const Window& window, ITensorPack& tensors) {
    auto get_num_threads = [](unsigned int num_iterations) {
#if OV_THREAD == OV_THREAD_OMP
        // In OpenMP case parallel_get_num_threads() method returns 1 here because it's called outside parallel section
        // This is the reason why this method isn't used to initialize _num_threads
        return num_iterations;
#else
        return std::min(num_iterations, static_cast<unsigned int>(parallel_get_num_threads()));
#endif
    };

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

    const Window& max_window = window;
    if (hints.split_dimension() == IScheduler::split_dimensions_all) {
        const unsigned int m = max_window.num_iterations(Window::DimX);
        const unsigned int n = max_window.num_iterations(Window::DimY);
        const unsigned int num_iterations = m * n;
        const auto num_threads = get_num_threads(num_iterations);

        auto [m_threads, n_threads] = scheduler_utils::split_2d(num_threads, m, n);

        unsigned int max_parallelism = std::min<unsigned int>(m, m_threads) * std::min<unsigned int>(n, n_threads);
        if (max_parallelism < num_threads) {
            m_threads = std::min<unsigned int>(m, m_threads);
            n_threads = std::min<unsigned int>(n, n_threads);
        }

        ov::parallel_for(m_threads * n_threads, [&, m_threads = m_threads, n_threads = n_threads](int wid) {
            int mi = wid / n_threads;
            int ni = wid % n_threads;
            Window win = max_window.split_window(Window::DimX, mi, m_threads).split_window(Window::DimY, ni, n_threads);
            win.validate();
            main_run(win, {wid, static_cast<int>(num_threads), &cpu_info()});
        });
    } else {
        const unsigned int num_iterations = max_window.num_iterations(hints.split_dimension());
        const auto _num_threads = get_num_threads(num_iterations);

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
        workloads[wid]({wid, parallel_get_num_threads(), &cpu_info()});
    });
}

}  // namespace ov::intel_cpu
