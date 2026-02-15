// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <arm_compute/core/CPP/ICPPKernel.h>
#include <arm_compute/core/ITensorPack.h>
#include <arm_compute/runtime/Scheduler.h>

#include "support/Mutex.h"

namespace ov::intel_cpu {

using namespace arm_compute;

class ACLScheduler final : public IScheduler {
public:
    ACLScheduler();
    ~ACLScheduler() override = default;
    [[nodiscard]] std::uint32_t num_threads() const override;
    void set_num_threads(unsigned int num_threads) override;
    void schedule(ICPPKernel* kernel, const Hints& hints) override;
    void schedule_op(ICPPKernel* kernel, const Hints& hints, const Window& window, ITensorPack& tensors) override;

protected:
    void run_workloads(std::vector<Workload>& workloads) override;

private:
    void schedule_custom(ICPPKernel* kernel, const Hints& hints, const Window& window, ITensorPack& tensors);
};
}  // namespace ov::intel_cpu
