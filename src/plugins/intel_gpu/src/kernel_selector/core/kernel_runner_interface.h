// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <chrono>

namespace kernel_selector {
class KernelRunnerInterface {
public:
    // Gets a list of kernels, executes them and returns the run time of each kernel (in nano-seconds).
    virtual std::vector<std::chrono::nanoseconds> run_kernels(const kernel_selector::KernelsData& kernelsData) = 0;

    virtual ~KernelRunnerInterface() = default;
};
}  // namespace kernel_selector
