// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/performance_counter.hpp"

namespace ov {
namespace util {
namespace monitor {
class CpuPerformanceCounter : public PerformanceCounter {
public:
    CpuPerformanceCounter(int n_cores = 0);
    virtual ~CpuPerformanceCounter() = default;
    std::map<std::string, double> get_utilization() override;

private:
    int n_cores;
    class PerformanceCounterImpl;
    std::shared_ptr<PerformanceCounterImpl> performance_counter = nullptr;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov