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
class GpuPerformanceCounter : public ov::util::monitor::PerformanceCounter {
public:
    GpuPerformanceCounter();
    ~GpuPerformanceCounter();
    std::vector<double> getLoad() override;

private:
    class PerformanceCounterImpl;
    PerformanceCounterImpl* performanceCounter = NULL;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov