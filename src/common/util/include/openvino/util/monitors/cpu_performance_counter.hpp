// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/cpu_performance_counter.hpp"
#include "openvino/util/monitors/performance_counter.hpp"

namespace ov {
namespace util {
namespace monitor {
class CpuPerformanceCounter : public PerformanceCounter {
public:
    CpuPerformanceCounter(int nCores = 0);
    virtual ~CpuPerformanceCounter() = default;
    std::map<std::string, double> get_load() override;

private:
    int nCores;
    class PerformanceCounterImpl;
    std::shared_ptr<PerformanceCounterImpl> performance_counter = nullptr;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov