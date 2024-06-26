// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/util/monitors/performance_counter.hpp"

namespace ov {
namespace util {
namespace monitor {
class GpuPerformanceCounter : public ov::util::monitor::PerformanceCounter {
public:
    GpuPerformanceCounter();
    virtual ~GpuPerformanceCounter() = default;
    std::map<std::string, double> get_load() override;

private:
    class PerformanceCounterImpl;
    std::shared_ptr<PerformanceCounterImpl> performance_counter = nullptr;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov