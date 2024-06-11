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
class DeviceMonitor {
public:
    DeviceMonitor(const std::shared_ptr<ov::util::monitor::PerformanceCounter>& PerformanceCounter,
                  unsigned historySize = 1);
    ~DeviceMonitor();
    void setHistorySize(std::size_t size);
    std::size_t getHistorySize() const;
    void collectData();
    std::deque<std::vector<double>> getLastHistory() const;
    std::vector<double> getMeanDeviceLoad() const;

private:
    unsigned samplesNumber;
    unsigned historySize;
    std::vector<double> deviceLoadSum;
    std::deque<std::vector<double>> deviceLoadHistory;
    const std::shared_ptr<ov::util::monitor::PerformanceCounter> performanceCounter;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov