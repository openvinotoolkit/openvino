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

    DeviceMonitor(unsigned historySize = 1);
    ~DeviceMonitor();
    void setHistorySize(std::size_t size);
    std::size_t getHistorySize() const;
    void collectData();
    void collectData(const std::string& deviceName);
    std::deque<std::map<std::string, double>> getLastHistory() const;
    std::map<std::string, double> getMeanDeviceLoad() const;
    std::map<std::string, double> getMeanDeviceLoad(const std::string& deviceName);

private:
    unsigned samplesNumber;
    unsigned historySize;
    std::map<std::string, double> deviceLoadSum;
    const std::vector<std::string> supportedDevices = {"CPU", "GPU"};
    std::deque<std::map<std::string, double>> deviceLoadHistory;
    const std::shared_ptr<ov::util::monitor::PerformanceCounter> performanceCounter;
    std::map<std::string, std::shared_ptr<ov::util::monitor::PerformanceCounter>> devicesPerformanceCounters;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov