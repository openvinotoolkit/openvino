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
    void set_history_size(std::size_t size);
    std::size_t get_history_size() const;
    void collect_data();
    void collect_data(const std::string& deviceName);
    std::deque<std::map<std::string, double>> get_last_history() const;
    std::map<std::string, double> get_mean_device_load() const;
    std::map<std::string, double> get_mean_device_load(const std::string& deviceName);

private:
    unsigned samplesNumber;
    std::size_t historySize;
    std::map<std::string, double> deviceLoadSum;
    const std::vector<std::string> supportedDevices = {"CPU", "GPU"};
    std::deque<std::map<std::string, double>> deviceLoadHistory;
    const std::shared_ptr<ov::util::monitor::PerformanceCounter> performance_counter;
    std::map<std::string, std::shared_ptr<ov::util::monitor::PerformanceCounter>> devicesPerformanceCounters;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov