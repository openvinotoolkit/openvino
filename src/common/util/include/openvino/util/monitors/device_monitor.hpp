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
    DeviceMonitor(std::size_t history_size = 1);
    virtual ~DeviceMonitor() = default;
    void set_history_size(std::size_t size);
    std::size_t get_history_size() const;
    void collect_data(const std::string& deviceName);
    std::deque<std::map<std::string, double>> get_last_history() const;
    std::map<std::string, double> get_mean_device_load(const std::string& deviceName);

private:
    std::size_t sample_number;
    std::size_t history_size;
    std::map<std::string, double> deviceLoad_sum;
    const std::vector<std::string> supported_devices = {"CPU", "GPU"};
    std::deque<std::map<std::string, double>> deviceLoad_history;
    const std::shared_ptr<ov::util::monitor::PerformanceCounter> performance_counter;
    std::map<std::string, std::shared_ptr<ov::util::monitor::PerformanceCounter>> devices_performance_counters;
};
}  // namespace monitor
}  // namespace util
}  // namespace ov