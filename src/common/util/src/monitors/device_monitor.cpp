// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/device_monitor.hpp"

#include <algorithm>
#include <iostream>

#include "openvino/util/monitors/cpu_performance_counter.hpp"
#include "openvino/util/monitors/gpu_performance_counter.hpp"

namespace ov {
namespace util {
namespace monitor {

DeviceMonitor::DeviceMonitor(std::size_t history_size)
    : sample_number{0},
      history_size{history_size > 0 ? history_size : 1} {}

void DeviceMonitor::set_history_size(std::size_t size) {
    history_size = size > 0 ? size : 1;
    std::ptrdiff_t newSize = static_cast<std::ptrdiff_t>(std::min(size, deviceLoad_history.size()));
    deviceLoad_history.erase(deviceLoad_history.begin(), deviceLoad_history.end() - newSize);
}

void DeviceMonitor::collect_data(const std::string& deviceName) {
    sample_number = 0;
    deviceLoad_history.clear();
    deviceLoad_sum.clear();
    auto isSupportedDev =
        std::find(supported_devices.begin(), supported_devices.end(), deviceName) != supported_devices.end();
    if (!isSupportedDev) {
        return;
    }
    if (devices_performance_counters.count(deviceName) == 0) {
        if (deviceName == "CPU")
            devices_performance_counters[deviceName] = std::make_shared<ov::util::monitor::CpuPerformanceCounter>();
        if (deviceName == "GPU")
            devices_performance_counters[deviceName] = std::make_shared<ov::util::monitor::GpuPerformanceCounter>();
    }
    while (deviceLoad_history.size() < history_size) {
        auto devicesLoad = devices_performance_counters[deviceName]->get_load();
        if (!devicesLoad.empty()) {
            for (auto item : devicesLoad) {
                if (deviceLoad_history.size() == 0)
                    deviceLoad_sum[item.first] = 0.0;
                if (history_size > 1)
                    deviceLoad_sum[item.first] += devicesLoad[item.first];
                else
                    deviceLoad_sum[item.first] = devicesLoad[item.first];
            }
            ++sample_number;
            deviceLoad_history.push_back(std::move(devicesLoad));
        }
    }
}

std::size_t DeviceMonitor::get_history_size() const {
    return history_size;
}

std::deque<std::map<std::string, double>> DeviceMonitor::get_last_history() const {
    return deviceLoad_history;
}

std::map<std::string, double> DeviceMonitor::get_mean_device_load(const std::string& deviceName) {
    collect_data(deviceName);
    std::map<std::string, double> meanDeviceLoad;
    for (auto item : deviceLoad_sum) {
        meanDeviceLoad[item.first] = (sample_number ? item.second / sample_number : 0) * 100;
    }
    return meanDeviceLoad;
}
}  // namespace monitor
}  // namespace util
}  // namespace ov
