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

DeviceMonitor::DeviceMonitor(unsigned historySize) : samplesNumber{0}, historySize{historySize > 0 ? historySize : 1} {
    for (auto& device : supportedDevices) {
        if (device == "CPU")
            devicesPerformanceCounters[device] = std::make_shared<ov::util::monitor::CpuPerformanceCounter>();
        if (device == "GPU")
            devicesPerformanceCounters[device] = std::make_shared<ov::util::monitor::GpuPerformanceCounter>();
        collect_data(device);
    }
}
DeviceMonitor::DeviceMonitor(const std::shared_ptr<ov::util::monitor::PerformanceCounter>& performanceCounter,
                             unsigned historySize)
    : samplesNumber{0},
      historySize{historySize > 0 ? historySize : 1},
      performanceCounter{performanceCounter} {
    while (deviceLoadHistory.size() < historySize)
        collect_data();
}

DeviceMonitor::~DeviceMonitor() = default;

void DeviceMonitor::set_history_size(std::size_t size) {
    historySize = size > 0 ? size : 1;
    std::ptrdiff_t newSize = static_cast<std::ptrdiff_t>(std::min(size, deviceLoadHistory.size()));
    deviceLoadHistory.erase(deviceLoadHistory.begin(), deviceLoadHistory.end() - newSize);
}

void DeviceMonitor::collect_data() {
    samplesNumber = 0;
    deviceLoadHistory.clear();
    while (deviceLoadHistory.size() < historySize) {
        auto devicesLoad = performanceCounter->get_load();
        if (!devicesLoad.empty()) {
            for (auto item : devicesLoad) {
                if (deviceLoadHistory.size() == 0)
                    deviceLoadSum[item.first] = 0.0;
                if (historySize > 1)
                    deviceLoadSum[item.first] += devicesLoad[item.first];
                else
                    deviceLoadSum[item.first] = devicesLoad[item.first];
            }
            ++samplesNumber;
            deviceLoadHistory.push_back(std::move(devicesLoad));
        }
    }
}

void DeviceMonitor::collect_data(const std::string& deviceName) {
    samplesNumber = 0;
    deviceLoadHistory.clear();
    deviceLoadSum.clear();
    if (devicesPerformanceCounters.count(deviceName) == 0)
        return;
    while (deviceLoadHistory.size() < historySize) {
        auto devicesLoad = devicesPerformanceCounters[deviceName]->get_load();
        if (!devicesLoad.empty()) {
            for (auto item : devicesLoad) {
                if (deviceLoadHistory.size() == 0)
                    deviceLoadSum[item.first] = 0.0;
                if (historySize > 1)
                    deviceLoadSum[item.first] += devicesLoad[item.first];
                else
                    deviceLoadSum[item.first] = devicesLoad[item.first];
            }
            ++samplesNumber;
            deviceLoadHistory.push_back(std::move(devicesLoad));
        }
    }
}

std::size_t DeviceMonitor::get_history_size() const {
    return historySize;
}

std::deque<std::map<std::string, double>> DeviceMonitor::get_last_history() const {
    return deviceLoadHistory;
}

std::map<std::string, double> DeviceMonitor::get_mean_device_load() const {
    std::map<std::string, double> meanDeviceLoad;
    for (auto item : deviceLoadSum) {
        meanDeviceLoad[item.first] = (samplesNumber ? item.second / samplesNumber : 0) * 100;
    }
    return meanDeviceLoad;
}

std::map<std::string, double> DeviceMonitor::get_mean_device_load(const std::string& deviceName) {
    collect_data(deviceName);
    std::map<std::string, double> meanDeviceLoad;
    for (auto item : deviceLoadSum) {
        meanDeviceLoad[item.first] = (samplesNumber ? item.second / samplesNumber : 0) * 100;
    }
    return meanDeviceLoad;
}
}  // namespace monitor
}  // namespace util
}  // namespace ov
