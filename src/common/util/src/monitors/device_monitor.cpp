// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/device_monitor.hpp"

#include <algorithm>
#include <iostream>

namespace ov {
namespace util {
namespace monitor {
DeviceMonitor::DeviceMonitor(const std::shared_ptr<ov::util::monitor::PerformanceCounter>& performanceCounter,
                             unsigned historySize)
    : samplesNumber{0},
      historySize{historySize > 0 ? historySize : 1},
      performanceCounter{performanceCounter} {
    while (deviceLoadHistory.size() < historySize)
        collectData();
}

DeviceMonitor::~DeviceMonitor() = default;

void DeviceMonitor::setHistorySize(std::size_t size) {
    historySize = size > 0 ? size : 1;
    std::ptrdiff_t newSize = static_cast<std::ptrdiff_t>(std::min(size, deviceLoadHistory.size()));
    deviceLoadHistory.erase(deviceLoadHistory.begin(), deviceLoadHistory.end() - newSize);
}

void DeviceMonitor::collectData() {
    samplesNumber = 0;
    deviceLoadHistory.clear();
    while (deviceLoadHistory.size() < historySize) {
        std::vector<double> deviceLoad = performanceCounter->getLoad();
        if (deviceLoadSum.empty())
            deviceLoadSum.resize(deviceLoad.size(), 0.0);
        if (!deviceLoad.empty()) {
            for (std::size_t i = 0; i < deviceLoad.size(); ++i) {
                if (deviceLoadHistory.size() == 0)
                    deviceLoadSum[i] = 0.0;
                if (historySize > 1)
                    deviceLoadSum[i] += deviceLoad[i];
                else
                    deviceLoadSum[i] = deviceLoad[i];
            }
            ++samplesNumber;
            deviceLoadHistory.push_back(std::move(deviceLoad));
        }
    }
}

std::size_t DeviceMonitor::getHistorySize() const {
    return historySize;
}

std::deque<std::vector<double>> DeviceMonitor::getLastHistory() const {
    return deviceLoadHistory;
}

std::vector<double> DeviceMonitor::getMeanDeviceLoad() const {
    std::vector<double> meanDeviceLoad;
    meanDeviceLoad.reserve(deviceLoadSum.size());
    for (double coreLoad : deviceLoadSum) {
        meanDeviceLoad.push_back(samplesNumber ? coreLoad / samplesNumber : 0);
    }
    return meanDeviceLoad;
}
}  // namespace monitor
}  // namespace util
}  // namespace ov
