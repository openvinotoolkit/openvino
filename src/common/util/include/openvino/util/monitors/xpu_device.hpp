// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/util/monitors/idevice.hpp"

namespace ov {
namespace util {
class XPUDevice : public IDevice {
    // This class is used to monitor GPU/NPU performance data.
    // It uses the PerformanceImpl class to get the actual performance data.
    // The user only needs to call the get_utilization() method to get the performance data.
public:
    XPUDevice(const std::string& device_luid);
    virtual ~XPUDevice() = default;
    std::map<std::string, double> get_utilization() override;

private:
    std::string m_device_luid;
    class PerformanceImpl;
    std::shared_ptr<PerformanceImpl> m_perf_impl = nullptr;
};
}  // namespace util
}  // namespace ov