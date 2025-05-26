// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <deque>
#include <memory>
#include <vector>

#include "openvino/util/monitors/idevice.hpp"

namespace ov {
namespace util {
class CPUDevice : public IDevice {
    // This class is used to monitor CPU performance data.
    // It uses the PerformanceImpl class to get the actual performance data.
    // The user only needs to call the get_utilization() method to get the performance data.
public:
    CPUDevice(int n_cores = 0);
    virtual ~CPUDevice() = default;
    std::map<std::string, double> get_utilization() override;

private:
    int n_cores;
    class PerformanceImpl;
    std::shared_ptr<PerformanceImpl> m_perf_impl = nullptr;
};
}  // namespace util
}  // namespace ov