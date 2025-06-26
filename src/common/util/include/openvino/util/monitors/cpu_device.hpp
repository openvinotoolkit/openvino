// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/util/monitors/idevice.hpp"

namespace ov::util {
class CPUDevice : public IDevice {
    // This class is used to monitor CPU performance data.
    // It uses the PerformanceImpl class to get the actual performance data.
    // The user only needs to call the get_utilization() method to get the performance data.
public:
    CPUDevice();
    virtual ~CPUDevice() = default;
    std::map<std::string, float> get_utilization() override;

private:
    class PerformanceImpl;
    std::shared_ptr<PerformanceImpl> m_perf_impl = nullptr;
};
}  // namespace ov::util