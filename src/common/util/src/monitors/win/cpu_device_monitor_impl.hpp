// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <map>

#include "openvino/util/idevice_monitor.hpp"

#define NOMINMAX
#include <pdh.h>
#include <pdhmsg.h>
#include <windows.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

#include "query_wrapper.hpp"

namespace ov::util {
class CPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    CPUDeviceMonitorImpl();
    constexpr int query_number_of_cores() {
        // Query the number of logical processors
        // Only focuses on the total usage, so we can use _Total counter
        return 0;
    }
    std::map<std::string, float> get_utilization() override;

private:
    QueryWrapper m_query;
    std::vector<PDH_HCOUNTER> m_core_time_counters;
    std::chrono::time_point<std::chrono::system_clock> m_last_time_stamp = std::chrono::system_clock::now();
    static constexpr int m_monitor_duration = 10;
};

}  // namespace ov::util
