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
class XPUDeviceMonitorImpl : public IDeviceMonitorImpl {
public:
    XPUDeviceMonitorImpl(const std::string& device_luid);
    void init_core_counters(const std::string& device_luid);
    std::vector<std::filesystem::path> expand_wild_card_path(const std::filesystem::path& wild_card_path);
    std::vector<PDH_HCOUNTER> add_counter(const std::vector<std::filesystem::path>& path_list);
    std::map<std::string, float> get_utilization() override;

private:
    QueryWrapper m_query;
    std::map<std::string, std::vector<std::vector<PDH_HCOUNTER>>> m_core_time_counters;
    std::chrono::time_point<std::chrono::system_clock> m_last_time_stamp = std::chrono::system_clock::now();
    std::string m_luid;
    static constexpr int m_monitor_duration = 500;
};
}  // namespace ov::util
