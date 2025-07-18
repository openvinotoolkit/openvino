// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_device_monitor_impl.hpp"

namespace ov {
namespace util {
CPUDeviceMonitorImpl::CPUDeviceMonitorImpl() {
    int n_cores = query_number_of_cores();
    if (n_cores == 0) {
        m_core_time_counters.resize(1);
        std::filesystem::path full_counter_path{L"\\Processor Information(_Total)\\% Processor Utility"};
        m_query.pdh_add_counterW(full_counter_path, 0, &m_core_time_counters[0]);
        m_query.pdh_set_counter_scale_factor(m_core_time_counters[0], -2);  // scale counter to [0, 1]
    } else {
        m_core_time_counters.resize(n_cores);
        for (std::size_t i = 0; i < n_cores; ++i) {
            std::filesystem::path full_counter_path{L"\\Processor Information(0," + std::to_wstring(i) +
                                                    L")\\% Processor Utility"};
            m_query.pdh_add_counterW(full_counter_path, 0, &m_core_time_counters[i]);
            m_query.pdh_set_counter_scale_factor(m_core_time_counters[i], -2);  // scale counter to [0, 1]
        }
    }
    m_query.pdh_collect_query_data();
}

std::map<std::string, float> CPUDeviceMonitorImpl::get_utilization() {
    auto ts = std::chrono::system_clock::now();
    if (ts > m_last_time_stamp) {
        auto delta =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - m_last_time_stamp);
        if (delta.count() < m_monitor_duration) {
            std::this_thread::sleep_for(std::chrono::milliseconds(m_monitor_duration - delta.count()));
        }
    }
    m_last_time_stamp = std::chrono::system_clock::now();
    auto ret = m_query.pdh_collect_query_data();
    if (!ret) {
        return {};
    }
    PDH_FMT_COUNTERVALUE display_value;
    std::vector<float> cpus_load(m_core_time_counters.size());
    for (std::size_t i = 0; i < m_core_time_counters.size(); ++i) {
        auto ret =
            m_query.pdh_get_formatted_counter_value(m_core_time_counters[i], PDH_FMT_DOUBLE, NULL, &display_value);
        if (ret != ERROR_SUCCESS)
            throw std::runtime_error("PdhGetFormattedCounterValue() failed. Error status: " + std::to_string(ret));
        if (PDH_CSTATUS_VALID_DATA != display_value.CStatus && PDH_CSTATUS_NEW_DATA != display_value.CStatus) {
            throw std::runtime_error("Error in counter data");
        }

        cpus_load[i] = static_cast<float>(display_value.doubleValue * 100.0f);
    }
    std::map<std::string, float> cpus_utilization;
    if (cpus_load.size() == 1) {
        cpus_utilization["Total"] = cpus_load.at(0);
        return cpus_utilization;
    }
    for (int index = 0; index < cpus_load.size(); index++) {
        cpus_utilization[std::to_string(index)] = cpus_load.at(index);
    }
    return cpus_utilization;
}
}  // namespace util
}  // namespace ov