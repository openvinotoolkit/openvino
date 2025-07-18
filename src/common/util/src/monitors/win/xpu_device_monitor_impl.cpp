// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xpu_device_monitor_impl.hpp"

constexpr int RENDER_ENGINE_COUNTER_INDEX = 0;
constexpr int COMPUTE_ENGINE_COUNTER_INDEX = 1;
constexpr int MAX_COUNTER_INDEX = 2;

namespace ov {
namespace util {
XPUDeviceMonitorImpl::XPUDeviceMonitorImpl(const std::string& device_luid) {
    m_luid = device_luid;
    if (!m_luid.empty()) {
        m_core_time_counters[m_luid] = {};
        m_core_time_counters[m_luid].resize(MAX_COUNTER_INDEX);
    }
    init_core_counters(device_luid);
}

void XPUDeviceMonitorImpl::init_core_counters(const std::string& device_luid) {
    if (device_luid.empty() || device_luid.length() % 2 != 0)
        return;
    auto device_luid_low = device_luid.substr(0, 8);
    std::string luid_win;
    for (std::size_t i = 0; i < device_luid_low.length(); i += 2) {
        luid_win.insert(0, device_luid_low.substr(i, 2));
    }
    std::transform(luid_win.begin(), luid_win.end(), luid_win.begin(), [](unsigned char c) {
        return std::toupper(c);
    });
    std::filesystem::path full_3d_counter_path =
        std::filesystem::path("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_3D)\\Utilization Percentage");
    std::filesystem::path full_compute_counter_path =
        std::filesystem::path("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_compute)\\Utilization Percentage");
    m_core_time_counters[m_luid][RENDER_ENGINE_COUNTER_INDEX] =
        add_counter(expand_wild_card_path(full_3d_counter_path));
    m_core_time_counters[m_luid][COMPUTE_ENGINE_COUNTER_INDEX] =
        add_counter(expand_wild_card_path(full_compute_counter_path));
    m_query.pdh_collect_query_data();
}

std::vector<std::filesystem::path> XPUDeviceMonitorImpl::expand_wild_card_path(
    const std::filesystem::path& wild_card_path) {
    DWORD path_list_length = 0;
    DWORD path_list_length_buf_len;
    std::vector<std::filesystem::path> path_list;
    auto ret = m_query.pdh_expand_wild_card_pathW(std::filesystem::path{}, wild_card_path, NULL, &path_list_length, 0);
    if (!ret) {
        return path_list;
    }
    path_list_length_buf_len = path_list_length + 100;
    std::vector<wchar_t> expanded_path_list(path_list_length_buf_len, 0);
    ret = m_query.pdh_expand_wild_card_pathW(std::filesystem::path{},
                                             wild_card_path,
                                             expanded_path_list.data(),
                                             &path_list_length,
                                             0);
    if (!ret) {
        return path_list;
    }
    for (size_t i = 0; i < path_list_length;) {
        std::wstring wpath(expanded_path_list.data() + i);
        if (wpath.length() > 0) {
            path_list.push_back(wpath);
            i += wpath.length() + 1;
        } else {
            break;
        }
    }
    return path_list;
}

std::vector<PDH_HCOUNTER> XPUDeviceMonitorImpl::add_counter(const std::vector<std::filesystem::path>& path_list) {
    std::vector<PDH_HCOUNTER> counter_list;
    for (const auto& path : path_list) {
        PDH_HCOUNTER counter;
        auto ret = m_query.pdh_add_counterW(path, NULL, &counter);
        if (!ret) {
            return counter_list;
        }
        ret = m_query.pdh_set_counter_scale_factor(counter, -2);
        if (!ret) {
            return counter_list;
        }
        counter_list.push_back(counter);
    }
    return counter_list;
}

std::map<std::string, float> XPUDeviceMonitorImpl::get_utilization() {
    if (m_luid.empty())
        return {};
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
    PDH_FMT_COUNTERVALUE display_value;
    std::map<std::string, float> utilization_map;
    for (auto item : m_core_time_counters) {
        float utilization = 0.0f;
        auto luid = item.first;
        auto core_counters = item.second;
        for (int counter_index = 0; counter_index < MAX_COUNTER_INDEX; counter_index++) {
            auto counters_list = core_counters[counter_index];
            for (auto counter : counters_list) {
                auto status = m_query.pdh_get_formatted_counter_value(counter, PDH_FMT_DOUBLE, NULL, &display_value);
                if (status != ERROR_SUCCESS) {
                    continue;
                }
                utilization += static_cast<float>(display_value.doubleValue);
            }
        }
        utilization_map[luid] = utilization * 100.0f;
    }
    return utilization_map;
}
}  // namespace util
}  // namespace ov
