// Copyright (C) 2019-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/monitors/xpu_device.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>

#include "openvino/util/monitors/idevice.hpp"

#ifdef _WIN32
#    define NOMINMAX
#    include <dxgi.h>
#    include <pdh.h>
#    include <pdhmsg.h>
#    include <windows.h>

#    include <chrono>
#    include <string>
#    include <system_error>
#    include <thread>

#    include "openvino/util/wstring_convert_util.hpp"
#    include "query_wrapper.hpp"
#    define RENDER_ENGINE_COUNTER_INDEX  0
#    define COMPUTE_ENGINE_COUNTER_INDEX 1
#    define MAX_COUNTER_INDEX            2

namespace ov {
namespace util {
class XPUDevice::PerformanceImpl {
public:
    PerformanceImpl(const std::string& device_luid) {
        m_luid = device_luid;
        if (!m_luid.empty()) {
            m_core_time_counters[m_luid] = {};
            m_core_time_counters[m_luid].resize(MAX_COUNTER_INDEX);
        }
        init_core_counters(device_luid);
    }

    void init_core_counters(const std::string& device_luid) {
        if (device_luid.empty() || device_luid.length() % 2 != 0)
            return;
        auto device_luid_low = device_luid.substr(0, 8);
        std::string luid_win;
        for (std::size_t i = 0; i < device_luid_low.length(); i += 2) {
            luid_win.insert(0, device_luid_low.substr(i, 2));
        }
        std::transform(luid_win.begin(), luid_win.end(), luid_win.begin(), ::toupper);
        std::string full_3d_counter_path =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_3D)\\Utilization Percentage");
        std::string full_compute_counter_path =
            std::string("\\GPU Engine(*_luid_*" + luid_win + "_phys*engtype_compute)\\Utilization Percentage");
        std::wstring full_3d_counter_path_w = ov::util::string_to_wstring(full_3d_counter_path);
        std::wstring full_compute_counter_path_w = ov::util::string_to_wstring(full_compute_counter_path);
        m_core_time_counters[m_luid][RENDER_ENGINE_COUNTER_INDEX] =
            add_counter(expand_wild_card_path(full_3d_counter_path_w.c_str()));
        m_core_time_counters[m_luid][COMPUTE_ENGINE_COUNTER_INDEX] =
            add_counter(expand_wild_card_path(full_compute_counter_path_w.c_str()));
        m_query.pdh_collect_query_data();
    }

    std::vector<std::wstring> expand_wild_card_path(LPCWSTR wild_card_path) {
        DWORD path_list_length = 0;
        DWORD path_list_length_buf_len;
        std::vector<std::wstring> path_list;
        auto ret = m_query.pdh_expand_wild_card_pathW(NULL, wild_card_path, NULL, &path_list_length, 0);
        if (!ret) {
            return path_list;
        }
        path_list_length_buf_len = path_list_length + 100;
        PZZWSTR expanded_path_list = (PZZWSTR)malloc(path_list_length_buf_len * sizeof(WCHAR));
        ret = m_query.pdh_expand_wild_card_pathW(NULL, wild_card_path, expanded_path_list, &path_list_length, 0);
        if (!ret) {
            free(expanded_path_list);
            return path_list;
        }
        for (size_t i = 0; i < path_list_length;) {
            std::wstring wpath(expanded_path_list + i);
            if (wpath.length() > 0) {
                path_list.push_back(wpath);
                i += wpath.length() + 1;
            } else {
                break;
            }
        }
        free(expanded_path_list);
        return path_list;
    }

    std::vector<PDH_HCOUNTER> add_counter(std::vector<std::wstring> path_list) {
        std::vector<PDH_HCOUNTER> counter_list;
        for (std::wstring path : path_list) {
            PDH_HCOUNTER counter;
            auto ret = m_query.pdh_add_counterW(path.c_str(), NULL, &counter);
            if (!ret) {
                return counter_list;
            }
            ret = m_query.pdh_set_counter_scale_factor(counter, -2);  // scale counter to [0, 1]
            if (!ret) {
                return counter_list;
            }
            counter_list.push_back(counter);
        }
        return counter_list;
    }

    std::map<std::string, float> get_utilization() {
        if (m_luid.empty())
            return {};
        auto ts = std::chrono::system_clock::now();
        if (ts > m_last_time_stamp) {
            auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() -
                                                                               m_last_time_stamp);
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
                    auto status =
                        m_query.pdh_get_formatted_counter_value(counter, PDH_FMT_DOUBLE, NULL, &display_value);
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

private:
    QueryWrapper m_query;
    std::map<std::string, std::vector<std::vector<PDH_HCOUNTER>>> m_core_time_counters;
    std::chrono::time_point<std::chrono::system_clock> m_last_time_stamp = std::chrono::system_clock::now();
    std::string m_luid;
    int m_monitor_duration = 500;
};

#elif defined(__linux__)
#    include <unistd.h>

#    include <chrono>
#    include <fstream>
#    include <regex>
#    include <utility>

namespace ov {
namespace util {
class XPUDevice::PerformanceImpl {
public:
    PerformanceImpl(const std::string& device_luid) {}

    std::map<std::string, float> get_utilization() {
        // TODO: Implement.
        return {};
    }
};

#else
namespace ov {
namespace util {
// not implemented
class XPUDevice::PerformanceImpl {
public:
    PerformanceImpl(const std::string& device_luid) {}
    std::map<std::string, float> get_utilization() {
        // TODO: Implement.
        return {};
    }
};
#endif
XPUDevice::XPUDevice(const std::string& device_luid) : IDevice("XPU"), m_device_luid(device_luid) {}
std::map<std::string, float> XPUDevice::get_utilization() {
    if (!m_perf_impl) {
        m_perf_impl = std::make_shared<PerformanceImpl>(m_device_luid);
    }
    return m_perf_impl->get_utilization();
}
}
}