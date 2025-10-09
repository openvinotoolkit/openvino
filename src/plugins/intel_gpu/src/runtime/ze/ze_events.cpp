// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_events.hpp"
#include "ze_common.hpp"

#include <cassert>
#include <chrono>
#include <list>

using namespace cldnn;
using namespace ze;

void ze_events::wait_impl() {
    if (m_last_event) {
        ZE_CHECK(zeEventHostSynchronize(m_last_event, default_timeout));
    }
}

void ze_events::set_impl() {
    // Call wait_impl to be in line with ocl_events
    wait_impl();
}

bool ze_events::is_set_impl() {
    if (!m_last_event) {
        return true;
    }

    auto ret = zeEventQueryStatus(m_last_event);
    switch (ret) {
    case ZE_RESULT_SUCCESS:
        return true;
        break;
    case ZE_RESULT_NOT_READY:
        return false;
        break;
    default:
        OPENVINO_THROW("[GPU] Query event returned unexpected value: ", std::to_string(ret));
        break;
    }
}

bool ze_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    // The goal is to sum up all disjoint durations of its projection on the time axis
    std::vector<ze_kernel_timestamp_data_t> all_global_timestamps;
    std::vector<ze_kernel_timestamp_data_t> all_context_timestamps;

    auto add_or_merge = [](std::vector<ze_kernel_timestamp_data_t>& all_timestamps, const ze_kernel_timestamp_data_t& ts) {
        auto it = all_timestamps.begin();
        bool merged = false;
        auto target_timestamp = ts;
        while (it != all_timestamps.end()) {
            auto& timestamp = *it;
            bool disjoint = timestamp.kernelEnd < target_timestamp.kernelStart || timestamp.kernelStart > target_timestamp.kernelEnd;
            bool equal = timestamp.kernelEnd == target_timestamp.kernelEnd && timestamp.kernelStart == target_timestamp.kernelStart;
            if (!disjoint) {
                if (equal) {
                    if (!merged) {
                        merged = true;
                        break;
                    } else {
                        it = all_timestamps.erase(it);
                    }
                } else {
                    if (!merged) {
                        timestamp.kernelStart = std::min(timestamp.kernelStart, target_timestamp.kernelStart);
                        timestamp.kernelEnd = std::max(timestamp.kernelEnd, target_timestamp.kernelEnd);
                        target_timestamp = timestamp;
                        merged = true;
                        it++;
                    } else {
                        if (timestamp.kernelEnd > target_timestamp.kernelEnd) {
                            it--;
                            it->kernelEnd = target_timestamp.kernelEnd;
                            it++;
                        }
                        it = all_timestamps.erase(it);
                    }
                }
            } else {
                it++;
            }
        }

        if (!merged) {
            all_timestamps.push_back(target_timestamp);
        }
    };

    if (m_events.empty())
        return false;

    auto device_info = m_engine.get_device_info();

    auto get_total_exec_time = [&device_info](std::vector<ze_kernel_timestamp_data_t>& all_timestamps) {
        std::chrono::nanoseconds total_time{0};
        for (const auto& ts : all_timestamps) {
            total_time += timestamp_to_duration(device_info, ts);
        }

        return total_time;
    };

    // Submission time is calculated as difference between merged context and wallclock intervals
    // May probably be more accurate if we sum all sub-intervals of wallclock timestamps not covered by execution intervals
    using intervals_t = std::vector<ze_kernel_timestamp_data_t>;
    auto get_submission_time = [&device_info](const intervals_t& s_timestamps,
                                              const intervals_t& e_timestamps) {
        auto get_minmax = [](const intervals_t& timestamps) {
            uint64_t min_val = std::min(timestamps.begin(), timestamps.end(),
                [](const intervals_t::const_iterator& lhs, const intervals_t::const_iterator& rhs) {
                    return lhs->kernelStart < rhs->kernelStart;
            })->kernelStart;
            uint64_t max_val = std::max(timestamps.begin(), timestamps.end(),
                [](const intervals_t::const_iterator& lhs, const intervals_t::const_iterator& rhs) {
                    return lhs->kernelEnd < rhs->kernelEnd;
            })->kernelEnd;

            return ze_kernel_timestamp_data_t{min_val, max_val};
        };

        auto submission_interval = get_minmax(s_timestamps);
        auto exec_interval = get_minmax(e_timestamps);

        auto wallclock_time = timestamp_to_duration(device_info, submission_interval);
        auto exec_time = timestamp_to_duration(device_info, exec_interval);

        return wallclock_time - exec_time;
    };

    for (size_t i = 0; i < m_events.size(); i++) {
        auto be = downcast<ze_base_event>(m_events[i].get());
        auto opt_timestamp = be->query_timestamp();
        if (!opt_timestamp.has_value()) {
            continue;
        }
        ze_kernel_timestamp_result_t timestamp = opt_timestamp.value();

        add_or_merge(all_global_timestamps, timestamp.global);
        add_or_merge(all_context_timestamps, timestamp.context);
    }

    auto submit_time = get_submission_time(all_global_timestamps, all_context_timestamps);
    auto exec_time = get_total_exec_time(all_context_timestamps);

    auto period_exec = std::make_shared<instrumentation::profiling_period_basic>(exec_time);
    auto period_submit = std::make_shared<instrumentation::profiling_period_basic>(submit_time);

    info.push_back({ instrumentation::profiling_stage::executing, period_exec });
    info.push_back({ instrumentation::profiling_stage::submission, period_submit });

    return true;
}