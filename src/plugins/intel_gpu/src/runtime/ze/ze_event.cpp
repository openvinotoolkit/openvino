// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_event.hpp"
#include "ze/ze_common.hpp"

#include <cassert>
#include <chrono>
#include <list>

using namespace cldnn;
using namespace ze;

void ze_event::reset() {
    event::reset();
    OV_ZE_EXPECT(zeEventHostReset(m_event));
}

void ze_event::wait_impl() {
    OV_ZE_EXPECT(zeEventHostSynchronize(m_event, default_timeout));
}

void ze_event::set_impl() {
    OV_ZE_EXPECT(zeEventHostSignal(m_event));
}

bool ze_event::is_set_impl() {
    auto ret = zeEventQueryStatus(m_event);
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

std::optional<ze_kernel_timestamp_result_t> ze_event::query_timestamp() {
    if (!m_factory.is_profiling_enabled()) {
        return std::nullopt;
    }
    ze_kernel_timestamp_result_t timestamp{};
    OV_ZE_EXPECT(zeEventQueryKernelTimestamp(m_event, &timestamp));
    return timestamp;
}

ze_event_handle_t ze_event::get_handle() const {
    return m_event;
}

bool ze_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    auto opt_timestamp = query_timestamp();
    if (!opt_timestamp.has_value()) {
        return true;
    }
    ze_kernel_timestamp_result_t timestamp = opt_timestamp.value();
    auto &dev_info = m_factory.get_engine().get_device_info();
    auto wallclock_time = timestamp_to_duration(dev_info, timestamp.global);
    auto exec_time = timestamp_to_duration(dev_info, timestamp.context);

    auto period_exec = std::make_shared<instrumentation::profiling_period_basic>(timestamp_to_duration(dev_info, timestamp.context));
    auto period_submit = std::make_shared<instrumentation::profiling_period_basic>(wallclock_time - exec_time);

    info.push_back({ instrumentation::profiling_stage::executing, period_exec });
    info.push_back({ instrumentation::profiling_stage::submission, period_submit });

    return true;
}

ze_event::~ze_event() {
    OV_ZE_WARN(zeEventDestroy(m_event));
}
