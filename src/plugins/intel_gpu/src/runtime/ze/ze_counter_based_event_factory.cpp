// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_counter_based_event_factory.hpp"
#include "ze_common.hpp"
#include "ze_counter_based_event.hpp"

#include "compute_runtime/zex_event.h"

#include <mutex>

using namespace cldnn;
using namespace ze;
namespace {
    std::once_flag counter_based_ev_init_flag;
    decltype(zexCounterBasedEventCreate2) *func_zexCounterBasedEventCreate2 = nullptr;
    void find_function_address(ze_driver_handle_t driver) {
        OV_ZE_EXPECT(ze_api->zeDriverGetExtensionFunctionAddress(driver,
                                                "zexCounterBasedEventCreate2",
                                                reinterpret_cast<void **>(&func_zexCounterBasedEventCreate2)));
    }
}

ze_counter_based_event_factory::ze_counter_based_event_factory(const ze_engine &engine, bool enable_profiling)
    : ze_base_event_factory(engine, enable_profiling) {
    std::call_once(counter_based_ev_init_flag, find_function_address, engine.get_driver());
}

event::ptr ze_counter_based_event_factory::create_event(uint64_t queue_stamp) {
    std::lock_guard<std::mutex> lock(_mutex);

    ze_event_handle_t event;
    auto desc = defaultIntelCounterBasedEventDesc;
    if (is_profiling_enabled()) {
        desc.flags |= ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP;
    }
    OV_ZE_EXPECT(func_zexCounterBasedEventCreate2(m_engine.get_context(), m_engine.get_device(), &desc, &event));
    auto cb_event = std::make_shared<ze_counter_based_event>(queue_stamp, *this, event);
    return cb_event;
}
