// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_cb_event_manager.hpp"
#include "ze_common.hpp"
#include "ze_event.hpp"

#include "zex_event.h"

using namespace cldnn;
using namespace ze;
namespace {
    decltype(zexCounterBasedEventCreate2) *func_zexCounterBasedEventCreate2 = nullptr;
    void find_function_address(ze_driver_handle_t driver) {
        ZE_CHECK(zeDriverGetExtensionFunctionAddress(driver,
                                                "zexCounterBasedEventCreate2",
                                                reinterpret_cast<void **>(&func_zexCounterBasedEventCreate2)));
    }
}

ze_cb_event_manager::ze_cb_event_manager(const ze_engine &engine, ze_command_list_handle_t cmd_list, bool enable_profiling)
    : ze_event_manager(engine, cmd_list, enable_profiling) {
    if (func_zexCounterBasedEventCreate2 == nullptr) {
        find_function_address(engine.get_driver());
    }
}

ze_cb_event_manager::~ze_cb_event_manager() {}

std::shared_ptr<ze_event> ze_cb_event_manager::create_event(uint64_t queue_stamp) {
    ze_event_handle_t event;
    auto desc = defaultIntelCounterBasedEventDesc;
    if (m_enable_profiling) {
        desc.flags |= ZEX_COUNTER_BASED_EVENT_FLAG_KERNEL_TIMESTAMP;
    }
    ZE_CHECK(func_zexCounterBasedEventCreate2(m_engine.get_context(), m_engine.get_device(), &desc, &event));
    return std::make_shared<ze_event>(this, event, queue_stamp);
}

void ze_cb_event_manager::destroy_event(ze_event *event) {
    zeEventDestroy(event->get());
}
