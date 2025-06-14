// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_event_pool.hpp"
#include "ze_event.hpp"
#include "ze_common.hpp"

namespace cldnn {
namespace ze {

ze_event_pool::ze_event_pool(const ze_engine& engine, uint32_t capacity, ze_event_pool_flags_t flags)
    : m_engine(engine) {
    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        flags,
        capacity
    };
    auto device = engine.get_device();
    ZE_CHECK(zeEventPoolCreate(engine.get_context(), &event_pool_desc, 1, &device, &m_handle));
}

ze_event_pool::~ze_event_pool() {
    zeEventPoolDestroy(m_handle);
}

ze_events_pool::ze_events_pool(const ze_engine& engine, bool enable_profiling)
    : m_engine(engine)
    , m_enable_profiling(enable_profiling) { }

std::shared_ptr<ze_event> ze_events_pool::create_event(uint64_t queue_stamp) {
    if (m_num_used >= m_capacity || !m_current_pool) {
        m_num_used = 0;
        ze_event_pool_flags_t flags = m_enable_profiling ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
        m_current_pool = std::make_shared<ze_event_pool>(m_engine, m_capacity, flags);
    }

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        m_num_used++,              // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_HOST   // no additional memory/cache coherency required on wait
    };
    ZE_CHECK(zeEventCreate(m_current_pool->m_handle, &event_desc, &event));

    return std::make_shared<ze_event>(m_current_pool, event, queue_stamp);
}

std::shared_ptr<ze_event> ze_events_pool::create_user_event() {
    if (m_num_used_user >= m_capacity || !m_current_user_pool) {
        m_num_used_user = 0;
        ze_event_pool_flags_t flags = m_enable_profiling ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
        flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        m_current_user_pool = std::make_shared<ze_event_pool>(m_engine, m_capacity, flags);
    }

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        m_num_used_user++,         // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_HOST   // no additional memory/cache coherency required on wait
    };
    ZE_CHECK(zeEventCreate(m_current_user_pool->m_handle, &event_desc, &event));

    return std::make_shared<ze_event>(m_current_user_pool, event);
}

}  // namespace ze
}  // namespace cldnn
