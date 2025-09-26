// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_event_pool_manager.hpp"
#include "ze_common.hpp"
#include "ze_event.hpp"

#include "zex_event.h"

using namespace cldnn;
using namespace ze;

ze_event_pool_manager::ze_event_pool_manager(const ze_engine &engine, bool enable_profiling, uint32_t capacity)
: ze_event_manager(engine, enable_profiling)
, m_current_pool(nullptr)
, m_capacity(capacity)
, m_num_used(0) {}

ze_event_pool_manager::~ze_event_pool_manager() {}

std::shared_ptr<ze_event> ze_event_pool_manager::create_event(uint64_t queue_stamp) {
    if (m_num_used >= m_capacity || !m_current_pool) {
        m_num_used = 0;
        ze_event_pool_flags_t flags = m_enable_profiling ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
        flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        m_current_pool = std::make_shared<ze_event_pool>(m_engine, m_capacity, flags);
    }

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        m_num_used++,
        ZE_EVENT_SCOPE_FLAG_HOST,
        0
    };
    ZE_CHECK(zeEventCreate(m_current_pool->m_handle, &event_desc, &event));

    return std::make_shared<ze_event>(this, event, queue_stamp, m_current_pool);
}

void ze_event_pool_manager::destroy_event(ze_event *event) {
    zeEventDestroy(event->get());
}
