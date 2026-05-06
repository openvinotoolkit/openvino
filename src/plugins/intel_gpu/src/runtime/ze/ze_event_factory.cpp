// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_event_factory.hpp"
#include "ze_common.hpp"
#include "ze_event.hpp"

#include "compute_runtime/zex_event.h"

using namespace cldnn;
using namespace ze;

ze_event_factory::ze_event_factory(const ze_engine &engine, bool enable_profiling, uint32_t capacity)
: ze_base_event_factory(engine, enable_profiling)
, m_capacity(capacity)
, m_num_used(0) { }

event::ptr ze_event_factory::create_event(uint64_t queue_stamp) {
    std::lock_guard<std::mutex> lock(_mutex);

    if (m_num_used >= m_capacity || m_current_pool.is_empty()) {
        m_num_used = 0;
        ze_event_pool_flags_t flags = is_profiling_enabled() ? ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP : 0;
        flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
        ze_event_pool_desc_t event_pool_desc = {
            ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
            nullptr,
            flags,
            m_capacity
        };
        auto device = m_engine.get_device();
        auto ctx_holder = m_engine.get_context_holder();
        ze_event_pool_handle_t event_pool;
        OV_ZE_EXPECT(ze::zeEventPoolCreate(ctx_holder.get_handle(), &event_pool_desc, 1, &device, &event_pool));
        m_current_pool = ze_holder<ze_resource_type::event_pool>(event_pool, ctx_holder);
    }

    ze_event_handle_t event;
    ze_event_desc_t event_desc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        m_num_used++,
        ZE_EVENT_SCOPE_FLAG_HOST,
        0
    };
    OV_ZE_EXPECT(ze::zeEventCreate(m_current_pool.get_handle(), &event_desc, &event));
    auto event_holder = ze_holder<ze_resource_type::event>(event, m_current_pool);

    return std::make_shared<ze_event>(queue_stamp, *this, event, m_current_pool);
}
