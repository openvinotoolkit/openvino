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
}  // namespace ze
}  // namespace cldnn
