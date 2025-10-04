// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_engine.hpp"

namespace cldnn {
namespace ze {

struct ze_event;

// Wrapper for ze events pool which is needed to track lifetime of the pool.
// I.e. the object is destoyed if no ze_events alive which refer to this pool
// and ze_events_pool doesn't refer to it as well
struct ze_event_pool {
    ze_event_pool(const ze_engine& engine, uint32_t capacity, ze_event_pool_flags_t flags);
    ~ze_event_pool();
    using ptr = std::shared_ptr<ze_event_pool>;

    ze_event_pool_handle_t m_handle;
    const ze_engine& m_engine;
};

}  // namespace ze
}  // namespace cldnn

