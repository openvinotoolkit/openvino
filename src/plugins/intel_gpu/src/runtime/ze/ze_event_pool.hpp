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

// Helper for events pool management
// Can hold multiple ze_event_pool objects and track their capacity with realloc when it's needed
struct ze_events_pool {
public:
    ze_events_pool(const ze_engine& engine, bool enable_profiling);

    std::shared_ptr<ze_event> create_event(uint64_t queue_stamp = 0);
    std::shared_ptr<ze_event> create_user_event();

protected:
    const ze_engine& m_engine;
    std::shared_ptr<ze_event_pool> m_current_user_pool = nullptr;
    std::shared_ptr<ze_event_pool> m_current_pool = nullptr;
    const uint32_t m_capacity = 100;
    uint32_t m_num_used = 0;
    uint32_t m_num_used_user = 0;
    const bool m_enable_profiling;
};

}  // namespace ze
}  // namespace cldnn
