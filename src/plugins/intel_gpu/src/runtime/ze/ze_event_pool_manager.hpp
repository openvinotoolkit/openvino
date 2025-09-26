// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_event_manager.hpp"
#include "ze_event_pool.hpp"

namespace cldnn {
namespace ze {

// Interface for creating and destroying l0 events using event pools
struct ze_event_pool_manager : public ze_event_manager {
public:
    ze_event_pool_manager(const ze_engine &engine, bool enable_profiling, uint32_t capacity = 255);
    ~ze_event_pool_manager();
    std::shared_ptr<ze_event> create_event(uint64_t queue_stamp) override;
    void destroy_event(ze_event *event) override;
protected:
    std::shared_ptr<ze_event_pool> m_current_pool;
    const uint32_t m_capacity;
    uint32_t m_num_used;
};
}  // namespace ze
}  // namespace cldnn
