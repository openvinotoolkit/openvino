// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event_factory.hpp"
#include "ze_event_pool.hpp"

namespace cldnn {
namespace ze {

// Interface for creating l0 events using event pools
struct ze_event_factory : public ze_base_event_factory {
public:
    ze_event_factory(const ze_engine &engine, bool enable_profiling, uint32_t capacity = 255);
    event::ptr create_event(uint64_t queue_stamp) override;
protected:
    std::shared_ptr<ze_event_pool> m_current_pool;
    const uint32_t m_capacity;
    uint32_t m_num_used;
};
}  // namespace ze
}  // namespace cldnn
