// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event_factory.hpp"
#include "ze_resource.hpp"

#include <mutex>

namespace cldnn {
namespace ze {

// Interface for creating ze events using event pools
struct ze_event_factory : public ze_base_event_factory {
public:
    ze_event_factory(const ze_engine &engine, bool enable_profiling, uint32_t capacity = 255);
    event::ptr create_event(uint64_t queue_stamp) override;
protected:
    std::mutex _mutex;
    ze_event_pool_resource m_current_pool;
    const uint32_t m_capacity;
    uint32_t m_num_used;
};
}  // namespace ze
}  // namespace cldnn
