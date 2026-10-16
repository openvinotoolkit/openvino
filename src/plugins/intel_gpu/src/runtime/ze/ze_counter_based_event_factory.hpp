// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event_factory.hpp"

#include <mutex>

namespace cldnn {
namespace ze {

// Interface for creating ze counter based events
// Should only be used with in-order queue
struct ze_counter_based_event_factory : public ze_base_event_factory {
public:
    ze_counter_based_event_factory(const ze_engine &ze_engine, bool enable_profiling);
    std::shared_ptr<ze_base_event> create_event(uint64_t queue_stamp) override;
protected:
    std::mutex _mutex;
};
}  // namespace ze
}  // namespace cldnn
