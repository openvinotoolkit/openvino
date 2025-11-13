// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event_factory.hpp"

namespace cldnn {
namespace ze {

// Interface for creating l0 counter based events
// Should only be used with in-order queue
struct ze_cb_event_factory : public ze_base_event_factory {
public:
    ze_cb_event_factory(const ze_engine &engine, bool enable_profiling);
    event::ptr create_event(uint64_t queue_stamp) override;
};
}  // namespace ze
}  // namespace cldnn
