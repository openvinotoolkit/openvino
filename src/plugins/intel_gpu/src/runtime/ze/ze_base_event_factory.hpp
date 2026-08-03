// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "ze_stream.hpp"

namespace cldnn {
namespace ze {

// Interface for creating Level Zero events
struct ze_base_event_factory {
public:
    ze_base_event_factory(const ze_stream &ze_stream)
    : _ze_stream(ze_stream) {}
    const ze_engine& get_engine() const { return _ze_stream.get_engine(); }
    bool is_profiling_enabled() const { return _ze_stream.is_profiling_enabled(); }

    virtual ~ze_base_event_factory() {}
    virtual event::ptr create_event(uint64_t queue_stamp) = 0;
protected:
    const ze_stream& _ze_stream;
};
}  // namespace ze
}  // namespace cldnn
