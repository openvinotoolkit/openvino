// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_engine.hpp"
#include "intel_gpu/runtime/event.hpp"

namespace cldnn {
namespace ze {

// Interface for creating Level Zero events
struct ze_base_event_factory {
public:
    ze_base_event_factory(const ze_engine &engine, bool enable_profiling)
    : m_engine(engine), m_profiling_enabled(enable_profiling) {}
    const ze_engine& get_engine() const { return m_engine; }
    bool is_profiling_enabled() const { return m_profiling_enabled; }

    virtual ~ze_base_event_factory() {}
    virtual event::ptr create_event(uint64_t queue_stamp) = 0;
protected:
    const ze_engine& m_engine;
    const bool m_profiling_enabled;
};
}  // namespace ze
}  // namespace cldnn
