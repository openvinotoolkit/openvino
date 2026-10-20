// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_engine.hpp"
#include "intel_gpu/runtime/event.hpp"

#include <memory>

namespace cldnn {
namespace ze {

struct ze_base_event;

// Interface for creating Level Zero events
struct ze_base_event_factory {
public:
    ze_base_event_factory(const ze_engine &ze_engine, bool enable_profiling)
    : m_engine(ze_engine), m_profiling_enabled(enable_profiling) {}
    const ze_engine& get_engine() const { return m_engine; }
    bool is_profiling_enabled() const { return m_profiling_enabled; }

    virtual ~ze_base_event_factory() {}
    virtual std::shared_ptr<ze_base_event> create_event(uint64_t queue_stamp) = 0;
protected:
    const ze_engine& m_engine;
    const bool m_profiling_enabled;
};
}  // namespace ze
}  // namespace cldnn
