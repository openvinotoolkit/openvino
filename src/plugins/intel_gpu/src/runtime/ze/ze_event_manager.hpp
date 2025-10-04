// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ze_engine.hpp"

namespace cldnn {
namespace ze {

struct ze_event;

// Interface for creating and destroying Level Zero events
struct ze_event_manager {
public:
    using ptr = std::shared_ptr<ze_event_manager>;
    ze_event_manager(const ze_engine &engine, ze_command_list_handle_t cmd_list, bool enable_profiling)
    : m_engine(engine), m_cmd_list(cmd_list), m_enable_profiling(enable_profiling) {}
    const ze_engine& get_engine() const { return m_engine; }
    bool is_profiling_enabled() const { return m_enable_profiling; }
    ze_command_list_handle_t get_cmd_list() { return m_cmd_list; }

    virtual ~ze_event_manager() {}
    virtual std::shared_ptr<ze_event> create_event(uint64_t queue_stamp) = 0;
    virtual void destroy_event(ze_event *event) = 0;
protected:
    const ze_engine& m_engine;
    ze_command_list_handle_t m_cmd_list;
    bool m_enable_profiling;
};
}  // namespace ze
}  // namespace cldnn
