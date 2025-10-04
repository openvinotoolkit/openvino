// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_event_manager.hpp"

namespace cldnn {
namespace ze {

// Interface for creating and destroying l0 counter based events
// Should only be used with in-order queue
struct ze_cb_event_manager : public ze_event_manager {
public:
    ze_cb_event_manager(const ze_engine &engine, ze_command_list_handle_t cmd_list, bool enable_profiling);
    ~ze_cb_event_manager();
    std::shared_ptr<ze_event> create_event(uint64_t queue_stamp) override;
    void destroy_event(ze_event *event) override;
};
}  // namespace ze
}  // namespace cldnn
