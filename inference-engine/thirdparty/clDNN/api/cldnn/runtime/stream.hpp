// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "event.hpp"
#include "kernel.hpp"
#include "kernel_args.hpp"

#include <memory>
#include <vector>

namespace cldnn {

class stream {
public:
    using ptr = std::shared_ptr<stream>;
    explicit stream(queue_types queue_type) : queue_type(queue_type) {}
    virtual ~stream() = default;

    virtual void flush() const = 0;
    virtual void finish() const = 0;

    virtual void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) = 0;
    virtual event::ptr enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& args,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output_event = false) = 0;
    virtual event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output_event = false) = 0;
    virtual void enqueue_barrier() = 0;
    virtual event::ptr group_events(std::vector<event::ptr> const& deps) = 0;
    virtual void wait_for_events(const std::vector<event::ptr>& events) = 0;
    virtual event::ptr create_user_event(bool set) = 0;
    virtual event::ptr create_base_event() = 0;

    queue_types get_queue_type() const { return queue_type; }

protected:
    queue_types queue_type;
};

}  // namespace cldnn
