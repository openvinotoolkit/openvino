// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "event.hpp"
#include "kernel.hpp"
#include "kernel_args.hpp"
#include "execution_config.hpp"

#include <memory>
#include <vector>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl.hpp>
#endif

namespace cldnn {

class stream {
public:
    using ptr = std::shared_ptr<stream>;
    explicit stream(QueueTypes queue_type) : queue_type(queue_type) {}
    virtual ~stream() = default;

    virtual void flush() const = 0;
    virtual void finish() const = 0;
    virtual void wait() = 0;

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

    QueueTypes get_queue_type() const { return queue_type; }

    static QueueTypes detect_queue_type(engine_types engine_type, void* queue_handle);

#ifdef ENABLE_ONEDNN_FOR_GPU
    virtual dnnl::stream& get_onednn_stream() = 0;
#endif

protected:
    QueueTypes queue_type;
};

}  // namespace cldnn
