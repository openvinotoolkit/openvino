// Copyright (C) 2018-2024 Intel Corporation
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

// Possible sync methods for kernels in stream
enum class SyncMethods {
    /* Build dependency graph using events. Each kernel creates proper ze_event which is set as dependency of users
       At this moment it requires multiple retain/release calls for ze_event after each enqueueNDRange
       which is less performant comparing to the barriers version
    */
    events = 0,
    /* Enqueue barriers between dependent kernels. For example consider the following dimond dependency graph:
             kernel_0
             /      \
         kernel_1   kernel_2
             \      /
             kernel_3
       In that case we do the following:
       1. Enqueue kernel_0
       2. Enqueue barrier (ensures kernel_0 is completed)
       3. Enqueue kernel_1
       4. Enqueue kernel_2 (doesn't depend on kernel_1)
       5. Enqueue barrier (ensures kernel_1 and kernel_2 are completed)
       6. Enqueue kernel_3
    */
    barriers = 1,
    /* No explicit syncronization is needed. Applicable for in-order queue only */
    none = 2
};

class stream {
public:
    using ptr = std::shared_ptr<stream>;
    explicit stream(QueueTypes queue_type, SyncMethods sync_method) : m_queue_type(queue_type), m_sync_method(sync_method) {}
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
    virtual event::ptr aggregate_events(const std::vector<event::ptr>& events, bool group = false, bool is_output = false);

    QueueTypes get_queue_type() const { return m_queue_type; }

    static QueueTypes detect_queue_type(engine_types engine_type, void* queue_handle);
    static SyncMethods get_expected_sync_method(const ExecutionConfig& config);

#ifdef ENABLE_ONEDNN_FOR_GPU
    virtual dnnl::stream& get_onednn_stream() = 0;
#endif

protected:
    QueueTypes m_queue_type;
    SyncMethods m_sync_method;
};

}  // namespace cldnn
