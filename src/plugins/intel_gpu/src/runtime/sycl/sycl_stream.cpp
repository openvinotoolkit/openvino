// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_stream.hpp"
#include "CL/cl.h"
#include "intel_gpu/runtime/stream.hpp"
#include "sycl_event.hpp"
#include "sycl_command_queues_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "sycl_kernel.hpp"
#include "sycl_common.hpp"

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "ocl/ocl_kernel.hpp"  // for testing purposes

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_sycl.hpp>
#endif

namespace cldnn {
namespace sycl {

sycl_stream::sycl_stream(const sycl_engine &engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config))
    , _engine(engine) {
    auto context = engine.get_sycl_context();
    auto device = engine.get_sycl_device();
    sycl::command_queues_builder queue_builder;
    queue_builder.set_profiling(config.get_enable_profiling());
    queue_builder.set_out_of_order(m_queue_type == QueueTypes::out_of_order);

    OPENVINO_ASSERT(m_sync_method != SyncMethods::none || m_queue_type == QueueTypes::in_order,
                    "[GPU] Unexpected sync method (none) is specified for out_of_order queue");

    _command_queue = queue_builder.build(context, device);
}

sycl_stream::sycl_stream(const sycl_engine &engine, const ExecutionConfig& config, void *handle)
    : stream(sycl_stream::detect_queue_type(handle), stream::get_expected_sync_method(config))
    , _engine(engine) {
    auto casted_handle = static_cast<::sycl::queue*>(handle);
    _command_queue = *casted_handle;
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& sycl_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order, "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID, "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        _onednn_stream = std::make_shared<dnnl::stream>(dnnl::sycl_interop::make_stream(_engine.get_onednn_engine(), _command_queue));
    }

    return *_onednn_stream;
}
#endif

QueueTypes sycl_stream::detect_queue_type(void *queue_handle) {
    auto queue = static_cast<::sycl::queue*>(queue_handle);
    return queue->is_in_order() ? QueueTypes::in_order : QueueTypes::out_of_order;
}

void sycl_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
}

event::ptr sycl_stream::enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& args,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output) {
    return nullptr;
}

void sycl_stream::enqueue_barrier() {
    try {
        _command_queue.ext_oneapi_submit_barrier();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

event::ptr sycl_stream::enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) {
    // Wait for all previously enqueued tasks if deps list is empty
    if (deps.empty()) {
        ::sycl::event ret_ev;
        try {
            ret_ev = _command_queue.submit([&](::sycl::handler &cgh) {
                cgh.single_task([]() {});
            });
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }

        return std::make_shared<sycl_event>(ret_ev, _command_queue);
    }

    if (m_sync_method == SyncMethods::events) {
        ::sycl::event ret_ev;
        std::vector<::sycl::event> dep_events;
        for (auto& dep : deps) {
            if (auto sycl_base_ev = dynamic_cast<sycl_base_event*>(dep.get()))
                dep_events.push_back(sycl_base_ev->get());
        }

        try {
            if (dep_events.empty()) {
                return create_user_event(true);
            }
            ret_ev = _command_queue.ext_oneapi_submit_barrier(dep_events);
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }

        return std::make_shared<sycl_event>(ret_ev, _command_queue, ++_queue_counter);
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        return std::make_shared<sycl_event>(_last_barrier_ev, _command_queue, _last_barrier);
    } else {
        // use sycl::event as a user event
        return std::make_shared<sycl_event>(::sycl::event(), _command_queue);
    }
}

event::ptr sycl_stream::group_events(std::vector<event::ptr> const& deps) {
    if (deps.size() == 1)
        return deps[0];
    return std::make_shared<sycl_events>(deps);
}

event::ptr sycl_stream::create_user_event(bool set) {
    OPENVINO_ASSERT(set, "[GPU] create user event with set=false is not supported in SYCL runtime");
    return std::make_shared<sycl_event>(::sycl::event(), _command_queue);
}

event::ptr sycl_stream::create_base_event() {
    ::sycl::event ret_ev;
    return std::make_shared<sycl_event>(ret_ev, _command_queue, ++_queue_counter);
}

event::ptr sycl_stream::create_base_event(::sycl::event& event) {
    return std::make_shared<sycl_event>(event, _command_queue, ++_queue_counter);
}

void sycl_stream::flush() const {
    // nothing to do
}
void sycl_stream::finish() {
    try {
        get_sycl_queue().wait_and_throw();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

void sycl_stream::wait() {
    // Enqueue barrier with empty wait list to wait for all previously enqueued tasks
    try {
        auto ev = _command_queue.ext_oneapi_submit_barrier();
        ev.wait();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

void sycl_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;

    std::vector<::sycl::event> syclevents;
    for (auto& ev : events) {
        if (!ev)
            continue;

        if (auto sycl_base_ev = downcast<sycl_base_event>(ev.get())) {
            syclevents.push_back(sycl_base_ev->get());
        }
    }

    if (!syclevents.empty()) {
        try {
            ::sycl::event::wait(syclevents);
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }
    }
}

void sycl_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* sycl_base_ev = downcast<sycl_base_event>(dep.get());
        if (sycl_base_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        try {
            if (is_output)
                _last_barrier_ev = _command_queue.ext_oneapi_submit_barrier();
            else
                _command_queue.ext_oneapi_submit_barrier();
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }

        _last_barrier = ++_queue_counter;
    }
}

}  // namespace sycl
}  // namespace cldnn
