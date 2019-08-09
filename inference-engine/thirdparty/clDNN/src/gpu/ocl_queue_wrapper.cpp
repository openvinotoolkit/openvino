/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "ocl_queue_wrapper.h"
#include "ocl_base_event.h"
#include "ocl_user_event.h"
#include "command_queues_builder.h"
#include "events_pool.h"

#include <cassert>
#include <iomanip>
#include <ios>

#include <fstream>
#include <thread>
#include <string>
#include <vector>
#include <memory>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

namespace {
std::string ndrange_to_string(cl::NDRange const& range) {
    std::string ret = "(";
    for (cl::size_type i = 0; i < range.dimensions(); ++i) ret += (!i ? "" : ", ") + std::to_string(range.get()[i]);

    ret += ")";
    return ret;
}

std::string events_list_to_string(std::vector<cldnn::event_impl::ptr> events) {
    std::string ret = "(";
    bool empty = true;
    for (auto& ev : events) {
        std::string id = "unk";
        if (auto* ocl_ev = dynamic_cast<cldnn::gpu::base_event*>(ev.get()))
            id = std::to_string(ocl_ev->get_queue_stamp());

        ret += (empty ? "" : ", ") + id;
        empty = false;
    }

    ret += ")";
    return ret;
}
}  // namespace

namespace cldnn {
namespace gpu {

gpu_queue::gpu_queue(int id, cl::CommandQueue queue, std::shared_ptr<gpu_toolkit> context)
    : id(id), _context(context), _command_queue(queue), _events_pool(new events_pool()) {}

event_impl::ptr gpu_queue::enqueue_kernel(cl::Kernel const& kern,
                                          cl::NDRange const& global,
                                          cl::NDRange const& local,
                                          std::vector<event_impl::ptr> const& deps) {
    std::vector<cl::Event> dep_events;
    auto dep_events_ptr = &dep_events;
    if (!_context->get_configuration().host_out_of_order) {
        for (auto& dep : deps)
            if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                dep_events.push_back(ocl_ev->get());
    } else {
        dep_events_ptr = nullptr;

        sync_events(deps);
    }

    cl::Event ret_ev;

    try {
        if (!_context->get_configuration().host_out_of_order || _output_event ||
            _context->get_configuration().enable_profiling) {
            _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, &ret_ev);
        } else {
            _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, nullptr);
        }
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }

    return _events_pool->get_from_base_pool(_context, ret_ev, ++_queue_counter);
}

event_impl::ptr gpu_queue::enqueue_marker(std::vector<event_impl::ptr> const& deps) {
    if (deps.empty())
        return _events_pool->get_from_user_pool(_context, true);

    bool enabled_single_kernel = _context->get_configuration().single_kernel_name == "" ? false : true;
    if (!_context->get_configuration().host_out_of_order) {
        cl::Event ret_ev;
        if (!enabled_single_kernel) {
            std::vector<cl::Event> dep_events;
            for (auto& dep : deps)
                if (auto ocl_ev = dynamic_cast<base_event*>(dep.get()))
                    dep_events.push_back(ocl_ev->get());

            try {
                _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
            } catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        } else {
            try {
                _command_queue.enqueueMarkerWithWaitList(nullptr, &ret_ev);
            } catch (cl::Error const& err) {
                throw ocl_error(err);
            }
        }

        return _events_pool->get_from_base_pool(_context, ret_ev, ++_queue_counter);
    } else {
        sync_events(deps);
        return _events_pool->get_from_base_pool(_context, _last_barrier_ev, _last_barrier);
    }
}

event_impl::ptr gpu_queue::group_events(std::vector<event_impl::ptr> const& deps) {
    return _events_pool->get_from_group_pool(_context, deps);
}

event_impl::ptr gpu_queue::create_user_event(bool set) { return _events_pool->get_from_user_pool(_context, set); }

void gpu_queue::reset_events() { _events_pool->reset_events(); }

void gpu_queue::release_events_pool() { _events_pool.reset(); }

void gpu_queue::flush() { queue().flush(); }

void gpu_queue::release_pending_memory() {
    /*
    TODO: Temp. solution, untill proper API calls from OpenCL are released.
    */
    void* ptr = nullptr;
    ptr = _mm_malloc(4096, 4096);
    queue().finish();
    try {
        cl::Buffer flusher(_context->context(), CL_MEM_USE_HOST_PTR, (size_t)4096, ptr);
        flusher = (cl_mem) nullptr;  // clear buffer
    } catch (...) {
        _mm_free(ptr);
        throw;
    }
    _mm_free(ptr);
}

void gpu_queue::sync_events(std::vector<event_impl::ptr> const& deps) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ocl_ev = dynamic_cast<ocl_base_event*>(dep.get());
        if (ocl_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        try {
            if (_output_event)
                _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
            else
                _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
        } catch (cl::Error const& err) {
            throw ocl_error(err);
        }

        _last_barrier = ++_queue_counter;
    }
}

}  // namespace gpu
}  // namespace cldnn
