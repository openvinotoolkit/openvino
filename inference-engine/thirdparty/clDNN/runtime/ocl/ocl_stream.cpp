// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_stream.hpp"
#include "ocl_event.hpp"
#include "ocl_user_event.hpp"
#include "ocl_command_queues_builder.hpp"
#include "ocl_kernel.hpp"
#include "ocl_common.hpp"

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

namespace cldnn {
namespace ocl {

namespace {
inline cl::NDRange toNDRange(const std::vector<size_t>& v) {
    switch (v.size()) {
        case 1:
            return cl::NDRange(v[0]);
        case 2:
            return cl::NDRange(v[0], v[1]);
        case 3:
            return cl::NDRange(v[0], v[1], v[2]);
        default:
            return cl::NullRange;
    }
}

void set_arguments_impl(ocl_kernel_type& kernel,
                        const arguments_desc& args,
                        const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    const auto& input_mem = data.inputs[args[i].index];
                    if (input_mem) {
                        if (input_mem->get_layout().format.is_image_2d())
                            status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_image2d>(input_mem)->get_buffer());
                        else if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(input_mem)->get_buffer());
                        else
                            status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(input_mem)->get_buffer());
                    }
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    const auto& input_mem = data.fused_op_inputs[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(input_mem)->get_buffer());
                        else
                            status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(input_mem)->get_buffer());
                    }
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    const auto& input_mem = data.intermediates[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(input_mem)->get_buffer());
                        else
                            status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(input_mem)->get_buffer());
                    }
                }
                break;
            case args_t::OUTPUT:
                if (data.output) {
                     if (data.output->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_image2d>(data.output)->get_buffer());
                     else if (memory_capabilities::is_usm_type(data.output->get_allocation_type()))
                         status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(data.output)->get_buffer());
                     else
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.output)->get_buffer());
                }
                break;
            case args_t::WEIGHTS:
                if (data.weights) {
                    if (data.weights->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_image2d>(data.weights)->get_buffer());
                    else if (memory_capabilities::is_usm_type(data.weights->get_allocation_type()))
                        status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(data.weights)->get_buffer());
                    else
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.weights)->get_buffer());
                }
                break;
            case args_t::BIAS:
                if (data.bias) {
                    if (memory_capabilities::is_usm_type(data.bias->get_allocation_type()))
                        status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(data.bias)->get_buffer());
                    else
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.bias)->get_buffer());
                }
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                if (data.weights_zero_points) {
                    if (memory_capabilities::is_usm_type(data.weights_zero_points->get_allocation_type()))
                        status = kernel.setArgUsm(
                            i,
                            std::dynamic_pointer_cast<const ocl::gpu_usm>(data.weights_zero_points)->get_buffer());
                    else
                        status = kernel.setArg(
                            i,
                            std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.weights_zero_points)->get_buffer());
                }
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                if (data.activations_zero_points) {
                    if (memory_capabilities::is_usm_type(data.activations_zero_points->get_allocation_type()))
                        status = kernel.setArgUsm(
                            i,
                            std::dynamic_pointer_cast<const ocl::gpu_usm>(data.activations_zero_points)->get_buffer());
                    else
                        status = kernel.setArg(
                            i,
                            std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.activations_zero_points)->get_buffer());
                }
                break;
            case args_t::COMPENSATION:
                if (data.compensation) {
                    if (memory_capabilities::is_usm_type(data.compensation->get_allocation_type()))
                        status = kernel.setArgUsm(
                                i,
                                std::dynamic_pointer_cast<const ocl::gpu_usm>(data.compensation)->get_buffer());
                    else
                        status = kernel.setArg(
                                 i,
                                 std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.compensation)->get_buffer());
                }
                break;
            case args_t::SCALE_TABLE:
                if (data.scale_table) {
                    if (memory_capabilities::is_usm_type(data.scale_table->get_allocation_type()))
                        status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(data.scale_table)->get_buffer());
                    else
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.scale_table)->get_buffer());
                }
                break;
            case args_t::SLOPE:
                if (data.slope) {
                    if (memory_capabilities::is_usm_type(data.slope->get_allocation_type()))
                        status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ocl::gpu_usm>(data.slope)->get_buffer());
                    else
                        status = kernel.setArg(i, std::dynamic_pointer_cast<const ocl::gpu_buffer>(data.slope)->get_buffer());
                }
                break;
            case args_t::SPLIT:
                status = kernel.setArg(i, data.split);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = kernel.setArg(i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = kernel.setArg(i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = kernel.setArg(i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = kernel.setArg(i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = kernel.setArg(i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = kernel.setArg(i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = kernel.setArg(i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = kernel.setArg(i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = kernel.setArg(i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = kernel.setArg(i, scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::RECURRENT:  // RNN/LSTM/GRU layers
                if (data.recurrent) {
                    if (data.recurrent->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_image2d&>(*data.recurrent).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.recurrent->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const ocl::gpu_usm&>(*data.recurrent).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_buffer&>(*data.recurrent).get_buffer());
                }
                break;
            case args_t::HIDDEN:  // RNN/LSTM/GRU layers
                if (data.hidden) {
                    if (data.hidden->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_image2d&>(*data.hidden).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.hidden->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const ocl::gpu_usm&>(*data.hidden).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_buffer&>(*data.hidden).get_buffer());
                }
                break;
            case args_t::CELL:  // LSTMlayers
                if (data.cell) {
                    if (data.cell->get_layout().format.is_image_2d())
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_image2d&>(*data.cell).get_buffer());
                    else if (memory_capabilities::is_usm_type(data.cell->get_allocation_type()))
                        status = kernel.setArgUsm(i, dynamic_cast<const ocl::gpu_usm&>(*data.cell).get_buffer());
                    else
                        status = kernel.setArg(i, dynamic_cast<const ocl::gpu_buffer&>(*data.cell).get_buffer());
                }
                break;
            default:
                break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}
}  // namespace

ocl_stream::ocl_stream(const ocl_engine& engine) : stream(engine.configuration().queue_type), _engine(engine) {
    auto context = engine.get_cl_context();
    auto device = engine.get_cl_device();
    auto config = engine.configuration();
    ocl::command_queues_builder queue_builder;
    queue_builder.set_profiling(config.enable_profiling);
    queue_builder.set_out_of_order((config.queue_type == queue_types::out_of_order));

    sync_method = _engine.configuration().enable_profiling ? sync_methods::events :
                  config.queue_type == queue_types::out_of_order ? sync_methods::barriers : sync_methods::none;

    if (sync_method == sync_methods::none && config.queue_type == queue_types::out_of_order) {
        throw std::runtime_error("[CLDNN] Unexpected sync method (none) is specified for out_of_order queue");
    }

    bool priorty_extensions = engine.extension_supported("cl_khr_priority_hints") && engine.extension_supported("cl_khr_create_command_queue");
    queue_builder.set_priority_mode(config.priority_mode, priorty_extensions);

    bool throttle_extensions = engine.extension_supported("cl_khr_throttle_hints") && engine.extension_supported("cl_khr_create_command_queue");
    queue_builder.set_throttle_mode(config.throttle_mode, throttle_extensions);

    _command_queue = queue_builder.build(context, device);
}

void ocl_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();

    try {
        set_arguments_impl(kern, args_desc.arguments, args);
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

event::ptr ocl_stream::enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& /* args */,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output) {
    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();
    auto global = toNDRange(args_desc.workGroups.global);
    auto local = toNDRange(args_desc.workGroups.local);
    std::vector<cl::Event> dep_events;
    std::vector<cl::Event>* dep_events_ptr = nullptr;
    if (sync_method == sync_methods::events) {
        for (auto& dep : deps) {
            if (auto ocl_base_ev = std::dynamic_pointer_cast<ocl_base_event>(dep)) {
                if (ocl_base_ev->get().get() != nullptr)
                    dep_events.push_back(ocl_base_ev->get());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (sync_method == sync_methods::barriers) {
        sync_events(deps, is_output);
    }

    cl::Event ret_ev;

    bool set_output_event = sync_method == sync_methods::events || is_output;

    try {
        _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, set_output_event ? &ret_ev : nullptr);
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }

    return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
}

void ocl_stream::enqueue_barrier() {
    _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
}

event::ptr ocl_stream::enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) {
    if (deps.empty())
        return std::make_shared<ocl_user_event>(_engine.get_cl_context(), true);

    if (sync_method == sync_methods::events) {
        cl::Event ret_ev;
        std::vector<cl::Event> dep_events;
        for (auto& dep : deps) {
            if (auto ocl_base_ev = dynamic_cast<ocl_base_event*>(dep.get()))
                if (ocl_base_ev->get().get() != nullptr)
                    dep_events.push_back(ocl_base_ev->get());
        }

        try {
            if (dep_events.empty()) {
                return create_user_event(true);
            }
            _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
        } catch (cl::Error const& err) {
            throw ocl_error(err);
        }

        return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
    } else if (sync_method == sync_methods::barriers) {
        sync_events(deps, is_output);
        return std::make_shared<ocl_event>(_last_barrier_ev, _last_barrier);
    } else {
        return std::make_shared<ocl_user_event>(_engine.get_cl_context(), true);
    }
}

event::ptr ocl_stream::group_events(std::vector<event::ptr> const& deps) {
    return std::make_shared<ocl_events>(deps);
}

event::ptr ocl_stream::create_user_event(bool set) {
    return std::make_shared<ocl_user_event>(_engine.get_cl_context(), set);
}

event::ptr ocl_stream::create_base_event() {
    cl::Event ret_ev;
    return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
}

void ocl_stream::flush() const { get_cl_queue().flush(); }
void ocl_stream::finish() const { get_cl_queue().finish(); }

void ocl_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;

    std::vector<cl::Event> clevents;
    for (auto& ev : events) {
        if (auto ocl_base_ev = dynamic_cast<ocl_base_event*>(ev.get()))
            clevents.push_back(ocl_base_ev->get());
    }

    try {
        cl::WaitForEvents(clevents);
    } catch (cl::Error const& err) {
        throw ocl_error(err);
    }
}

void ocl_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ocl_base_ev = dynamic_cast<ocl_base_event*>(dep.get());
        if (ocl_base_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        try {
            if (is_output)
                _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
            else
                _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
        } catch (cl::Error const& err) {
            throw ocl_error(err);
        }

        _last_barrier = ++_queue_counter;
    }
}

}  // namespace ocl
}  // namespace cldnn
