// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_stream.hpp"
#include "ze_event.hpp"
#include "ze_kernel.hpp"
#include "ze_common.hpp"

#include <cassert>
#include <iomanip>
#include <ios>

#include <fstream>
#include <thread>
#include <string>
#include <vector>
#include <memory>

namespace cldnn {
namespace ze {

namespace {
inline ze_group_count_t to_group_count(const std::vector<size_t>& v) {
     switch (v.size()) {
         case 1:
            return {uint32_t(v[0]), uint32_t(1), uint32_t(1)};
        case 2:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(1)};
        case 3:
            return {uint32_t(v[0]), uint32_t(v[1]), uint32_t(v[2])};
        default:
            return {uint32_t(0), uint32_t(0), uint32_t(0)};
    }
}

void set_arguments_impl(ze_kernel_handle_t kernel,
                         const arguments_desc& args,
                         const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        ze_result_t status = ZE_RESULT_NOT_READY;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    const auto& input_mem = data.inputs[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            //status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
                            status = zeKernelSetArgumentValue(kernel, i, args.size(), input_mem.get());
                    }
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    const auto& input_mem = data.fused_op_inputs[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            //status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
                            status = zeKernelSetArgumentValue(kernel, i, args.size(), input_mem.get());
                    }
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    const auto& input_mem = data.intermediates[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                            //status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
                            status = zeKernelSetArgumentValue(kernel, i, args.size(), input_mem.get());
                    }
                }
                break;
            case args_t::OUTPUT:
                if (data.output) {
                    status = zeKernelSetArgumentValue(kernel, i, args.size(), data.output.get());
                }
                break;
            case args_t::WEIGHTS:
                if (data.weights) {
                    //status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.weights)->get_buffer());
                    status = zeKernelSetArgumentValue(kernel, i, args.size(), data.weights.get());
                }
                break;
            case args_t::BIAS:
                if (data.bias) {
                    //status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.bias)->get_buffer());
                    status = zeKernelSetArgumentValue(kernel, i, args.size(), data.bias.get());
                }
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                if (data.weights_zero_points) {
                    status = zeKernelSetArgumentValue(kernel, i, args.size(), data.weights_zero_points.get());
                }
                break;
            default:
                break;
        }
    }


//     using args_t = argument_desc::Types;
//     using scalar_t = scalar_desc::Types;
//     for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
//         cl_int status = CL_INVALID_ARG_VALUE;
//         switch (args[i].t) {
//             case args_t::INPUT:
//                 if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
//                     const auto& input_mem = data.inputs[args[i].index];
//                     if (input_mem) {
//                         if (input_mem->get_layout().format.is_image_2d())
//                             status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_image2d>(input_mem)->get_buffer());
//                         else if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
//                             status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
//                         else
//                             status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(input_mem)->get_buffer());
//                     }
//                 }
//                 break;
//             case args_t::INPUT_OF_FUSED_PRIMITIVE:
//                 if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
//                     const auto& input_mem = data.fused_op_inputs[args[i].index];
//                     if (input_mem) {
//                         if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
//                             status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
//                         else
//                             status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(input_mem)->get_buffer());
//                     }
//                 }
//                 break;
//             case args_t::INTERNAL_BUFFER:
//                 if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
//                     const auto& input_mem = data.intermediates[args[i].index];
//                     if (input_mem) {
//                         if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
//                             status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer());
//                         else
//                             status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(input_mem)->get_buffer());
//                     }
//                 }
//                 break;
//             case args_t::OUTPUT:
//                 if (data.output) {
//                      if (data.output->get_layout().format.is_image_2d())
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_image2d>(data.output)->get_buffer());
//                      else if (memory_capabilities::is_usm_type(data.output->get_allocation_type()))
//                          status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.output)->get_buffer());
//                      else
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(data.output)->get_buffer());
//                 }
//                 break;
//             case args_t::WEIGHTS:
//                 if (data.weights) {
//                     if (data.weights->get_layout().format.is_image_2d())
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_image2d>(data.weights)->get_buffer());
//                     else if (memory_capabilities::is_usm_type(data.weights->get_allocation_type()))
//                         status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.weights)->get_buffer());
//                     else
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(data.weights)->get_buffer());
//                 }
//                 break;
//             case args_t::BIAS:
//                 if (data.bias) {
//                     if (memory_capabilities::is_usm_type(data.bias->get_allocation_type()))
//                         status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.bias)->get_buffer());
//                     else
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(data.bias)->get_buffer());
//                 }
//                 break;
//             case args_t::WEIGHTS_ZERO_POINTS:
//                 if (data.weights_zero_points) {
//                     if (memory_capabilities::is_usm_type(data.weights_zero_points->get_allocation_type()))
//                         status = kernel.setArgUsm(
//                             i,
//                             std::dynamic_pointer_cast<const ze::gpu_usm>(data.weights_zero_points)->get_buffer());
//                     else
//                         status = kernel.setArg(
//                             i,
//                             std::dynamic_pointer_cast<const ze::gpu_buffer>(data.weights_zero_points)->get_buffer());
//                 }
//                 break;
//             case args_t::ACTIVATIONS_ZERO_POINTS:
//                 if (data.activations_zero_points) {
//                     if (memory_capabilities::is_usm_type(data.activations_zero_points->get_allocation_type()))
//                         status = kernel.setArgUsm(
//                             i,
//                             std::dynamic_pointer_cast<const ze::gpu_usm>(data.activations_zero_points)->get_buffer());
//                     else
//                         status = kernel.setArg(
//                             i,
//                             std::dynamic_pointer_cast<const ze::gpu_buffer>(data.activations_zero_points)->get_buffer());
//                 }
//                 break;
//             case args_t::COMPENSATION:
//                 if (data.compensation) {
//                     if (memory_capabilities::is_usm_type(data.compensation->get_allocation_type()))
//                         status = kernel.setArgUsm(
//                                 i,
//                                 std::dynamic_pointer_cast<const ze::gpu_usm>(data.compensation)->get_buffer());
//                     else
//                         status = kernel.setArg(
//                                  i,
//                                  std::dynamic_pointer_cast<const ze::gpu_buffer>(data.compensation)->get_buffer());
//                 }
//                 break;
//             case args_t::SCALE_TABLE:
//                 if (data.scale_table) {
//                     if (memory_capabilities::is_usm_type(data.scale_table->get_allocation_type()))
//                         status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.scale_table)->get_buffer());
//                     else
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(data.scale_table)->get_buffer());
//                 }
//                 break;
//             case args_t::SLOPE:
//                 if (data.slope) {
//                     if (memory_capabilities::is_usm_type(data.slope->get_allocation_type()))
//                         status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const ze::gpu_usm>(data.slope)->get_buffer());
//                     else
//                         status = kernel.setArg(i, std::dynamic_pointer_cast<const ze::gpu_buffer>(data.slope)->get_buffer());
//                 }
//                 break;
//             case args_t::SPLIT:
//                 status = kernel.setArg(i, data.split);
//                 break;
//             case args_t::SCALAR:
//                 if (data.scalars && args[i].index < data.scalars->size()) {
//                     const auto& scalar = (*data.scalars)[args[i].index];
//                     switch (scalar.t) {
//                         case scalar_t::UINT8:
//                             status = kernel.setArg(i, scalar.v.u8);
//                             break;
//                         case scalar_t::UINT16:
//                             status = kernel.setArg(i, scalar.v.u16);
//                             break;
//                         case scalar_t::UINT32:
//                             status = kernel.setArg(i, scalar.v.u32);
//                             break;
//                         case scalar_t::UINT64:
//                             status = kernel.setArg(i, scalar.v.u64);
//                             break;
//                         case scalar_t::INT8:
//                             status = kernel.setArg(i, scalar.v.s8);
//                             break;
//                         case scalar_t::INT16:
//                             status = kernel.setArg(i, scalar.v.s16);
//                             break;
//                         case scalar_t::INT32:
//                             status = kernel.setArg(i, scalar.v.s32);
//                             break;
//                         case scalar_t::INT64:
//                             status = kernel.setArg(i, scalar.v.s64);
//                             break;
//                         case scalar_t::FLOAT32:
//                             status = kernel.setArg(i, scalar.v.f32);
//                             break;
//                         case scalar_t::FLOAT64:
//                             status = kernel.setArg(i, scalar.v.f64);
//                             break;
//                         default:
//                             break;
//                     }
//                 }
//                 break;
//             case args_t::RECURRENT:  // RNN/LSTM/GRU layers
//                 if (data.recurrent) {
//                     if (data.recurrent->get_layout().format.is_image_2d())
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_image2d&>(*data.recurrent).get_buffer());
//                     else if (memory_capabilities::is_usm_type(data.recurrent->get_allocation_type()))
//                         status = kernel.setArgUsm(i, dynamic_cast<const ze::gpu_usm&>(*data.recurrent).get_buffer());
//                     else
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_buffer&>(*data.recurrent).get_buffer());
//                 }
//                 break;
//             case args_t::HIDDEN:  // RNN/LSTM/GRU layers
//                 if (data.hidden) {
//                     if (data.hidden->get_layout().format.is_image_2d())
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_image2d&>(*data.hidden).get_buffer());
//                     else if (memory_capabilities::is_usm_type(data.hidden->get_allocation_type()))
//                         status = kernel.setArgUsm(i, dynamic_cast<const ze::gpu_usm&>(*data.hidden).get_buffer());
//                     else
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_buffer&>(*data.hidden).get_buffer());
//                 }
//                 break;
//             case args_t::CELL:  // LSTMlayers
//                 if (data.cell) {
//                     if (data.cell->get_layout().format.is_image_2d())
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_image2d&>(*data.cell).get_buffer());
//                     else if (memory_capabilities::is_usm_type(data.cell->get_allocation_type()))
//                         status = kernel.setArgUsm(i, dynamic_cast<const ze::gpu_usm&>(*data.cell).get_buffer());
//                     else
//                         status = kernel.setArg(i, dynamic_cast<const ze::gpu_buffer&>(*data.cell).get_buffer());
//                 }
//                 break;
//             default:
//                 break;
//         }

//         if (status != CL_SUCCESS) {
//             throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
//         }
//     }
}
}  // namespace

ze_stream::ze_stream(const ze_engine& engine) : stream(engine.configuration().queue_type), _engine(engine) {
    auto context = engine.get_context();
    auto device = engine.get_device();
    auto config = engine.configuration();

    sync_method = _engine.configuration().enable_profiling ? sync_methods::events :
                  config.queue_type == queue_types::out_of_order ? sync_methods::barriers : sync_methods::none;

    if (sync_method == sync_methods::none && config.queue_type == queue_types::out_of_order) {
        throw std::runtime_error("[CLDNN] Unexpected sync method (none) is specified for out_of_order queue");
    }

    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.flags = 0;
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;

    command_queue_desc.pNext = nullptr;
    command_queue_desc.mode = config.queue_type == queue_types::out_of_order ? ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS : ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    ZE_CHECK(zeCommandListCreateImmediate(context, device, &command_queue_desc, &_command_list));

    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
        100 // count
    };

    ZE_CHECK(zeEventPoolCreate(context, &event_pool_desc, 0, nullptr, &_event_pool));
}

void ze_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);

    auto& kern = ze_kernel.get_handle();

    set_arguments_impl(kern, args_desc.arguments, args);
}

event::ptr ze_stream::enqueue_kernel(kernel& kernel,
                                     const kernel_arguments_desc& args_desc,
                                     const kernel_arguments_data& /* args */,
                                     std::vector<event::ptr> const& deps,
                                     bool is_output) {
    auto& ze_kernel = downcast<ze::ze_kernel>(kernel);

    auto& kern = ze_kernel.get_handle();
    //_command_list
    //zeKernelSetArgumentValue

    //zeKernelSetArgumentValue
    //zeCommandListAppendLaunchKernel
    // auto global = toNDRange(args_desc.workGroups.global);
    // auto local = toNDRange(args_desc.workGroups.local);
    // std::vector<cl::Event> dep_events;
    // std::vector<cl::Event>* dep_events_ptr = nullptr;
    // if (sync_method == sync_methods::events) {
    //     for (auto& dep : deps) {
    //         if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
    //             if (ze_base_ev->get().get() != nullptr)
    //                 dep_events.push_back(ze_base_ev->get());
    //         }
    //     }
    //     dep_events_ptr = &dep_events;
    // } else if (sync_method == sync_methods::barriers) {
    //     sync_events(deps, is_output);
    // }

    // cl::Event ret_ev;

    std::vector<ze_event_handle_t> dep_events;
    std::vector<ze_event_handle_t>* dep_events_ptr = nullptr;
    if (sync_method == sync_methods::events) {
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (sync_method == sync_methods::barriers) {
        sync_events(deps, is_output);
    }
    bool set_output_event = sync_method == sync_methods::events || is_output;

    // try {
    //     _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, set_output_event ? &ret_ev : nullptr);
    // } catch (cl::Error const& err) {
    //     throw ze_error(err);
    // }
    ze_event_handle_t ret_ev;
    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        event_idx++ % 100,       // index
        0,                       // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_DEVICE // ensure memory coherency across device and Host after event completes
    };
    zeEventCreate(_event_pool, &eventDesc, &ret_ev);
    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    zeKernelSetGroupSize(kern, global.groupCountX/local.groupCountX, global.groupCountY/local.groupCountX, global.groupCountZ/local.groupCountX);

    ZE_CHECK(zeCommandListAppendLaunchKernel(_command_list,
                                    kern,
                                    &local,
                                    set_output_event ? ret_ev : nullptr,
                                    dep_events_ptr == nullptr ? 0 : dep_events_ptr->size(),
                                    &dep_events_ptr->at(0)));

    return std::make_shared<ze_event>(ret_ev, ++_queue_counter);
    //return nullptr;
}

void ze_stream::enqueue_barrier() {
    ZE_CHECK(zeCommandListAppendBarrier(_command_list, nullptr, 0, nullptr));
}

ze_event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    //if (deps.empty())
    //    return std::make_shared<ze_event>(_engine.get_context(), true);

    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        event_idx++ % 100,                 // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_DEVICE // ensure memory coherency across device and Host after event completes
    };
    ze_event_handle_t hEvent;
    ZE_CHECK(zeEventCreate(_event_pool, &eventDesc, &hEvent));

    return std::make_shared<ze_event>(hEvent, false);
/*
    if (sync_method  == sync_methods::events) {
        //cl::Event ret_ev;
        //std::vector<cl::Event> dep_events;
        ze_event_desc_t tsEventDesc = {
            ZE_STRUCTURE_TYPE_EVENT_DESC,
            nullptr,
            0, // index
            0, // no additional memory/cache coherency required on signal
            0  // no additional memory/cache coherency required on wait
        };
        ze_event_handle_t hTSEvent;
        for (auto& dep : deps) {
            if (auto ze_base_ev = dynamic_cast<ze_base_event*>(dep.get()))
                if (ze_base_ev->get().get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
        }

        try {
            if (dep_events.empty()) {
                return create_user_event(true);
            }
            _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
        } catch (cl::Error const& err) {
            throw ze_error(err);
        }

        return std::make_shared<ze_event>(ret_ev, ++_queue_counter);
    } else if (sync_method == sync_methods::barriers) {
        sync_events(deps, is_output);
        return std::make_shared<ze_event>(_last_barrier_ev, _last_barrier);
    } else {
        return std::make_shared<ze_user_event>(_engine.get_cl_context(), true);
    }
    immidiatly_comand_queue!!!
*/
}

ze_event::ptr ze_stream::group_events(std::vector<ze_events::ptr> const& deps) {
    return std::make_shared<ze_events>(deps);
}

ze_event::ptr ze_stream::create_user_event(bool set) {
    //return nullptr;
    ze_event_handle_t hEvent;
    ze_event_desc_t tsEventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        event_idx++ % 100,         // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_DEVICE // no additional memory/cache coherency required on wait
    };
    ZE_CHECK(zeEventCreate(_event_pool, &tsEventDesc, &hEvent));
    return std::make_shared<ze_event>(hEvent, set);//_engine.get_context(), set);
}

ze_event::ptr ze_stream::create_base_event() {
    //cl::Event ret_ev;
    ze_event_handle_t hEvent;
    ze_event_desc_t tsEventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        event_idx++ % 100,         // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_DEVICE // no additional memory/cache coherency required on wait
    };
    ZE_CHECK(zeEventCreate(_event_pool, &tsEventDesc, &hEvent));
    return std::make_shared<ze_event>(hEvent, ++_queue_counter);
}

void ze_stream::flush() const {
    ZE_CHECK(zeCommandListAppendBarrier(get_queue(), nullptr, 0, nullptr));
}

void ze_stream::finish() const {
    //zeCommandListClose(get_queue());
    ZE_CHECK(zeCommandListAppendBarrier(get_queue(), nullptr, 0, nullptr));
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;

    std::vector<ze::ze_base_event> clevents;
    for (auto& ev : events) {
        if (auto ze_base_ev = dynamic_cast<ze_base_event*>(ev.get())) {
            //clevents.push_back(*ze_base_ev);
            ZE_CHECK(zeEventHostSynchronize(ze_base_ev->get(), 0));
        }
    }

    // std::vector<cl::Event> clevents;
    // for (auto& ev : events) {
    //     if (auto ze_base_ev = dynamic_cast<ze_base_event*>(ev.get()))
    //         clevents.push_back(ze_base_ev->get());
    // }

    // try {
    //     cl::WaitForEvents(clevents);
    // } catch (cl::Error const& err) {
    //     throw ze_error(err);
    // }
}

void ze_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(dep.get());
        if (ze_base_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        //try {
            if (is_output) {
                ze_event_desc_t event_desc = {
                    ZE_STRUCTURE_TYPE_EVENT_DESC,
                    nullptr,
                    event_idx++ % 100,         // index
                    0,                         // no additional memory/cache coherency required on signal
                    ZE_EVENT_SCOPE_FLAG_DEVICE // ensure memory coherency across device and Host after event completes
                };
                ze_event_handle_t event;
                ZE_CHECK(zeEventCreate(_event_pool, &event_desc, &event));
                ZE_CHECK(zeCommandListAppendBarrier(_command_list, event, 0, nullptr));

            } else {
                ZE_CHECK(zeCommandListAppendBarrier(_command_list, nullptr, 0, nullptr));
            }
        //} catch (ze::Error const& err) {
        //    throw ze_error(err);
        //}

        _last_barrier = ++_queue_counter;
    }
}

}  // namespace ze
}  // namespace cldnn