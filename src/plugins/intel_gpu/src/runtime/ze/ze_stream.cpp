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

//#define IMMIDIATLY

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
            return {uint32_t(1), uint32_t(1), uint32_t(1)};
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
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type())) {
                            auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer().get();
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                            ZE_CHECK(status);
                        }
                    }
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    const auto& input_mem = data.fused_op_inputs[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type())) {
                            auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer().get();
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                            ZE_CHECK(status);
                        }
                    }
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    const auto& input_mem = data.intermediates[args[i].index];
                    if (input_mem) {
                        if (memory_capabilities::is_usm_type(input_mem->get_allocation_type())) {
                            auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(input_mem)->get_buffer().get();
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                            ZE_CHECK(status);
                        }
                    }
                }
                break;
            case args_t::OUTPUT:
                if (data.output) {
                    auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.output)->get_buffer().get();
                    status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    ZE_CHECK(status);
                }
                break;
            case args_t::WEIGHTS:
                if (data.weights) {
                    auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.weights)->get_buffer().get();
                    status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    ZE_CHECK(status);
                }
                break;
            case args_t::BIAS:
                if (data.bias) {
                    auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.bias)->get_buffer().get();
                    status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    ZE_CHECK(status);
                }
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                if (data.weights_zero_points) {
                    auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.weights_zero_points)->get_buffer().get();
                    status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    ZE_CHECK(status);
                }
                else {
                    throw std::runtime_error("WEIGHTS_ZERO_POINTS");
                }
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                if (data.activations_zero_points) {
                    if (memory_capabilities::is_usm_type(data.activations_zero_points->get_allocation_type())){
                        auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.activations_zero_points)->get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("ACTIVATIONS_ZERO_POINTS");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::COMPENSATION:
                if (data.compensation) {
                    if (memory_capabilities::is_usm_type(data.compensation->get_allocation_type())) {
                        auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.compensation)->get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("COMPENSATION");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::SCALE_TABLE:
                if (data.scale_table) {
                    if (memory_capabilities::is_usm_type(data.scale_table->get_allocation_type())) {
                        auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.scale_table)->get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("SCALE_TABLE");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::SLOPE:
                if (data.slope) {
                    if (memory_capabilities::is_usm_type(data.slope->get_allocation_type())) {
                        auto ptr = std::dynamic_pointer_cast<const ze::gpu_usm>(data.slope)->get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("SLOPE");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::SPLIT:
                {
                    status = zeKernelSetArgumentValue(kernel, i, sizeof(data.split), &data.split);
                    ZE_CHECK(status);
                }
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.u8), &scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.u16), &scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.u32), &scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.u64), &scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.s8), &scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.s16), &scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.s32), &scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.s64), &scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.f32), &scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = zeKernelSetArgumentValue(kernel, i, sizeof(scalar.v.f64), &scalar.v.f64);
                            break;
                        default:
                            break;
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::RECURRENT:  // RNN/LSTM/GRU layers
                if (data.recurrent) {
                    if (data.recurrent->get_layout().format.is_image_2d()) {
                        throw std::runtime_error("RECURRENT 1");
                    }
                    else if (memory_capabilities::is_usm_type(data.recurrent->get_allocation_type())) {
                        auto ptr = dynamic_cast<const ze::gpu_usm&>(*data.recurrent).get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("RECURRENT 2");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::HIDDEN:  // RNN/LSTM/GRU layers
                if (data.hidden) {
                    if (data.hidden->get_layout().format.is_image_2d()) {
                        throw std::runtime_error("HIDDEN 1");
                    }
                    else if (memory_capabilities::is_usm_type(data.hidden->get_allocation_type())) {
                        auto ptr = dynamic_cast<const ze::gpu_usm&>(*data.hidden).get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("HIDDEN 2");
                    }
                    ZE_CHECK(status);
                }
                break;
            case args_t::CELL:  // LSTMlayers
                if (data.cell) {
                    if (data.cell->get_layout().format.is_image_2d()) {
                        throw std::runtime_error("CELL 1");
                    }
                    else if (memory_capabilities::is_usm_type(data.cell->get_allocation_type())) {
                        auto ptr = dynamic_cast<const ze::gpu_usm&>(*data.cell).get_buffer().get();
                        status = zeKernelSetArgumentValue(kernel, i, sizeof(void*), &ptr);
                    }
                    else {
                        throw std::runtime_error("CELL 2");
                    }
                    ZE_CHECK(status);
                }
                break;
            default:
                break;
        }
        if (status != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
            //std::cout << "Error set arg " + std::to_string(i)  + " " + std::to_string(int(args[i].t)) + " error code: " + std::to_string(status) + "\n" << std::endl;
        }
    }
}

sync_methods get_expected_sync_method(const engine_configuration &config) {
    return config.enable_profiling ? sync_methods::events : config.queue_type == queue_types::out_of_order ? sync_methods::barriers
                                                                                                           : sync_methods::none;
}

}  // namespace

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ze_stream::get_onednn_stream() {
    if (!_onednn_stream)
        throw std::runtime_error("[GPU] onednn stream is nullptr");
    return *_onednn_stream;
}
#endif

ze_stream::ze_stream(const ze_engine& engine)
    : stream(engine.configuration().queue_type)
    , _engine(engine)
    , sync_method(get_expected_sync_method(engine.configuration())) {
    auto context = engine.get_context();
    auto device = engine.get_device();
    auto config = engine.configuration();

    sync_method = _engine.configuration().enable_profiling ? sync_methods::events :
                  config.queue_type == queue_types::out_of_order ? sync_methods::barriers : sync_methods::none;

    if (sync_method == sync_methods::none && config.queue_type == queue_types::out_of_order) {
        throw std::runtime_error("[CLDNN] Unexpected sync method (none) is specified for out_of_order queue");
    }
#ifdef SINGLE_EVENT_POOL
    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
        pool_size // count
    };
    event_idx = 0;
    ZE_CHECK(zeEventPoolCreate(_engine.get_context(), &event_pool_desc, 0, nullptr, &_event_pool));

    ze_event_pool_desc_t event_pool_desc_last = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
        1 // count
    };

    ZE_CHECK(zeEventPoolCreate(_engine.get_context(), &event_pool_desc_last, 0, nullptr, &_last_barrier_pool));
    ze_event_desc_t eventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
        0,         // index
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_HOST // ensure memory coherency across device and Host after event completes
    };
    ZE_CHECK(zeEventCreate(_last_barrier_pool, &eventDesc, &_last_barrier_ev));
#endif
#ifdef IMMIDIATLY
    command_queue_desc.flags = 0;
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = 0;
    command_queue_desc.ordinal = 0;
    command_queue_desc.mode = config.queue_type == queue_types::out_of_order ? ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS : ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    
    ZE_CHECK(zeCommandListCreateImmediate(context, device, &command_queue_desc, &_command_list));
#else
    ze_command_list_desc_t command_list_desc = {};
    command_queue_desc.flags = 0;
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = 0;
    command_queue_desc.ordinal = 0;
    command_queue_desc.mode = config.queue_type == queue_types::out_of_order ? ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS : ZE_COMMAND_QUEUE_MODE_SYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;
    ZE_CHECK(zeCommandListCreate(context, device, &command_list_desc, &_command_list));
#endif
}

ze_stream::ze_stream(const ze_engine &engine, void *handle)
    : stream(engine.configuration().queue_type)
    , _engine(engine)
    , sync_method(get_expected_sync_method(engine.configuration())) {
    throw std::runtime_error("ze_stream::ze_stream(const ze_engine &engine, void *handle)");

#ifdef ENABLE_ONEDNN_FOR_GPU
    // auto config = engine.configuration();
    // if (config.queue_type == queue_types::in_order) {
    //     auto onednn_engine = engine.get_onednn_engine();
    //     _onednn_stream = std::make_shared<dnnl::stream>(dnnl::ocl_interop::make_stream(engine.get_onednn_engine(), _command_queue.get()));
    // }
#endif
}

queue_types ze_stream::detect_queue_type(void *queue_handle) {
    return queue_types::out_of_order;
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

    auto ev =  create_base_event();
    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t launchArgs = { global.groupCountX/local.groupCountX, global.groupCountY/local.groupCountY, global.groupCountZ/local.groupCountZ };
    ZE_CHECK(zeKernelSetGroupSize(kern, local.groupCountX, local.groupCountY, local.groupCountZ));
    ZE_CHECK(zeCommandListAppendLaunchKernel(_command_list,
                                    kern,
                                    &launchArgs,
                                    set_output_event ? std::dynamic_pointer_cast<ze_base_event>(ev)->get() : nullptr,
                                    dep_events_ptr == nullptr ? 0 : dep_events_ptr->size(),
                                    dep_events_ptr == nullptr ? 0 : &dep_events_ptr->front()));
    if (!set_output_event) {
        std::dynamic_pointer_cast<ze_base_event>(ev)->set();
    }
    return ev;
}

void ze_stream::enqueue_barrier() {
    ZE_CHECK(zeCommandListAppendBarrier(_command_list, nullptr, 0, nullptr));
}

ze_event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    if (deps.empty()) {
        return create_user_event(true);
    }

    if (sync_method  == sync_methods::events) {
        std::vector<ze_event_handle_t> dep_events;
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
            }
        }
        if (dep_events.empty())
            return create_user_event(true);

        auto ev = create_base_event();
        ZE_CHECK(zeCommandListAppendBarrier(_command_list, std::dynamic_pointer_cast<ze_base_event>(ev)->get(), dep_events.size(), &dep_events.front()));
        return ev;
    } else if (sync_method == sync_methods::barriers) {
        sync_events(deps, is_output);
        auto result = std::make_shared<ze_event>(_last_barrier_pool, _last_barrier_ev, _last_barrier);
        return result;
    } else {
        return create_user_event(true);
    }
    //immidiatly_comand_queue!!!
}

ze_event::ptr ze_stream::group_events(std::vector<ze_events::ptr> const& deps) {
    return std::make_shared<ze_events>(deps);
}

ze_event::ptr ze_stream::create_user_event(bool set) {
#ifndef SINGLE_EVENT_POOL
    ze_event_pool_handle_t _event_pool;
    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
        1 // count
    };
    ZE_CHECK(zeEventPoolCreate(_engine.get_context(), &event_pool_desc, 0, nullptr, &_event_pool));
#endif
    ze_event_handle_t hEvent;
    ze_event_desc_t tsEventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
#ifdef SINGLE_EVENT_POOL
        event_idx++ % pool_size,         // index
#else
        0,         // index
#endif
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_HOST // no additional memory/cache coherency required on wait
    };
    ZE_CHECK(zeEventCreate(_event_pool, &tsEventDesc, &hEvent));
    auto result = std::make_shared<ze_event>(_event_pool, hEvent, set);
    if (set) {
        result->set();
    }
    return result;//_engine.get_context(), set);
}

ze_event::ptr ze_stream::create_base_event() {
    //cl::Event ret_ev;
#ifndef SINGLE_EVENT_POOL
    ze_event_pool_handle_t _event_pool;
    ze_event_pool_desc_t event_pool_desc = {
        ZE_STRUCTURE_TYPE_EVENT_POOL_DESC,
        nullptr,
        ZE_EVENT_POOL_FLAG_HOST_VISIBLE, // all events in pool are visible to Host
        1 // count
    };
    ZE_CHECK(zeEventPoolCreate(_engine.get_context(), &event_pool_desc, 0, nullptr, &_event_pool));
#endif
    ze_event_handle_t hEvent;
    ze_event_desc_t tsEventDesc = {
        ZE_STRUCTURE_TYPE_EVENT_DESC,
        nullptr,
#ifdef SINGLE_EVENT_POOL
        event_idx++ % pool_size,         // index
#else
        0,         // index
#endif
        0,                         // no additional memory/cache coherency required on signal
        ZE_EVENT_SCOPE_FLAG_HOST // no additional memory/cache coherency required on wait
    };

    ZE_CHECK(zeEventCreate(_event_pool, &tsEventDesc, &hEvent));
    auto result =  std::make_shared<ze_event>(_event_pool, hEvent, ++_queue_counter);
    return result;
}

void ze_stream::flush() const {
    ZE_CHECK(zeCommandListClose(_command_list));
    ze_command_queue_handle_t hCommandQueue;
    ZE_CHECK(zeCommandQueueCreate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &hCommandQueue));

    ze_fence_desc_t fenceDesc = {
        ZE_STRUCTURE_TYPE_FENCE_DESC ,
        nullptr,
        0
    };
    ze_fence_handle_t hFence;
    ZE_CHECK(zeFenceCreate(hCommandQueue, &fenceDesc, &hFence));
    ZE_CHECK(zeCommandQueueExecuteCommandLists(hCommandQueue, 1, &_command_list, hFence));
    ZE_CHECK(zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX));
    ZE_CHECK(zeFenceHostSynchronize(hFence, UINT32_MAX));
    ZE_CHECK(zeFenceReset(hFence));
    ZE_CHECK(zeCommandListReset(_command_list));

    //_queue_counter.store(uint64_t(0));
}

void ze_stream::finish() const {
    flush();
    // ZE_CHECK(zeCommandQueueSynchronize(hCommandQueue, UINT32_MAX));
    // //ZE_CHECK(zeFenceHostSynchronize(hFence, UINT32_MAX));
    // //ZE_CHECK(zeFenceReset(hFence));
    // for (auto it_command_list : vec_command_list) {
    //     ZE_CHECK(zeCommandListReset(it_command_list));
    // }
    // vec_command_list.clear();
    // ZE_CHECK(zeCommandQueueDestroy(hCommandQueue));
    // ze_command_queue_desc_t commandQueueDesc = {
    //     ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC ,
    //     nullptr,
    //     0,
    //     0,
    //     0,
    //     ZE_COMMAND_QUEUE_MODE_DEFAULT ,
    //     ZE_COMMAND_QUEUE_PRIORITY_NORMAL
    // };
    // ZE_CHECK(zeCommandQueueCreate(_engine.get_context(), _engine.get_device(), &commandQueueDesc, &hCommandQueue));
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;
    std::vector<ze_event_handle_t> _ze_events;
    for (auto& ev : events) {
        if (auto ze_base_ev = dynamic_cast<ze_base_event*>(ev.get())) {
            _ze_events.push_back(ze_base_ev->get());
        }
    }
    ZE_CHECK(zeCommandListAppendWaitOnEvents(_command_list, _ze_events.size(), &_ze_events.front()));
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
                ZE_CHECK(zeCommandListAppendBarrier(_command_list, _last_barrier_ev, 0, nullptr));//_last_barrier_ev
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