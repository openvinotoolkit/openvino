// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_stream.hpp"
#include "sycl_event.hpp"
#include "sycl_kernel.hpp"
#include "sycl_common.hpp"

#include <cassert>
#include <iomanip>
#include <ios>

#include <fstream>
#include <thread>
#include <string>
#include <vector>
#include <memory>

namespace cldnn {
namespace sycl {

namespace {

inline cl::sycl::nd_range<3> toNDRange(const work_group_sizes& range) {
    auto global_range = range.global;
    auto local_range = range.local;

    auto sycl_global_range = cl::sycl::range<3>(global_range[2], global_range[1], global_range[0]);
    auto sycl_local_range = cl::sycl::range<3>(local_range[2], local_range[1], local_range[0]);
    return cl::sycl::nd_range<3>(sycl_global_range, sycl_local_range);
}

}  // namespace

sycl_stream::sycl_stream(const sycl_engine& engine) : stream(engine.configuration().queue_type), _engine(engine) {
    _command_queue = cl::sycl::queue(engine.get_sycl_context(), engine.get_sycl_device());
}

void sycl_stream::set_arguments(kernel& /* kernel */, const kernel_arguments_desc& /* args_desc */, const kernel_arguments_data& /* args */) {
}

event::ptr sycl_stream::enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& data,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output_event) {
    auto& sycl_kernel = dynamic_cast<sycl::sycl_kernel&>(kernel);

    auto kern = sycl_kernel.get_handle();
    auto range = toNDRange(args_desc.workGroups);
    std::vector<cl::sycl::event> dep_events;
    for (auto& dep : deps) {
        if (auto sycl_ev = dynamic_cast<sycl_event*>(dep.get())) {
            dep_events.push_back(sycl_ev->get());
        }
    }

    try {

        auto event = _command_queue.submit([&](cl::sycl::handler &cgh) {
            cgh.depends_on(dep_events);
            using args_t = argument_desc::Types;
            using scalar_t = scalar_desc::Types;
            for (uint32_t i = 0; i < static_cast<uint32_t>(args_desc.arguments.size()); i++) {
                bool success = false;
                switch (args_desc.arguments[i].t) {
                    case args_t::INPUT:
                        if (args_desc.arguments[i].index < data.inputs.size() && data.inputs[args_desc.arguments[i].index]) {
                            const auto& input_mem = data.inputs[args_desc.arguments[i].index];
                            if (input_mem) {
                                // if (input_mem->get_layout().format.is_image_2d())
                                //     kernel.set_arg(i, std::dynamic_pointer_cast<const sycl::gpu_image2d>(input_mem)->get_buffer());
                                // else if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                                //     kernel.set_arg(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(input_mem)->get_buffer());
                                // else
                                auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(input_mem)->get_buffer();
                                cgh.set_arg(i, buf.get_access<cl::sycl::access::mode::read_write>(cgh));
                            }
                            success = true;
                        }
                        break;
                //     case args_t::INPUT_OF_FUSED_PRIMITIVE:
                //         if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                //             const auto& input_mem = data.fused_op_inputs[args[i].index];
                //             if (input_mem) {
                //                 if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                //                     status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(input_mem)->get_buffer());
                //                 else
                //                     status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(input_mem)->get_buffer());
                //             }
                //         }
                //         break;
                //     case args_t::INTERNAL_BUFFER:
                //         if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                //             const auto& input_mem = data.intermediates[args[i].index];
                //             if (input_mem) {
                //                 if (memory_capabilities::is_usm_type(input_mem->get_allocation_type()))
                //                     status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(input_mem)->get_buffer());
                //                 else
                //                     status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(input_mem)->get_buffer());
                //             }
                //         }
                //         break;
                    case args_t::OUTPUT:
                        if (data.output) {
                                // if (data.output->get_layout().format.is_image_2d())
                                // status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_image2d>(data.output)->get_buffer());
                                // else if (memory_capabilities::is_usm_type(data.output->get_allocation_type()))
                                //     status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(data.output)->get_buffer());
                                // else
                                auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.output)->get_buffer();
                                cgh.set_arg(i, buf.get_access<cl::sycl::access::mode::read_write>(cgh));
                        }
                        break;
                    case args_t::WEIGHTS:
                        if (data.weights) {
                            // if (data.weights->get_layout().format.is_image_2d())
                            //     status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_image2d>(data.weights)->get_buffer());
                            // else if (memory_capabilities::is_usm_type(data.weights->get_allocation_type()))
                            //     status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(data.weights)->get_buffer());
                            // else
                                // status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.weights)->get_buffer());
                            auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.weights)->get_buffer();
                            cgh.set_arg(i, buf.get_access<cl::sycl::access::mode::read_write>(cgh));
                        }
                        break;
                    case args_t::BIAS:
                        if (data.bias) {
                //             if (memory_capabilities::is_usm_type(data.bias->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(data.bias)->get_buffer());
                //             else
                //                 status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.bias)->get_buffer());
                            auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.bias)->get_buffer();
                            cgh.set_arg(i, buf.get_access<cl::sycl::access::mode::read_write>(cgh));
                        }
                        break;
                //     case args_t::WEIGHTS_ZERO_POINTS:
                //         if (data.weights_zero_points) {
                //             if (memory_capabilities::is_usm_type(data.weights_zero_points->get_allocation_type()))
                //                 status = kernel.setArgUsm(
                //                     i,
                //                     std::dynamic_pointer_cast<const sycl::gpu_usm>(data.weights_zero_points)->get_buffer());
                //             else
                //                 status = kernel.setArg(
                //                     i,
                //                     std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.weights_zero_points)->get_buffer());
                //         }
                //         break;
                //     case args_t::ACTIVATIONS_ZERO_POINTS:
                //         if (data.activations_zero_points) {
                //             if (memory_capabilities::is_usm_type(data.activations_zero_points->get_allocation_type()))
                //                 status = kernel.setArgUsm(
                //                     i,
                //                     std::dynamic_pointer_cast<const sycl::gpu_usm>(data.activations_zero_points)->get_buffer());
                //             else
                //                 status = kernel.setArg(
                //                     i,
                //                     std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.activations_zero_points)->get_buffer());
                //         }
                //         break;
                //     case args_t::COMPENSATION:
                //         if (data.compensation) {
                //             if (memory_capabilities::is_usm_type(data.compensation->get_allocation_type()))
                //                 status = kernel.setArgUsm(
                //                         i,
                //                         std::dynamic_pointer_cast<const sycl::gpu_usm>(data.compensation)->get_buffer());
                //             else
                //                 status = kernel.setArg(
                //                             i,
                //                             std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.compensation)->get_buffer());
                //         }
                //         break;
                //     case args_t::SCALE_TABLE:
                //         if (data.scale_table) {
                //             if (memory_capabilities::is_usm_type(data.scale_table->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(data.scale_table)->get_buffer());
                //             else
                //                 status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.scale_table)->get_buffer());
                //         }
                //         break;
                //     case args_t::SLOPE:
                //         if (data.slope) {
                //             if (memory_capabilities::is_usm_type(data.slope->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, std::dynamic_pointer_cast<const sycl::gpu_usm>(data.slope)->get_buffer());
                //             else
                //                 status = kernel.setArg(i, std::dynamic_pointer_cast<const sycl::gpu_buffer>(data.slope)->get_buffer());
                //         }
                //         break;
                    case args_t::SPLIT:
                        cgh.set_arg(i, data.split);
                        break;
                //     case args_t::SCALAR:
                //         if (data.scalars && args[i].index < data.scalars->size()) {
                //             const auto& scalar = (*data.scalars)[args[i].index];
                //             switch (scalar.t) {
                //                 case scalar_t::UINT8:
                //                     status = kernel.setArg(i, scalar.v.u8);
                //                     break;
                //                 case scalar_t::UINT16:
                //                     status = kernel.setArg(i, scalar.v.u16);
                //                     break;
                //                 case scalar_t::UINT32:
                //                     status = kernel.setArg(i, scalar.v.u32);
                //                     break;
                //                 case scalar_t::UINT64:
                //                     status = kernel.setArg(i, scalar.v.u64);
                //                     break;
                //                 case scalar_t::INT8:
                //                     status = kernel.setArg(i, scalar.v.s8);
                //                     break;
                //                 case scalar_t::INT16:
                //                     status = kernel.setArg(i, scalar.v.s16);
                //                     break;
                //                 case scalar_t::INT32:
                //                     status = kernel.setArg(i, scalar.v.s32);
                //                     break;
                //                 case scalar_t::INT64:
                //                     status = kernel.setArg(i, scalar.v.s64);
                //                     break;
                //                 case scalar_t::FLOAT32:
                //                     status = kernel.setArg(i, scalar.v.f32);
                //                     break;
                //                 case scalar_t::FLOAT64:
                //                     status = kernel.setArg(i, scalar.v.f64);
                //                     break;
                //                 default:
                //                     break;
                //             }
                //         }
                //         break;
                //     case args_t::RECURRENT:  // RNN/LSTM/GRU layers
                //         if (data.recurrent) {
                //             if (data.recurrent->get_layout().format.is_image_2d())
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_image2d&>(*data.recurrent).get_buffer());
                //             else if (memory_capabilities::is_usm_type(data.recurrent->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, dynamic_cast<const sycl::gpu_usm&>(*data.recurrent).get_buffer());
                //             else
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_buffer&>(*data.recurrent).get_buffer());
                //         }
                //         break;
                //     case args_t::HIDDEN:  // RNN/LSTM/GRU layers
                //         if (data.hidden) {
                //             if (data.hidden->get_layout().format.is_image_2d())
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_image2d&>(*data.hidden).get_buffer());
                //             else if (memory_capabilities::is_usm_type(data.hidden->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, dynamic_cast<const sycl::gpu_usm&>(*data.hidden).get_buffer());
                //             else
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_buffer&>(*data.hidden).get_buffer());
                //         }
                //         break;
                //     case args_t::CELL:  // LSTMlayers
                //         if (data.cell) {
                //             if (data.cell->get_layout().format.is_image_2d())
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_image2d&>(*data.cell).get_buffer());
                //             else if (memory_capabilities::is_usm_type(data.cell->get_allocation_type()))
                //                 status = kernel.setArgUsm(i, dynamic_cast<const sycl::gpu_usm&>(*data.cell).get_buffer());
                //             else
                //                 status = kernel.setArg(i, dynamic_cast<const sycl::gpu_buffer&>(*data.cell).get_buffer());
                //         }
                //         break;
                    default:
                        break;
                }
            }
            cgh.parallel_for(range, kern);
        });
        return std::make_shared<sycl_event>(_engine.get_sycl_context(), event);
    } catch (std::exception& e) {
        std::cerr << "EXEC FAILS: " << e.what() << std::endl;
        throw;
    }

}

void sycl_stream::enqueue_barrier() {
    queue().submit_barrier();
}

event::ptr sycl_stream::enqueue_marker(std::vector<event::ptr> const& deps, bool is_output_event) {
    if (deps.empty())
        return std::make_shared<sycl_event>(_engine.get_sycl_context());

    std::vector<cl::sycl::event> dep_events;
    for (auto& dep : deps) {
        if (auto sycl_ev = dynamic_cast<sycl_event*>(dep.get()))
            dep_events.push_back(sycl_ev->get());
    }

    auto event = _command_queue.submit([&](cl::sycl::handler &cgh) {
        cgh.barrier(dep_events);
    });
    return std::make_shared<sycl_event>(_engine.get_sycl_context(), event);
}

event::ptr sycl_stream::group_events(std::vector<event::ptr> const& deps) {
    throw std::runtime_error("stream method is not implemented");
}

event::ptr sycl_stream::create_user_event(bool set) {
    return std::make_shared<sycl_event>(_engine.get_sycl_context());
}

event::ptr sycl_stream::create_base_event() {
    return std::make_shared<sycl_event>(_engine.get_sycl_context());
}

void sycl_stream::reset_events() {
    // throw std::runtime_error("stream method is not implemented");
}

void sycl_stream::release_events_pool() {
    throw std::runtime_error("stream method is not implemented");
}

void sycl_stream::flush() const { }

void sycl_stream::finish() const {
    _command_queue.wait();
}

void sycl_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;

    std::vector<cl::sycl::event> sycl_events;
    for (auto& ev : events) {
        if (auto sycl_ev = dynamic_cast<sycl_event*>(ev.get()))
            sycl_events.push_back(sycl_ev->get());
    }

    cl::sycl::event::wait_and_throw(sycl_events);
}

void sycl_stream::release_pending_memory() {
    // /*
    // TODO: Temp. solution, untill proper API calls from OpenCL are released.
    // */
    // void* ptr = nullptr;
    // ptr = _mm_malloc(4096, 4096);
    // queue().finish();
    // try {
    //     cl::Buffer flusher(context()->context(), CL_MEM_USE_HOST_PTR, (size_t)4096, ptr);
    //     flusher = (cl_mem) nullptr;  // clear buffer
    // } catch (...) {
    //     _mm_free(ptr);
    //     throw;
    // }
    // _mm_free(ptr);
}

void sycl_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output_event) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* sycl_ev = dynamic_cast<sycl_event*>(dep.get());
        if (sycl_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        _command_queue.wait();
        _last_barrier = ++_queue_counter;
    }
}

}  // namespace sycl
}  // namespace cldnn
