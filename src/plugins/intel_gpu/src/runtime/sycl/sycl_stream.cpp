// Copyright (C) 2019-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_stream.hpp"
#include "CL/cl.h"
#include "intel_gpu/runtime/stream.hpp"
#include "sycl_event.hpp"
// #include "sycl_user_event.hpp"
#include "sycl_command_queues_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "sycl_kernel.hpp"
#include "sycl_common.hpp"

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "ocl/ocl_common.hpp"  // for testing purposes
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
#include <oneapi/dnnl/dnnl_ocl.hpp>
#endif

namespace cldnn {
namespace sycl {

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

cl_int set_kernel_arg(ocl::ocl_kernel_type& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return CL_INVALID_ARG_VALUE;

    if (mem->get_layout().format.is_image_2d()) {
        OPENVINO_NOT_IMPLEMENTED;
        // TODO: implement
        // auto buf = std::dynamic_pointer_cast<const sycl::gpu_image2d>(mem)->get_buffer();
        // GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (image) " << idx << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        // return kernel.setArg(idx, buf);
    } else if (memory_capabilities::is_usm_type(mem->get_allocation_type())) {
        OPENVINO_NOT_IMPLEMENTED;
        // TODO: implement
        // auto buf = std::dynamic_pointer_cast<const sycl::gpu_usm>(mem)->get_buffer();
        // auto mem_type = std::dynamic_pointer_cast<const sycl::gpu_usm>(mem)->get_allocation_type();
        // GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (" << mem_type << ") " << idx
        //                        << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        // return kernel.setArgUsm(idx, buf);
    } else {
        auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(mem)->get_buffer();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (buffer) " << idx << " mem: " << &buf
                               << " size: " << mem->size() << std::endl;
        std::vector<cl_mem> cl_buf = ::sycl::get_native<::sycl::backend::opencl>(buf);
        GPU_DEBUG_TRACE_DETAIL << "# of cl_buf: " << cl_buf.size() << std::endl;
        OPENVINO_ASSERT(cl_buf.size() >= 1 && cl_buf[0] != nullptr, "[GPU] SYCL buffer should have one OpenCL buffer handle");
        {
            cl::Buffer cl_buf0(cl_buf[0], true);
            auto cl_buf0_size = cl_buf0.getInfo<CL_MEM_SIZE>();
            GPU_DEBUG_TRACE_DETAIL << "cl_mem[0] = " << cl_buf[0]
                                   << ", size = " << cl_buf0_size
                                   << ", ctx = " << cl_buf0.getInfo<CL_MEM_CONTEXT>().get() << std::endl;
        }

        return kernel.setArg(idx, cl::Buffer(cl_buf[0], true));
    }

    return CL_INVALID_ARG_VALUE;
}

void set_arguments_impl(ocl::ocl_kernel_type& kernel,
                        const arguments_desc& args,
                        const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        switch (args[i].t) {
            case args_t::INPUT:
                if (args[i].index < data.inputs.size() && data.inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
                }
                break;
            case args_t::INPUT_OF_FUSED_PRIMITIVE:
                if (args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
                }
                break;
            case args_t::INTERNAL_BUFFER:
                if (args[i].index < data.intermediates.size() && data.intermediates[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
                }
                break;
            case args_t::OUTPUT:
                if (args[i].index < data.outputs.size() && data.outputs[args[i].index]) {
                    status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
                }
                break;
            case args_t::WEIGHTS:
                status = set_kernel_arg(kernel, i, data.weights);
                break;
            case args_t::BIAS:
                status = set_kernel_arg(kernel, i, data.bias);
                break;
            case args_t::WEIGHTS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.weights_zero_points);
                break;
            case args_t::ACTIVATIONS_ZERO_POINTS:
                status = set_kernel_arg(kernel, i, data.activations_zero_points);
                break;
            case args_t::COMPENSATION:
                status = set_kernel_arg(kernel, i, data.compensation);
                break;
            case args_t::SCALE_TABLE:
                status = set_kernel_arg(kernel, i, data.scale_table);
                break;
            case args_t::SLOPE:
                status = set_kernel_arg(kernel, i, data.slope);
                break;
            case args_t::SCALAR:
                if (data.scalars && args[i].index < data.scalars->size()) {
                    const auto& scalar = (*data.scalars)[args[i].index];
                    switch (scalar.t) {
                        case scalar_t::UINT8:
                            status = kernel.setArg(i, scalar.v.u8);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u8): " << static_cast<int>(scalar.v.u8)
                                << "\n";
                            break;
                        case scalar_t::UINT16:
                            status = kernel.setArg(i, scalar.v.u16);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u16): " << scalar.v.u16 << "\n";
                            break;
                        case scalar_t::UINT32:
                            status = kernel.setArg(i, scalar.v.u32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u32): " << scalar.v.u32 << "\n";
                            break;
                        case scalar_t::UINT64:
                            status = kernel.setArg(i, scalar.v.u64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u64): " << scalar.v.u64 << "\n";
                            break;
                        case scalar_t::INT8:
                            status = kernel.setArg(i, scalar.v.s8);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s8): " << static_cast<int>(scalar.v.s8)
                                << "\n";
                            break;
                        case scalar_t::INT16:
                            status = kernel.setArg(i, scalar.v.s16);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s16): " << scalar.v.s16 << "\n";
                            break;
                        case scalar_t::INT32:
                            status = kernel.setArg(i, scalar.v.s32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s32): " << scalar.v.s32 << "\n";
                            break;
                        case scalar_t::INT64:
                            status = kernel.setArg(i, scalar.v.s64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s64): " << scalar.v.s64 << "\n";
                            break;
                        case scalar_t::FLOAT32:
                            status = kernel.setArg(i, scalar.v.f32);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (f32): " << scalar.v.f32 << "\n";
                            break;
                        case scalar_t::FLOAT64:
                            status = kernel.setArg(i, scalar.v.f64);
                            GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (f64): " << scalar.v.f64 << "\n";
                            break;
                        default:
                            break;
                    }
                }
                break;
            case args_t::CELL:
                status = set_kernel_arg(kernel, i, data.cell);
                break;
            case args_t::SHAPE_INFO:
                status = set_kernel_arg(kernel, i, data.shape_info);
                break;
            default:
                break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i)
                                     + ", kernel: " + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>()
                                     + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

}  // namespace

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

    // bool priorty_extensions = engine.extension_supported("cl_khr_priority_hints") && engine.extension_supported("cl_khr_create_command_queue");
    // queue_builder.set_priority_mode(config.get_queue_priority(), priorty_extensions);

    // bool throttle_extensions = engine.extension_supported("cl_khr_throttle_hints") && engine.extension_supported("cl_khr_create_command_queue");
    // queue_builder.set_throttle_mode(config.get_queue_throttle(), throttle_extensions);

    // bool queue_families_extension = engine.get_device_info().supports_queue_families;
    // queue_builder.set_supports_queue_families(queue_families_extension);

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
        _onednn_stream = std::make_shared<dnnl::stream>(dnnl::ocl_interop::make_stream(_engine.get_onednn_engine(), _command_queue.get()));
    }

    return *_onednn_stream;
}
#endif

QueueTypes sycl_stream::detect_queue_type(void *queue_handle) {
    auto queue = static_cast<::sycl::queue*>(queue_handle);
    return queue->is_in_order() ? QueueTypes::in_order : QueueTypes::out_of_order;
}

void sycl_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    // TODO: add sycl kernel case
    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();

    try {
        // TODO: implement debug trace
        GPU_DEBUG_TRACE_DETAIL << "Set arguments for primitive: " << args_desc.layerID << " (" << kernel.get_id() << " = " << kern.get() << ")\n";
        set_arguments_impl(kern, args_desc.arguments, args);
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

event::ptr sycl_stream::enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& args,
                                      std::vector<event::ptr> const& deps,
                                      bool is_output) {
    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();
    auto global = toNDRange(args_desc.workGroups.global);
    auto local = toNDRange(args_desc.workGroups.local);
    std::vector<::sycl::event> dep_events;
    if (m_sync_method == SyncMethods::events) {
        for (auto& dep : deps) {
            if (auto sycl_base_ev = std::dynamic_pointer_cast<sycl_base_event>(dep)) {
                dep_events.push_back(sycl_base_ev->get());
            }
        }
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }

    // collect memory objects to create accessors
    std::vector<memory::cptr> mems;
    mems.insert(mems.end(), args.inputs.begin(), args.inputs.end());
    mems.insert(mems.end(), args.intermediates.begin(), args.intermediates.end());
    mems.insert(mems.end(), args.outputs.begin(), args.outputs.end());
    mems.insert(mems.end(), args.fused_op_inputs.begin(), args.fused_op_inputs.end());
    mems.insert(mems.end(), {args.weights, args.recurrent, args.hidden, args.cell, args.bias,
                             args.weights_zero_points, args.activations_zero_points,
                             args.compensation, args.lookup_table, args.scale_table,
                             args.slope, args.shape_info});

    try {
        auto sycl_ev = _command_queue.submit([&](::sycl::handler& cgh) {
            cgh.depends_on(dep_events);

            std::vector<::sycl::accessor<std::byte, 1, ::sycl::access::mode::read_write>> accessors;
            for (const auto& mem : mems) {
                if (mem == nullptr) {
                    continue;
                }
                auto buf = std::dynamic_pointer_cast<const sycl::gpu_buffer>(mem)->get_buffer();
                accessors.push_back(buf.get_access<::sycl::access::mode::read_write>(cgh));
                GPU_DEBUG_TRACE_DETAIL << "get accessor for cl_mem" << ::sycl::get_native<::sycl::backend::opencl>(buf)[0] << std::endl;
            }

            cgh.host_task([=](const ::sycl::interop_handle &ih) {
                cl::Event ret_ev;
                auto cl_queue = ih.get_native_queue<::sycl::backend::opencl>();
                auto command_queue = cl::CommandQueue(cl_queue, true);
                try {
                    command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, nullptr, &ret_ev);
                    ret_ev.wait();
                } catch (cl::Error const& err) {
                    ocl::rethrow(err, _engine.get_device_info());
                }
            });
        });

        return create_base_event(sycl_ev);
    } catch (cl::Error const& err) {
        ocl::rethrow(err, _engine.get_device_info());
    } catch (::sycl::exception const& err) {
        sycl::rethrow(err, _engine.get_device_info());
    }
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
        ::sycl::event ev = _command_queue.ext_oneapi_submit_barrier();
        ev.wait();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

void sycl_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty())
        return;

    GPU_DEBUG_TRACE_DETAIL << "sycl_stream::wait_for_events: waiting for " << events.size() << " events" << std::endl;

    bool needs_barrier = false;
    std::vector<::sycl::event> syclevents;
    for (auto& ev : events) {
        if (!ev)
            continue;

        if (auto sycl_base_ev = downcast<sycl_base_event>(ev.get())) {
            syclevents.push_back(sycl_base_ev->get());
        }
    }

    // dead code, because we always use events sync method
    if (needs_barrier) {
        try {
            ::sycl::event barrier_ev = _command_queue.ext_oneapi_submit_barrier();
            syclevents.push_back(barrier_ev);
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
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
