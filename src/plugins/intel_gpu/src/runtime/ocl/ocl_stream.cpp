// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_stream.hpp"
#include "CL/cl.h"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/primitives/paged_attention.hpp"
#include "intel_gpu/runtime/kv_chunk_table.hpp"
#include "ocl_event.hpp"
#include "ocl_user_event.hpp"
#include "ocl_command_queues_builder.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "ocl_kernel.hpp"
#include "ocl_common.hpp"
#include "ocl_memory.hpp"

#include <cassert>
#include <memory>
#include <algorithm>

// NOTE: Due to buggy scope transition of warnings we need to disable warning in place of use/instantation
//       of some types (even though we already disabled them in scope of definition of these types).
//       Moreover this warning is pretty much now only for annoyance: it is generated due to lack
//       of proper support for mangling of custom GCC attributes into type name (usually when used
//       with templates, even from standard library).
#if defined __GNUC__ && __GNUC__ >= 6
#    pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#ifdef ENABLE_ONEDNN_FOR_GPU
#    include <oneapi/dnnl/dnnl_ocl.hpp>
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

cl_int set_kernel_arg(ocl_kernel_type& kernel, uint32_t idx, uint32_t size) {
    if (size == 0) {
        return CL_INVALID_ARG_VALUE;
    }

    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg " << idx << " local memory size : " << size << std::endl;
    return kernel.setArg(idx, size, nullptr);
}

cl_int set_kernel_arg(ocl_kernel_type& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem) {
        return CL_INVALID_ARG_VALUE;
    }

    if (mem->get_layout().format.is_image_2d()) {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_image2d>(mem)->get_buffer();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (image) " << idx << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
        return kernel.setArg(idx, buf);
    }
    if (memory_capabilities::is_usm_type(mem->get_allocation_type())) {
        auto buf = std::dynamic_pointer_cast<const ocl::gpu_usm>(mem)->get_buffer();
        auto mem_type = std::dynamic_pointer_cast<const ocl::gpu_usm>(mem)->get_allocation_type();
        GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (" << mem_type << ") " << idx << " mem: " << buf.get() << " size: " << mem->size()
                               << std::endl;
        return kernel.setArgUsm(idx, buf);
    }
    auto buf = std::dynamic_pointer_cast<const ocl::gpu_buffer>(mem)->get_buffer();
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set arg (buffer) " << idx << " mem: " << buf.get() << " size: " << mem->size() << std::endl;
    return kernel.setArg(idx, buf);

    return CL_INVALID_ARG_VALUE;
}

void set_arguments_impl(ocl_kernel_type& kernel, const arguments_desc& args, const kernel_arguments_data& data) {
    using args_t = argument_desc::Types;
    using scalar_t = scalar_desc::Types;
    for (uint32_t i = 0; i < static_cast<uint32_t>(args.size()); i++) {
        cl_int status = CL_INVALID_ARG_VALUE;
        switch (args[i].t) {
        case args_t::INPUT:
            OPENVINO_ASSERT(args[i].index < data.inputs.size() && data.inputs[args[i].index],
                            "The allocated input memory is necessary to set kernel arguments.");
            status = set_kernel_arg(kernel, i, data.inputs[args[i].index]);
            break;
        case args_t::INPUT_OF_FUSED_PRIMITIVE:
            OPENVINO_ASSERT(args[i].index < data.fused_op_inputs.size() && data.fused_op_inputs[args[i].index],
                            "The allocated fused_op_input memory is necessary to set kernel arguments.");
            status = set_kernel_arg(kernel, i, data.fused_op_inputs[args[i].index]);
            break;
        case args_t::INTERNAL_BUFFER:
            OPENVINO_ASSERT(args[i].index < data.intermediates.size() && data.intermediates[args[i].index],
                            "The allocated intermediate memory is necessary to set kernel arguments.");
            status = set_kernel_arg(kernel, i, data.intermediates[args[i].index]);
            break;
        case args_t::OUTPUT:
            OPENVINO_ASSERT(args[i].index < data.outputs.size() && data.outputs[args[i].index],
                            "The allocated output memory is necessary to set kernel arguments.");
            status = set_kernel_arg(kernel, i, data.outputs[args[i].index]);
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
                    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (u8): " << static_cast<int>(scalar.v.u8) << "\n";
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
                    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel.get() << " set scalar " << i << " (s8): " << static_cast<int>(scalar.v.s8) << "\n";
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
        case args_t::LOCAL_MEMORY_SIZE:
            OPENVINO_ASSERT(args[i].index < data.local_memory_args->size() && data.local_memory_args->at(args[i].index),
                            "The allocated local memory is necessary to set kernel arguments.");
            status = set_kernel_arg(kernel, i, static_cast<uint32_t>(data.local_memory_args->at(args[i].index)));
            break;
            break;
        default:
            break;
        }

        if (status != CL_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", kernel: " + kernel.getInfo<CL_KERNEL_FUNCTION_NAME>() +
                                     ", error code: " + std::to_string(status) + "\n");
        }
    }
}

std::shared_ptr<device_clock_sync> make_device_clock(const ocl_engine& engine, const ExecutionConfig& config) {
    if (!config.get_enable_profiling())
        return nullptr;

    auto clock = std::make_shared<device_clock_sync>(engine.get_cl_device());
    return clock->is_valid() ? clock : nullptr;
}

}  // namespace

ocl_stream::ocl_stream(const ocl_engine& engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config)),
      _engine(engine) {
    auto context = engine.get_cl_context();
    auto device = engine.get_cl_device();
    ocl::command_queues_builder queue_builder;
    queue_builder.set_profiling(config.get_enable_profiling());
    queue_builder.set_out_of_order(m_queue_type == QueueTypes::out_of_order);

    OPENVINO_ASSERT(m_sync_method != SyncMethods::none || m_queue_type == QueueTypes::in_order,
                    "[GPU] Unexpected sync method (none) is specified for out_of_order queue");

    bool priorty_extensions = engine.extension_supported("cl_khr_priority_hints") && engine.extension_supported("cl_khr_create_command_queue");
    queue_builder.set_priority_mode(config.get_queue_priority(), priorty_extensions);

    bool throttle_extensions = engine.extension_supported("cl_khr_throttle_hints") && engine.extension_supported("cl_khr_create_command_queue");
    queue_builder.set_throttle_mode(config.get_queue_throttle(), throttle_extensions);

    bool queue_families_extension = engine.get_device_info().supports_queue_families;
    queue_builder.set_supports_queue_families(queue_families_extension);

    _command_queue = queue_builder.build(context, device);
    _device_clock = make_device_clock(_engine, config);
}

ocl_stream::ocl_stream(const ocl_engine& engine, const ExecutionConfig& config, void* handle)
    : stream(ocl_stream::detect_queue_type(handle), stream::get_expected_sync_method(config)),
      _engine(engine) {
    auto* casted_handle = static_cast<cl_command_queue>(handle);
    _command_queue = ocl_queue_type(casted_handle, true);
    _device_clock = make_device_clock(_engine, config);
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ocl_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order, "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID, "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
#    ifdef OV_GPU_WITH_ZE_RT
        OPENVINO_THROW("[GPU] Using OCL OneDNN API with Level Zero runtime");
#    else
        _onednn_stream = std::make_shared<dnnl::stream>(dnnl::ocl_interop::make_stream(_engine.get_onednn_engine(), _command_queue.get()));
#    endif
    }

    return *_onednn_stream;
}
#endif

QueueTypes ocl_stream::detect_queue_type(void* queue_handle) {
    cl_command_queue queue = static_cast<cl_command_queue>(queue_handle);
    cl_command_queue_properties properties;
    auto status = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), &properties, nullptr);
    if (status != CL_SUCCESS) {
        throw std::runtime_error("Can't get queue properties for user handle\n");
    }

    return (properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) ? QueueTypes::out_of_order : QueueTypes::in_order;
}

void ocl_stream::set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    static std::mutex m;
    std::lock_guard<std::mutex> guard(m);

    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();

    try {
        GPU_DEBUG_TRACE_DETAIL << "Set arguments for primitive: " << args_desc.layerID << " (" << kernel.get_id() << " = " << kern.get() << ")\n";
        set_arguments_impl(kern, args_desc.arguments, args);

        // The global KV chunk table (PagedAttentionExtension input 28) holds USM device pointers that the kernels
        // dereference indirectly (loaded from a regular argument buffer, not passed as direct kernel arguments),
        // which the driver cannot discover through normal argument analysis. They must be registered through
        // CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL, otherwise the first dispatch faults with CL_OUT_OF_RESOURCES.
        // Only this layer's K/V chunk pointers are registered (the layer index travels as the SCALAR argument that
        // directly follows the table INPUT argument). The set only changes when a chunk is appended or released, so
        // it is compared against what was last registered for this kernel and the driver call is skipped otherwise.
        // NOTE: several different kernels share the primitive's layerID and each binds its own subset of the inputs,
        // so the table is matched by its declared input index, never by position.
        const bool is_paged_attention = args_desc.layerID.find("pagedattentionextension") != std::string::npos;
        if (is_paged_attention) {
            using InputIdx = cldnn::paged_attention::PagedAttentionInputIdx;
            namespace kv_chunk_table = ov::intel_gpu::kv_chunk_table;
            const auto& arguments = args_desc.arguments;
            for (size_t arg_pos = 0; arg_pos < arguments.size(); ++arg_pos) {
                if (arguments[arg_pos].t != argument_desc::Types::INPUT ||
                    arguments[arg_pos].index != static_cast<uint32_t>(InputIdx::KV_CHUNK_TABLE)) {
                    continue;
                }
                const size_t table_index = static_cast<size_t>(InputIdx::KV_CHUNK_TABLE);
                OPENVINO_ASSERT(table_index < args.inputs.size() && args.inputs[table_index] && args.scalars != nullptr &&
                                    arg_pos + 1 < arguments.size() && arguments[arg_pos + 1].t == argument_desc::Types::SCALAR,
                                "[GPU] A chunked PagedAttentionExtension dispatch needs the kv_chunk_table input followed by the layer index scalar");
                const size_t layer = args.scalars->at(arguments[arg_pos + 1].index).v.u32;

                mem_lock<uint64_t, mem_lock_type::read> lock(std::const_pointer_cast<memory>(args.inputs[table_index]), *this);
                OPENVINO_ASSERT(lock.size() * sizeof(uint64_t) >= kv_chunk_table::kHeaderSlots * sizeof(uint64_t) && kv_chunk_table::is_valid(lock.data()),
                                "[GPU] kv_chunk_table input is not initialized (missing header)");
                const auto header = kv_chunk_table::read_header(lock.data());
                OPENVINO_ASSERT(layer < header.num_layers && header.live_chunks <= header.max_chunks &&
                                    lock.size() >= kv_chunk_table::total_slots(header.num_layers, header.max_chunks, header.max_granules),
                                "[GPU] kv_chunk_table header is inconsistent with the PagedAttentionExtension layer ", layer);

                std::vector<void*> ptrs;
                ptrs.reserve(2 * header.live_chunks);
                const size_t base = kv_chunk_table::layer_base_slot(header, layer);
                for (uint32_t i = 0; i < header.live_chunks; ++i) {
                    ptrs.push_back(reinterpret_cast<void*>(lock[base + i]));
                }
                for (uint32_t i = 0; i < header.live_chunks; ++i) {
                    ptrs.push_back(reinterpret_cast<void*>(lock[base + header.max_chunks + i]));
                }
                if (!ptrs.empty()) {
                    auto& last_registered = _last_registered_chunk_exec_info[kern.get()];
                    if (last_registered != ptrs) {
                        cl_int exec_info_err = clSetKernelExecInfo(kern.get(),
                                                                    CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                                                                    ptrs.size() * sizeof(void*),
                                                                    ptrs.data());
                        OPENVINO_ASSERT(exec_info_err == CL_SUCCESS,
                                        "[GPU] Failed to set USM_PTRS_INTEL exec info for ",
                                        ptrs.size(),
                                        " chunk pointers, error code: ",
                                        exec_info_err);
                        last_registered = std::move(ptrs);
                    }
                }
                break;
            }
        }
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

event::ptr ocl_stream::enqueue_kernel(kernel& kernel,
                                      const kernel_arguments_desc& args_desc,
                                      const kernel_arguments_data& /* args */,
                                      const std::vector<event::ptr>& deps,
                                      bool is_output) {
    auto& ocl_kernel = downcast<ocl::ocl_kernel>(kernel);

    auto& kern = ocl_kernel.get_handle();
    auto global = toNDRange(args_desc.workGroups.global);
    auto local = toNDRange(args_desc.workGroups.local);
    std::vector<cl::Event> dep_events;
    std::vector<cl::Event>* dep_events_ptr = nullptr;
    if (m_sync_method == SyncMethods::events) {
        dep_events = utils::get_cl_events(deps);
        dep_events_ptr = &dep_events;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }

    cl::Event ret_ev;

    bool set_output_event = m_sync_method == SyncMethods::events || is_output;

    try {
        _command_queue.enqueueNDRangeKernel(kern, cl::NullRange, global, local, dep_events_ptr, set_output_event ? &ret_ev : nullptr);
    } catch (const cl::Error& err) {
        ocl::rethrow(err, _engine.get_device_info());
    }

    // Diagnostic: force per-kernel completion so CL_OUT_OF_RESOURCES surfaces at the exact cldnn OCL kernel.
    GPU_DEBUG_IF(ov::intel_gpu::ExecutionConfig::get_finish_after_enqueue()) {
        try {
            _command_queue.finish();
        } catch (cl::Error const& err) {
            std::string kname;
            try { kname = kern.getInfo<CL_KERNEL_FUNCTION_NAME>(); } catch (...) {}
            std::fprintf(stderr, "[FINISH_AFTER_ENQUEUE][cldnn] kernel='%s' finish err=%d\n", kname.c_str(), err.err());
            std::fflush(stderr);
            ocl::rethrow(err, _engine.get_device_info());
        }
    }

    return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
}

void ocl_stream::enqueue_barrier(const std::vector<event::ptr>& deps) {
    if (deps.empty()) {
        try {
            _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    } else {
        auto dep_events = utils::get_cl_events(deps);
        if (dep_events.empty()) {
            return;
        }
        try {
            _command_queue.enqueueBarrierWithWaitList(&dep_events, nullptr);
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
}

event::ptr ocl_stream::enqueue_marker(const std::vector<event::ptr>& deps, bool is_output) {
    // Wait for all previously enqueued tasks if deps list is empty
    if (deps.empty()) {
        cl::Event ret_ev;
        try {
            _command_queue.enqueueMarkerWithWaitList(nullptr, &ret_ev);
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }

        return std::make_shared<ocl_event>(ret_ev);
    }

    if (m_sync_method == SyncMethods::events) {
        cl::Event ret_ev;
        std::vector<cl::Event> dep_events = utils::get_cl_events(deps);
        try {
            if (dep_events.empty()) {
                return create_user_event(true);
            }
            _command_queue.enqueueMarkerWithWaitList(&dep_events, &ret_ev);
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }

        return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
    }
    if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        return std::make_shared<ocl_event>(_last_barrier_ev, _last_barrier);
    }
    return std::make_shared<ocl_user_event>(_engine.get_cl_context(), true, _device_clock);
}

event::ptr ocl_stream::group_events(const std::vector<event::ptr>& deps) {
    if (deps.size() == 1) {
        return deps[0];
    }
    return std::make_shared<ocl_events>(deps);
}

event::ptr ocl_stream::create_user_event(bool set) {
    return std::make_shared<ocl_user_event>(_engine.get_cl_context(), set, _device_clock);
}

event::ptr ocl_stream::create_base_event() {
    cl::Event ret_ev;
    return std::make_shared<ocl_event>(ret_ev, ++_queue_counter);
}

std::unique_ptr<surfaces_lock> ocl_stream::create_surfaces_lock(const std::vector<memory::ptr>& mem) const {
    return std::unique_ptr<ocl::ocl_surfaces_lock>(new ocl::ocl_surfaces_lock(mem, *this));
}

void ocl_stream::flush() const {
    try {
        get_cl_queue().flush();
    } catch (const cl::Error& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}
void ocl_stream::finish() const {
    try {
        get_cl_queue().finish();
    } catch (const cl::Error& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
}

void ocl_stream::wait() {
    cl::Event ev;

    // Enqueue barrier with empty wait list to wait for all previously enqueued tasks
    try {
        _command_queue.enqueueBarrierWithWaitList(nullptr, &ev);
    } catch (const cl::Error& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    ev.wait();
}

void ocl_stream::wait_for_events(const std::vector<event::ptr>& events) {
    if (events.empty()) {
        return;
    }

    bool needs_barrier = false;
    std::vector<cl_event> clevents;
    for (const auto& ev : events) {
        if (!ev) {
            continue;
        }

        if (auto* ocl_base_ev = downcast<ocl_base_event>(ev.get())) {
            if (ocl_base_ev->get().get() != nullptr) {
                clevents.push_back(ocl_base_ev->get().get());
            } else {
                needs_barrier = true;
            }
        }
    }

    cl::Event barrier_ev;
    if (needs_barrier) {
        try {
            _command_queue.enqueueBarrierWithWaitList(nullptr, &barrier_ev);
            clevents.push_back(barrier_ev.get());
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }

    if (!clevents.empty()) {
        auto err = clWaitForEvents(static_cast<cl_uint>(clevents.size()), &clevents[0]);
        if (err != CL_SUCCESS) {
            OPENVINO_THROW("[GPU] clWaitForEvents failed with ", err, " code");
        }
    }
}

void ocl_stream::sync_events(const std::vector<event::ptr>& deps, bool is_output) {
    bool needs_barrier = false;
    for (const auto& dep : deps) {
        auto* ocl_base_ev = downcast<ocl_base_event>(dep.get());
        if (ocl_base_ev->get_queue_stamp() > _last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        try {
            if (is_output) {
                _command_queue.enqueueBarrierWithWaitList(nullptr, &_last_barrier_ev);
            } else {
                _command_queue.enqueueBarrierWithWaitList(nullptr, nullptr);
            }
        } catch (const cl::Error& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }

        _last_barrier = ++_queue_counter;
    }
}

}  // namespace ocl
}  // namespace cldnn
