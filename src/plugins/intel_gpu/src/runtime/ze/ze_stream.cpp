// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_stream.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/properties.hpp"
#include "ze_event_pool.hpp"
#include "ze_event.hpp"
#include "ze_kernel.hpp"
#include "ze_memory.hpp"
#include "ze_common.hpp"

#include <ze_api.h>
#include <ze_intel_gpu.h>
#include <ze_stypes.h>

#include <cassert>
#include <string>
#include <vector>
#include <memory>

#ifdef ENABLE_ONEDNN_FOR_GPU
#include <oneapi/dnnl/dnnl_l0.hpp>
#endif

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

template<typename T>
ze_result_t set_kernel_arg_scalar(ze_kernel_handle_t& kernel, uint32_t idx, const T& val) {
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set scalar " << idx << " (" << ov::element::from<T>().get_type_name() << ")" << val << "\n";
    return zeKernelSetArgumentValue(kernel, idx, sizeof(T), &val);
}

ze_result_t set_kernel_arg(ze_kernel_handle_t& kernel, uint32_t idx, cldnn::memory::cptr mem) {
    if (!mem)
        return ZE_RESULT_ERROR_INVALID_ARGUMENT;

    OPENVINO_ASSERT(memory_capabilities::is_usm_type(mem->get_allocation_type()), "Unsupported alloc type");
    const auto& buf = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_buffer();
    auto mem_type = std::dynamic_pointer_cast<const ze::gpu_usm>(mem)->get_allocation_type();
    GPU_DEBUG_TRACE_DETAIL << "kernel: " << kernel << " set arg (" << mem_type << ") " << idx
                            << " mem: " << buf.get() << " size: " << mem->size() << std::endl;

    auto ptr = buf.get();
    return zeKernelSetArgumentValue(kernel, idx, sizeof(ptr), &ptr);
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
                            status = set_kernel_arg_scalar<uint8_t>(kernel, i, scalar.v.u8);
                            break;
                        case scalar_t::UINT16:
                            status = set_kernel_arg_scalar<uint16_t>(kernel, i, scalar.v.u16);
                            break;
                        case scalar_t::UINT32:
                            status = set_kernel_arg_scalar<uint32_t>(kernel, i, scalar.v.u32);
                            break;
                        case scalar_t::UINT64:
                            status = set_kernel_arg_scalar<uint64_t>(kernel, i, scalar.v.u64);
                            break;
                        case scalar_t::INT8:
                            status = set_kernel_arg_scalar<int8_t>(kernel, i, scalar.v.s8);
                            break;
                        case scalar_t::INT16:
                            status = set_kernel_arg_scalar<int16_t>(kernel, i, scalar.v.s16);
                            break;
                        case scalar_t::INT32:
                            status = set_kernel_arg_scalar<int32_t>(kernel, i, scalar.v.s32);
                            break;
                        case scalar_t::INT64:
                            status = set_kernel_arg_scalar<int64_t>(kernel, i, scalar.v.s64);
                            break;
                        case scalar_t::FLOAT32:
                            status = set_kernel_arg_scalar<float>(kernel, i, scalar.v.f32);
                            break;
                        case scalar_t::FLOAT64:
                            status = set_kernel_arg_scalar<double>(kernel, i, scalar.v.f64);
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
        if (status != ZE_RESULT_SUCCESS) {
            throw std::runtime_error("Error set arg " + std::to_string(i) + ", error code: " + std::to_string(status) + "\n");
        }
    }
}

}  // namespace

ze_stream::ze_stream(const ze_engine &engine, const ExecutionConfig& config)
    : stream(config.get_queue_type(), stream::get_expected_sync_method(config))
    , _engine(engine)
    , m_pool(engine, config.get_enable_profiling()) {
    const auto &info = engine.get_device_info();

    ze_command_queue_desc_t command_queue_desc = {};
    command_queue_desc.stype = ZE_STRUCTURE_TYPE_COMMAND_QUEUE_DESC;
    command_queue_desc.pNext = nullptr;
    command_queue_desc.index = 0;
    command_queue_desc.ordinal = info.compute_queue_group_ordinal;
    command_queue_desc.flags = m_queue_type == QueueTypes::out_of_order ? 0 : ZE_COMMAND_QUEUE_FLAG_IN_ORDER;
    command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
    command_queue_desc.priority = ZE_COMMAND_QUEUE_PRIORITY_NORMAL;

    zex_intel_queue_copy_operations_offload_hint_exp_desc_t cp_offload_desc = {};
    cp_offload_desc.stype = ZEX_INTEL_STRUCTURE_TYPE_QUEUE_COPY_OPERATIONS_OFFLOAD_HINT_EXP_PROPERTIES;
    cp_offload_desc.copyOffloadEnabled = true;
    cp_offload_desc.pNext = nullptr;
    if (info.supports_cp_offload) {
        command_queue_desc.pNext = &cp_offload_desc;
    }

    ZE_CHECK(zeCommandListCreateImmediate(_engine.get_context(), _engine.get_device(), &command_queue_desc, &m_command_list));
}

ze_stream::~ze_stream() {
    // Destroy OneDNN stream before destroying command list
    _onednn_stream.reset();
    zeCommandListDestroy(m_command_list);
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
    if (m_sync_method == SyncMethods::events) {
        for (auto& dep : deps) {
            if (auto ze_base_ev = std::dynamic_pointer_cast<ze_base_event>(dep)) {
                if (ze_base_ev->get() != nullptr)
                    dep_events.push_back(ze_base_ev->get());
            }
        }
        dep_events_ptr = &dep_events;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
    }
    bool set_output_event = m_sync_method == SyncMethods::events || is_output;

    auto ev = set_output_event ? create_base_event() : std::make_shared<ze_event>(nullptr, nullptr, ++m_queue_counter);
    auto global = to_group_count(args_desc.workGroups.global);
    auto local = to_group_count(args_desc.workGroups.local);
    ze_group_count_t args = { global.groupCountX / local.groupCountX, global.groupCountY / local.groupCountY, global.groupCountZ / local.groupCountZ };
    ZE_CHECK(zeKernelSetGroupSize(kern, local.groupCountX, local.groupCountY, local.groupCountZ));
    ZE_CHECK(zeCommandListAppendLaunchKernel(m_command_list,
                                             kern,
                                             &args,
                                             set_output_event ? std::dynamic_pointer_cast<ze_base_event>(ev)->get() : nullptr,
                                             dep_events_ptr == nullptr ? 0 : static_cast<uint32_t>(dep_events_ptr->size()),
                                             dep_events_ptr == nullptr ? 0 : &dep_events_ptr->front()));

    return ev;
}

void ze_stream::enqueue_barrier() {
    ZE_CHECK(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
}

event::ptr ze_stream::enqueue_marker(std::vector<ze_event::ptr> const& deps, bool is_output) {
    if (deps.empty()) {
        auto ev = create_base_event();
        ZE_CHECK(zeCommandListAppendBarrier(m_command_list, std::dynamic_pointer_cast<ze_base_event>(ev)->get(), 0, nullptr));
        return ev;
    }

    if (m_sync_method  == SyncMethods::events) {
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
        ZE_CHECK(zeCommandListAppendBarrier(m_command_list,
                                            std::dynamic_pointer_cast<ze_base_event>(ev)->get(),
                                            static_cast<uint32_t>(dep_events.size()),
                                            &dep_events.front()));
        return ev;
    } else if (m_sync_method == SyncMethods::barriers) {
        sync_events(deps, is_output);
        assert(m_last_barrier_ev != nullptr);
        return m_last_barrier_ev;
    } else {
        return create_user_event(true);
    }
}

ze_event::ptr ze_stream::group_events(std::vector<ze_events::ptr> const& deps) {
    return std::make_shared<ze_events>(deps);
}

void ze_stream::wait() {
    finish();
}

event::ptr ze_stream::create_user_event(bool set) {
    auto ev = m_pool.create_user_event();
    if (set)
        ev->set();

    return ev;
}

event::ptr ze_stream::create_base_event() {
    return m_pool.create_event(++m_queue_counter);
}

void ze_stream::flush() const {
    //Immediate Command List submits commands immediately - no flush impl
}

void ze_stream::finish() const {
    ZE_CHECK(zeCommandListHostSynchronize(m_command_list, default_timeout));
}

void ze_stream::wait_for_events(const std::vector<event::ptr>& events) {
    bool needs_sync = false;
    for (auto& ev : events) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(ev.get());
        if (ze_base_ev->get() != nullptr) {
            ze_base_ev->wait();
        } else {
            needs_sync = true;
        }
        // Block thread and wait for event signal
        ev->wait();
    }

    if (needs_sync) {
        finish();
    }
}

void ze_stream::sync_events(std::vector<event::ptr> const& deps, bool is_output) {
    bool needs_barrier = false;
    for (auto& dep : deps) {
        auto* ze_base_ev = dynamic_cast<ze_base_event*>(dep.get());
        assert(ze_base_ev != nullptr);
        if (ze_base_ev->get_queue_stamp() > m_last_barrier) {
            needs_barrier = true;
        }
    }

    if (needs_barrier) {
        if (is_output) {
            m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_base_event());
            m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
            ZE_CHECK(zeCommandListAppendBarrier(m_command_list, m_last_barrier_ev->get(), 0, nullptr));
        } else {
            ZE_CHECK(zeCommandListAppendBarrier(m_command_list, nullptr, 0, nullptr));
        }
        m_last_barrier = ++m_queue_counter;
    }

    if (!m_last_barrier_ev) {
        m_last_barrier_ev = std::dynamic_pointer_cast<ze_event>(create_user_event(true));
        m_last_barrier_ev->set_queue_stamp(m_queue_counter.load());
    }
}

#ifdef ENABLE_ONEDNN_FOR_GPU
dnnl::stream& ze_stream::get_onednn_stream() {
    OPENVINO_ASSERT(m_queue_type == QueueTypes::in_order, "[GPU] Can't create onednn stream handle as onednn doesn't support out-of-order queue");
    OPENVINO_ASSERT(_engine.get_device_info().vendor_id == INTEL_VENDOR_ID, "[GPU] Can't create onednn stream handle as for non-Intel devices");
    if (!_onednn_stream) {
        _onednn_stream = std::make_shared<dnnl::stream>(dnnl::l0_interop::make_stream(_engine.get_onednn_engine(), m_command_list));
    }

    return *_onednn_stream;
}
#endif

}  // namespace ze
}  // namespace cldnn
