// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include <memory>
#include <limits>
#include <utility>

namespace ov::intel_gpu {

VariableState::VariableState(const VariableStateInfo& info, RemoteContextImpl::Ptr context,
                             std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                             std::shared_ptr<cldnn::tensor_conversion> conversion)
    : VariableStateBase{info.m_id, context}
    , m_layout(info.m_layout)
    , m_user_specified_type(info.m_user_specified_type)
    , m_shape_predictor(shape_predictor)
    , m_prim_inst(info.m_release_variable_inst)
    , m_transpose_required(info.transpose_required)
    , m_conversion(std::move(conversion))
    , m_initial_layout(info.m_layout) {
    update_device_buffer();
}

VariableState::~VariableState() {
    try {
        wait_for_gpu_conversion();
    } catch (...) {
        try {
            m_context->get_engine().get_service_stream().finish();
            m_pending.clear_after_completion();
        } catch (...) {
        }
    }
}

void VariableState::wait_for_gpu_conversion() const {
    m_pending.wait();
}

void VariableState::reset() {
    wait_for_gpu_conversion();
    m_is_set = false;
    set_layout(m_initial_layout);
    for (auto& user : m_prim_inst) {
        if (const auto prim = user.lock(); prim) {
            prim->release_variable();
        }
    }
}

cldnn::memory::ptr VariableState::get_memory() const {
    wait_for_gpu_conversion();
    return m_memory;
}

const cldnn::layout& VariableState::get_layout() const {
    return m_layout;
}

void VariableState::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    wait_for_gpu_conversion();
    GPU_DEBUG_TRACE_DETAIL << m_name << " : Update memory (Ptr : " << new_mem->buffer_ptr()
                           << ", layout : " << actual_layout.to_short_string() << ")" << std::endl;
    m_memory = new_mem;
    m_layout = actual_layout;
    actual_size = m_memory->size();
    update_device_buffer();
}

void VariableState::set_layout(const cldnn::layout& new_layout) {
    wait_for_gpu_conversion();
    if (m_layout == new_layout) {
        return;
    }
    m_layout = new_layout;
    GPU_DEBUG_TRACE_DETAIL << m_name << " : " << "Update state layout to " << new_layout.to_short_string() << std::endl;
    update_device_buffer();
}

void VariableState::set_state(const ov::SoPtr<ov::ITensor>& state) {
    wait_for_gpu_conversion();
    auto src_shape = state->get_shape();
    size_t src_rank = src_shape.size();
    cldnn::padding::DynamicDimsMask dynamic_pad_dims;
    for (size_t i = 0; i < src_rank; i++) {
        dynamic_pad_dims[i] = m_layout.data_padding._dynamic_dims_mask[i];
    }
    m_layout.data_padding = cldnn::padding(std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           std::vector<ov::Dimension::value_type>(src_rank, 0),
                                           dynamic_pad_dims);
    auto src_stride = state->get_strides();
    OPENVINO_ASSERT(src_stride.size() == src_rank, "[GPU] State source stride rank mismatch");
    for (size_t i = 0; i < src_rank; ++i) {
        src_stride[i] /= state->get_element_type().bitwidth() / 8;
    }
    m_layout.set_partial_shape(src_shape);
    update_device_buffer();

    if (actual_size == 0) {
        set();
        return;
    }

    // Derive upper padding from adjacent strides to preserve gaps in tensor views.
    std::vector<ov::Dimension::value_type> upper_pad(src_rank, 0);
    std::vector<ov::Dimension::value_type> lower_pad(src_rank, 0);
    OPENVINO_ASSERT(src_stride.empty() || src_stride.back() == 1,
                    "[GPU] State source innermost stride must be one element");
    for (size_t i = src_rank; i-- > 1;) {
        OPENVINO_ASSERT(src_stride[i] > 0 && src_stride[i - 1] % src_stride[i] == 0,
                        "[GPU] State source strides cannot be represented as padding");
        const size_t padded_size = src_stride[i - 1] / src_stride[i];
        OPENVINO_ASSERT(padded_size >= src_shape[i], "[GPU] Overlapping state source strides are unsupported");
        OPENVINO_ASSERT(padded_size - src_shape[i] <=
                        static_cast<size_t>(std::numeric_limits<ov::Dimension::value_type>::max()),
                        "[GPU] State source padding is too large");
        upper_pad[i] = static_cast<ov::Dimension::value_type>(padded_size - src_shape[i]);
    }
    cldnn::padding src_padd = cldnn::padding(lower_pad, upper_pad);
    auto src_fmt = cldnn::format::get_default_format(src_rank);
    auto src_layout = cldnn::layout(ov::PartialShape(src_shape), state->get_element_type(), src_fmt, src_padd);

    auto& stream = m_context->get_engine().get_service_stream();
    bool gpu_conversion_submitted = false;
    if (state->get_element_type() == get_user_specified_type() &&
        state->get_element_type() != m_layout.data_type) {
        gpu_conversion_submitted = convert_and_copy_async(state._ptr.get(), m_memory, stream, src_layout, m_conversion,
                                                         m_pending, m_transpose_required);
    }
    if (gpu_conversion_submitted) {
        GPU_DEBUG_INFO << "[state_conversion] GPU state=" << m_name
                       << " dtype=" << state->get_element_type() << "->" << ov::element::Type(m_layout.data_type)
                       << " count=" << ov::shape_size(src_shape) << " padded=" << static_cast<bool>(src_padd)
                       << " transpose=" << m_transpose_required << std::endl;
        set();
        return;
    }
    m_pending.clear_after_completion();
    convert_and_copy(state._ptr.get(), m_memory, stream, src_layout, m_transpose_required);
    set();
}

void VariableState::update_device_buffer() {
    OPENVINO_ASSERT(m_context != nullptr, "m_context should not be null.");
    if (m_layout.is_dynamic() || m_layout.bytes_count() == 0) {
        m_shape_predictor->reset();
        m_memory.reset();
        actual_size = 0;
        return;
    }

    if (actual_size < m_layout.bytes_count()) {
        const auto alloc_type = m_context->get_engine().use_unified_shared_memory() ? cldnn::allocation_type::usm_device : cldnn::allocation_type::cl_mem;
        const auto current_buf_size = m_layout.get_padded_dims();
        ov::Shape current_shape(current_buf_size.begin(), current_buf_size.end());
        const auto alloc_shape = predict_shape(m_name, cldnn::layout(current_shape, m_layout.data_type, m_layout.format), *m_shape_predictor);
        const auto alloc_layout = cldnn::layout(alloc_shape, m_layout.data_type, m_layout.format);
        m_memory = m_context->get_engine().allocate_memory(alloc_layout, alloc_type, false);
        actual_size = std::max(actual_size, alloc_layout.bytes_count());
    }

    OPENVINO_ASSERT(m_memory != nullptr, "m_memory is nullptr!!!");
    m_memory = m_context->get_engine().reinterpret_buffer(*m_memory, m_layout);
}

ov::element::Type VariableState::get_user_specified_type() const {
    return m_user_specified_type != ov::element::dynamic ? m_user_specified_type : ov::element::Type(m_layout.data_type);
}

ov::SoPtr<ov::ITensor> VariableState::get_state() const {
    wait_for_gpu_conversion();
    if (m_memory == nullptr) {
        const auto& pshape = m_layout.get_partial_shape();
        const auto& shape = get_tensor_shape(pshape);
        return m_context->create_host_tensor(get_user_specified_type(), shape);
    }

    auto tensor = m_context->create_host_tensor(get_user_specified_type(), m_memory->get_layout().get_shape());

    convert_and_copy(m_memory, tensor._ptr.get(), m_context->get_engine().get_service_stream());

    return tensor;
}

}  // namespace ov::intel_gpu
