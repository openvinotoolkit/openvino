// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "intel_gpu/plugin/variable_state.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

MultiTensorState::MultiTensorState(const std::vector<VariableStateInfo>& infos,
                                   std::shared_ptr<RemoteContextImpl> context,
                                   ShapePredictor::Ptr shape_predictor) : ov::intel_gpu::VariableStateBase(infos[0].m_id, context) {
    for (auto& info : infos) {
        m_hidden_states.push_back(std::make_shared<VariableState>(info, context, shape_predictor));
    }
}

VariableStateIndirectKVCache::VariableStateIndirectKVCache(const VariableStateInfo& info,
                                                           RemoteContextImpl::Ptr context,
                                                           std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                                           size_t beam_axis,
                                                           size_t concat_axis)
    : MultiTensorState { {info}, context, shape_predictor}
    , m_beam_axis(beam_axis)
    , m_concat_axis(concat_axis) {
    cldnn::layout beam_table_layout(get_beam_table_shape(info.m_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    VariableStateInfo beam_table_state_info(info.m_id + "/beam_table", beam_table_layout);
    m_hidden_states.push_back(std::make_shared<VariableState>(beam_table_state_info, context, shape_predictor));
    OPENVINO_ASSERT(m_hidden_states.size() == 2, "[GPU] VariableStateIndirectKVCache expects 2 internal states to be initialized");
}

void VariableStateIndirectKVCache::reset() {
    for (auto& state : m_hidden_states) {
        state->reset();
    }
    m_is_set = false;
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_memory() const {
    return m_hidden_states[0]->get_memory();
}

const cldnn::layout& VariableStateIndirectKVCache::get_layout() const {
    return m_hidden_states[0]->get_layout();
}

void VariableStateIndirectKVCache::set_state(const ov::SoPtr<ov::ITensor>& state) {
    OPENVINO_ASSERT(m_hidden_states.size() == 2, "[GPU] Corrupted VariableStateIndirectKVCache. Expected 2 internal states. Got: ", m_hidden_states.size());
    m_hidden_states[0]->set_state(state); // user can set only KV cache

    // Beam table is reset to cleanup rearranges history
    cldnn::layout bt_layout(get_beam_table_shape(state->get_shape()), ov::element::i32, cldnn::format::bfyx);
    m_hidden_states[1]->reset();
    m_hidden_states[1]->set_layout(bt_layout);
}

template<typename T>
void copy_element(const void* src, void* dst, size_t src_offset, size_t dst_offset) {
    static_cast<T*>(dst)[dst_offset] = static_cast<const T*>(src)[src_offset];
}

static void rearrange_cache(cldnn::memory::ptr kv_in_mem, cldnn::memory::ptr bt_mem, cldnn::memory::ptr kv_out_mem, cldnn::stream& stream, size_t concat_axis) {
    auto kv_layout = kv_in_mem->get_layout();
    auto kv_shape = kv_layout.get_shape();
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::read> kv_in_ptr(kv_in_mem, stream);
    cldnn::mem_lock<int32_t, cldnn::mem_lock_type::read> bt_in_ptr(bt_mem, stream);
    cldnn::mem_lock<uint8_t, cldnn::mem_lock_type::write> kv_out_ptr(kv_out_mem, stream);

    OPENVINO_ASSERT(kv_shape.size() == 4);

    for (size_t b = 0; b < kv_shape[0]; b++) {
        for (size_t f = 0; f < kv_shape[1]; f++) {
            for (size_t y = 0; y < kv_shape[2]; y++) {
                for (size_t x = 0; x < kv_shape[3]; x++) {
                    size_t b_kv = bt_in_ptr[b* kv_shape[concat_axis] + y];

                    auto in_idx = std::vector<int>{static_cast<int>(b_kv), static_cast<int>(f), static_cast<int>(y), static_cast<int>(x)};
                    auto out_idx = std::vector<int>{static_cast<int>(b), static_cast<int>(f), static_cast<int>(y), static_cast<int>(x)};

                    cldnn::tensor in(cldnn::format::bfyx, in_idx, 0);
                    cldnn::tensor out(cldnn::format::bfyx, out_idx, 0);

                    size_t out_offset = kv_out_mem->get_layout().get_linear_offset(out);
                    size_t in_offset = kv_layout.get_linear_offset(in);

                    if (ov::element::Type(kv_layout.data_type).size() == 2)
                        copy_element<uint16_t>(kv_in_ptr.data(), kv_out_ptr.data(), in_offset, out_offset);
                    else if (ov::element::Type(kv_layout.data_type).size() == 2)
                        copy_element<uint32_t>(kv_in_ptr.data(), kv_out_ptr.data(), in_offset, out_offset);
                }
            }
        }
    }
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCache::get_state() const {
    auto kv_layout = m_hidden_states[0]->get_layout();
    auto bt_mem = m_hidden_states[1]->get_memory();
    if (kv_layout.get_partial_shape()[m_beam_axis].get_length() > 1 && bt_mem) {
        auto kv_mem = m_hidden_states[0]->get_memory();
        auto tensor = m_context->create_host_tensor(m_hidden_states[0]->get_user_specified_type(), kv_layout.get_shape());

        auto& engine = m_context->get_engine();
        auto tmp_mem = engine.allocate_memory(kv_layout, engine.get_lockable_preferred_memory_allocation_type(), false);

        rearrange_cache(kv_mem, bt_mem, tmp_mem, m_context->get_engine().get_service_stream(), m_concat_axis);

        convert_and_copy(tmp_mem, tensor._ptr.get(), m_context->get_engine().get_service_stream());

        return tensor;
    } else {
        return m_hidden_states[0]->get_state();
    }
}

void VariableStateIndirectKVCache::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    m_hidden_states[0]->set_memory(new_mem, actual_layout);
}

void VariableStateIndirectKVCache::set_layout(const cldnn::layout& new_layout) {
    m_hidden_states[0]->set_layout(new_layout);
}

size_t VariableStateIndirectKVCache::get_actual_mem_size() const {
    return m_hidden_states[0]->get_actual_mem_size();
}

ov::PartialShape VariableStateIndirectKVCache::get_beam_table_shape(const ov::PartialShape& kv_cache_shape) {
    auto rank = kv_cache_shape.size();
    ov::PartialShape beam_table_shape(std::vector<size_t>(rank, 1));
    beam_table_shape[m_beam_axis] = kv_cache_shape[m_beam_axis];
    beam_table_shape[m_concat_axis] = kv_cache_shape[m_concat_axis];
    return beam_table_shape;
}

VariableState::Ptr VariableStateIndirectKVCache::get_beam_table_state() const {
    return m_hidden_states[1];
}

VariableStateIndirectKVCacheCompressed::VariableStateIndirectKVCacheCompressed(
    const VariableStateInfo& info,
    std::shared_ptr<RemoteContextImpl> context,
    std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
    size_t beam_idx,
    size_t concat_idx,
    bool has_zp_state = false)
    : VariableStateIndirectKVCache(info, context, shape_predictor, beam_idx, concat_idx),
      m_has_zp_state(has_zp_state) {
    const auto compression_scale_shape = ov::PartialShape::dynamic(info.m_layout.get_partial_shape().size());

    // TODO: set proper data type and shape
    cldnn::layout compression_scale_layout(compression_scale_shape,
                                           ov::element::f16,
                                           cldnn::format::bfyx);
    VariableStateInfo compression_scale_state_info(info.m_id + "/comp_scale", compression_scale_layout);
    m_hidden_states.push_back(std::make_shared<VariableState>(compression_scale_state_info, context, shape_predictor));

    if (has_zp_state) {
        cldnn::layout compression_zp_layout(compression_scale_shape,
                                            ov::element::f16,
                                            cldnn::format::bfyx);
        VariableStateInfo compression_scale_state_info(info.m_id + "/comp_zp", compression_zp_layout);
        m_hidden_states.push_back(std::make_shared<VariableState>(compression_scale_state_info, context, shape_predictor));
    }

    OPENVINO_ASSERT((!m_has_zp_state && m_hidden_states.size() == 3) || (m_has_zp_state && m_hidden_states.size() == 4),
                    "[GPU] VariableStateIndirectKVCacheCompressed expects 3 or 4 internal states to be initialized, "
                    "actual number is ", m_hidden_states.size());
}

VariableState::Ptr VariableStateIndirectKVCacheCompressed::get_compression_scale_state() const {
    return m_hidden_states[2];
}

void VariableStateIndirectKVCacheCompressed::set_compression_scale_layout(const cldnn::layout& new_layout) {
    m_hidden_states[2]->set_layout(new_layout);
}

VariableState::Ptr VariableStateIndirectKVCacheCompressed::get_compression_zp_state() const {
    OPENVINO_ASSERT(m_has_zp_state);
    return m_hidden_states[3];
}

void VariableStateIndirectKVCacheCompressed::set_compression_zp_layout(const cldnn::layout& new_layout) {
    OPENVINO_ASSERT(m_has_zp_state);
    m_hidden_states[3]->set_layout(new_layout);
}

bool VariableStateIndirectKVCacheCompressed::has_zp_state() const {
    return m_has_zp_state;
}

void VariableStateIndirectKVCacheCompressed::set_state(const ov::SoPtr<ov::ITensor>& state) {
    OPENVINO_THROW("[GPU] set_state API is supported only when KV-cache compression is disabled");
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCacheCompressed::get_state() const {
    OPENVINO_THROW("[GPU] get_state API is supported only when KV-cache compression is disabled");
}

}  // namespace intel_gpu
}  // namespace ov
