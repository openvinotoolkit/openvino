// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/variable_state.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/make_tensor.hpp"
#include "openvino/runtime/tensor.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/remote_tensor.hpp"
#include "intel_gpu/plugin/multi_tensor_variable_state.hpp"
#include "intel_gpu/runtime/memory_caps.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

MultiTensorState::MultiTensorState(const std::vector<VariableStateInfo>& infos,
                                   std::shared_ptr<RemoteContextImpl> context,
                                   ShapePredictor::Ptr shape_predictor) : ov::intel_gpu::GPUVariableState(infos[0].m_id, context) {
    for (auto& info : infos) {
        m_states.push_back(std::make_shared<VariableState>(info, context, shape_predictor));
    }
}

VariableStateIndirectKVCache::VariableStateIndirectKVCache(const VariableStateInfo& info,
                                                           RemoteContextImpl::Ptr context,
                                                           std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                                           size_t beam_idx,
                                                           size_t concat_idx)
    : MultiTensorState { {info}, context, shape_predictor}
    , m_beam_idx(beam_idx)
    , m_concat_idx(concat_idx) {
    cldnn::layout beam_table_layout(get_beam_table_shape(info.m_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    VariableStateInfo beam_table_state_info(info.m_id + "/beam_table", beam_table_layout);
    m_states.push_back(std::make_shared<VariableState>(beam_table_state_info, context, shape_predictor));
    OPENVINO_ASSERT(m_states.size() == 2, "[GPU] VariableStateIndirectKVCache expects 2 internal states to be initialized");
}

void VariableStateIndirectKVCache::reset() {
    for (auto& state : m_states) {
        state->reset();
    }
    m_is_set = false;
}

bool VariableStateIndirectKVCache::is_set() const {
    return m_is_set;
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_memory() const {
    return m_states[0]->get_memory();
}

const cldnn::layout& VariableStateIndirectKVCache::get_layout() const {
    return m_states[0]->get_layout();
}

void VariableStateIndirectKVCache::set_state(const ov::SoPtr<ov::ITensor>& state) {
    OPENVINO_ASSERT(m_states.size() == 2, "[GPU] Corrupted VariableStateIndirectKVCache. Expected 2 internal states. Got: ", m_states.size());
    auto kv_cache_state = m_states[0];
    m_states[0]->set_state(state); // user can set only KV cache itself
    ov::Tensor default_beam_table(ov::element::i32, ov::Shape{1});
    m_states[1]->set_state(ov::get_tensor_impl(default_beam_table));
    m_states[1]->set();
}

ov::SoPtr<ov::ITensor> VariableStateIndirectKVCache::get_state() const {
    return m_states[0]->get_state();
}

void VariableStateIndirectKVCache::set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) {
    m_states[0]->set_memory(new_mem, actual_layout);

    cldnn::layout beam_table_layout(get_beam_table_shape(actual_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    std::cerr << "set_memory: update beam table to" << beam_table_layout.to_short_string() << std::endl;
    auto prev_table = m_states[1]->get_memory();
    m_states[1]->set_layout(beam_table_layout);
    auto curr_table = m_states[1]->get_memory();

    if (prev_table && curr_table && !prev_table->get_engine()->is_the_same_buffer(*prev_table, *curr_table)) {
        curr_table->copy_from(m_context->get_engine().get_service_stream(), *prev_table, true);
    }
}

void VariableStateIndirectKVCache::set_layout(const cldnn::layout& new_layout) {
    m_states[0]->set_layout(new_layout);
    cldnn::layout beam_table_layout(get_beam_table_shape(new_layout.get_partial_shape()), ov::element::i32, cldnn::format::bfyx);
    std::cerr << "set_layout: update beam table to" << beam_table_layout.to_short_string() << std::endl;
    auto prev_table = m_states[1]->get_memory();
    m_states[1]->set_layout(beam_table_layout);
    auto curr_table = m_states[1]->get_memory();

    if (prev_table && curr_table && !prev_table->get_engine()->is_the_same_buffer(*prev_table, *curr_table)) {
        curr_table->copy_from(m_context->get_engine().get_service_stream(), *prev_table, true);
    }
}

size_t VariableStateIndirectKVCache::get_actual_mem_size() const {
    return m_states[0]->get_actual_mem_size();
}

cldnn::memory::ptr VariableStateIndirectKVCache::get_beam_table_mem() const {
    return m_states[1]->get_memory();
}

ov::PartialShape VariableStateIndirectKVCache::get_beam_table_shape(const ov::PartialShape& kv_cache_shape) {
    return ov::PartialShape{kv_cache_shape[m_beam_idx], kv_cache_shape[m_concat_idx]};
}

VariableState::Ptr VariableStateIndirectKVCache::get_kv_cache_state() const {
    return m_states[0];
}

VariableState::Ptr VariableStateIndirectKVCache::get_beam_table_state() const {
    return m_states[1];
}

}  // namespace intel_gpu
}  // namespace ov
