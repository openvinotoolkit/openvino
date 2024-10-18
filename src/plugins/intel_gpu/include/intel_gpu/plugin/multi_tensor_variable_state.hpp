// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "intel_gpu/plugin/variable_state.hpp"
#include "openvino/core/partial_shape.hpp"

namespace ov {
namespace intel_gpu {

class MultiTensorState : public VariableStateBase {
public:
    MultiTensorState(const std::vector<VariableStateInfo>& infos, std::shared_ptr<RemoteContextImpl> context, ShapePredictor::Ptr shape_predictor);

protected:
    std::vector<std::shared_ptr<VariableState>> m_hidden_states = {};
};

// This is multi-tensor state for Indirect KV-Cache + Gemm pattern
// Internally it stores KV Cache state + Beam Table state
class VariableStateIndirectKVCache : public MultiTensorState {
public:
    VariableStateIndirectKVCache(const VariableStateInfo& info,
                                 std::shared_ptr<RemoteContextImpl> context,
                                 std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                 size_t beam_idx,
                                 size_t concat_idx);
    using Ptr = std::shared_ptr<VariableStateIndirectKVCache>;

    void reset() override;
    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

    cldnn::memory::ptr get_memory() const override;
    const cldnn::layout& get_layout() const override;
    void set_layout(const cldnn::layout& new_layout) override;
    void set_memory(const cldnn::memory::ptr& new_mem, const cldnn::layout& actual_layout) override;
    size_t get_actual_mem_size() const override;

    VariableState::Ptr get_beam_table_state() const;
    ov::PartialShape get_beam_table_shape(const ov::PartialShape& kv_cache_shape);

private:
    size_t m_beam_axis = 0;
    size_t m_concat_axis = 0;
};

// This is multi-tensor state for Indirect KV-Cache + Gemm pattern
// Internally it stores KV Cache state + Beam Table state + compression scales state + (optional compression zero points)
class VariableStateIndirectKVCacheCompressed : public VariableStateIndirectKVCache {
public:
    VariableStateIndirectKVCacheCompressed(const VariableStateInfo& info,
                                           std::shared_ptr<RemoteContextImpl> context,
                                           std::shared_ptr<cldnn::ShapePredictor> shape_predictor,
                                           size_t beam_idx,
                                           size_t concat_idx,
                                           bool has_zp_state);
    using Ptr = std::shared_ptr<VariableStateIndirectKVCacheCompressed>;

    void set_state(const ov::SoPtr<ov::ITensor>& state) override;
    ov::SoPtr<ov::ITensor> get_state() const override;

    VariableState::Ptr get_compression_scale_state() const;
    void set_compression_scale_layout(const cldnn::layout& new_layout);

    VariableState::Ptr get_compression_zp_state() const;
    void set_compression_zp_layout(const cldnn::layout& new_layout);
    bool has_zp_state() const;

private:
    bool m_has_zp_state = false;
};
}  // namespace intel_gpu
}  // namespace ov
