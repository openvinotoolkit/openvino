// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_sdpa.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../qsa_host_common.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "primitive_inst.h"
#include "sparse_attention_common.hpp"

namespace ov::intel_gpu::ocl {

using namespace cldnn;
using SparseSDPAIn = ov::intel_gpu::op::SparseSDPA::InputIdx;

namespace {

// Decode for the dense op is L == 1; below this history the scalar kernel wins over the split pair.
constexpr size_t split_decode_min_past = 2048;

qsa::SparseAttentionSlots sdpa_slots() {
    qsa::SparseAttentionSlots s;
    s.paged = false;
    s.query = SparseSDPAIn::QUERY;
    s.key = SparseSDPAIn::KEY;
    s.value = SparseSDPAIn::VALUE;
    s.selected_blocks = SparseSDPAIn::SELECTED_BLOCK_INDICES;
    s.selected_count = SparseSDPAIn::SELECTED_BLOCK_COUNT;
    s.key_length = SparseSDPAIn::KEY_LENGTH;
    return s;
}

bool is_decode_shape(const kernel_impl_params& params) {
    const auto& q_layout = params.get_input_layout(SparseSDPAIn::QUERY);
    if (q_layout.is_dynamic()) {
        return false;
    }
    const auto& shape = q_layout.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "[GPU][OCL] sparse_sdpa expects a rank-4 [B,H,*,D] query input");
    return shape[2] == 1;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

/// Allocated sequence pitch of a [B, H, S, D] K/V input. KVCache hands its consumers a buffer with dynamic upper padding
/// on S, so this can exceed the logical S.
size_t seq_pitch(const layout& l, const char* name) {
    const auto& pad = l.data_padding;
    for (size_t axis : {size_t{0}, size_t{1}, size_t{3}}) {
        OPENVINO_ASSERT(pad._lower_size[axis] == 0 && pad._upper_size[axis] == 0,
                        "[GPU][OCL] sparse_sdpa: ",
                        name,
                        " padding is only supported on the sequence axis, got ",
                        l.to_short_string());
    }
    OPENVINO_ASSERT(pad._lower_size[2] == 0, "[GPU][OCL] sparse_sdpa: ", name, " lower sequence padding is not supported, got ", l.to_short_string());
    return l.get_shape()[2] + static_cast<size_t>(pad._upper_size[2]);
}

using Gen = qsa::AttentionGenerator<sparse_sdpa>;

class SparseSDPAImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SparseSDPAImpl)

    Stage::Ptr attention = make_stage<Gen>(qsa::AttKind::Scalar, sdpa_slots());
    Stage::Ptr attention_dpas = make_stage<Gen>(qsa::AttKind::Dpas, sdpa_slots());
    Stage::Ptr attention_gqa_dpas = make_stage<Gen>(qsa::AttKind::Gqa, sdpa_slots());
    Stage::Ptr attention_gqa_split = make_stage<Gen>(qsa::AttKind::GqaSplit, sdpa_slots());
    Stage::Ptr split = make_stage<Gen>(qsa::AttKind::Split, sdpa_slots());
    Stage::Ptr split_reduce = make_stage<Gen>(qsa::AttKind::Reduce, sdpa_slots());

    SparseSDPAImpl() : PrimitiveImplOCL(SparseSDPAImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<qsa::SparseAttentionRuntimeParams>();
    }

    qsa::SparseAttentionRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<qsa::SparseAttentionRuntimeParams>();
        }
        return *static_cast<qsa::SparseAttentionRuntimeParams*>(m_rt_params.get());
    }

    explicit SparseSDPAImpl(const RuntimeParams& params) : SparseSDPAImpl() {
        add_stage(attention, params);
        if (use_gqa_dpas_attention(params)) {
            add_stage(attention_gqa_dpas, params);
            if (may_decode(params)) {
                add_stage(attention_gqa_split, params);
                add_stage(split_reduce, params);
            }
        } else {
            if (use_dpas_attention(params)) {
                add_stage(attention_dpas, params);
            }
            if (may_decode(params)) {
                add_stage(split, params);
                add_stage(split_reduce, params);
            }
        }
    }

    [[nodiscard]] std::unique_ptr<primitive_impl> clone() const override {
        return make_deep_copy<SparseSDPAImpl>(this);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        update_rt_params(inst);
        PrimitiveImplOCL::update(inst, impl_params);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    static bool use_dpas_attention(const kernel_impl_params& params) {
        return qsa::can_use_dpas_attention(*params.typed_desc<sparse_sdpa>(), std::nullopt, params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return qsa::can_use_gqa_dpas_attention(*params.typed_desc<sparse_sdpa>(), std::nullopt, params.get_device_info());
    }

    // Split-selected-KV is only a decode fallback for configs without the GQA systolic kernel.
    bool use_split_attention() {
        return has_stage(split) && rt().is_decode && rt().history_len >= split_decode_min_past;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto& query_shape = params.get_input_layout(SparseSDPAIn::QUERY).get_shape();
        const auto batch = std::max<size_t>(1, query_shape[0]);
        const auto seq_len = std::max<size_t>(1, query_shape[2]);
        const auto desc = params.typed_desc<sparse_sdpa>();

        std::vector<BufferDescriptor> bufs;
        // Only decode-shaped calls dispatch the split kernels, where batch * seq_len == batch is small.
        const auto split_rows = is_decode_shape(params) ? batch * seq_len : size_t{1};
        const auto partitions = qsa::decode_split_partitions(*desc, std::nullopt, params.get_device_info());
        bufs.emplace_back(split_rows * desc->num_heads * partitions * desc->v_head_size, ov::element::f32);  // SPLIT_ACC
        bufs.emplace_back(split_rows * desc->num_heads * partitions * 2, ov::element::f32);                  // SPLIT_ML
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);
        const auto& params = *instance.get_impl_params();
        if (use_gqa_dpas_attention(params)) {
            if (has_stage(attention_gqa_split) && rt().is_decode) {
                auto split_event = execute_stage(events, instance, attention_gqa_split);
                return execute_stage({split_event}, instance, split_reduce);
            }
            return execute_stage(events, instance, attention_gqa_dpas);
        }
        if (use_split_attention()) {
            auto split_event = execute_stage(events, instance, split);
            return execute_stage({split_event}, instance, split_reduce);
        }
        // Plain dpas needs a multi-token query tile; decode keeps the scalar kernel.
        if (has_stage(attention_dpas) && !rt().is_decode) {
            return execute_stage(events, instance, attention_dpas);
        }
        return execute_stage(events, instance, attention);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        qsa::force_stages_refresh(*this);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto& query_shape = params.get_input_layout(SparseSDPAIn::QUERY).get_shape();
        const auto& key_shape = params.get_input_layout(SparseSDPAIn::KEY).get_shape();
        OPENVINO_ASSERT(query_shape.size() == 4 && key_shape.size() == 4, "[GPU][OCL] sparse_sdpa expects rank-4 [B,H,*,D] query/key inputs");
        OPENVINO_ASSERT(!params.get_input_layout(SparseSDPAIn::QUERY).data_padding && !params.get_output_layout(0).data_padding,
                        "[GPU][OCL] sparse_sdpa: padded query/output buffers are not supported");

        rtp->batch = query_shape[0];
        rtp->seq_len = query_shape[2];
        rtp->history_len = key_shape[2];
        rtp->num_tokens = rtp->batch * rtp->seq_len;
        rtp->k_seq_pitch = seq_pitch(params.get_input_layout(SparseSDPAIn::KEY), "key");
        rtp->v_seq_pitch = seq_pitch(params.get_input_layout(SparseSDPAIn::VALUE), "value");
        rtp->is_decode = rtp->seq_len == 1;
        rtp->tile_count = rtp->batch * ((rtp->seq_len + qsa::dpas_query_tile - 1) / qsa::dpas_query_tile);
    }
};

}  // namespace

bool SparseSDPAImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::sparse_sdpa>());

    if (!qsa::ocl_impl_allowed() || node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!qsa::subgroups_available(info)) {
        return false;
    }

    const auto desc = node.as<cldnn::sparse_sdpa>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::SparseSDPA::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // Every dense kernel adds the causal tail and applies the causal bound; head rows must be 16-byte aligned and tile
    // the subgroup.
    const size_t sg = qsa::subgroup_size(info);
    if (desc->num_selection_heads != 1 || !desc->causal || desc->k_head_size % std::max<size_t>(16, sg) != 0 ||
        desc->v_head_size % std::max<size_t>(16, sg) != 0) {
        return false;
    }

    static constexpr std::array f16_inputs = {SparseSDPAIn::QUERY, SparseSDPAIn::KEY, SparseSDPAIn::VALUE};
    for (auto idx : f16_inputs) {
        if (node.get_input_layout(idx).data_type != ov::element::f16) {
            return false;
        }
    }
    static constexpr std::array i32_inputs = {SparseSDPAIn::SELECTED_BLOCK_INDICES, SparseSDPAIn::SELECTED_BLOCK_COUNT, SparseSDPAIn::KEY_LENGTH};
    for (auto idx : i32_inputs) {
        if (node.get_input_layout(idx).data_type != ov::element::i32) {
            return false;
        }
    }
    return true;
}

std::unique_ptr<cldnn::primitive_impl> SparseSDPAImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                    const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::sparse_sdpa>());
    return std::make_unique<SparseSDPAImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SparseSDPAImpl)
