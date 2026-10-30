// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.hpp"

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
#include "intel_gpu/primitives/sparse_pa.hpp"
#include "primitive_inst.h"
#include "sparse_attention_common.hpp"

namespace ov::intel_gpu::ocl {

using namespace cldnn;
using SparsePAIn = cldnn::sparse_pa::InputIdx;

namespace {

qsa::SparseAttentionSlots pa_slots() {
    qsa::SparseAttentionSlots s;
    s.paged = true;
    s.query = SparsePAIn::QUERY;
    s.key = SparsePAIn::KEY_CACHE;
    s.value = SparsePAIn::VALUE_CACHE;
    s.past_lens = SparsePAIn::PAST_LENS;
    s.subsequence_begins = SparsePAIn::SUBSEQUENCE_BEGINS;
    s.block_indices = SparsePAIn::BLOCK_INDICES;
    s.block_indices_begins = SparsePAIn::BLOCK_INDICES_BEGINS;
    s.selected_blocks = SparsePAIn::SELECTED_BLOCK_INDICES;
    s.selected_count = SparsePAIn::SELECTED_BLOCK_COUNT;
    return s;
}

size_t token_pitch(const layout& l) {
    const auto& shape = l.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "[GPU][OCL] sparse_pa expects a rank-2 (or legacy-padded) token-major input, got ", l.to_short_string());
    return shape[1];
}

/// Tokens per physical page: dim 2 of key_cache (there is no block_size attribute).
size_t page_size_of(const kernel_impl_params& params) {
    const auto& ps = params.get_input_layout(SparsePAIn::KEY_CACHE).get_partial_shape();
    OPENVINO_ASSERT(ps.size() == 4 && ps[2].is_static(), "[GPU][OCL] sparse_pa: key_cache must have a static page_size dim");
    return static_cast<size_t>(ps[2].get_length());
}

bool is_decode_shape(const kernel_impl_params& params) {
    const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
    const auto& past_lens_layout = params.get_input_layout(SparsePAIn::PAST_LENS);
    if (q_layout.is_dynamic() || past_lens_layout.is_dynamic()) {
        return false;
    }
    const auto tokens = q_layout.get_shape()[0];
    const auto seqs = past_lens_layout.get_shape()[0];
    return seqs > 0 && tokens == seqs;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

// ===========================================================================
// Q0 - main KV cache write
// ===========================================================================
class KvWriteGenerator : public qsa::QsaGenerator {
public:
    KvWriteGenerator() : QsaGenerator("sparse_attention_kv_write", "sparse_attention_kv_write") {}

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = QsaGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_pa>();
        jit.make("QSA_KV_HEADS", desc->num_kv_heads);
        jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
        jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
        jit.make("QSA_PAGE_SIZE", page_size_of(params));
        jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        for (auto idx : {SparsePAIn::KEY,
                         SparsePAIn::VALUE,
                         SparsePAIn::PAST_LENS,
                         SparsePAIn::BLOCK_INDICES,
                         SparsePAIn::BLOCK_INDICES_BEGINS,
                         SparsePAIn::SUBSEQUENCE_BEGINS,
                         SparsePAIn::KEY_CACHE,
                         SparsePAIn::VALUE_CACHE}) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        }
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<sparse_pa>();
            auto* rtp = static_cast<qsa::SparseAttentionRuntimeParams*>(rt);
            OPENVINO_ASSERT(rtp != nullptr, "[GPU][OCL] sparse_pa runtime params are not initialized");
            const auto sg = qsa::subgroup_size(params.get_device_info());

            auto& wgs = kd.params.workGroups;
            wgs.global = {sg, desc->num_kv_heads, std::max<size_t>(1, rtp->num_tokens)};
            wgs.local = {sg, 1, 1};

            kd.params.scalars.clear();
            qsa::push_u32(kd, token_pitch(params.get_input_layout(SparsePAIn::KEY)));
            qsa::push_u32(kd, params.get_input_layout(SparsePAIn::KEY).get_linear_offset());
            qsa::push_u32(kd, token_pitch(params.get_input_layout(SparsePAIn::VALUE)));
            qsa::push_u32(kd, params.get_input_layout(SparsePAIn::VALUE).get_linear_offset());
            qsa::push_u32(kd, rtp->num_seqs);
        }};
    }
};

using Gen = qsa::AttentionGenerator<sparse_pa>;

class SparsePAImpl : public PrimitiveImplOCL {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::ocl::SparsePAImpl)

    Stage::Ptr kv_write = make_stage<KvWriteGenerator>();
    Stage::Ptr attention = make_stage<Gen>(qsa::AttKind::Scalar, pa_slots());
    Stage::Ptr attention_dpas = make_stage<Gen>(qsa::AttKind::Dpas, pa_slots());
    Stage::Ptr attention_gqa_dpas = make_stage<Gen>(qsa::AttKind::Gqa, pa_slots());
    Stage::Ptr attention_gqa_split = make_stage<Gen>(qsa::AttKind::GqaSplit, pa_slots());
    Stage::Ptr split = make_stage<Gen>(qsa::AttKind::Split, pa_slots());
    Stage::Ptr split_reduce = make_stage<Gen>(qsa::AttKind::Reduce, pa_slots());

    SparsePAImpl() : PrimitiveImplOCL(SparsePAImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<qsa::SparseAttentionRuntimeParams>();
    }

    qsa::SparseAttentionRuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<qsa::SparseAttentionRuntimeParams>();
        }
        return *static_cast<qsa::SparseAttentionRuntimeParams*>(m_rt_params.get());
    }

    explicit SparsePAImpl(const RuntimeParams& params) : SparsePAImpl() {
        add_stage(kv_write, params);
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
        return make_deep_copy<SparsePAImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    static bool use_dpas_attention(const kernel_impl_params& params) {
        return qsa::can_use_dpas_attention(*params.typed_desc<sparse_pa>(), page_size_of(params), params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return qsa::can_use_gqa_dpas_attention(*params.typed_desc<sparse_pa>(), page_size_of(params), params.get_device_info());
    }

    // Split-selected-KV is only a decode fallback for configs without the GQA systolic kernel.
    bool use_split_attention() {
        return has_stage(split) && rt().is_decode;
    }

    [[nodiscard]] std::vector<BufferDescriptor> get_internal_buffer_descs(const kernel_impl_params& params) const override {
        if (params.is_dynamic()) {
            return {};
        }
        const auto tokens = std::max<size_t>(1, params.get_input_layout(SparsePAIn::QUERY).get_shape()[0]);
        const auto desc = params.typed_desc<sparse_pa>();

        std::vector<BufferDescriptor> bufs;
        // Only decode-shaped calls dispatch the split kernels, where `tokens` equals the (small) sequence count.
        const auto split_rows = is_decode_shape(params) ? tokens : size_t{1};
        const auto partitions = qsa::decode_split_partitions(*desc, page_size_of(params), params.get_device_info());
        bufs.emplace_back(split_rows * desc->num_heads * partitions * desc->v_head_size, ov::element::f32);  // SPLIT_ACC
        bufs.emplace_back(split_rows * desc->num_heads * partitions * 2, ov::element::f32);                  // SPLIT_ML
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        // The attention stages read the cache the K/V write just produced, so they wait for both the input events and
        // the write.
        auto kv_event = execute_stage(events, instance, kv_write);
        std::vector<event::ptr> attn_deps = events;
        attn_deps.push_back(kv_event);

        const auto& params = *instance.get_impl_params();
        if (use_gqa_dpas_attention(params)) {
            if (has_stage(attention_gqa_split) && rt().is_decode) {
                auto split_event = execute_stage(attn_deps, instance, attention_gqa_split);
                return execute_stage({split_event}, instance, split_reduce);
            }
            return execute_stage(attn_deps, instance, attention_gqa_dpas);
        }
        if (use_split_attention()) {
            auto split_event = execute_stage(attn_deps, instance, split);
            return execute_stage({split_event}, instance, split_reduce);
        }
        // Plain dpas needs a multi-token query tile; decode keeps the scalar kernel.
        if (has_stage(attention_dpas) && !rt().is_decode) {
            return execute_stage(attn_deps, instance, attention_dpas);
        }
        return execute_stage(attn_deps, instance, attention);
    }

    void update_rt_params(const primitive_inst& instance) override {
        update_stages_flags(instance);
        qsa::force_stages_refresh(*this);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
        rtp->num_seqs = params.get_input_layout(SparsePAIn::PAST_LENS).get_shape()[0];
        rtp->num_tokens = q_layout.get_shape()[0];
        rtp->is_decode = is_decode_shape(params);
        // At most tokens / 16 + num_seqs query tiles (a tile never straddles two sequences).
        rtp->tile_count = rtp->num_tokens / qsa::dpas_query_tile + rtp->num_seqs;
        rtp->q_base = q_layout.get_linear_offset();
        rtp->q_pitch = token_pitch(q_layout);
    }
};

}  // namespace

bool SparsePAImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::sparse_pa>());

    if (!qsa::ocl_impl_allowed() || node.has_fused_primitives()) {
        return false;
    }

    const auto& info = node.get_program().get_engine().get_device_info();
    if (!qsa::subgroups_available(info)) {
        return false;
    }

    const auto desc = node.as<cldnn::sparse_pa>().get_primitive();
    std::string reason;
    if (!cldnn::sparse_pa::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // Single selection head; head rows must be 16-byte aligned and tile the subgroup.
    const size_t sg = qsa::subgroup_size(info);
    if (desc->num_selection_heads != 1 || desc->k_head_size % std::max<size_t>(16, sg) != 0 || desc->v_head_size % std::max<size_t>(16, sg) != 0) {
        return false;
    }

    const auto& key_cache_ps = node.get_input_layout(SparsePAIn::KEY_CACHE).get_partial_shape();
    if (key_cache_ps.size() != 4 || !key_cache_ps[2].is_static() || key_cache_ps[2].get_length() % static_cast<int64_t>(desc->key_block_size) != 0) {
        return false;
    }

    static constexpr std::array f16_inputs = {SparsePAIn::QUERY, SparsePAIn::KEY, SparsePAIn::VALUE, SparsePAIn::KEY_CACHE, SparsePAIn::VALUE_CACHE};
    for (auto idx : f16_inputs) {
        if (node.get_input_layout(idx).data_type != ov::element::f16) {
            return false;
        }
    }
    return node.get_input_layout(SparsePAIn::SELECTED_BLOCK_INDICES).data_type == ov::element::i32 &&
           node.get_input_layout(SparsePAIn::SELECTED_BLOCK_COUNT).data_type == ov::element::i32;
}

std::unique_ptr<cldnn::primitive_impl> SparsePAImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                  const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::sparse_pa>());
    return std::make_unique<SparsePAImpl>(params);
}

}  // namespace ov::intel_gpu::ocl

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::ocl::SparsePAImpl)
