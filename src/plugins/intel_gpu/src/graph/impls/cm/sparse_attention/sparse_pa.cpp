// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sparse_pa.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "../utils/qsa_host_common.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/sparse_pa.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using SparsePAIn = cldnn::sparse_pa::InputIdx;

namespace {

constexpr size_t kv_write_wg_size = 16;

SparsePARuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<SparsePARuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] sparse_pa runtime params are not initialized");
    return p;
}

size_t token_pitch(const layout& l) {
    const auto& shape = l.get_shape();
    OPENVINO_ASSERT(shape.size() >= 2, "[GPU][CM] sparse_pa expects a rank-2 (or legacy-padded) token-major input, got ", l.to_short_string());
    return shape[1];
}

/// Tokens per physical page: dim 2 of key_cache (there is no block_size attribute).
size_t page_size_of(const kernel_impl_params& params) {
    const auto& ps = params.get_input_layout(SparsePAIn::KEY_CACHE).get_partial_shape();
    OPENVINO_ASSERT(ps.size() == 4 && ps[2].is_static(), "[GPU][CM] sparse_pa: key_cache must have a static page_size dim");
    return static_cast<size_t>(ps[2].get_length());
}

JitConstants make_jit(const std::shared_ptr<const sparse_pa>& desc, const RuntimeParams& params) {
    JitConstants jit;
    jit.make("QSA_HEADS", desc->num_heads);
    jit.make("QSA_KV_HEADS", desc->num_kv_heads);
    jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
    jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
    jit.make("QSA_PAGE_SIZE", page_size_of(params));
    jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
    jit.add(make_jit_constant("QSA_SCALE", desc->scale));
    return jit;
}

// Only the attention kernel reads QSA_BLOCK_TOPK (the kv-write kernel never selects
// blocks), so it is added here only, not in the shared make_jit() above -- avoids
// defining it twice with different values for the same compiled program.
JitConstants make_jit_with_topk(const std::shared_ptr<const sparse_pa>& desc, const RuntimeParams& params) {
    auto jit = make_jit(desc, params);
    const auto& sel_ps = params.get_input_layout(SparsePAIn::SELECTED_BLOCK_INDICES).get_partial_shape();
    OPENVINO_ASSERT(sel_ps.rank().is_static() && sel_ps.size() >= 2 && sel_ps[sel_ps.size() - 1].is_static(),
                    "[GPU][CM] sparse_pa expects a rank>=2 selected_block_indices input with a static K_max");
    jit.make("QSA_BLOCK_TOPK", sel_ps[sel_ps.size() - 1].get_length());
    return jit;
}

// Decode-only split producer/reducer additionally need the partition count.
JitConstants make_jit_with_split(const std::shared_ptr<const sparse_pa>& desc, const RuntimeParams& params, size_t split_partitions) {
    auto jit = make_jit_with_topk(desc, params);
    jit.make("QSA_SPLIT_PARTITIONS", split_partitions);
    return jit;
}

constexpr auto get_sparse_attention_build_options() {
    return qsa_cm_build_options;
}

/// Whether every sequence in this call contributes exactly one query token. A
/// shape-agnostic impl keeps the decode stages too and decides per call.
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
class KvWriteGenerator : public KernelGenerator {
public:
    KvWriteGenerator() : KernelGenerator("sparse_attention_kv_write", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<sparse_pa>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE_CACHE});
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
            auto* rtp = rt_of(rt);

            const size_t wg_count = (rtp->num_tokens + kv_write_wg_size - 1) / kv_write_wg_size;
            auto& wgs = kd.params.workGroups;
            wgs.global = {1, desc->num_kv_heads, std::max<size_t>(1, wg_count) * kv_write_wg_size};
            wgs.local = {1, 1, kv_write_wg_size};

            kd.params.scalars.clear();
            push_u32(kd, token_pitch(params.get_input_layout(SparsePAIn::KEY)));
            push_u32(kd, params.get_input_layout(SparsePAIn::KEY).get_linear_offset());
            push_u32(kd, token_pitch(params.get_input_layout(SparsePAIn::VALUE)));
            push_u32(kd, params.get_input_layout(SparsePAIn::VALUE).get_linear_offset());
            push_u32(kd, rtp->num_seqs);
        }};
    }
};

// ===========================================================================
// Q3 - selected-block sparse attention (paged)
// ===========================================================================
class AttentionPagedGenerator : public KernelGenerator {
public:
    AttentionPagedGenerator() : KernelGenerator("sparse_attention_paged", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_with_topk(params.typed_desc<sparse_pa>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 4; i++) {
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
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->num_tokens), 1};
            wgs.local = {1, 1, 1};

            const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
        }};
    }
};

// ===========================================================================
// Q3 prefill, systolic variant - selected-block sparse attention (paged) on cm_dpas
// ===========================================================================
class AttentionPagedDpasGenerator : public KernelGenerator {
public:
    AttentionPagedDpasGenerator() : KernelGenerator("sparse_attention_paged_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_pa>();
        jit.add(make_jit_with_topk(desc, params));
        jit.make("QSA_ATT_WORKERS", std::max<size_t>(1, dpas_attention_workers(*desc)));
        jit.make("QSA_HEADS_PER_TILE", dpas_heads_per_tile(*desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 4; i++) {
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
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            const size_t workers = std::max<size_t>(1, dpas_attention_workers(*desc));
            const size_t heads_per_tile = dpas_heads_per_tile(*desc);
            wgs.global = {(desc->num_heads / heads_per_tile) * workers, std::max<size_t>(1, rtp->tile_count) * heads_per_tile, 1};
            wgs.local = {workers, 1, 1};

            const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
            kd.params.scalars.clear();
            push_u32(kd, rtp->tile_count);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
        }};
    }
};

// ===========================================================================
// Q3 prefill, GQA-aware systolic variant - one token x one KV head per WG on cm_dpas
// ===========================================================================
// Preferred over AttentionPagedDpasGenerator whenever can_use_gqa_dpas_attention() holds:
// the dpas N axis is filled with the GQA replication group (all n_rep query heads of one
// token share one KV read), not a union of several tokens' independent top-k selections.
class AttentionPagedGqaDpasGenerator : public KernelGenerator {
public:
    explicit AttentionPagedGqaDpasGenerator(bool split = false)
        : KernelGenerator("sparse_attention_paged_gqa_dpas", split ? "_split_cm" : "_cm"),
          m_split(split) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_pa>();
        jit.add(make_jit_with_topk(desc, params));
        jit.make("QSA_IS_CAUSAL", desc->causal ? 1 : 0);
        jit.make("QSA_GQA_HEAD_MAJOR", 1);
        jit.make("QSA_GQA_VECTOR_META", 1);
        if (m_split) {
            jit.make("QSA_GQA_SPLIT", 1);
            jit.make("QSA_SPLIT_PARTITIONS", gqa_split_partitions);
        }
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_COUNT});
        if (m_split) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ACC});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ML});
        } else {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        }
        for (uint32_t i = 0; i < 4; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[split = m_split](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<sparse_pa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            // QSA_GQA_HEAD_MAJOR=1: dim0 enumerates (token, W4 worker), dim1 the KV head,
            // so consecutive work-groups share a KV head.
            const size_t tokens = std::max<size_t>(1, rtp->num_tokens);
            wgs.global = {tokens * 4, desc->num_kv_heads, split ? gqa_split_partitions : 1};
            wgs.local = {4, 1, 1};

            const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
        }};
    }

private:
    bool m_split;
};

// ===========================================================================
// Q3 decode, split-selected-KV variant (paged), pass 1 - per-partition partial softmax
// ===========================================================================
class SplitPagedGenerator : public KernelGenerator {
public:
    SplitPagedGenerator() : KernelGenerator("sparse_attention_paged_split", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_with_split(params.typed_desc<sparse_pa>(), params, split_partitions));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::KEY_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::VALUE_CACHE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::PAST_LENS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SUBSEQUENCE_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::BLOCK_INDICES_BEGINS});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparsePAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ML});
        for (uint32_t i = 0; i < 4; i++) {
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
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->num_tokens), split_partitions};
            wgs.local = {1, 1, 1};

            const auto& q_layout = params.get_input_layout(SparsePAIn::QUERY);
            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
            push_u32(kd, rtp->num_seqs);
            push_u32(kd, q_layout.get_linear_offset());
            push_u32(kd, token_pitch(q_layout));
        }};
    }
};

// ===========================================================================
// Q3 decode, split-selected-KV variant (paged), pass 2 - merge pass
// ===========================================================================
class SplitReducePagedGenerator : public KernelGenerator {
public:
    SplitReducePagedGenerator() : KernelGenerator("sparse_attention_paged_split_reduce", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<sparse_pa>(), params));
        jit.make("QSA_SPLIT_PARTITIONS", decode_split_partitions(*params.typed_desc<sparse_pa>(), page_size_of(params), params.get_device_info()));
        jit.make("QSA_REDUCE_COLS", split_reduce_cols(params.typed_desc<sparse_pa>()->v_head_size));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparsePAInternalBufIdx::SPLIT_ML});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        args.push_back({ArgumentDescriptor::Types::SCALAR, 0});
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<sparse_pa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->num_tokens), desc->v_head_size / split_reduce_cols(desc->v_head_size)};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->num_tokens);
        }};
    }
};

class SparsePACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::SparsePACmImpl)

    Stage::Ptr kv_write = make_stage<KvWriteGenerator>();
    Stage::Ptr attention = make_stage<AttentionPagedGenerator>();
    Stage::Ptr attention_dpas = make_stage<AttentionPagedDpasGenerator>();
    Stage::Ptr attention_gqa_dpas = make_stage<AttentionPagedGqaDpasGenerator>();
    Stage::Ptr attention_gqa_split = make_stage<AttentionPagedGqaDpasGenerator>(true);
    Stage::Ptr split = make_stage<SplitPagedGenerator>();
    Stage::Ptr split_reduce = make_stage<SplitReducePagedGenerator>();

    SparsePACmImpl() : PrimitiveImplCM(SparsePACMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<SparsePARuntimeParams>();
    }

    SparsePARuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<SparsePARuntimeParams>();
        }
        return *static_cast<SparsePARuntimeParams*>(m_rt_params.get());
    }

    explicit SparsePACmImpl(const RuntimeParams& params) : SparsePACmImpl() {
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
        return make_deep_copy<SparsePACmImpl>(this);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    // Recomputed rather than cached: make_deep_copy() rebuilds the impl with the default
    // constructor and copies only _order and the per-stage kernel data.
    static bool use_dpas_attention(const kernel_impl_params& params) {
        return can_use_dpas_attention(*params.typed_desc<sparse_pa>(), page_size_of(params), params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return can_use_gqa_dpas_attention(*params.typed_desc<sparse_pa>(), page_size_of(params), params.get_device_info());
    }

    // Split-selected-KV is only a decode fallback for configs without GQA-DPAS: on BMG
    // (Qwen3.8 dims) GQA-DPAS decode measured ~2.5x faster than split+reduce at 4K-16K.
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
        // SPLIT_ACC/SPLIT_ML: only ever dispatched for decode-shaped calls, where `tokens`
        // equals the (small) sequence count, so this is cheap even at the full partition
        // width regardless of whether split actually fires for this particular call.
        const auto split_rows = is_decode_shape(params) ? tokens : size_t{1};
        const auto partitions = decode_split_partitions(*desc, page_size_of(params), params.get_device_info());
        bufs.emplace_back(split_rows * desc->num_heads * partitions * desc->v_head_size, ov::element::f32);  // SPLIT_ACC
        bufs.emplace_back(split_rows * desc->num_heads * partitions * 2, ov::element::f32);                  // SPLIT_ML
        return bufs;
    }

    event::ptr execute(const std::vector<event::ptr>& events, primitive_inst& instance) override {
        kernel_dump_info.clear_entries();
        update_rt_params(instance);

        // Q0 (main K/V write) and Q3 (selection read + attention) are independent of
        // each other's own inputs, but Q3 reads the cache Q0 just wrote, so Q3 must wait
        // for BOTH the original input events and Q0's completion.
        auto kv_event = execute_stage(events, instance, kv_write);
        std::vector<event::ptr> attn_deps = events;
        attn_deps.push_back(kv_event);

        const auto& params = *instance.get_impl_params();
        // GQA-DPAS wins for decode too (fills dpas N with the n_rep heads of one token).
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
        force_stages_refresh(*this);
        auto* rtp = &rt();

        // past_lens / subsequence_begins stay in GPU memory: sizes are shape-derived upper bounds
        // and the kernels resolve the real sequences (dropping the excess work-groups).
        const auto& params = *instance.get_impl_params();
        rtp->num_seqs = params.get_input_layout(SparsePAIn::PAST_LENS).get_shape()[0];
        rtp->num_tokens = params.get_input_layout(SparsePAIn::QUERY).get_shape()[0];
        rtp->is_decode = is_decode_shape(params);
        // At most tokens / 16 + num_seqs query tiles (a tile never straddles two sequences).
        rtp->tile_count = rtp->num_tokens / dpas_query_tile + rtp->num_seqs;
    }
};

}  // namespace

bool SparsePACMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::sparse_pa>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::sparse_pa>().get_primitive();
    std::string reason;
    if (!cldnn::sparse_pa::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // v1 capability gate: only a single selection head is supported.
    if (desc->num_selection_heads != 1) {
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
    if (node.get_input_layout(SparsePAIn::SELECTED_BLOCK_INDICES).data_type != ov::element::i32 ||
        node.get_input_layout(SparsePAIn::SELECTED_BLOCK_COUNT).data_type != ov::element::i32) {
        return false;
    }

    return true;
}

std::unique_ptr<cldnn::primitive_impl> SparsePACMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                    const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::sparse_pa>());
    return std::make_unique<SparsePACmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sparse_pa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::SparsePACmImpl)
