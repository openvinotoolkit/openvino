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

#include "../primitive_cm_base.hpp"
#include "../utils/kernel_generator.hpp"
#include "common_utils/jitter.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/primitives/sparse_sdpa.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu::cm {

using namespace cldnn;
using SparseSDPAIn = ov::intel_gpu::op::SparseSDPA::InputIdx;

namespace {

SparseSDPARuntimeParams* rt_of(ImplRuntimeParams* rt) {
    auto* p = static_cast<SparseSDPARuntimeParams*>(rt);
    OPENVINO_ASSERT(p != nullptr, "[GPU][CM] sparse_sdpa runtime params are not initialized");
    return p;
}

JitConstants make_jit(const std::shared_ptr<const sparse_sdpa>& desc, const RuntimeParams& params) {
    const auto& sel_ps = params.get_input_layout(SparseSDPAIn::SELECTED_BLOCK_INDICES).get_partial_shape();
    OPENVINO_ASSERT(sel_ps.rank().is_static() && sel_ps.size() >= 2 && sel_ps[sel_ps.size() - 1].is_static(),
                    "[GPU][CM] sparse_sdpa expects a rank>=2 selected_block_indices input with a static K_max");

    JitConstants jit;
    jit.make("QSA_HEADS", desc->num_heads);
    jit.make("QSA_KV_HEADS", desc->num_kv_heads);
    jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
    jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
    jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
    jit.make("QSA_BLOCK_TOPK", sel_ps[sel_ps.size() - 1].get_length());
    jit.add(make_jit_constant("QSA_SCALE", desc->scale));
    return jit;
}

// Decode-only split producer/reducer additionally need the partition count.
JitConstants make_jit_with_split(const std::shared_ptr<const sparse_sdpa>& desc, const RuntimeParams& params, size_t split_partitions) {
    auto jit = make_jit(desc, params);
    jit.make("QSA_SPLIT_PARTITIONS", split_partitions);
    return jit;
}

void push_u32(KernelData& kd, size_t value) {
    scalar_desc s;
    s.t = scalar_desc::Types::UINT32;
    s.v.u32 = static_cast<uint32_t>(value);
    kd.params.scalars.push_back(s);
}

constexpr auto get_sparse_attention_build_options() {
    return " -cmc -Qxcm_register_file_size=256";
}

constexpr size_t dpas_query_tile = 16;  // dpas N axis == GRF width on Xe2
constexpr size_t dpas_k_step = 16;      // fp16 K elements per dpas
constexpr size_t dpas_wide_cols = 32;   // halfs per 2D block row (64-byte hardware cap)

// Decode-only split-selected-KV attention (see sparse_pa.cpp's identical constants /
// rationale). "Decode" for the dense op is L == 1 (no past_lens/subsequence_begins here,
// the Q shape alone tells the whole story).
constexpr size_t split_partitions = 16;
// One GQA-DPAS WG serves all n_rep heads of a KV head, so decode needs more WGs per row.
constexpr size_t gqa_split_partitions = 64;
constexpr size_t split_decode_min_past = 2048;

/// Dv slice merged by one split_reduce thread.
size_t split_reduce_cols(size_t v_head_size) {
    return v_head_size % 32 == 0 ? 32 : 16;
}

/// Whether this call's query length is 1. A shape-agnostic impl must keep both the
/// decode and prefill stages, so it is "maybe decode" while L is unknown.
bool is_decode_shape(const kernel_impl_params& params) {
    const auto& q_layout = params.get_input_layout(SparseSDPAIn::QUERY);
    if (q_layout.is_dynamic()) {
        return false;
    }
    const auto& shape = q_layout.get_shape();
    OPENVINO_ASSERT(shape.size() == 4, "[GPU][CM] sparse_sdpa expects a rank-4 [B,H,*,D] query input");
    return shape[2] == 1;
}

bool may_decode(const kernel_impl_params& params) {
    return params.is_dynamic() || is_decode_shape(params);
}

/// Allocated sequence pitch of a [B, H, S, D] K/V input. KVCache hands its consumers a
/// buffer with dynamic upper padding on S, so this can exceed the logical S.
size_t seq_pitch(const layout& l, const char* name) {
    const auto& pad = l.data_padding;
    for (size_t axis : {size_t{0}, size_t{1}, size_t{3}}) {
        OPENVINO_ASSERT(pad._lower_size[axis] == 0 && pad._upper_size[axis] == 0,
                        "[GPU][CM] sparse_sdpa: ", name, " padding is only supported on the sequence axis, got ", l.to_short_string());
    }
    OPENVINO_ASSERT(pad._lower_size[2] == 0, "[GPU][CM] sparse_sdpa: ", name, " lower sequence padding is not supported, got ", l.to_short_string());
    return l.get_shape()[2] + static_cast<size_t>(pad._upper_size[2]);
}

/// Heads that share one dpas tile. See qsa.cpp's dpas_heads_per_tile() for the rationale.
size_t dpas_heads_per_tile(const sparse_sdpa& desc) {
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    // 16 heads would leave one token per tile, and load_vnni_cols<1> does not compile.
    for (size_t ht : {size_t{8}, size_t{4}, size_t{2}}) {
        if (n_rep % ht == 0) {
            return ht;
        }
    }
    return 1;
}

/// Threads that cooperate on one (query tile, head). See qsa.cpp's dpas_attention_workers().
size_t dpas_attention_workers(const sparse_sdpa& desc) {
    constexpr size_t reg_budget = 12 * 1024;
    for (size_t w : {size_t{1}, size_t{2}, size_t{4}, size_t{8}}) {
        if (desc.k_head_size % (w * dpas_k_step) != 0 || desc.v_head_size % (w * dpas_query_tile) != 0) {
            continue;
        }
        const auto dh_w = desc.k_head_size / w;
        const auto dv_w = desc.v_head_size / w;
        if (dh_w % dpas_wide_cols != 0 || dv_w % dpas_wide_cols != 0) {
            continue;
        }
        const auto rq = dh_w * dpas_query_tile * sizeof(ov::float16);
        const auto ro = dpas_query_tile * dv_w * sizeof(float);
        const auto kv = dpas_k_step * (dh_w + dv_w) * sizeof(ov::float16);
        if (rq + ro + kv <= reg_budget) {
            return w;
        }
    }
    return 0;
}

/// Whether the systolic sparse_attention_dense_dpas kernel can be used. Unlike the
/// paged capability gate, there is no page/compress-ratio page-boundary check at all:
/// dense K/V for one (batch, kv_head) is one contiguous region.
bool can_use_dpas_attention(const sparse_sdpa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.k_head_size % dpas_k_step != 0 || desc.v_head_size % dpas_query_tile != 0) {
        return false;
    }
    if (desc.k_head_size * sizeof(ov::float16) < 64 || desc.v_head_size * sizeof(ov::float16) < 64) {
        return false;
    }
    if (desc.key_block_size % 2 != 0 || dpas_k_step % desc.key_block_size != 0) {
        return false;  // four QSA blocks must pack exactly one 16-token dpas K step
    }
    return dpas_attention_workers(desc) != 0;
}

/// Whether the GQA-aware systolic sparse_attention_dense_gqa_dpas kernel can be used.
/// Preferred over the plain dpas kernel whenever it holds - this is what matters for GQA
/// ratios like Qwen4's Hq/Hkv = 12 (see qsa.cpp's identical can_use_gqa_dpas_attention()).
bool can_use_gqa_dpas_attention(const sparse_sdpa& desc, const cldnn::device_info& info) {
    if (info.arch < gpu_arch::xe2) {
        return false;
    }
    if (desc.num_kv_heads == 0 || desc.num_heads % desc.num_kv_heads != 0) {
        return false;
    }
    const auto n_rep = desc.num_heads / desc.num_kv_heads;
    if (n_rep < 1 || n_rep > 16) {
        return false;
    }
    constexpr size_t gqa_workers = 4;  // GQA_W in sparse_attention_dense_gqa_dpas.cm
    if (desc.k_head_size % (gqa_workers * 32) != 0 || desc.v_head_size % (gqa_workers * 32) != 0) {
        return false;
    }
    if (std::max(desc.k_head_size, desc.v_head_size) > 256) {
        return false;
    }
    if (desc.key_block_size != 2 && desc.key_block_size != 4 && desc.key_block_size != 8 && desc.key_block_size != 16) {
        return false;
    }
    return desc.causal;
}

/// Partitions behind SPLIT_ACC/SPLIT_ML and split_reduce for this config's decode producer.
size_t decode_split_partitions(const sparse_sdpa& desc, const cldnn::device_info& info) {
    return can_use_gqa_dpas_attention(desc, info) ? gqa_split_partitions : split_partitions;
}

// ===========================================================================
// Q3 - selected-block sparse attention (dense, read-only main K/V)
// ===========================================================================
class AttentionDenseGenerator : public KernelGenerator {
public:
    AttentionDenseGenerator() : KernelGenerator("sparse_attention_dense", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<sparse_sdpa>(), params));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->batch * rtp->seq_len), 1};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
            push_u32(kd, rtp->batch * rtp->seq_len);
            push_u32(kd, rtp->k_seq_pitch);
            push_u32(kd, rtp->v_seq_pitch);
        }};
    }
};

// ===========================================================================
// Q3 prefill, systolic variant - selected-block sparse attention (dense) on cm_dpas
// ===========================================================================
class AttentionDenseDpasGenerator : public KernelGenerator {
public:
    AttentionDenseDpasGenerator() : KernelGenerator("sparse_attention_dense_dpas", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_sdpa>();
        jit.add(make_jit(desc, params));
        jit.make("QSA_ATT_WORKERS", std::max<size_t>(1, dpas_attention_workers(*desc)));
        jit.make("QSA_HEADS_PER_TILE", dpas_heads_per_tile(*desc));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            const size_t workers = std::max<size_t>(1, dpas_attention_workers(*desc));
            const size_t heads_per_tile = dpas_heads_per_tile(*desc);
            wgs.global = {(desc->num_heads / heads_per_tile) * workers, std::max<size_t>(1, rtp->tile_count) * heads_per_tile, 1};
            wgs.local = {workers, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->tile_count);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
            push_u32(kd, rtp->k_seq_pitch);
            push_u32(kd, rtp->v_seq_pitch);
        }};
    }
};

// ===========================================================================
// Q3 prefill, GQA-aware systolic variant - one token x one KV head per WG on cm_dpas
// ===========================================================================
class AttentionDenseGqaDpasGenerator : public KernelGenerator {
public:
    explicit AttentionDenseGqaDpasGenerator(bool split = false)
        : KernelGenerator("sparse_attention_dense_gqa_dpas", split ? "_split_cm" : "_cm"),
          m_split(split) {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<sparse_sdpa>();
        jit.add(make_jit(desc, params));
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
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY_LENGTH});
        if (m_split) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ACC});
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ML});
        } else {
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        }
        for (uint32_t i = 0; i < 5; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[split = m_split](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            // QSA_GQA_HEAD_MAJOR=1: dim0 enumerates (token, W4 worker), dim1 the KV head.
            const size_t tokens = std::max<size_t>(1, rtp->batch * rtp->seq_len);
            wgs.global = {tokens * 4, desc->num_kv_heads, split ? gqa_split_partitions : 1};
            wgs.local = {4, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, tokens);
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
            push_u32(kd, rtp->k_seq_pitch);
            push_u32(kd, rtp->v_seq_pitch);
        }};
    }

private:
    bool m_split;
};

// ===========================================================================
// Q3 decode, split-selected-KV variant (dense), pass 1 - per-partition partial softmax
// ===========================================================================
class SplitDenseGenerator : public KernelGenerator {
public:
    SplitDenseGenerator() : KernelGenerator("sparse_attention_dense_split", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit_with_split(params.typed_desc<sparse_sdpa>(), params, split_partitions));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::QUERY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::VALUE});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_INDICES});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::SELECTED_BLOCK_COUNT});
        args.push_back({ArgumentDescriptor::Types::INPUT, SparseSDPAIn::KEY_LENGTH});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ML});
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
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->batch * rtp->seq_len), split_partitions};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->history_len);
            push_u32(kd, rtp->batch * rtp->seq_len);
            push_u32(kd, rtp->k_seq_pitch);
            push_u32(kd, rtp->v_seq_pitch);
        }};
    }
};

// ===========================================================================
// Q3 decode, split-selected-KV variant (dense), pass 2 - merge pass
// ===========================================================================
class SplitReduceDenseGenerator : public KernelGenerator {
public:
    SplitReduceDenseGenerator() : KernelGenerator("sparse_attention_dense_split_reduce", "_cm") {}

protected:
    [[nodiscard]] std::string get_build_options(const RuntimeParams& params) const override {
        return KernelGenerator::get_build_options(params) + get_sparse_attention_build_options();
    }

    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = KernelGenerator::get_jit_constants(params);
        jit.add(make_jit(params.typed_desc<sparse_sdpa>(), params));
        jit.make("QSA_SPLIT_PARTITIONS", decode_split_partitions(*params.typed_desc<sparse_sdpa>(), params.get_device_info()));
        jit.make("QSA_REDUCE_COLS", split_reduce_cols(params.typed_desc<sparse_sdpa>()->v_head_size));
        jit.add(make_jit_constant("KERNEL_NAME", get_entry_point(params)));
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ACC});
        args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, SparseSDPAInternalBufIdx::SPLIT_ML});
        args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        for (uint32_t i = 0; i < 2; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<sparse_sdpa>();
            auto* rtp = rt_of(rt);

            auto& wgs = kd.params.workGroups;
            wgs.global = {desc->num_heads, std::max<size_t>(1, rtp->batch * rtp->seq_len), desc->v_head_size / split_reduce_cols(desc->v_head_size)};
            wgs.local = {1, 1, 1};

            kd.params.scalars.clear();
            push_u32(kd, rtp->seq_len);
            push_u32(kd, rtp->batch * rtp->seq_len);
        }};
    }
};

class SparseSDPACmImpl : public PrimitiveImplCM {
public:
    DECLARE_OBJECT_TYPE_SERIALIZATION(ov::intel_gpu::cm::SparseSDPACmImpl)

    Stage::Ptr attention = make_stage<AttentionDenseGenerator>();
    Stage::Ptr attention_dpas = make_stage<AttentionDenseDpasGenerator>();
    Stage::Ptr attention_gqa_dpas = make_stage<AttentionDenseGqaDpasGenerator>();
    Stage::Ptr attention_gqa_split = make_stage<AttentionDenseGqaDpasGenerator>(true);
    Stage::Ptr split = make_stage<SplitDenseGenerator>();
    Stage::Ptr split_reduce = make_stage<SplitReduceDenseGenerator>();

    SparseSDPACmImpl() : PrimitiveImplCM(SparseSDPACMImplementationManager::get_type_info_static()) {
        m_rt_params = std::make_unique<SparseSDPARuntimeParams>();
    }

    SparseSDPARuntimeParams& rt() {
        if (m_rt_params == nullptr) {
            m_rt_params = std::make_unique<SparseSDPARuntimeParams>();
        }
        return *static_cast<SparseSDPARuntimeParams*>(m_rt_params.get());
    }

    explicit SparseSDPACmImpl(const RuntimeParams& params) : SparseSDPACmImpl() {
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
        return make_deep_copy<SparseSDPACmImpl>(this);
    }

    void update(primitive_inst& inst, const kernel_impl_params& impl_params) override {
        update_rt_params(inst);
        PrimitiveImplCM::update(inst, impl_params);
    }

    [[nodiscard]] bool requires_update(primitive_inst& /*inst*/, const kernel_impl_params& /*impl_params*/) const override {
        return true;
    }

    // Recomputed rather than cached: make_deep_copy() rebuilds the impl with the default
    // constructor and copies only _order and the per-stage kernel data.
    static bool use_dpas_attention(const kernel_impl_params& params) {
        return can_use_dpas_attention(*params.typed_desc<sparse_sdpa>(), params.get_device_info());
    }
    static bool use_gqa_dpas_attention(const kernel_impl_params& params) {
        return can_use_gqa_dpas_attention(*params.typed_desc<sparse_sdpa>(), params.get_device_info());
    }

    // Split-selected-KV is only a decode fallback for configs without GQA-DPAS (see
    // sparse_pa.cpp's identical gate and BMG measurement).
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
        // SPLIT_ACC/SPLIT_ML: only ever dispatched for decode-shaped calls (seq_len == 1),
        // where batch*seq_len == batch (small), so this is cheap regardless of whether
        // split actually fires for this particular call.
        const auto split_rows = is_decode_shape(params) ? batch * seq_len : size_t{1};
        const auto partitions = decode_split_partitions(*desc, params.get_device_info());
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
        force_stages_refresh(*this);
        auto* rtp = &rt();

        const auto& params = *instance.get_impl_params();
        const auto& query_shape = params.get_input_layout(SparseSDPAIn::QUERY).get_shape();
        const auto& key_shape = params.get_input_layout(SparseSDPAIn::KEY).get_shape();
        OPENVINO_ASSERT(query_shape.size() == 4 && key_shape.size() == 4,
                        "[GPU][CM] sparse_sdpa expects rank-4 [B,H,*,D] query/key inputs");

        OPENVINO_ASSERT(!params.get_input_layout(SparseSDPAIn::QUERY).data_padding && !params.get_output_layout(0).data_padding,
                        "[GPU][CM] sparse_sdpa: padded query/output buffers are not supported");

        rtp->batch = query_shape[0];
        rtp->seq_len = query_shape[2];
        rtp->history_len = key_shape[2];
        rtp->k_seq_pitch = seq_pitch(params.get_input_layout(SparseSDPAIn::KEY), "key");
        rtp->v_seq_pitch = seq_pitch(params.get_input_layout(SparseSDPAIn::VALUE), "value");
        rtp->is_decode = rtp->seq_len == 1;

        rtp->tile_count = rtp->batch * ((rtp->seq_len + dpas_query_tile - 1) / dpas_query_tile);
    }
};

}  // namespace

bool SparseSDPACMImplementationManager::validate_impl(const cldnn::program_node& node) const {
    assert(node.is_type<cldnn::sparse_sdpa>());

    if (node.has_fused_primitives()) {
        return false;
    }

    auto& engine = node.get_program().get_engine();
    const auto& config = node.get_program().get_config();
    const auto& info = engine.get_device_info();
    if (!check_cm_jit_support(engine, config) || !info.supports_immad || !config.get_use_cm()) {
        return false;
    }

    const auto desc = node.as<cldnn::sparse_sdpa>().get_primitive();
    std::string reason;
    if (!ov::intel_gpu::op::SparseSDPA::validate_config(desc->get_config(), reason)) {
        return false;
    }

    // Every dense kernel adds the causal tail and applies the causal bound.
    if (desc->num_selection_heads != 1 || !desc->causal) {
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

std::unique_ptr<cldnn::primitive_impl> SparseSDPACMImplementationManager::create_impl(const cldnn::program_node& node,
                                                                                       const cldnn::kernel_impl_params& params) const {
    OPENVINO_ASSERT(node.is_type<cldnn::sparse_sdpa>());
    return std::make_unique<SparseSDPACmImpl>(params);
}

}  // namespace ov::intel_gpu::cm

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::sparse_sdpa)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_gpu::cm::SparseSDPACmImpl)
