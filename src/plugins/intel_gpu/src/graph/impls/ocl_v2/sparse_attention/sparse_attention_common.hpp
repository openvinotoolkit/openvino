// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Kernel generators shared by the OpenCL SparsePA (paged) and SparseSDPA (dense) implementations. The two primitives
// run the same kernels with a different QSA_ADDRESS_MODE; only the input slots and the scalar lists differ.

#pragma once

#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "../primitive_ocl_base.hpp"
#include "../qsa_host_common.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"

namespace ov::intel_gpu::ocl::qsa {

struct SparseAttentionInternalBufIdx {
    static constexpr size_t SPLIT_ACC = 0;  ///< f32 [rows, Hq, partitions, Dv]; decode split attention only
    static constexpr size_t SPLIT_ML = 1;   ///< f32 [rows, Hq, partitions, 2]; decode split attention only

    static constexpr size_t COUNT = 2;
};

/// All sizes are upper bounds derived from shapes: the real sequence lengths live in GPU memory.
struct SparseAttentionRuntimeParams : public ImplRuntimeParams {
    size_t num_tokens = 0;  ///< paged: tokens; dense: B * L
    size_t num_seqs = 0;
    size_t tile_count = 0;
    bool is_decode = false;
    size_t batch = 0;
    size_t seq_len = 0;
    size_t history_len = 0;
    size_t k_seq_pitch = 0;
    size_t v_seq_pitch = 0;
    size_t q_base = 0;
    size_t q_pitch = 0;
};

enum class AttKind { Scalar, Dpas, Gqa, GqaSplit, Split, Reduce };

/// Input slots of the primitive the generators serve.
struct SparseAttentionSlots {
    bool paged = true;
    size_t query = 0;
    size_t key = 0;  ///< paged: key cache
    size_t value = 0;
    size_t past_lens = 0;
    size_t subsequence_begins = 0;
    size_t block_indices = 0;
    size_t block_indices_begins = 0;
    size_t selected_blocks = 0;
    size_t selected_count = 0;
    size_t key_length = 0;
};

inline std::optional<size_t> page_size_of(const SparseAttentionSlots& slots, const RuntimeParams& params) {
    if (!slots.paged) {
        return std::nullopt;
    }
    const auto& ps = params.get_input_layout(slots.key).get_partial_shape();
    OPENVINO_ASSERT(ps.size() == 4 && ps[2].is_static(), "[GPU][OCL] sparse attention: key_cache must have a static page_size dim");
    return static_cast<size_t>(ps[2].get_length());
}

template <class Desc>
class AttentionGenerator : public QsaGenerator {
public:
    AttentionGenerator(AttKind kind, const SparseAttentionSlots& slots)
        : QsaGenerator(template_name(kind), entry_name(kind, slots.paged), kind == AttKind::Dpas || kind == AttKind::Gqa || kind == AttKind::GqaSplit),
          m_kind(kind),
          m_slots(slots) {}

    static std::string template_name(AttKind kind) {
        switch (kind) {
        case AttKind::Scalar:
            return "sparse_attention_scalar";
        case AttKind::Dpas:
            return "sparse_attention_dpas";
        case AttKind::Gqa:
        case AttKind::GqaSplit:
            return "sparse_attention_gqa_dpas";
        case AttKind::Split:
            return "sparse_attention_split";
        case AttKind::Reduce:
            return "sparse_attention_split_reduce";
        }
        return {};
    }

    // Entry names carry the addressing mode so the kernel is recognisable in dumps and tests.
    static std::string entry_name(AttKind kind, bool paged) {
        const std::string base = paged ? "sparse_attention_paged" : "sparse_attention_dense";
        switch (kind) {
        case AttKind::Scalar:
            return base;
        case AttKind::Dpas:
            return base + "_dpas";
        case AttKind::Gqa:
            return base + "_gqa_dpas";
        case AttKind::GqaSplit:
            return base + "_gqa_dpas_split";
        case AttKind::Split:
            return base + "_split";
        case AttKind::Reduce:
            return base + "_split_reduce";
        }
        return base;
    }

protected:
    [[nodiscard]] JitConstants get_jit_constants(const RuntimeParams& params) const override {
        auto jit = QsaGenerator::get_jit_constants(params);
        const auto desc = params.typed_desc<Desc>();
        const auto& sel_ps = params.get_input_layout(m_slots.selected_blocks).get_partial_shape();
        OPENVINO_ASSERT(sel_ps.rank().is_static() && sel_ps.size() >= 2 && sel_ps[sel_ps.size() - 1].is_static(),
                        "[GPU][OCL] sparse attention expects a rank>=2 selected_block_indices input with a static K_max");
        const auto page = page_size_of(m_slots, params);

        jit.make("QSA_HEADS", desc->num_heads);
        jit.make("QSA_KV_HEADS", desc->num_kv_heads);
        jit.make("QSA_K_HEAD_SIZE", desc->k_head_size);
        jit.make("QSA_V_HEAD_SIZE", desc->v_head_size);
        jit.make("QSA_COMPRESS_RATIO", desc->key_block_size);
        jit.make("QSA_BLOCK_TOPK", sel_ps[sel_ps.size() - 1].get_length());
        jit.make("QSA_ADDRESS_MODE", m_slots.paged ? 0 : 1);
        jit.add(float_constant("QSA_SCALE", desc->scale));
        if (page) {
            jit.make("QSA_PAGE_SIZE", *page);
        }
        switch (m_kind) {
        case AttKind::Dpas:
            jit.make("QSA_ATT_WORKERS", std::max<size_t>(1, dpas_attention_workers(*desc)));
            jit.make("QSA_HEADS_PER_TILE", dpas_heads_per_tile(*desc));
            break;
        case AttKind::Gqa:
            jit.make("QSA_IS_CAUSAL", desc->causal ? 1 : 0);
            break;
        case AttKind::GqaSplit:
            jit.make("QSA_IS_CAUSAL", desc->causal ? 1 : 0);
            jit.make("QSA_GQA_SPLIT", 1);
            jit.make("QSA_SPLIT_PARTITIONS", gqa_split_partitions);
            break;
        case AttKind::Split:
            jit.make("QSA_SPLIT_PARTITIONS", split_partitions);
            break;
        case AttKind::Reduce:
            jit.make("QSA_SPLIT_PARTITIONS", decode_split_partitions(*desc, page, params.get_device_info()));
            jit.make("QSA_REDUCE_COLS", split_reduce_cols(desc->v_head_size));
            break;
        default:
            break;
        }
        return jit;
    }

    [[nodiscard]] Arguments get_arguments_desc(const RuntimeParams& /*params*/) const override {
        Arguments args;
        const auto input = [&](size_t idx) {
            args.push_back({ArgumentDescriptor::Types::INPUT, static_cast<uint32_t>(idx)});
        };
        const auto internal = [&](size_t idx) {
            args.push_back({ArgumentDescriptor::Types::INTERNAL_BUFFER, static_cast<uint32_t>(idx)});
        };
        if (m_kind == AttKind::Reduce) {
            internal(SparseAttentionInternalBufIdx::SPLIT_ACC);
            internal(SparseAttentionInternalBufIdx::SPLIT_ML);
            args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
        } else {
            input(m_slots.query);
            input(m_slots.key);
            input(m_slots.value);
            if (m_slots.paged) {
                input(m_slots.past_lens);
                input(m_slots.subsequence_begins);
                input(m_slots.block_indices);
                input(m_slots.block_indices_begins);
                input(m_slots.selected_blocks);
                input(m_slots.selected_count);
            } else {
                input(m_slots.selected_blocks);
                input(m_slots.selected_count);
                input(m_slots.key_length);
            }
            if (m_kind == AttKind::GqaSplit || m_kind == AttKind::Split) {
                internal(SparseAttentionInternalBufIdx::SPLIT_ACC);
                internal(SparseAttentionInternalBufIdx::SPLIT_ML);
            } else {
                args.push_back({ArgumentDescriptor::Types::OUTPUT, 0});
            }
        }
        const uint32_t scalars = m_kind == AttKind::Reduce ? (m_slots.paged ? 1 : 2) : (m_slots.paged ? 4 : 5);
        for (uint32_t i = 0; i < scalars; i++) {
            args.push_back({ArgumentDescriptor::Types::SCALAR, i});
        }
        return args;
    }

    [[nodiscard]] DispatchDataFunc get_dispatch_data_func() const override {
        return DispatchDataFunc{[kind = m_kind, paged = m_slots.paged](const RuntimeParams& params, KernelData& kd, ImplRuntimeParams* rt) {
            if (params.is_dynamic()) {
                return;
            }
            const auto desc = params.typed_desc<Desc>();
            auto* rtp = static_cast<SparseAttentionRuntimeParams*>(rt);
            OPENVINO_ASSERT(rtp != nullptr, "[GPU][OCL] sparse attention runtime params are not initialized");
            const size_t sg = subgroup_size(params.get_device_info());
            const size_t rows = std::max<size_t>(1, rtp->num_tokens);
            const size_t hq = desc->num_heads;
            auto& wgs = kd.params.workGroups;

            switch (kind) {
            case AttKind::Scalar:
                wgs.global = {hq * sg, rows, 1};
                wgs.local = {sg, 1, 1};
                break;
            case AttKind::Dpas: {
                const size_t workers = std::max<size_t>(1, dpas_attention_workers(*desc));
                const size_t ht = dpas_heads_per_tile(*desc);
                wgs.global = {(hq / ht) * workers * sg, std::max<size_t>(1, rtp->tile_count) * ht, 1};
                wgs.local = {workers * sg, 1, 1};
                break;
            }
            case AttKind::Gqa:
            case AttKind::GqaSplit:
                wgs.global = {rows * gqa_workers * sg, desc->num_kv_heads, kind == AttKind::GqaSplit ? gqa_split_partitions : 1};
                wgs.local = {gqa_workers * sg, 1, 1};
                break;
            case AttKind::Split:
                wgs.global = {hq * sg, rows, split_partitions};
                wgs.local = {sg, 1, 1};
                break;
            case AttKind::Reduce:
                wgs.global = {hq * sg, rows, desc->v_head_size / split_reduce_cols(desc->v_head_size)};
                wgs.local = {sg, 1, 1};
                break;
            }

            kd.params.scalars.clear();
            if (kind == AttKind::Reduce) {
                if (!paged) {
                    push_u32(kd, rtp->seq_len);
                }
                push_u32(kd, rtp->num_tokens);
                return;
            }
            if (paged) {
                push_u32(kd, kind == AttKind::Dpas ? rtp->tile_count : rtp->num_tokens);
                push_u32(kd, rtp->num_seqs);
                push_u32(kd, rtp->q_base);
                push_u32(kd, rtp->q_pitch);
            } else {
                // scalar / split: (seq_len, history_len, rows, ...); dpas: (tiles, seq_len, history_len, ...);
                // gqa: (rows, seq_len, history_len, ...)
                if (kind == AttKind::Dpas || kind == AttKind::Gqa || kind == AttKind::GqaSplit) {
                    push_u32(kd, kind == AttKind::Dpas ? rtp->tile_count : rtp->num_tokens);
                    push_u32(kd, rtp->seq_len);
                    push_u32(kd, rtp->history_len);
                } else {
                    push_u32(kd, rtp->seq_len);
                    push_u32(kd, rtp->history_len);
                    push_u32(kd, rtp->num_tokens);
                }
                push_u32(kd, rtp->k_seq_pitch);
                push_u32(kd, rtp->v_seq_pitch);
            }
        }};
    }

private:
    AttKind m_kind;
    SparseAttentionSlots m_slots;
};

}  // namespace ov::intel_gpu::ocl::qsa
