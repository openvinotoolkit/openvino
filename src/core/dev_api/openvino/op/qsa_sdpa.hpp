// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cstddef>

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"
#include "openvino/op/qsa.hpp"

namespace ov::op::internal {

/// \note QSASDPA op class is under development and subject to change.
///
/// \brief SDPA-style (dense, KVCache-managed) variant of QSA (Qwen Sparse Attention).
///
/// QSASDPA is the "no physical page table" sibling of `ov::op::internal::QSA`: it exposes
/// the same math (indexer block scoring + top-k selection + selected-block sparse main
/// attention, see `QSA`'s own documentation), but the caller never constructs or manages a
/// paged KV cache / block table. Instead:
///   - the main K/V cache inputs are expected to be the "present" output of an ordinary
///     `ov::intel_gpu::op::KVCache` node (the same mechanism `SDPA` already uses), already
///     containing the full history for this call;
///   - the indexer raw-K/summary cache inputs are expected to be the outputs of a
///     `QSAIndexerCache` node (a `VariableExtension` that internally reuses the existing
///     Q1 kernel unchanged).
///
/// Lowering (`ops/qsa_sdpa.cpp::CreateQSASDPAOp`) reinterprets these dense
/// `[num_seqs, ..., S, ...]` tensors as a "degenerate" page table (one physical page per
/// sequence, `pa_block_size == S`, identity `block_indices`) and dispatches the SAME
/// `cldnn::qsa` primitive / CM kernels that `QSA` uses, with `skip_kv_write=true` (the
/// main K/V append is already done by the upstream `KVCache` node). No new `.cm` kernel
/// exists or is needed for this op.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API QSASDPA : public ov::op::Op {
public:
    OPENVINO_OP("QSASDPA");

    /// Reuses QSA's Config verbatim: the underlying geometry/attention math is identical,
    /// only the input contract (paged vs. dense/KVCache-managed) differs.
    using Config = ov::op::internal::QSA::Config;

    /// @brief Fixed input order of the QSASDPA op.
    ///
    /// Compared to `ov::op::internal::QSA::InputIdx`, the four paged-cache-table inputs
    /// (`BLOCK_INDICES`, `BLOCK_INDICES_BEGINS`) are gone entirely (synthesized internally
    /// at lowering time as an identity mapping) and `KEY_CACHE`/`VALUE_CACHE`/
    /// `INDEXER_RAW_K_CACHE`/`INDEXER_SUMMARY_CACHE` become dense, KVCache/QSAIndexerCache
    /// -managed tensors instead of physically-paged ones. `KEY`/`VALUE` (the current-step
    /// new tokens) also disappear: they are already folded into `KEY_CACHE_DENSE`/
    /// `VALUE_CACHE_DENSE` by the upstream `KVCache` node's Concat before this op runs.
    struct InputIdx {
        static constexpr size_t MIXED = 0;                        ///< [tokens, D] GR READ/MIX output
        static constexpr size_t QUERY = 1;                        ///< [tokens, Hq*Dh] (or [tokens, 2*Hq*Dh] when gate is fused)
        static constexpr size_t PAST_LENS = 2;                    ///< [num_seqs], same semantics as QSA::InputIdx::PAST_LENS
        static constexpr size_t SUBSEQUENCE_BEGINS = 3;           ///< [num_seqs+1], same semantics as QSA::InputIdx::SUBSEQUENCE_BEGINS
        static constexpr size_t KEY_CACHE_DENSE = 4;              ///< [num_seqs, Hkv, S, Dh], KVCache "present" output
        static constexpr size_t VALUE_CACHE_DENSE = 5;            ///< [num_seqs, Hkv, S, Dv], KVCache "present" output
        static constexpr size_t INDEXER_RAW_K_CACHE_DENSE = 6;    ///< [num_seqs, S, Di], QSAIndexerCache output 0
        static constexpr size_t INDEXER_SUMMARY_CACHE_DENSE = 7;  ///< [num_seqs, S/r, Di], QSAIndexerCache output 1
        static constexpr size_t INDEXER_QK_WEIGHT = 8;            ///< [Hidx*Di + Di, D] row-major
        static constexpr size_t INDEXER_Q_NORM_WEIGHT = 9;        ///< [Di]
        static constexpr size_t INDEXER_K_NORM_WEIGHT = 10;       ///< [Di]
        static constexpr size_t ROTARY_COS_SIN = 11;              ///< [max_positions, indexer_rotary_dim], (cos,sin) interleaved pairs
        static constexpr size_t POSITION_IDS = 12;                ///< [tokens] absolute position / mRoPE rank

        static constexpr size_t COUNT = 13;
    };

    QSASDPA() = default;

    QSASDPA(const ov::OutputVector& args, const Config& config, const ov::element::Type output_type = ov::element::dynamic);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    const Config& get_config() const {
        return m_config;
    }
    ov::element::Type get_output_type() const {
        return m_output_type;
    }

    /// Delegates to `ov::op::internal::QSA::validate_config` verbatim: the geometry
    /// constraints do not depend on how the caches are managed.
    static bool validate_config(const Config& config, std::string& reason) {
        return ov::op::internal::QSA::validate_config(config, reason);
    }

private:
    Config m_config{};
    ov::element::Type m_output_type = ov::element::dynamic;
};

}  // namespace ov::op::internal
