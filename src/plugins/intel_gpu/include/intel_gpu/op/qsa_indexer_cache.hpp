// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/qsa.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal stateful cache for the QSA indexer's raw-K and block-summary tensors,
/// used by the SDPA-style `QSASDPA` op (see QSA_SDPA_DESIGN.md section 4.3, "Option 2").
///
/// `QSA` (PagedAttention-style) keeps the indexer raw-K/summary caches as physically-paged
/// graph inputs managed by an external continuous-batching scheduler. `QSASDPA` instead
/// needs `SDPA`-style dense, `Variable`-backed state, analogous to `ov::intel_gpu::op::
/// KVCache` for the main K/V. Unlike the main K/V, the indexer raw-K append and the Q1
/// (`qsa_prepare.cm`) indexer-K projection are fused into a single CM kernel, so the "new"
/// raw-K/summary values are NOT plain precomputed dense tensors that a generic `Concat`
/// could append (this is why `QSAIndexerCache` exists instead of reusing `KVCache`
/// directly). `QSAIndexerCache` computes its own "new" data internally by re-running Q1
/// unchanged (lowered to the same `cldnn::qsa` primitive with `prepare_only=true`), the same
/// way `ReadValue` derives its output purely from `Variable` state rather than from an
/// explicit "new value" graph input.
///
/// This op owns TWO independent pieces of state (raw-K and summary), but
/// `ov::op::util::VariableExtension` only supports a single `Variable` per node. This class
/// therefore holds a second `Variable` (`m_summary_variable`) directly and exposes it via
/// `get_summary_variable()`/`get_summary_variable_id()`; `get_variable()`/`get_variable_id()`
/// (the `VariableExtension` overrides) refer to the raw-K `Variable` only. Downstream
/// consumers that need both (state reset, beam reorder, ProgramBuilder lowering) must use
/// the explicit accessors rather than the generic `VariableExtension` interface.
///
/// Beam search reordering (see design section 4.9) and `trim`/dynamic-B support are NOT
/// implemented by this class yet; it currently only supports a single, growing sequence
/// per batch slot, mirroring the scope already implemented for `QSASDPA` itself.
class QSAIndexerCache : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("QSAIndexerCache", "gpu_opset");

    /// Reuses QSA's Config verbatim: `indexer_head_dim` (Di) and `compress_ratio` (r)
    /// drive the raw-K/summary output shapes, the remaining fields are unused by this op
    /// but kept so the same Config object can be threaded through unchanged from QSASDPA.
    using Config = ov::op::internal::QSA::Config;

    /// @brief Fixed input order of the QSAIndexerCache op.
    struct InputIdx {
        static constexpr size_t MIXED = 0;                   ///< [tokens, D], same graph output QSASDPA also consumes
        static constexpr size_t INDEXER_QK_WEIGHT = 1;       ///< [Hidx*Di + Di, D] row-major
        static constexpr size_t INDEXER_Q_NORM_WEIGHT = 2;   ///< [Di], unused output but needed by the fused Q1 kernel
        static constexpr size_t INDEXER_K_NORM_WEIGHT = 3;   ///< [Di]
        static constexpr size_t ROTARY_COS_SIN = 4;          ///< [max_positions, indexer_rotary_dim]
        static constexpr size_t PAST_LENS = 5;               ///< [num_seqs]
        static constexpr size_t SUBSEQUENCE_BEGINS = 6;      ///< [num_seqs+1]
        static constexpr size_t POSITION_IDS = 7;            ///< [tokens] absolute position / mRoPE rank, needed by Q1's RoPE
                                                              ///< application -- not synthesizable in general (mRoPE), so unlike
                                                              ///< QUERY/KEY/VALUE this is a genuine, non-dead input.

        static constexpr size_t COUNT = 8;
    };

    QSAIndexerCache() = default;

    QSAIndexerCache(const ov::OutputVector& args,
                     const std::shared_ptr<ov::op::util::Variable>& raw_k_variable,
                     const std::shared_ptr<ov::op::util::Variable>& summary_variable,
                     const Config& config);

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    /// Refers to the raw-K state (VariableExtension's single-variable contract).
    std::string get_variable_id() const override {
        OPENVINO_ASSERT(m_variable, "Variable is not initialized. Variable_id is unavailable");
        return m_variable->get_info().variable_id;
    }

    std::shared_ptr<ov::op::util::Variable> get_summary_variable() const {
        return m_summary_variable;
    }
    void set_summary_variable(const std::shared_ptr<ov::op::util::Variable>& variable) {
        m_summary_variable = variable;
    }
    std::string get_summary_variable_id() const {
        OPENVINO_ASSERT(m_summary_variable, "Summary variable is not initialized. Variable_id is unavailable");
        return m_summary_variable->get_info().variable_id;
    }

    const Config& get_config() const {
        return m_config;
    }

private:
    std::shared_ptr<ov::op::util::Variable> m_summary_variable;
    Config m_config{};
};

}  // namespace ov::intel_gpu::op
