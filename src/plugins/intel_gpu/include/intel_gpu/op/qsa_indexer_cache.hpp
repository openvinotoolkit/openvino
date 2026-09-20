// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/qsa.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/op/util/variable_extension.hpp"

namespace ov::intel_gpu::op {

/// @brief GPU-internal state DECLARATION for the QSA indexer's raw-K and block-summary
/// tensors, used by the SDPA-style `QSASDPA` op.
///
/// \note `QSAIndexerCache` does not compute or write cache
/// content. It has NO runtime data inputs and is not an independent Q1 producer: it only
/// declares the identity of the two persistent Variables and carries the graph-level QSA
/// `Config`. The corresponding `qsa_inst` that executes `QSASDPA` (see
/// ops/qsa_sdpa.cpp::CreateQSASDPAOp) is the sole owner of reserve/grow, Q1 dispatch, and
/// state publication (`Assign`) for both Variables. This mirrors `ov::intel_gpu::op::
/// KVCache` at the OV level (a state declaration), while the runtime growth/write logic
/// lives in `qsa_inst`, analogous to `kv_cache_inst`.
///
/// Output 0/1 report the CURRENT (pre-call) dense content of the raw-K/summary Variables,
/// exactly like a bare `ReadValue` node would report a single Variable's current value.
///
/// This op owns TWO independent pieces of state (raw-K and summary), but
/// `ov::op::util::VariableExtension` only supports a single `Variable` per node. This class
/// therefore holds a second `Variable` (`m_summary_variable`) directly and exposes it via
/// `get_summary_variable()`/`get_summary_variable_id()`; `get_variable()`/`get_variable_id()`
/// (the `VariableExtension` overrides) refer to the raw-K `Variable` only. Downstream
/// consumers that need both (state reset, beam reorder, ProgramBuilder lowering) must use
/// the explicit accessors rather than the generic `VariableExtension` interface.
///
/// Beam-search reordering and `trim`/dynamic-batch support are not implemented. The
/// current implementation supports one growing sequence per batch slot.
class QSAIndexerCache : public ov::op::Op, public ov::op::util::VariableExtension {
public:
    OPENVINO_OP("QSAIndexerCache", "gpu_opset");

    /// Reuses QSA's Config verbatim: `indexer_head_dim` (Di) and `compress_ratio` (r)
    /// drive the raw-K/summary output shapes, the remaining fields are unused by this op
    /// but kept so the same Config object can be threaded through unchanged from QSASDPA.
    using Config = ov::op::internal::QSA::Config;

    QSAIndexerCache() = default;

    /// @brief Pure state declaration: zero data inputs. `MIXED`, the indexer projection
    /// weights, `ROTARY_COS_SIN`, `PAST_LENS`, `SUBSEQUENCE_BEGINS`, and `POSITION_IDS`
    /// are NOT inputs of this op -- they are only needed by the actual Q1/Q2/Q3
    /// computation, which is owned by `QSASDPA`'s own `qsa_inst`.
    QSAIndexerCache(const std::shared_ptr<ov::op::util::Variable>& raw_k_variable,
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
