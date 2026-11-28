// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

/// \brief MoEExpertSort — GPU-internal op that groups the (token, top-k slot) pairs of a
/// Mixture-of-Experts block by expert, as needed by GroupedMatMul.
///
/// Let S = num_tokens * top_k and s = token * top_k + slot be a flat position. The pairs are
/// ordered by expert id (counting sort, stable by s).
///
/// Inputs:
///   0: topk_indices  [num_tokens, top_k]  — selected expert ids
///
/// Outputs (all i32):
///   0: token_index   [S]            — token of each sorted row
///   1: offsets       [num_experts]  — inclusive end row of each expert (GroupedMatMul offsets)
///   2: row_lut       [S]            — sorted row of each flat position s (inverse permutation)
///
class MoEExpertSort : public ov::op::Op {
public:
    OPENVINO_OP("MoEExpertSort", "gpu_opset");

    MoEExpertSort() = default;
    MoEExpertSort(const ov::Output<ov::Node>& topk_indices, size_t num_experts);

    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    size_t get_num_experts() const {
        return m_num_experts;
    }

private:
    size_t m_num_experts = 0;
};

}  // namespace ov::intel_gpu::op
