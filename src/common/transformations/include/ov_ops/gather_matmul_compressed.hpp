// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "ov_ops/gather_matmul.hpp"

namespace ov::op::internal {

class TRANSFORMATIONS_API GatherMatmulCompressed : public GatherMatmul {
public:
    OPENVINO_OP("GatherMatmulCompressed", "", GatherMatmul);

    GatherMatmulCompressed() = default;

    /// \param weight_scale_global  One per-tensor factor the per-group `weight_scales` must be
    ///     multiplied by. 1.0 for the usual single-level f16 scale; only ever != 1 when the group
    ///     scale is stored in a low-precision float type that cannot hold the weights' magnitude on
    ///     its own (see CompressedWeightsBlock's allow_two_level_scale). Carried as a member rather
    ///     than a 7th input on purpose: the input list is pattern-matched by exact arity in
    ///     moe_op_fusion.cpp and keep_moe_3gemm_const_precision.cpp, and a new input would silently
    ///     stop those patterns from matching.
    GatherMatmulCompressed(const ov::Output<Node>& A,
                           const ov::Output<Node>& B,
                           const ov::Output<Node>& indices,
                           const ov::Output<Node>& bias,
                           const ov::Output<Node>& weight_scales,
                           const ov::Output<Node>& weight_zero_points,
                           float weight_scale_global = 1.0f);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    float get_weight_scale_global() const {
        return m_weight_scale_global;
    }

private:
    float m_weight_scale_global = 1.0f;
};

}  // namespace ov::op::internal
