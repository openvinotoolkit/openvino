// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::op::v17 {
/// \brief Grouped Matrix Multiplication operation for Mixture of Experts (MoE).
///
/// Computes multiple matrix multiplications where each group processes a subset
/// of the input data. This operation supports three input combinations:
///
/// - **Case 1 (2D × 3D)**: MoE forward pass
///   - mat_a: (total_tokens, K) - rows partitioned by offsets
///   - mat_b: (G, K, N) - per-group weights
///   - output: (total_tokens, N) - each group's output in corresponding rows
///
/// - **Case 2 (3D × 3D)**: Batched uniform (no offsets needed)
///   - mat_a: (G, M, K) - per-group inputs
///   - mat_b: (G, K, N) - per-group weights
///   - output: (G, M, N) - per-group outputs
///
/// - **Case 3 (2D × 2D)**: MoE weight gradient
///   - mat_a: (K, total_tokens) - trailing dim partitioned by offsets
///   - mat_b: (total_tokens, N) - leading dim partitioned by offsets
///   - output: (G, K, N) - per-group gradient matrices
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GroupedMatMul : public ov::op::Op {
public:
    OPENVINO_OP("GroupedMatMul", "opset17", ov::op::Op);

    GroupedMatMul() = default;

    /// \brief Constructs a GroupedMatMul operation without offsets (3D × 3D case).
    ///
    /// \param mat_a First input tensor (G, M, K)
    /// \param mat_b Second input tensor (G, K, N)
    GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b);

    /// \brief Constructs a GroupedMatMul operation with offsets (2D × 3D or 2D × 2D).
    ///
    /// \param mat_a First input tensor
    /// \param mat_b Second input tensor
    /// \param offsets Cumulative offsets tensor of shape (G,) indicating group boundaries.
    ///                For 2D×3D: partitions rows of mat_a.
    ///                For 2D×2D: partitions shared dimension of both operands.
    GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b, const Output<Node>& offsets);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace ov::op::v17
