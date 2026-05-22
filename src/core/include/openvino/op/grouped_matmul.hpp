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
///   - mat_b: (G, N, K) - per-group weights (stored transposed)
///   - output: (total_tokens, N) - each group's output in corresponding rows
///
/// - **Case 2 (3D × 3D)**: Batched uniform (no offsets needed)
///   - mat_a: (G, M, K) - per-group inputs
///   - mat_b: (G, N, K) - per-group weights (stored transposed)
///   - output: (G, M, N) - per-group outputs
///
/// - **Case 3 (2D × 2D)**: MoE weight gradient
///   - mat_a: (K, total_tokens) - trailing dim partitioned by offsets
///   - mat_b: (N, total_tokens) - columns partitioned by offsets (stored transposed)
///   - output: (G, K, N) - per-group gradient matrices
///
/// All four inputs are always required. Pass a 1D tensor of Shape{0} to indicate
/// "not applicable":
///   - offsets: Shape{0} for Case 2 (3D×3D, no group offsets needed)
///   - bias: Shape{0} when no per-group bias is desired
///
/// Bias shape (when provided): [G, N] — added after the matmul for each group.
///
/// \ingroup ov_ops_cpp_api
class OPENVINO_API GroupedMatMul : public ov::op::Op {
public:
    OPENVINO_OP("GroupedMatMul", "opset17", ov::op::Op);

    GroupedMatMul() = default;

    /// \brief Constructs a GroupedMatMul operation without offsets (3D × 3D case).
    ///        Injects empty Shape{0} placeholder constants for offsets and bias.
    ///
    /// \param mat_a First input tensor (G, M, K)
    /// \param mat_b Second input tensor (G, N, K)
    GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b);

    /// \brief Constructs a GroupedMatMul operation with offsets (2D × 3D or 2D × 2D).
    ///        Injects an empty Shape{0} placeholder constant for bias.
    ///
    /// \param mat_a First input tensor
    /// \param mat_b Second input tensor
    /// \param offsets Cumulative offsets tensor of shape (G,) indicating group boundaries.
    GroupedMatMul(const Output<Node>& mat_a, const Output<Node>& mat_b, const Output<Node>& offsets);

    /// \brief Constructs a GroupedMatMul operation with all four inputs.
    ///
    /// \param mat_a First input tensor
    /// \param mat_b Second input tensor (stored transposed)
    /// \param offsets Cumulative offsets tensor of shape (G,), or Shape{0} for Case 2.
    /// \param bias Per-group bias of shape (G, N), or Shape{0} for no bias.
    GroupedMatMul(const Output<Node>& mat_a,
                  const Output<Node>& mat_b,
                  const Output<Node>& offsets,
                  const Output<Node>& bias);

    bool visit_attributes(AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool evaluate(TensorVector& outputs, const TensorVector& inputs) const override;
    bool has_evaluate() const override;
};

}  // namespace ov::op::v17
