// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/matmul.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface LoadConvertSaturation
 * @brief Fused operation to represent computations equal to consecutive Load and ConvertSaturation operations.
 *        The operation is used for peephole optimization during subgraph lowering.
 * @ingroup snippets
 */
class MatMulCPU : public ngraph::op::v0::MatMul {
public:
    OPENVINO_OP("MatMulCPU", "SnippetsOpset", ngraph::op::v0::MatMul);
    MatMulCPU(const Output<Node>& A, const Output<Node>& B);
    MatMulCPU() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
};

} // namespace op
} // namespace snippets
} // namespace ngraph