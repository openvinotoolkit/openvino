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
 * @interface Brgemm
 * @brief Brgemm is a matrix multiplication, but it allows for strided input-output access
 * @ingroup snippets
 */
class Brgemm : public ngraph::op::v0::MatMul {
public:
    OPENVINO_OP("Brgemm", "SnippetsOpset", ngraph::op::v0::MatMul);
    Brgemm(const Output<Node>& A, const Output<Node>& B);
    Brgemm() = default;

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
};

} // namespace op
} // namespace snippets
} // namespace ngraph