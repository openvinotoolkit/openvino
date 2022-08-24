// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/op.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Truncation
 * @brief TODO
 * @ingroup snippets
 */
class Truncation : public ngraph::op::Op {
public:
    OPENVINO_OP("Truncation", "SnippetsOpset");

    Truncation(const Output<Node>& x);
    Truncation() = default;

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_evaluate() const override { return false; }
};

} // namespace op
} // namespace snippets
} // namespace ngraph
