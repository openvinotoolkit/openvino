// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface FMA
 * @brief Fused Multiply Add
 * @ingroup snippets
 */
class FMA : public ngraph::op::Op {
public:
    OPENVINO_OP("FMA", "SnippetsOpset");

    FMA() = default;
    FMA(const Output<Node>& a, const Output<Node>& b, const Output<Node>& c);

    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void validate_and_infer_types() override;

    OPENVINO_SUPPRESS_DEPRECATED_START
    bool evaluate(const HostTensorVector& output_values, const HostTensorVector& input_values) const override;
    OPENVINO_SUPPRESS_DEPRECATED_END
};

} // namespace op
} // namespace snippets
} // namespace ngraph
