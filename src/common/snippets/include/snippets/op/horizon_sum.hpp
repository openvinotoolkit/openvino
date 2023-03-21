// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface HorizonSum
 * @brief The operation calculates a horizon sum of a vector register
 * @ingroup snippets
 */
class HorizonSum : public ngraph::op::Op {
public:
    OPENVINO_OP("HorizonSum", "SnippetsOpset");

    HorizonSum(const Output<Node>& x);
    HorizonSum() = default;

    bool visit_attributes(AttributeVisitor& visitor) override { return true;}
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

} // namespace op
} // namespace snippets
} // namespace ngraph
