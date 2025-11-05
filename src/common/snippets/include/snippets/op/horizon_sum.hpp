// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {

/**
 * @interface HorizonSum
 * @brief The operation calculates a horizon sum of a vector register
 * @ingroup snippets
 */
class SNIPPETS_API HorizonSum : public ov::op::Op {
public:
    OPENVINO_OP("HorizonSum", "SnippetsOpset");

    explicit HorizonSum(const Output<Node>& x);
    HorizonSum() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
