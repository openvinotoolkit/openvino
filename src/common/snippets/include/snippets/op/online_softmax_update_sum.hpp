// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/op/op.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {

/**
 * @interface OnlineSoftmaxUpdateSum
 * @brief OnlineSoftmaxUpdateSum handles the current sum computation and update parts in online softmax algo.
 * @ingroup snippets
 * scheme:
 * -----> Buffer  In1  In0(sum_local)
 * |          \   /    |
 * | Result1---Mul     |
 * |            \     /
 * ---------------Add
 *                 |
 *              Result0
 */
class SNIPPETS_API OnlineSoftmaxUpdateSum : public ov::op::Op {
public:
    OPENVINO_OP("OnlineSoftmaxUpdateSum", "SnippetsOpset");

    explicit OnlineSoftmaxUpdateSum(const Output<Node>& sum_local, const Output<Node>& coeff);
    OnlineSoftmaxUpdateSum() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
