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

namespace ov::snippets::op {

/**
 * @interface OnlineSoftmaxUpdateSum
 * @brief OnlineSoftmaxUpdateSum handle the current sum computation and update part in online softmax algo.
 * @ingroup snippets
 */
class OnlineSoftmaxUpdateSum : public ov::op::Op {
public:
    OPENVINO_OP("OnlineSoftmaxUpdateSum", "SnippetsOpset");

    explicit OnlineSoftmaxUpdateSum(const Output<Node>& A, const Output<Node>& B);
    OnlineSoftmaxUpdateSum() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
