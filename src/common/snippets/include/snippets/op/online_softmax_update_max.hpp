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
 * @interface OnlineSoftmaxUpdateMax
 * @brief OnlineSoftmaxUpdateMax handle the current max computation and update part in online softmax algo.
 * It also handle substract(max_past, max_current) part by fix this substract order before store inplace max buffer in
 * control flow.
 * @ingroup snippets
 */
class OnlineSoftmaxUpdateMax : public ov::op::Op {
public:
    OPENVINO_OP("OnlineSoftmaxUpdateMax", "SnippetsOpset");

    explicit OnlineSoftmaxUpdateMax(const Output<Node>& x);
    OnlineSoftmaxUpdateMax() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op
