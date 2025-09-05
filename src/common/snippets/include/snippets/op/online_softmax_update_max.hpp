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
 * @brief OnlineSoftmaxUpdateMax handles the current max computation and update parts in online softmax algo.
 * @ingroup snippets
 * scheme:
 * ---------------
 * |             |
 * |            \|/
 * |   Input  Buffer
 * |      \  /    |
 * ------- Max    |
 *         | \   /
 *         |  Sub
 *         |   |
 *    Result0  Result1
 *
 * Note that "Max-->Buffer" should be after Sub.
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
