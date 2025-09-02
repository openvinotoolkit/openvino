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
 * @interface OnlineSoftmax
 * @brief OnlineSoftmax is a softmax that evaluate with online manner with axis on last dimension.
 * @ingroup snippets
 */
class OnlineSoftmax : public ov::op::Op {
public:
    OPENVINO_OP("OnlineSoftmax", "SnippetsOpset");

    explicit OnlineSoftmax(const Output<Node>& x);
    OnlineSoftmax() = default;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;
    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op