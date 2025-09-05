// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/softmax.hpp"

namespace ov::snippets::op {

/**
 * @interface OnlineSoftmax
 * @brief OnlineSoftmax is a softmax that evaluate with online manner with axis on last dimension.
 * @ingroup snippets
 */
class OnlineSoftmax : public ov::op::v8::Softmax {
public:
    OPENVINO_OP("OnlineSoftmax", "SnippetsOpset");

    explicit OnlineSoftmax(const Output<Node>& x, const int64_t axis = 1);
    OnlineSoftmax() = default;

    void validate_and_infer_types() override;
};

}  // namespace ov::snippets::op