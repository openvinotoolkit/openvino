// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/softmax.hpp>

namespace ngraph {
namespace snippets {
namespace op {

/**
 * @interface Softmax
 * @brief This is simply a copy of the ov::op::v8::Softmax, which is needed to indicate that the Softmax operation was
 *        scheduled appropriately and can de decomposed to a set of low-level operations.
 * @ingroup snippets
 */
class Softmax : public ov::op::v8::Softmax {
public:
    OPENVINO_OP("Softmax", "SnippetsOpset", ov::op::v8::Softmax);
    Softmax() = default;
    Softmax(const Output<Node>& arg, const int64_t axis = 1) : ov::op::v8::Softmax(arg, axis) {}
};

} // namespace op
} // namespace snippets
} // namespace ngraph
