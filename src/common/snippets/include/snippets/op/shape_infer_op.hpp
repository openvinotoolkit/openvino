// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov {
namespace snippets {
namespace op {

/**
 * @interface ShapeInferOp
 * @brief Op which infers shape without actually moving data
 * @ingroup snippets
 */
class ShapeInferOp : public ov::op::Op {
public:
    OPENVINO_OP("ShapeInferOp", "SnippetsOpset");
    ShapeInferOp() = default;
    ShapeInferOp(const OutputVector& args) : ov::op::Op(args) {}
};

} // namespace op
} // namespace snippets
} // namespace ov
