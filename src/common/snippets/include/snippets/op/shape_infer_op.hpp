// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "snippets/snippets_visibility.hpp"

namespace ov::snippets::op {

/**
 * @interface ShapeInferOp
 * @brief Op which infers shape without actually moving data
 * @ingroup snippets
 */
class SNIPPETS_API ShapeInferOp : public ov::op::Op {
public:
    OPENVINO_OP("ShapeInferOp", "SnippetsOpset");
    ShapeInferOp() = default;
    explicit ShapeInferOp(const OutputVector& args) : ov::op::Op(args) {}
};

}  // namespace ov::snippets::op
