// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/pattern/op/block.hpp"
#include "openvino/pass/pattern/op/predicate.hpp"
#include "transformations_visibility.hpp"

namespace ov::pass::pattern::op {

class TRANSFORMATIONS_API QDQBlock;

}  // namespace ov::pass::pattern::op

/**
 * @brief QDQBlock is a reusable pattern Block for Quantize-Dequantize subgraphs.
 *
 * The block matches the following subgraph:
 *   data -> FakeQuantize
 *        -> Convert (to low precision)            [q_convert]
 *        -> Convert (back to original precision)  [dq_convert]
 *        -> [optional Subtract(zero_point)]       [sub]
 *        -> Multiply(scale)                       [mul]
 *
 * Callers supply full predicates for data, q_convert, and dq_convert.
 * Pass an empty Predicate{} when no constraint is needed.
 */
class ov::pass::pattern::op::QDQBlock : public ov::pass::pattern::op::Block {
public:
    QDQBlock(ov::pass::pattern::op::Predicate data_pred,
             ov::pass::pattern::op::Predicate convert1_pred,
             ov::pass::pattern::op::Predicate dq_convert_pred);
};
