// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "gru_sequence_shape_inference.hpp"
#include "ngraph_ops/augru_sequence.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace internal {
template <class ShapeType>
void shape_infer(const ov::op::internal::AUGRUSequence* op,
                 const std::vector<ShapeType>& input_shapes,
                 std::vector<ShapeType>& output_shapes) {
    rnn_seq::gru_shape_infer(op, input_shapes, output_shapes);
    // TODO: Add attention input validation
}
}  // namespace internal
}  // namespace op
}  // namespace ov
