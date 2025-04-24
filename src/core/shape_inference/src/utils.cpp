// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "eltwise_shape_inference.hpp"

namespace ov {
namespace op {

ov::Shape infer_broadcast_shape(const ov::Node* const op, const ov::Shape& first, const ov::Shape& second) {
    return eltwise_shape_infer(op, std::vector<ov::PartialShape>{first, second}).front().to_shape();
}

ov::Shape infer_broadcast_shape(const ov::Node* const op, const ov::TensorVector& inputs) {
    return eltwise_shape_infer(op, ov::util::get_tensors_partial_shapes(inputs)).front().to_shape();
}
}  // namespace op
}  // namespace ov
