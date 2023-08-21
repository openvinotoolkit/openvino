// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "prior_box_shape_inference_util.hpp"

namespace ov {
namespace op {
namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const PriorBox* const op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    return prior_box::shape_infer(op, input_shapes, ta);
}
}  // namespace v0

namespace v8 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const PriorBox* const op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    return prior_box::shape_infer(op, input_shapes, ta);
}
}  // namespace v8

}  // namespace op
}  // namespace ov
