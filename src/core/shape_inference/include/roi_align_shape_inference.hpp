// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/roi_align.hpp"
#include "roi_align_shape_utils.hpp"

namespace ov {
namespace op {

namespace v3 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ROIAlign* op, const std::vector<TShape>& input_shapes) {
    return roi_align::shape_infer<TShape, TRShape>(op, input_shapes);
}
}  // namespace v3

namespace v9 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ROIAlign* op, const std::vector<TShape>& input_shapes) {
    return roi_align::shape_infer<TShape, TRShape>(op, input_shapes);
}
}  // namespace v9
}  // namespace op
}  // namespace ov
