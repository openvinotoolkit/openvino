// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "roi_align_shape_utils.hpp"

namespace ov {
namespace op {
namespace v15 {
class ROIAlignRotated;
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ROIAlignRotated* op, const std::vector<TShape>& input_shapes) {
    return roi_align::shape_infer<TShape, TRShape>(op, input_shapes);
}
}  // namespace v15
}  // namespace op
}  // namespace ov
