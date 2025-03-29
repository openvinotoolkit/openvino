// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/op/roi_pooling.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace roi_pooling {
namespace validate {
template <class TROIPooling, class TShape>
void feat_intput_shape(const TROIPooling* op, const TShape& feat_shape) {
    NODE_VALIDATION_CHECK(op,
                          feat_shape.rank().compatible(4),
                          "Expected a 4D tensor for the feature maps input. Got: ",
                          feat_shape);
}

template <class TROIPooling, class TShape>
void rois_input_shape(const TROIPooling* op, const TShape rois_shape) {
    if (rois_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(op,
                              rois_shape.size() == 2,
                              "Expected a 2D tensor for the ROIs input with box coordinates. Got: ",
                              rois_shape);

        NODE_VALIDATION_CHECK(op,
                              rois_shape[1].compatible(5),
                              "The second dimension of ROIs input should contain batch id and box coordinates. ",
                              "This dimension is expected to be equal to 5. Got: ",
                              rois_shape[1]);
    }
}

template <class TROIPooling>
void output_roi_attr(const TROIPooling* op) {
    const auto& out_roi = op->get_output_roi();

    NODE_VALIDATION_CHECK(op,
                          out_roi.size() == 2,
                          "The dimension of pooled size is expected to be equal to 2. Got: ",
                          out_roi.size());

    NODE_VALIDATION_CHECK(op,
                          std::none_of(out_roi.cbegin(), out_roi.cend(), cmp::Less<size_t>(1)),
                          "Pooled size attributes pooled_h and pooled_w should should be positive integers. Got: ",
                          out_roi[0],
                          " and: ",
                          out_roi[1],
                          "respectively");
}

template <class TROIPooling>
void scale_attr(const TROIPooling* op) {
    const auto scale = op->get_spatial_scale();
    NODE_VALIDATION_CHECK(op,
                          std::isnormal(scale) && !std::signbit(scale),
                          "The spatial scale attribute should be a positive floating point number. Got: ",
                          scale);
}

template <class TROIPooling>
void method_attr(const TROIPooling* op) {
    const auto& method = op->get_method();
    NODE_VALIDATION_CHECK(op,
                          method == "max" || method == "bilinear",
                          "Pooling method attribute should be either \'max\' or \'bilinear\'. Got: ",
                          method);
}
}  // namespace validate
}  // namespace roi_pooling

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ROIPooling* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using namespace ov::util;

    const auto& feat_shape = input_shapes[0];
    const auto& rois_shape = input_shapes[1];
    const auto& feat_rank = feat_shape.rank();

    roi_pooling::validate::feat_intput_shape(op, feat_shape);
    roi_pooling::validate::rois_input_shape(op, rois_shape);
    roi_pooling::validate::output_roi_attr(op);
    roi_pooling::validate::scale_attr(op);
    roi_pooling::validate::method_attr(op);

    auto output_shapes = std::vector<TRShape>(1);
    auto& out_shape = output_shapes.front();
    out_shape.reserve(4);

    out_shape.emplace_back(rois_shape.rank().is_static() ? rois_shape[0] : dim::inf_bound);
    out_shape.emplace_back(feat_rank.is_static() ? feat_shape[1] : dim::inf_bound);
    std::copy(op->get_output_roi().cbegin(), op->get_output_roi().cend(), std::back_inserter(out_shape));

    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
