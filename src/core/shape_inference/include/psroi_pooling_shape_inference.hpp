// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>

#include "compare.hpp"
#include "dimension_util.hpp"
#include "openvino/op/roi_pooling.hpp"
#include "roi_pooling_shape_inference.hpp"

namespace ov {
namespace op {
namespace psroi_pooling {
namespace validate {
template <class TROIPooling, class TShape>
void feat_input_shape(const TROIPooling* op, const TShape feat_shape) {
    using namespace ov::util;

    roi_pooling::validate::feat_intput_shape(op, feat_shape);

    if (feat_shape.rank().is_static()) {
        const auto& mode = op->get_mode();
        const auto& num_channels = feat_shape[1];
        if (mode == "average") {
            const auto group_area = op->get_group_size() * op->get_group_size();
            NODE_VALIDATION_CHECK(
                op,
                num_channels.compatible(group_area * op->get_output_dim()),
                "Number of input's channels must be a multiply of output_dim * group_size * group_size");
        } else if (mode == "bilinear") {
            const auto bins_area = op->get_spatial_bins_x() * op->get_spatial_bins_y();
            NODE_VALIDATION_CHECK(
                op,
                num_channels.compatible(bins_area * op->get_output_dim()),
                "Number of input's channels must be a multiply of output_dim * spatial_bins_x * spatial_bins_y");
        }
    }
}

template <class TROIPooling>
void output_group_attr(const TROIPooling* op) {
    NODE_VALIDATION_CHECK(op, op->get_group_size() > 0, "group_size has to be greater than 0");
}

template <class TROIPooling>
void bins_attr(const TROIPooling* op) {
    if (op->get_mode() == "bilinear") {
        NODE_VALIDATION_CHECK(op, op->get_spatial_bins_x() > 0, "spatial_bins_x has to be greater than 0");
        NODE_VALIDATION_CHECK(op, op->get_spatial_bins_y() > 0, "spatial_bins_y has to be greater than 0");
    }
}

template <class TROIPooling>
void mode_attr(const TROIPooling* op) {
    const auto& mode = op->get_mode();
    NODE_VALIDATION_CHECK(op,
                          mode == "average" || mode == "bilinear",
                          "Expected 'average' or 'bilinear' mode. Got " + mode);
}
}  // namespace validate
}  // namespace psroi_pooling

namespace v0 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const PSROIPooling* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using namespace ov::util;

    const auto& feat_shape = input_shapes[0];
    const auto& rois_shape = input_shapes[1];

    psroi_pooling::validate::mode_attr(op);
    psroi_pooling::validate::output_group_attr(op);
    psroi_pooling::validate::bins_attr(op);
    roi_pooling::validate::scale_attr(op);

    psroi_pooling::validate::feat_input_shape(op, feat_shape);
    roi_pooling::validate::rois_input_shape(op, rois_shape);

    auto output_shapes = std::vector<TRShape>(1);
    auto& out_shape = output_shapes.front();
    out_shape.reserve(4);

    out_shape.emplace_back(rois_shape.rank().is_static() ? rois_shape[0] : dim::inf_bound);
    out_shape.emplace_back(op->get_output_dim());
    out_shape.insert(out_shape.end(), 2, op->get_group_size());

    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
