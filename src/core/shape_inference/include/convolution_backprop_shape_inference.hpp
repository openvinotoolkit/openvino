// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolution_backprop_shape_inference_util.hpp"
#include "convolution_shape_inference_util.hpp"
#include "openvino/op/convolution.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const ConvolutionBackpropData* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    const auto inputs_count = input_shapes.size();
    const auto has_spatial_shape = inputs_count >= 3;
    NODE_VALIDATION_CHECK(op, inputs_count >= 2);
    using namespace ov::util;

    ov::optional<TRShape> out_spatial_shape;
    if (has_spatial_shape) {
        const auto& spatial_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              spatial_shape.rank().compatible(1),
                              "Input delivering output shape must have rank 1.");
        out_spatial_shape = get_input_const_data_as_shape<TRShape>(op, 2, ta);
        if (!out_spatial_shape) {
            if (spatial_shape.is_static()) {
                out_spatial_shape.emplace();
                out_spatial_shape->resize(spatial_shape[0].get_length());
            } else {
                out_spatial_shape = PartialShape::dynamic();
            }
        }
    } else {
        out_spatial_shape.emplace();
    }

    const auto num_spatial = convolution::calculate_num_spatial(op, input_shapes, *out_spatial_shape);

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    if (num_spatial != convolution::num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];

        NODE_VALIDATION_CHECK(
            op,
            !has_spatial_shape || out_spatial_shape->rank().is_dynamic() || out_spatial_shape->size() == num_spatial,
            "Output shape should be defined for all and only spatial dimensions.");

        convolution::resize_empty_padding(num_spatial, pads_begin, pads_end);
        convolution::validate::filter_shape(op, filters_shape, data_shape);
        if (is_attr_validation_required(op)) {
            convolution::validate::data_shape(op, data_shape);
            convolution::validate::common_attributes(op, num_spatial, pads_begin, pads_end);
        }
        convolution::apply_padding(op, input_shapes, *out_spatial_shape, pads_begin, pads_end);

        output_shape.reserve(util::spatial_dim_offset + num_spatial);
        output_shape.emplace_back(data_shape.rank().is_static() ? data_shape[0] : dim::inf_bound);
        output_shape.emplace_back(filters_shape.rank().is_static() ? filters_shape[1] : dim::inf_bound);

        if (has_spatial_shape) {
            output_shape.insert(output_shape.end(),
                                std::make_move_iterator(out_spatial_shape->begin()),
                                std::make_move_iterator(out_spatial_shape->end()));
        } else {
            convolution::append_spatial_shape(op, data_shape, filters_shape, pads_begin, pads_end, output_shape);
        }
    } else {
        output_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
