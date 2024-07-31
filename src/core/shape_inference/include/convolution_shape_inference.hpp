// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolution_shape_inference_util.hpp"
#include "openvino/op/convolution.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
template <class TOp,
          class TShape,
          class TRShape = result_shape_t<TShape>,
          typename std::enable_if<std::is_same<TOp, Convolution>::value ||
                                  std::is_same<TOp, BinaryConvolution>::value>::type* = nullptr>
std::vector<TRShape> shape_infer(const TOp* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 2);
    using namespace ov::util;

    const auto num_spatial = convolution::calculate_num_spatial(op, input_shapes);

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    if (num_spatial != util::num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        const auto data_rank = data_shape.rank();
        const auto filters_rank = filters_shape.rank();

        convolution::resize_empty_padding(num_spatial, pads_begin, pads_end);
        convolution::validate::filter_shape(op, filters_shape, data_shape);
        if (is_attr_validation_required(op)) {
            convolution::validate::data_shape(op, data_shape);
            convolution::validate::common_attributes(op, num_spatial, pads_begin, pads_end);
        }
        convolution::apply_padding(op, data_shape, filters_shape, pads_begin, pads_end);

        output_shape.reserve(util::spatial_dim_offset + num_spatial);
        output_shape.emplace_back(data_rank.is_static() ? data_shape[0] : dim::inf_bound);
        output_shape.emplace_back(filters_rank.is_static() ? filters_shape[0] : dim::inf_bound);
        convolution::append_spatial_shape(op, data_shape, filters_shape, pads_begin, pads_end, output_shape);
    } else {
        output_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
