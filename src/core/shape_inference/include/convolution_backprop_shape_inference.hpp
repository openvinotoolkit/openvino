// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolution_shape_inference.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {
template <class TShape>
std::vector<TShape> shape_infer(const ConvolutionBackpropData* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    const auto inputs_count = input_shapes.size();
    const auto has_spatial_shape = inputs_count == 3;
    NODE_VALIDATION_CHECK(op, inputs_count == 2 || has_spatial_shape);
    using namespace ov::util;

    int64_t num_spatial;
    TShape out_spatial_shape;

    if (has_spatial_shape) {
        const auto& spatial_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              spatial_shape.rank().compatible(1),
                              "Input delivering output shape must have rank 1.");

        if (!get_data_as_shape(2, op, out_spatial_shape, constant_data)) {
            if (spatial_shape.is_static()) {
                out_spatial_shape.resize(spatial_shape[0].get_length());
            } else {
                out_spatial_shape = PartialShape::dynamic();
            }
        }
        num_spatial = convolution::get_num_spatial(op, input_shapes, out_spatial_shape);
    } else {
        num_spatial = convolution::get_num_spatial(op, input_shapes);
        if (num_spatial > 0) {
            out_spatial_shape.resize(num_spatial);
        }
    }

    TShape output_shape;
    if (num_spatial != dim::inf_bound) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        const auto data_rank = data_shape.rank();
        const auto filters_rank = filters_shape.rank();

        NODE_VALIDATION_CHECK(op,
                              data_rank.compatible(filters_rank),
                              "Data and filters rank do not match (data batch shape: ",
                              data_shape,
                              ", filters shape: ",
                              filters_shape,
                              ").");

        NODE_VALIDATION_CHECK(
            op,
            data_rank.is_dynamic() || filters_rank.is_dynamic() || data_shape[1].compatible(filters_shape[0]),
            "Data batch channel count (",
            data_shape[1],
            ") does not match filter input channel count (",
            filters_shape[0],
            ").");

        if (out_spatial_shape.rank().is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  static_cast<int64_t>(out_spatial_shape.size()) == num_spatial,
                                  "Output shape should be defined for all and only spatial dimensions.");
        } else {
            out_spatial_shape.resize(num_spatial);
        }

        update_and_validate_attributes(const_cast<ConvolutionBackpropData*>(op), input_shapes, out_spatial_shape);

        output_shape.reserve(convolution::spatial_dim_offset + num_spatial);
        output_shape.emplace_back(data_rank.is_static() ? data_shape[0] : dim::inf_bound);
        output_shape.emplace_back(filters_rank.is_static() ? filters_shape[1] : dim::inf_bound);

        if (has_spatial_shape) {
            output_shape.insert(output_shape.end(),
                                std::make_move_iterator(out_spatial_shape.begin()),
                                std::make_move_iterator(out_spatial_shape.end()));
        } else {
            convolution::backprop::append_spatial_shape(op, input_shapes, output_shape);
        }
    } else {
        output_shape = PartialShape::dynamic();
    }

    return {output_shape};
}
}  // namespace v1
}  // namespace op
}  // namespace ov
