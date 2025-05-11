// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolution_shape_inference_util.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace convolution {

/**
 * @brief Defines non-spatial dimension for filters for group convolution operator.
 * @return Value of non-spatial filter dimensions (3).
 */
template <>
constexpr size_t filter_non_spatial_dims_count<v1::GroupConvolution>() {
    return 3;
}
}  // namespace convolution

namespace v1 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const GroupConvolution* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() >= 2);
    using namespace ov::util;

    const auto num_spatial = convolution::calculate_num_spatial(op, input_shapes);

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    if (num_spatial != convolution::num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        const auto data_rank = data_shape.rank();
        const auto filters_rank = filters_shape.rank();

        convolution::resize_empty_padding(num_spatial, pads_begin, pads_end);
        if (is_attr_validation_required(op)) {
            convolution::validate::data_shape(op, data_shape);

            NODE_VALIDATION_CHECK(op,
                                  data_rank.compatible(filters_rank - 1),
                                  "Data batch and filters rank do not match (data batch shape: ",
                                  data_shape,
                                  ", filters shape: ",
                                  filters_shape,
                                  ").");

            convolution::validate::common_attributes(op, num_spatial, pads_begin, pads_end);
        }
        convolution::apply_padding(op, data_shape, filters_shape, pads_begin, pads_end);

        output_shape.reserve(util::spatial_dim_offset + num_spatial);
        output_shape.emplace_back(data_rank.is_static() ? data_shape[0] : dim::inf_bound);

        if (filters_rank.is_static()) {
            auto groups = filters_shape[0];

            if (data_rank.is_static() && filters_shape[2].is_static()) {
                NODE_VALIDATION_CHECK(
                    op,
                    groups.merge(groups, groups, (data_shape[1] / filters_shape[2].get_length())),
                    "Input channels dimension of data batch is incompatible with filter groups or input channels.");
            }

            groups *= filters_shape[1];
            output_shape.push_back(std::move(groups));
        } else {
            output_shape.emplace_back(dim::inf_bound);
        }

        convolution::append_spatial_shape(op, data_shape, filters_shape, pads_begin, pads_end, output_shape);
    } else {
        output_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace v1
}  // namespace op
}  // namespace ov
