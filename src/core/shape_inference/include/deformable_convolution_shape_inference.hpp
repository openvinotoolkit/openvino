// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>

#include "convolution_shape_inference_util.hpp"
#include "openvino/op/util/deformable_convolution_base.hpp"

namespace ov {
namespace op {
namespace deformable_conv {
template <class TShape>
size_t calculate_num_spatial(const util::DeformableConvolutionBase* op, const std::vector<TShape>& input_shapes) {
    constexpr auto non_spatial_count = convolution::filter_non_spatial_dims_count<util::DeformableConvolutionBase>();

    auto num_spatial = util::num_spatial_from_shapes(input_shapes[0], input_shapes[2], non_spatial_count);

    if (num_spatial == convolution::num_spatial_undefined && input_shapes[1].rank().is_static()) {
        constexpr size_t offsets_shape_rank = 4;
        num_spatial = offsets_shape_rank - non_spatial_count;
    }

    return num_spatial;
}

namespace validate {
template <class TDeformableConv, class TShape>
void input_shape(const TDeformableConv* op, const TShape& shape, const std::string& name) {
    const auto& shape_rank = shape.rank();
    NODE_VALIDATION_CHECK(op, shape_rank.compatible(4), name, " must be of rank 4. Got: ", shape_rank);
}

template <class TDeformableConv>
void group_attribute(const TDeformableConv* op, int64_t group, const std::string& name) {
    NODE_VALIDATION_CHECK(op, group > 0, "Attribute '", name, "' must be any value starting from 1. Got: ", group);
}

template <class TDeformableConv, class TDim>
void group_divisible_dimension(const TDeformableConv* op, const TDim& dim, const std::string name) {
    const auto group = op->get_group();
    NODE_VALIDATION_CHECK(op,
                          ov::util::dim::is_divisible(dim, group),
                          name,
                          " channels dimension (",
                          dim,
                          ") must be evenly divisible by the 'group': ",
                          group);
}

template <class TDeformableConv, class TDim>
void deformable_group_divisible_dimension(const TDeformableConv* op, const TDim& dim, const std::string name) {
    const auto group = op->get_deformable_group();
    NODE_VALIDATION_CHECK(op,
                          ov::util::dim::is_divisible(dim, group),
                          name,
                          " channels dimension (",
                          dim,
                          ") must be evenly divisible by the 'deformable group': ",
                          group);
}
}  // namespace validate
}  // namespace deformable_conv

namespace util {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DeformableConvolutionBase* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end) {
    static constexpr std::array<const char*, 4> names{"Input", "Offsets", "Filters", "Mask"};
    using namespace ov::util;
    using TDim = typename TShape::value_type;

    const auto num_spatial = deformable_conv::calculate_num_spatial(op, input_shapes);

    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    if (num_spatial != convolution::num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& offsets_shape = input_shapes[1];
        const auto& filters_shape = input_shapes[2];

        const auto data_rank = data_shape.rank();
        const auto filters_rank = filters_shape.rank();
        const auto offsets_rank = offsets_shape.rank();

        output_shape.reserve(num_spatial + util::spatial_dim_offset);

        convolution::resize_empty_padding(num_spatial, pads_begin, pads_end);
        for (size_t i = 0; i < input_shapes.size(); ++i) {
            deformable_conv::validate::input_shape(op, input_shapes[i], names[i]);
        }
        deformable_conv::validate::group_attribute(op, op->get_group(), "group");
        deformable_conv::validate::group_attribute(op, op->get_deformable_group(), "deformable group");
        convolution::validate::common_attributes(op, num_spatial, pads_begin, pads_end);
        convolution::apply_padding(op, data_shape, filters_shape, pads_begin, pads_end);

        // add to output shape number of batches
        if (data_rank.is_static()) {
            deformable_conv::validate::group_divisible_dimension(op, data_shape[1], names[0]);

            output_shape.push_back(data_shape[0]);
        } else {
            output_shape.emplace_back(dim::inf_bound);
        }
        if (offsets_rank.is_static()) {
            if (filters_rank.is_static()) {
                auto offsets_channels = filters_shape[2] * filters_shape[3] * 2 * op->get_deformable_group();

                NODE_VALIDATION_CHECK(op,
                                      offsets_shape[1].compatible(offsets_channels),
                                      "The channels dimension of offsets input is not compatible with filters and "
                                      "'deformable group' attribute. Offsets input shape: ",
                                      offsets_shape,
                                      ", deformable 'group' attribute value: ",
                                      op->get_deformable_group(),
                                      ", filters shape: ",
                                      filters_shape);
            }
            deformable_conv::validate::deformable_group_divisible_dimension(op, offsets_shape[1], names[1]);

            NODE_VALIDATION_CHECK(op,
                                  TDim::merge(output_shape[0], offsets_shape[0], output_shape[0]),
                                  "Data batch and offsets batch dimension must be same value. Got: ",
                                  output_shape[0],
                                  " and ",
                                  data_shape[0]);
        }

        // add to output shape number output channels
        if (filters_rank.is_static()) {
            deformable_conv::validate::group_divisible_dimension(op, filters_shape[0], names[2]);

            NODE_VALIDATION_CHECK(
                op,
                data_rank.is_dynamic() || data_shape[1].compatible(filters_shape[1] * op->get_group()),
                "Data batch channel count (",
                data_shape[1],
                ") does not match filter input channel count (",
                filters_shape[1] * op->get_group(),
                ")");

            output_shape.push_back(filters_shape[0]);
        } else {
            output_shape.emplace_back(dim::inf_bound);
        }
        convolution::append_spatial_shape(op, data_shape, filters_shape, pads_begin, pads_end, output_shape);

        // post infer check.
        if (offsets_rank.is_static()) {
            auto offset_dim = offsets_shape.begin() + util::spatial_dim_offset;
            NODE_VALIDATION_CHECK(op,
                                  std::all_of(output_shape.begin() + util::spatial_dim_offset,
                                              output_shape.end(),
                                              [&offset_dim](const TDim& d) {
                                                  return d.compatible(*offset_dim++);
                                              }),
                                  "Spatial dimensions of offsets and output must be compatible.",
                                  output_shape);
        }
    } else {
        output_shape = PartialShape::dynamic();
    }

    return output_shapes;
}
}  // namespace util

namespace v1 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DeformableConvolution* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3);
    return util::shape_infer(op, input_shapes, pads_begin, pads_end);
}
}  // namespace v1

namespace v8 {
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const DeformableConvolution* op,
                                 const std::vector<TShape>& input_shapes,
                                 CoordinateDiff& pads_begin,
                                 CoordinateDiff& pads_end) {
    const auto has_mask_shape = input_shapes.size() == 4;
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 3 || has_mask_shape);
    using TDim = typename TShape::value_type;

    const auto& data_shape = input_shapes[0];
    const auto& offsets_shape = input_shapes[1];
    const auto& filters_shape = input_shapes[2];

    const auto data_rank = data_shape.rank();
    const auto filters_rank = filters_shape.rank();
    const auto offsets_rank = offsets_shape.rank();

    if (has_mask_shape) {
        const auto& mask_shape = input_shapes[3];
        if (mask_shape.rank().is_static()) {
            if (filters_rank.is_static()) {
                auto offsets_channels = filters_shape[2] * filters_shape[3] * op->get_deformable_group();

                NODE_VALIDATION_CHECK(op,
                                      mask_shape[1].compatible(offsets_channels),
                                      "The channels dimension of mask input is not "
                                      "compatible with filters and 'deformable group' attribute. "
                                      "Mask input shape: ",
                                      mask_shape,
                                      ", deformable 'group' attribute value: ",
                                      op->get_deformable_group(),
                                      ", filters shape: ",
                                      filters_shape);
            }

            deformable_conv::validate::deformable_group_divisible_dimension(op, mask_shape[1], "Mask");

            NODE_VALIDATION_CHECK(op,
                                  data_rank.is_dynamic() || mask_shape[0].compatible(data_shape[0]),
                                  "Data batch and mask batch dimension must be same value. Got: ",
                                  mask_shape[0],
                                  " and ",
                                  data_shape[0]);
        }
    }

    auto output_shapes = util::shape_infer(op, input_shapes, pads_begin, pads_end);
    // post infer checks
    if (has_mask_shape && input_shapes[3].rank().is_static() && output_shapes[0].rank().is_static()) {
        auto mask_dim = input_shapes[3].begin() + util::spatial_dim_offset;
        NODE_VALIDATION_CHECK(op,
                              std::all_of(output_shapes[0].begin() + util::spatial_dim_offset,
                                          output_shapes[0].end(),
                                          [&mask_dim](const TDim& d) {
                                              return d.compatible(*mask_dim++);
                                          }),
                              "Spatial dimensions of mask and output must be compatible.");
    }
    return output_shapes;
}
}  // namespace v8
}  // namespace op
}  // namespace ov
