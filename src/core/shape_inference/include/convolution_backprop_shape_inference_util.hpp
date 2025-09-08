// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_shape_inference_util.hpp"
#include "openvino/op/util/convolution_backprop_base.hpp"

namespace ov {
namespace op {
namespace convolution {
namespace validate {
template <class TShape>
void filter_shape(const ov::op::util::ConvolutionBackPropBase* op,
                  const TShape& filters_shape,
                  const TShape& data_shape) {
    const auto& data_rank = data_shape.rank();
    const auto& filters_rank = filters_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          data_rank.compatible(filters_rank),
                          "Data batch and filters rank do not match (data batch shape: ",
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
}
}  // namespace validate

template <class TOp,
          class TShape,
          typename std::enable_if<std::is_base_of<util::ConvolutionBackPropBase, TOp>::value>::type* = nullptr>
size_t calculate_num_spatial(const TOp* op,
                             const std::vector<TShape>& input_shapes,
                             const result_shape_t<TShape>& out_spatial_shape) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() > 1);

    auto num_spatial = util::get_num_spatial(op);
    if (num_spatial == num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        num_spatial = util::num_spatial_from_shapes(data_shape, filters_shape, filter_non_spatial_dims_count<TOp>());
    }

    if (num_spatial == num_spatial_undefined && out_spatial_shape.rank().is_static() && out_spatial_shape.size() > 0) {
        num_spatial = out_spatial_shape.size();
    }

    if (num_spatial == num_spatial_undefined) {
        num_spatial = num_spatial_from_attr(op);
    }
    return num_spatial;
}

/**
 * @brief Apply auto padding for backward convolution.
 *
 * The auto padding can be applied only if inputs and attributes of operator are validated.
 * The input shapes must have got static ranks.
 *
 * @param op                 Pointer to convolution operator.
 * @param data_shape         Input data shape (must be static rank).
 * @param filters_shape      Input filter shape (must be static rank).
 * @param out_spatial_shape  Reference to input with out spatial shape.
 * @param pads_begin         Iterator to begin of pads begin.
 * @param pads_end           Iterator to begin of pads end.
 */
template <class TOp, class TShape, class TIter>
void apply_auto_pad(const TOp* op,
                    const TShape& data_shape,
                    const TShape& filters_shape,
                    const result_shape_t<TShape>& out_spatial_shape,
                    TIter pads_begin,
                    TIter pads_end) {
    const auto& strides = op->get_strides();
    const auto& dilations = op->get_dilations();
    const auto& out_padding = op->get_output_padding();

    const auto num_spatial = strides.size();
    auto data_dim = data_shape.cend() - num_spatial;
    auto filter_dim = filters_shape.cend() - num_spatial;

    const auto padding_swap = op->get_auto_pad() == PadType::SAME_UPPER;
    auto& pad_b = padding_swap ? pads_end : pads_begin;
    auto& pad_e = padding_swap ? pads_begin : pads_end;

    for (size_t i = 0; i < num_spatial; ++i, ++pad_b, ++pad_e, ++data_dim, ++filter_dim) {
        using namespace ov::util;
        if (dim::is_static(*data_dim) && dim::is_static(*filter_dim) && out_spatial_shape[i].is_static()) {
            const auto dilated_filter = dim::dilated(*filter_dim, dilations[i]);
            const auto dim_len = static_cast<int64_t>(dim::get_length(*data_dim) - 1);
            const auto padding = std::max<int64_t>(dim_len * strides[i] + dim::get_length(dilated_filter) -
                                                       out_spatial_shape[i].get_length() + out_padding[i],
                                                   0);

            *pad_b = padding / 2;
            *pad_e = padding - *pad_b;
        } else {
            *pad_b = 0;
            *pad_e = 0;
        }
    }
}

/**
 * @brief  Apply auto padding for back propagation convolutions.
 *
 * @tparam TShape            Shape type.
 * @param op                 Pointer to back propagation convolution operator.
 * @param data_shape         Input data shape.
 * @param filters_shape      Input filter shape.
 * @param out_spatial_shape  Input output spatial shape.
 */
template <class TShape>
void apply_padding(const util::ConvolutionBackPropBase* op,
                   const std::vector<TShape>& input_shapes,
                   const result_shape_t<TShape>& out_spatial_shape,
                   CoordinateDiff& pads_begin,
                   CoordinateDiff& pads_end) {
    const auto& data_shape = input_shapes[0];
    const auto& filters_shape = input_shapes[1];

    // apply padding if required
    if (input_shapes.size() >= 3 && convolution::is_auto_pad(op) && data_shape.rank().is_static() &&
        filters_shape.rank().is_static()) {
        convolution::apply_auto_pad(op,
                                    data_shape,
                                    filters_shape,
                                    out_spatial_shape,
                                    pads_begin.begin(),
                                    pads_end.begin());
    } else if (convolution::is_auto_pad(op) || op->get_auto_pad() == op::PadType::VALID) {
        std::fill(pads_begin.begin(), pads_begin.end(), 0);
        std::fill(pads_end.begin(), pads_end.end(), 0);
    } else if (op->get_auto_pad() == op::PadType::EXPLICIT) {
        std::copy(op->get_pads_begin().begin(), op->get_pads_begin().end(), pads_begin.begin());
        std::copy(op->get_pads_end().begin(), op->get_pads_end().end(), pads_end.begin());
    }
}

/**
 * @brief Append spatial dimension at end of output shape of back propagation convolution.
 *
 * @tparam TOp           Back propagation convolution operator type.
 * @tparam TShape        Type of shape.
 * @param op             Pointer to operator.
 * @param data_shape     Input data shape.
 * @param filters_shape  Input filter shape.
 * @param out_shape      Output shape to append spatial dimensions.
 */
template <class TOp,
          class TShape,
          class TContainer,
          typename std::enable_if<std::is_base_of<ov::op::util::ConvolutionBackPropBase, TOp>::value>::type* = nullptr>
void append_spatial_shape(const TOp* op,
                          const TShape& data_shape,
                          const TShape& filters_shape,
                          const TContainer& pads_begin,
                          const TContainer& pads_end,
                          result_shape_t<TShape>& out_shape) {
    using namespace ov::util;

    const auto& strides = op->get_strides();
    const auto& dilations = op->get_dilations();
    const auto& output_padding = op->get_output_padding();

    const auto spatial_num = strides.size();

    const auto& d_shape = data_shape.rank().is_static() ? data_shape : PartialShape::dynamic(spatial_num);
    auto data_dim = d_shape.cend() - spatial_num;

    const auto& f_shape = filters_shape.rank().is_static() ? filters_shape : PartialShape::dynamic(spatial_num);
    auto filters_dim = f_shape.cend() - spatial_num;

    for (size_t i = 0; i < spatial_num; ++i, ++data_dim, ++filters_dim) {
        auto dim = (*data_dim - 1) * strides[i];
        dim += dim::dilated(*filters_dim, dilations[i]);
        out_shape.push_back(dim::padded(dim, output_padding[i] - pads_begin[i] - pads_end[i]));
    }
}
}  // namespace convolution
}  // namespace op
}  // namespace ov
