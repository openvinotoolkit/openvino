// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace convolution {

constexpr size_t spatial_dim_offset = 2;  //!< Spatial dimension offset for data.

/**
 * @brief Provides convolution filter non spatial dimension count.
 *
 * @note If Specific convolution operator requires different value provide specialization for this operator.
 * @tparam TConv  Type of convolution operator.
 * @return Default value for convolution operators (2).
 */
template <class TConv>
constexpr size_t filter_non_spatial_dims_count() {
    return 2;
}

/**
 * @brief Get num of spatial form convolution operator.
 *
 * Tries get value from operator member if is not deduced (has -1 value) then tries evaluate it from input shapes.
 *
 * @tparam TConv        Convolution type (this function must be a friend of TConv to access private member).
 * @tparam TShape       Shape type.
 * @param op            Pointer to convolution operator.
 * @param input_shapes  Input shapes (must have two) to spatial dim number evaluation.
 * @return Value of spatial dimension number or infinite bound (-1) if cannot evaluate.
 */
template <class TConv, class TShape>
int64_t get_num_spatial(const TConv* op, const std::vector<TShape>& input_shapes) {
    const auto& data_rank = input_shapes[0].rank();
    const auto& filters_rank = input_shapes[1].rank();

    auto result = op->m_num_spatial;

    if (result == ov::util::dim::inf_bound) {
        if (data_rank.is_static()) {
            result = data_rank.get_length() - spatial_dim_offset;
        } else if (filters_rank.is_static()) {
            result = filters_rank.get_length() - filter_non_spatial_dims_count<TConv>();
        }
    }

    return result;
}

/**
 * @brief Checks if Op property auto_pad is set to same lower or upper.
 *
 * @tparam TOp  Type of operator (must have get_auto_pad member function).
 * @param op    Pointer to operator.
 * @return True if auto pad enabled.
 */
template <class TOp>
bool is_auto_pad(const TOp* op) {
    return (op->get_auto_pad() == PadType::SAME_LOWER) || (op->get_auto_pad() == PadType::SAME_UPPER);
}

template <class T, class TOp, class TShape>
void apply_auto_pad(const TOp* op, const std::vector<TShape>& input_shapes, T pads_begin, T pads_end) {
    const auto num_spatial = get_num_spatial(op, input_shapes);
    auto data_dim = input_shapes[0].cend() - num_spatial;
    auto kernel_dim = input_shapes[1].cend() - num_spatial;
    const auto& dilations = op->get_dilations();
    const auto& strides = op->get_strides();

    const auto same_upper_padding = op->get_auto_pad() == PadType::SAME_UPPER;
    auto& pad_left = same_upper_padding ? pads_begin : pads_end;
    auto& pad_right = same_upper_padding ? pads_end : pads_begin;

    for (int64_t i = 0; i < num_spatial; ++i, ++pad_left, ++pad_right, ++data_dim, ++kernel_dim) {
        using namespace ov::util;
        if (kernel_dim->is_static()) {
            std::tie(*pad_left, *pad_right) =
                dim::padding(*data_dim, kernel_dim->get_length(), dilations[i], strides[i]);
        } else {
            *pad_left = 0;
            *pad_right = 0;
        }
    }
}

/**
 * @brief Append convolution spatial dimension at end of output shape.
 *
 * @tparam TOp          Convolution operator type.
 * @tparam TShape       Type of shape.
 * @param op            Pointer to operator.
 * @param input_shapes  Input shape of convolution shape inference.
 * @param out_shape     Output shape to append spatial dimensions.
 */
template <class TOp, class TShape>
void append_spatial_shape(const TOp* op, const std::vector<TShape>& input_shapes, TShape& out_shape) {
    using namespace ov::util;
    using TDim = typename TShape::value_type;

    const auto& strides = op->get_strides();
    const auto spatial_num = strides.size();

    const auto& data_shape = input_shapes[0].rank().is_static() ? input_shapes[0] : PartialShape::dynamic(spatial_num);
    auto data_dim = data_shape.cend() - spatial_num;

    if (is_auto_pad(op)) {
        std::transform(data_dim,
                       data_shape.cend(),
                       strides.cbegin(),
                       std::back_inserter(out_shape),
                       &dim::ceil_div<TDim>);
    } else {
        const auto& filters_shape =
            input_shapes[1].rank().is_static() ? input_shapes[1] : PartialShape::dynamic(spatial_num);
        auto filters_dim = filters_shape.cend() - spatial_num;
        const auto& pads_begin = op->get_pads_begin();
        const auto& pads_end = op->get_pads_end();
        const auto& dilations = op->get_dilations();

        for (size_t i = 0; i < spatial_num; ++i, ++data_dim, ++filters_dim) {
            auto dim = *data_dim + (pads_begin[i] + pads_end[i]);
            const auto filter_dilated = dim::dilated(*filters_dim, dilations[i]);

            if (dim.is_static() && filter_dilated.is_static()) {
                // Use check from pooling op as it is same.
                pooling::valid_dilated_kernel_with_dim(op, filter_dilated.get_length(), dim, i);
            }

            dim = dim::floor_div(dim - filter_dilated, strides[i]);
            dim += 1;
            out_shape.push_back(std::move(dim));
        }
    }
}
}  // namespace convolution
}  // namespace op
}  // namespace ov
