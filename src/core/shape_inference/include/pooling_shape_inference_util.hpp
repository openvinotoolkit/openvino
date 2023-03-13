// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace pooling {
constexpr size_t spatial_dim_offset = 2;

/**
 * @brief Calculate dimension padding required by filter/kernel properties.
 *
 * Provides pair of padding values as left padding is total value of required padding divided by 2 and right as
 * total required padding minus left padding.
 *
 * @param dim          input dimension to calculate its padding.
 * @param filter_size  Kernel size for input dimension.
 * @param dilation     Kernel dilation.
 * @param stride       Kernel stride.
 * @return Pair of left, right padding values for input dimension.
 */
template <class TDim, class T = typename TDim::value_type>
inline std::pair<T, T> dim_padding(const TDim& dim, const int64_t kernel_size, const int64_t dilation, int64_t stride) {
    if (dim.is_static()) {
        const auto dim_size = static_cast<int64_t>(dim.get_length());
        const auto dilated_kernel = ov::util::dim::dilated(kernel_size, dilation);
        const int64_t tmp = (dim_size + stride - 1) / stride;

        const auto padding = std::max<int64_t>(0, (tmp - 1) * stride + dilated_kernel - dim_size);
        const auto left_padding = padding / 2;
        return {left_padding, padding - left_padding};
    } else {
        // If input dimension is infinite or interval the padding will be set to 0
        // as operator cannot store paddings for both bounds.
        return {0, 0};
    }
}

template <class TOp, class TShape>
void update_and_validate_attributes(TOp* op, const TShape& data_shape, const Strides& dilations) {
    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    const auto& kernel = op->get_kernel();
    const auto& auto_pad = op->get_auto_pad();
    const auto num_spatial = kernel.size();
    const auto& strides = op->get_strides();

    if (auto_pad == PadType::VALID || op->get_pads_begin().empty()) {
        op->set_pads_begin(Shape(num_spatial, 0));
    }
    if (auto_pad == PadType::VALID || op->get_pads_end().empty()) {
        op->set_pads_end(Shape(num_spatial, 0));
    }

    NODE_VALIDATION_CHECK(op,
                          op->get_pads_begin().size() == num_spatial,
                          "Expected pads_begin size to be equal to input size - 2. Got: ",
                          op->get_pads_begin().size());
    NODE_VALIDATION_CHECK(op,
                          op->get_pads_end().size() == num_spatial,
                          "Expected pads_end size to be equal to input size - 2. Got: ",
                          op->get_pads_end().size());
    NODE_VALIDATION_CHECK(op,
                          strides.size() == num_spatial,
                          "Expected strides size to be equal to input size - 2. Got: ",
                          strides.size());
    NODE_VALIDATION_CHECK(op,
                          dilations.size() == num_spatial,
                          "Expected dilations size to be equal to kernel size. Got: ",
                          dilations.size());

    if (data_rank.is_static()) {
        NODE_VALIDATION_CHECK(op,
                              num_spatial == (data_shape.size() - spatial_dim_offset),
                              "Expected kernel size to be equal to input size - 2. Got: ",
                              num_spatial);

        if (auto_pad == PadType::SAME_UPPER || auto_pad == PadType::SAME_LOWER) {
            Shape pads_begin, pads_end;
            pads_begin.reserve(num_spatial);
            pads_end.reserve(num_spatial);

            auto data_dim = data_shape.cbegin() + spatial_dim_offset;
            auto pad_begin_ins = std::back_inserter(pads_begin);
            auto pad_end_ins = std::back_inserter(pads_end);
            auto& pad_left = auto_pad == PadType::SAME_UPPER ? pad_begin_ins : pad_end_ins;
            auto& pad_right = auto_pad == PadType::SAME_UPPER ? pad_end_ins : pad_begin_ins;

            for (size_t i = 0; i < num_spatial; ++i, ++pad_left, ++pad_right, ++data_dim) {
                std::tie(*pad_left, *pad_right) = dim_padding(*data_dim, kernel[i], dilations[i], strides[i]);
            }

            op->set_pads_begin(pads_begin);
            op->set_pads_end(std::move(pads_end));
        }
    }

    constexpr auto is_zero = cmp::Equal<size_t>(0);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(strides.cbegin(), strides.cend(), is_zero),
                          "Strides has zero dimension(s). ",
                          strides);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(dilations.cbegin(), dilations.cend(), is_zero),
                          "Kernel dilations has zero dimension(s). ",
                          dilations);
}

template <class TOp, class TDim>
void valid_dilated_kernel_with_dim(const TOp* op, const size_t kernel, const TDim& dim, const size_t axis) {
    NODE_VALIDATION_CHECK(op,
                          kernel > 0,
                          "Kernel after dilation has dimension less than 1 (dim: ",
                          kernel,
                          ") at axis ",
                          axis,
                          ".");

    NODE_VALIDATION_CHECK(op,
                          cmp::le(kernel, dim.get_length()),
                          "Kernel after dilation has size (dim: ",
                          kernel,
                          ") larger than the data shape after padding (dim: ",
                          dim,
                          ") at axis ",
                          axis,
                          ".");
}

template <class TOp>
void valid_dilated_kernel_with_padding(const TOp* op,
                                       const size_t kernel,
                                       const size_t pad_begin,
                                       const size_t pad_end,
                                       const size_t axis) {}

template <class TOp, class TShape>
TShape spatial_shape_infer(const TOp* op, const TShape& data_shape, const Strides& dilations) {
    using namespace ov::util;
    const auto spatial_num = data_shape.size() - spatial_dim_offset;
    const auto is_ceil_mode = op->get_rounding_type() == RoundingType::CEIL;
    const auto is_auto_pad = (op->get_auto_pad() == PadType::SAME_UPPER) || (op->get_auto_pad() == PadType::SAME_LOWER);

    using TDim = typename TShape::value_type;
    const auto& dim_divide = is_ceil_mode ? dim::ceil_div<TDim> : dim::floor_div<TDim>;

    TShape out_shape;
    out_shape.reserve(spatial_num);

    auto data_dim = data_shape.cbegin() + spatial_dim_offset;
    const auto& pads_begin = op->get_pads_begin();
    const auto& pads_end = op->get_pads_end();
    const auto& kernel = op->get_kernel();
    const auto& stride = op->get_strides();

    for (size_t i = 0; i < spatial_num; ++i, ++data_dim) {
        if (data_dim->is_static() || !is_auto_pad) {
            auto dim = *data_dim + (pads_begin[i] + pads_end[i]);
            const auto kernel_dilated = dim::dilated(kernel[i], dilations[i]);

            if (data_dim->is_static()) {
                valid_dilated_kernel_with_dim(op, kernel_dilated, dim, i);
                valid_dilated_kernel_with_padding(op, kernel_dilated, pads_begin[i], pads_end[i], i);
            }

            dim = dim - kernel_dilated;
            dim = dim_divide(dim, stride[i]);
            dim += 1;
            out_shape.push_back(std::move(dim));
        } else {
            // If dimension is interval and is auto pad then result is dynamic shape as padding values are not correct.
            // Operator cannot keep separate auto padding values for upper, lower bounds.
            out_shape.emplace_back(dim::inf_bound);
        }
    }

    return out_shape;
}

/**
 * @brief Shape inference helper used for pooling operators such Max Pool, Avg Pool.
 */
template <class TOp, class TShape>
TShape out_shape_infer(const TOp* op, const TShape& data_shape, const Strides& dilations) {
    TShape out_shape;
    if (data_shape.rank().is_static()) {
        const auto& batch_size = data_shape[0];
        const auto& channel_count = data_shape[1];

        NODE_VALIDATION_CHECK(op, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");
        NODE_VALIDATION_CHECK(op,
                              channel_count.is_dynamic() || channel_count.get_length() > 0,
                              "Channel count is zero.");

        out_shape = spatial_shape_infer(op, data_shape, dilations);
        out_shape.insert(out_shape.begin(), data_shape.begin(), data_shape.begin() + spatial_dim_offset);
    } else {
        out_shape.insert(out_shape.begin(), spatial_dim_offset + op->get_kernel().size(), Dimension::dynamic());
    }

    return out_shape;
}

/**
 * @brief Shape inference helper used for adaptive pooling operators.
 */
template <class TShape,
          class TOp,
          typename std::enable_if<std::is_same<TOp, v8::AdaptiveAvgPool>::value ||
                                  std::is_same<TOp, v8::AdaptiveMaxPool>::value>::type* = nullptr>
TShape out_shape_infer(const TOp* op,
                       const std::vector<TShape>& input_shapes,
                       const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    const auto& out_spatial_shape = input_shapes[1];

    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    TShape output_shape;
    if (data_rank.is_static()) {
        auto num_of_spatial_dims = data_shape.size() - spatial_dim_offset;

        NODE_VALIDATION_CHECK(
            op,
            out_spatial_shape.rank().is_dynamic() || out_spatial_shape[0].compatible(num_of_spatial_dims),
            "Output shape for spatial dimension not compatible with data shape.");

        output_shape.reserve(data_shape.size());
        std::copy_n(data_shape.begin(), spatial_dim_offset, std::back_inserter(output_shape));

        if (const auto spatial_dims = get_input_const_data_as_shape<TShape>(op, 1, constant_data)) {
            NODE_VALIDATION_CHECK(op,
                                  num_of_spatial_dims == spatial_dims->size(),
                                  "Number of spatial dimensions is not compatible with input data rank");

            output_shape.insert(output_shape.end(), spatial_dims->begin(), spatial_dims->end());
        } else {
            output_shape.insert(output_shape.end(), num_of_spatial_dims, ov::util::dim::inf_bound);
        }
    } else {
        output_shape = PartialShape::dynamic();
    }
    return output_shape;
}
}  // namespace pooling
}  // namespace op
}  // namespace ov
