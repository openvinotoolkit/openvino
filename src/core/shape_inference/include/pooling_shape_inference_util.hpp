// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace pooling {
constexpr size_t spatial_dim_offset = 2;

namespace validate {
template <class TOp, class TContainer>
void padding(const TOp* op, const TContainer& pads_begin, const TContainer& pads_end) {
    const auto num_spatial = op->get_kernel().size();
    NODE_VALIDATION_CHECK(op,
                          pads_begin.size() == num_spatial,
                          "Expected pads_begin size to be equal to input size - 2. Got: ",
                          pads_begin.size());
    NODE_VALIDATION_CHECK(op,
                          pads_end.size() == num_spatial,
                          "Expected pads_end size to be equal to input size - 2. Got: ",
                          pads_end.size());
}

template <class TOp>
constexpr bool has_torch_ceil_mode() {
    return std::is_same<TOp, v14::AvgPool>::value || std::is_same<TOp, v14::MaxPool>::value;
}

template <class TOp, class TShape>
void attributes(const TOp* op, const TShape& data_shape, const Strides& dilations) {
    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          ov::util::is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    const auto& kernel = op->get_kernel();
    const auto num_spatial = kernel.size();
    const auto& strides = op->get_strides();

    NODE_VALIDATION_CHECK(op,
                          strides.size() == num_spatial,
                          "Expected strides size to be equal to input size - 2. Got: ",
                          strides.size());
    NODE_VALIDATION_CHECK(op,
                          dilations.size() == num_spatial,
                          "Expected dilations size to be equal to kernel size. Got: ",
                          dilations.size());

    NODE_VALIDATION_CHECK(op,
                          data_rank.is_dynamic() || num_spatial == (data_shape.size() - spatial_dim_offset),
                          "Expected kernel size to be equal to input size - 2. Got: ",
                          num_spatial);

    constexpr auto is_zero = cmp::Equal<size_t>(0);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(strides.cbegin(), strides.cend(), is_zero),
                          "Strides has zero dimension(s). ",
                          strides);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(dilations.cbegin(), dilations.cend(), is_zero),
                          "Kernel dilations has zero dimension(s). ",
                          dilations);
    if (!has_torch_ceil_mode<TOp>()) {
        const auto is_ceil_torch = op->get_rounding_type() == RoundingType::CEIL_TORCH;
        NODE_VALIDATION_CHECK(op, !is_ceil_torch, "Rounding CEIL_TORCH is not supported.");
    }
}
}  // namespace validate

/**
 * @brief Resize paddings if empty to number of spatial dimensions.
 *
 * @param num_spatial  Number of spatial dimensions.
 * @param pads_begin   Begin padding to resize.
 * @param pads_end     End padding to resize.
 */
template <class TContainer>
void resize_empty_padding(const size_t num_spatial, TContainer& pads_begin, TContainer& pads_end) {
    if (pads_begin.empty()) {
        pads_begin.resize(num_spatial);
    }

    if (pads_end.empty()) {
        pads_end.resize(num_spatial);
    }
}

/**
 * @brief Apply pooling operator padding depends on auto pad value.
 *
 * @param op          Pointer to Pooling operator to apply padding.
 * @param data_shape  Shape infer data input shape.
 * @param dilations   Kernel dilations.
 * @param pads_begin  Padding begin to update.
 * @param pads_end    Padding end to update.
 */
template <class TOp, class TShape, class TContainer>
void apply_padding(const TOp* op,
                   const TShape& data_shape,
                   const Strides& dilations,
                   TContainer& pads_begin,
                   TContainer& pads_end) {
    const auto& auto_pad = op->get_auto_pad();
    if (data_shape.rank().is_static() && (auto_pad == PadType::SAME_UPPER || auto_pad == PadType::SAME_LOWER)) {
        const auto& kernel = op->get_kernel();
        const auto& strides = op->get_strides();
        const auto num_spatial = kernel.size();
        pads_begin.reserve(num_spatial);
        pads_end.reserve(num_spatial);

        auto data_dim = &data_shape[spatial_dim_offset];
        auto pad_b = auto_pad == PadType::SAME_UPPER ? pads_begin.begin() : pads_end.begin();
        auto pad_e = auto_pad == PadType::SAME_UPPER ? pads_end.begin() : pads_begin.begin();

        for (size_t i = 0; i < num_spatial; ++i, ++pad_b, ++pad_e, ++data_dim) {
            using namespace ov::util;
            std::tie(*pad_b, *pad_e) = dim::padding(*data_dim, kernel[i], dilations[i], strides[i]);
        }
    } else if (auto_pad == PadType::VALID) {
        std::fill_n(pads_begin.begin(), pads_begin.size(), 0);
        std::fill_n(pads_end.begin(), pads_end.size(), 0);
    } else if (op->get_auto_pad() == op::PadType::EXPLICIT) {
        std::copy(op->get_pads_begin().begin(), op->get_pads_begin().end(), pads_begin.begin());
        std::copy(op->get_pads_end().begin(), op->get_pads_end().end(), pads_end.begin());
    }
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

template <class TDim>
void align_ceil_torch_dimension_size(TDim& dim,
                                     const size_t last_pooling_start_index,
                                     const size_t data_dim_length,
                                     const size_t pads_begin) {
    if (!(last_pooling_start_index > data_dim_length + pads_begin - 1) && !ov::util::dim::is_inf_bound(dim)) {
        dim += 1;
    }
}

template <class TDim>
TDim disallow_pooling_start_in_padding(const TDim& dim,
                                       const size_t stride,
                                       const TDim* data_dim,
                                       const size_t pads_begin) {
    // Ensure the last pooling doesn't start in padding.
    auto dim_min_length = dim.get_min_length();
    const auto last_pooling_min_start_index = dim_min_length * stride;
    const auto data_dim_min_length = data_dim->get_min_length();
    align_ceil_torch_dimension_size(dim_min_length, last_pooling_min_start_index, data_dim_min_length, pads_begin);
    if (data_dim->is_static()) {
        return TDim(dim_min_length);
    } else {
        Dimension::value_type dim_max_length;
        if (data_dim->get_interval().has_upper_bound()) {
            dim_max_length = dim.get_max_length();
            const auto last_pooling_max_start_index = dim_max_length * stride;
            const auto data_dim_max_length = data_dim->get_max_length();
            align_ceil_torch_dimension_size(dim_max_length,
                                            last_pooling_max_start_index,
                                            data_dim_max_length,
                                            pads_begin);
        } else {
            dim_max_length = -1;
        }
        return TDim(dim_min_length, dim_max_length);
    }
}

template <class TDim>
TDim allow_pooling_start_in_padding(const TDim& dim, const size_t, const TDim*, const size_t) {
    return dim + 1;
}

/**
 * @brief Append spatial shape to the end of output shape for pooling operator shape inference result.
 *
 * @param op          Pointer to pooling operator.
 * @param data_shape  Shape inference input pooling data shape.
 * @param pads_begin  Pooling pads begin.
 * @param pads_end    Pooling pads end.
 * @param dilations   Kernel dilations.
 * @param out_shape   Output shape for appending the spatial shape of pooling
 */
template <class TOp, class TShape, class TContainer, class TRShape>
void append_spatial_shape(const TOp* op,
                          const TShape& data_shape,
                          const TContainer& pads_begin,
                          const TContainer& pads_end,
                          const Strides& dilations,
                          TRShape& out_shape) {
    using namespace ov::util;
    const auto spatial_num = data_shape.size() - spatial_dim_offset;
    const auto is_ceil_torch_mode = op->get_rounding_type() == RoundingType::CEIL_TORCH;
    const auto is_ceil_mode = op->get_rounding_type() == RoundingType::CEIL || is_ceil_torch_mode;
    const auto is_auto_pad = (op->get_auto_pad() == PadType::SAME_UPPER) || (op->get_auto_pad() == PadType::SAME_LOWER);

    using TDim = typename TShape::value_type;
    const auto& dim_divide = is_ceil_mode ? dim::ceil_div<TDim> : dim::floor_div<TDim>;

    auto data_dim = &data_shape[spatial_dim_offset];
    const auto& kernel = op->get_kernel();
    const auto& stride = op->get_strides();

    // Torch CEIL rounding disallows the last pooling operation from starting in the pads area.
    auto set_pooling_ceil_behavior =
        is_ceil_torch_mode ? &disallow_pooling_start_in_padding<TDim> : &allow_pooling_start_in_padding<TDim>;

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
            out_shape.push_back(set_pooling_ceil_behavior(dim, stride[i], data_dim, pads_begin[i]));
        } else {
            // If dimension is interval and is auto pad then result is dynamic shape as padding values are not correct.
            // Operator cannot keep separate auto padding values for upper, lower bounds.
            out_shape.emplace_back(dim::inf_bound);
        }
    }
}

/**
 * @brief Shape inference helper used for pooling operators such Max Pool, Avg Pool.
 */
template <class TOp, class TShape, class TContainer, class TRShape = result_shape_t<TShape>>
TRShape out_shape_infer(const TOp* op,
                        const TShape& data_shape,
                        const TContainer& pads_begin,
                        const TContainer& pads_end,
                        const Strides& dilations) {
    const auto out_rank_size = spatial_dim_offset + op->get_kernel().size();
    TRShape out_shape;
    if (data_shape.rank().is_static()) {
        const auto& batch_size = data_shape[0];
        const auto& channel_count = data_shape[1];

        NODE_VALIDATION_CHECK(op, batch_size.is_dynamic() || batch_size.get_length() > 0, "Batch size is zero.");
        NODE_VALIDATION_CHECK(op,
                              channel_count.is_dynamic() || channel_count.get_length() > 0,
                              "Channel count is zero.");

        out_shape.reserve(out_rank_size);
        std::copy_n(data_shape.begin(), spatial_dim_offset, std::back_inserter(out_shape));
        pooling::append_spatial_shape(op, data_shape, pads_begin, pads_end, dilations, out_shape);
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
          class TRShape = result_shape_t<TShape>,
          typename std::enable_if<std::is_same<TOp, v8::AdaptiveAvgPool>::value ||
                                  std::is_same<TOp, v8::AdaptiveMaxPool>::value>::type* = nullptr>
TRShape out_shape_infer(const TOp* op, const std::vector<TShape>& input_shapes, const ITensorAccessor& ta) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    const auto& out_spatial_shape = input_shapes[1];

    const auto& data_rank = data_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          ov::util::is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);

    TRShape output_shape;
    if (data_rank.is_static()) {
        auto num_of_spatial_dims = data_shape.size() - spatial_dim_offset;

        NODE_VALIDATION_CHECK(
            op,
            out_spatial_shape.rank().is_dynamic() || out_spatial_shape[0].compatible(num_of_spatial_dims),
            "Output shape for spatial dimension not compatible with data shape.");

        output_shape.reserve(data_shape.size());
        std::copy_n(data_shape.begin(), spatial_dim_offset, std::back_inserter(output_shape));

        if (const auto spatial_dims = get_input_const_data_as_shape<TRShape>(op, 1, ta)) {
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
