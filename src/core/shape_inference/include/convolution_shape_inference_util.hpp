// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "dimension_util.hpp"
#include "openvino/op/util/convolution_backprop_base.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "pooling_shape_inference_util.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace util {
constexpr size_t num_spatial_undefined = std::numeric_limits<size_t>::max();
constexpr size_t spatial_dim_offset = 2;

/**
 * @brief Get num of spatial form convolution operator.
 *
 * Tries to get value from operator member, if not deduced (has -1 value) then tries evaluate it from input shapes.
 *
 * @tparam TShape   Shape type.
 * @param data_shape    Input data shape.
 * @param filter_shape  Input filter shape.
 * @param filter_non_spatial_dims_count Number of non spatial dimensions in filter input
 * @return Value of spatial dimension number or infinite bound (-1) if cannot evaluate.
 */
template <class TShape>
size_t num_spatial_from_shapes(const TShape& data_shape,
                               const TShape& filter_shape,
                               const size_t filter_non_spatial_dims_count) {
    const auto& data_rank = data_shape.rank();
    const auto& filters_rank = filter_shape.rank();

    size_t num_spatial;

    if (data_rank.is_static()) {
        num_spatial = data_rank.get_length() - spatial_dim_offset;
    } else if (filters_rank.is_static()) {
        num_spatial = filters_rank.get_length() - filter_non_spatial_dims_count;
    } else {
        num_spatial = num_spatial_undefined;
    }

    return num_spatial;
}

/**
 * @brief Checks if validation attributes is required.
 *
 * @param op  Pointer to convolution base operator.
 * @return True if internal number of spatial dimension not defined otherwise false.
 */
inline bool is_attr_validation_required(const ConvolutionBase* op) {
    return num_spatial_undefined == op->m_num_spatial;
}

/**
 * @brief Get the num spatil object
 *
 * @param op
 * @return size_t
 */
inline size_t get_num_spatial(const ConvolutionBase* op) {
    return op->m_num_spatial;
}
}  // namespace util

namespace convolution {

constexpr auto num_spatial_undefined = util::num_spatial_undefined;
constexpr size_t spatial_dim_offset = 2;

/**
 * @brief Provides convolution filter non spatial dimension count.
 *
 * @note If specific convolution operator requires different value provide specialization for this operator.
 * @tparam TConv  Type of convolution operator.
 * @return Default value for convolution operators (2).
 */
template <class TConv>
constexpr size_t filter_non_spatial_dims_count() {
    return 2;
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

/**
 * @brief Resize paddings if empty to number of spatial dimensions.
 *
 * @param num_spatial  Number of spatial dimensions.
 * @param pads_begin   Begin padding to resize.
 * @param pads_end     End padding to resize.
 */
inline void resize_empty_padding(const size_t num_spatial, CoordinateDiff& pads_begin, CoordinateDiff& pads_end) {
    if (pads_begin.empty()) {
        pads_begin.resize(num_spatial);
    }

    if (pads_end.empty()) {
        pads_end.resize(num_spatial);
    }
}

inline size_t num_spatial_from_attr(const util::ConvolutionBase* op) {
    size_t num_spatial;

    if (!op->get_strides().empty()) {
        num_spatial = op->get_strides().size();
    } else if (!op->get_dilations().empty()) {
        num_spatial = op->get_dilations().size();
    } else if (!op->get_pads_begin().empty()) {
        num_spatial = op->get_pads_begin().size();
    } else if (!op->get_pads_end().empty()) {
        num_spatial = op->get_pads_end().size();
    } else {
        num_spatial = num_spatial_undefined;
    }

    return num_spatial;
}

template <class TOp,
          class TShape,
          typename std::enable_if<std::is_base_of<util::ConvolutionFwdPropBase, TOp>::value>::type* = nullptr>
size_t calculate_num_spatial(const TOp* op, const std::vector<TShape>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() > 1);
    auto num_spatial = get_num_spatial(op);

    if (num_spatial == num_spatial_undefined) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        num_spatial = util::num_spatial_from_shapes(data_shape, filters_shape, filter_non_spatial_dims_count<TOp>());
    }

    if (num_spatial == num_spatial_undefined) {
        num_spatial = num_spatial_from_attr(op);
    }

    return num_spatial;
}

/**
 * @brief Apply auto padding for forward convolution.
 *
 * The auto padding can be applied only if inputs and attributes of operator are validated.
 * The input shapes must have got static ranks.
 *
 * @param op             Pointer to convolution operator.
 * @param data_shape     Input data shape (must be static rank).
 * @param filters_shape  Input filter shape (must be static rank).
 * @param pads_begin     Iterator to begin of pads begin.
 * @param pads_end       Iterator to begin of pads end.
 */
template <class TOp,
          class TShape,
          class TIter,
          typename std::enable_if<std::is_base_of<util::ConvolutionFwdPropBase, TOp>::value ||
                                  std::is_base_of<util::DeformableConvolutionBase, TOp>::value>::type* = nullptr>
void apply_auto_pad(const TOp* op,
                    const TShape& data_shape,
                    const TShape& filters_shape,
                    TIter pads_begin,
                    TIter pads_end) {
    const auto& dilations = op->get_dilations();
    const auto& strides = op->get_strides();

    const auto num_spatial = strides.size();
    auto data_dim = data_shape.cend() - num_spatial;
    auto kernel_dim = filters_shape.cend() - num_spatial;

    const auto padding_swap = op->get_auto_pad() == PadType::SAME_UPPER;
    auto& pad_b = padding_swap ? pads_begin : pads_end;
    auto& pad_e = padding_swap ? pads_end : pads_begin;

    for (size_t i = 0; i < num_spatial; ++i, ++pad_b, ++pad_e, ++data_dim, ++kernel_dim) {
        using namespace ov::util;
        if (dim::is_static(*kernel_dim)) {
            std::tie(*pad_b, *pad_e) = dim::padding(*data_dim, dim::get_length(*kernel_dim), dilations[i], strides[i]);
        } else {
            *pad_b = 0;
            *pad_e = 0;
        }
    }
}

/**
 * @brief Apply padding to forward propagation convolution besed on padding.
 *
 * @tparam TShape
 *
 * @param op            Pointer to coevolution operator.
 * @param data_shape    Input data shapes for shape inference.
 * @param filters_shape Input filters shape for shape inference.
 * @param pads_begin    Begin padding to updated.
 * @param pads_end      End padding to update.
 */
template <class TOp,
          class TShape,
          typename std::enable_if<std::is_base_of<util::ConvolutionFwdPropBase, TOp>::value ||
                                  std::is_base_of<util::DeformableConvolutionBase, TOp>::value>::type* = nullptr>
void apply_padding(const TOp* op,
                   const TShape& data_shape,
                   const TShape& filters_shape,
                   CoordinateDiff& pads_begin,
                   CoordinateDiff& pads_end) {
    if (convolution::is_auto_pad(op) && data_shape.rank().is_static() && filters_shape.rank().is_static()) {
        convolution::apply_auto_pad(op, data_shape, filters_shape, pads_begin.begin(), pads_end.begin());
    } else if (op->get_auto_pad() == op::PadType::VALID) {
        std::fill(pads_begin.begin(), pads_begin.end(), 0);
        std::fill(pads_end.begin(), pads_end.end(), 0);
    } else if (op->get_auto_pad() == op::PadType::EXPLICIT) {
        std::copy(op->get_pads_begin().begin(), op->get_pads_begin().end(), pads_begin.begin());
        std::copy(op->get_pads_end().begin(), op->get_pads_end().end(), pads_end.begin());
    }
}

/**
 * @brief Append spatial dimension at end of output shape of forward propagation convolution.
 *
 * @tparam TOp           Forward propagation convolution operator type.
 * @tparam TShape        Type of shape.
 * @param op             Pointer to operator.
 * @param data_shape     Input data shape.
 * @param filters_shape  Input filter shape.
 * @param out_shape      Output shape to append spatial dimensions.
 */
template <class TOp,
          class TShape,
          class TRShape = result_shape_t<TShape>,
          typename std::enable_if<std::is_base_of<util::ConvolutionFwdPropBase, TOp>::value ||
                                  std::is_base_of<util::DeformableConvolutionBase, TOp>::value>::type* = nullptr>
void append_spatial_shape(const TOp* op,
                          const TShape& data_shape,
                          const TShape& filters_shape,
                          CoordinateDiff& pads_begin,
                          CoordinateDiff& pads_end,
                          TRShape& out_shape) {
    using namespace ov::util;
    using TDim = typename TShape::value_type;

    const auto& strides = op->get_strides();
    const auto spatial_num = strides.size();

    const auto& d_shape = data_shape.rank().is_static() ? data_shape : PartialShape::dynamic(spatial_num);
    auto data_dim = d_shape.cend() - spatial_num;

    if (is_auto_pad(op)) {
        std::transform(data_dim, d_shape.cend(), strides.cbegin(), std::back_inserter(out_shape), &dim::ceil_div<TDim>);
    } else {
        const auto& f_shape = filters_shape.rank().is_static() ? filters_shape : PartialShape::dynamic(spatial_num);
        auto filters_dim = f_shape.cend() - spatial_num;
        const auto& dilations = op->get_dilations();

        for (size_t i = 0; i < spatial_num; ++i, ++data_dim, ++filters_dim) {
            TDim dim = *data_dim + (pads_begin[i] + pads_end[i]);
            const TDim filter_dilated = dim::dilated(*filters_dim, dilations[i]);

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

namespace validate {
template <class TShape>
void data_shape(const ov::op::util::ConvolutionBase* op, const TShape& data_shape) {
    NODE_VALIDATION_CHECK(op,
                          ov::util::is_rank_compatible_any_of(data_shape.rank(), {3, 4, 5}),
                          "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                          data_shape);
}

template <class TShape>
void filter_shape(const ov::op::util::ConvolutionBase* op, const TShape& filters_shape, const TShape& data_shape) {
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
        data_rank.is_dynamic() || filters_rank.is_dynamic() || data_shape[1].compatible(filters_shape[1]),
        "Data batch channel count (",
        data_shape[1],
        ") does not match filter input channel count (",
        filters_shape[1],
        ").");
}

inline void common_attributes(const util::ConvolutionBase* op,
                              const size_t num_spatial,
                              const CoordinateDiff& pads_begin,
                              const CoordinateDiff& pads_end) {
    auto& strides = op->get_strides();
    auto& dilations = op->get_dilations();

    NODE_VALIDATION_CHECK(op,
                          strides.size() == num_spatial,
                          "Strides should be defined for all and only spatial dimensions.");
    NODE_VALIDATION_CHECK(op,
                          dilations.size() == num_spatial,
                          "Dilations should be defined for all and only spatial dimensions.");
    NODE_VALIDATION_CHECK(op,
                          pads_begin.size() == num_spatial && pads_end.size() == pads_begin.size(),
                          "Pads begin and end should be defined for all and only spatial dimensions.");

    constexpr auto is_zero = cmp::Equal<size_t>(0);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(strides.cbegin(), strides.cend(), is_zero),
                          "Strides has zero dimension(s). ",
                          strides);
    NODE_VALIDATION_CHECK(op,
                          std::none_of(dilations.cbegin(), dilations.cend(), is_zero),
                          "Filter dilations has zero dimension(s). ",
                          dilations);
}

inline void common_attributes(const util::ConvolutionBackPropBase* op,
                              const size_t num_spatial,
                              const CoordinateDiff& pads_begin,
                              const CoordinateDiff& pads_end) {
    common_attributes(static_cast<const util::ConvolutionBase*>(op), num_spatial, pads_begin, pads_end);
    NODE_VALIDATION_CHECK(op,
                          op->get_output_padding().size() == num_spatial,
                          "Output padding should be defined for all and only spatial dimensions.");
}
}  // namespace validate
}  // namespace convolution

}  // namespace op
}  // namespace ov
