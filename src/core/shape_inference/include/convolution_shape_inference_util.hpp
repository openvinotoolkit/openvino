// Copyright (C) 2018-2023 Intel Corporation
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
namespace convolution {

constexpr size_t spatial_dim_offset = 2;
constexpr size_t num_spatial_undefined = std::numeric_limits<size_t>::max();

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
 * @brief Get num of spatial form convolution operator.
 *
 * Tries get value from operator member if is not deduced (has -1 value) then tries evaluate it from input shapes.
 *
 * @tparam TConv       Convolution type (this function must be a friend of TConv to access private member).
 * @tparam TShape      Shape type.
 * @param op           Pointer to convolution operator.
 * @param data_shape   Input data shape.
 * @param flter_shape  Input filter shape.
 * @return Value of spatial dimension number or infinite bound (-1) if cannot evaluate.
 */
template <class TShape>
size_t get_num_spatial(const TShape& data_shape,
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
 * @brief Get num of spatial form convolution operator.
 *
 * @tparam TConv        Convolution type.
 * @tparam TShape       Shape type.
 * @param op            Pointer to convolution operator.
 * @param input_shapes  Input shapes (must have two) to spatial dim number evaluation.
 * @return Value of spatial dimension number or `num_spatial_undefined` if cannot evaluate.
 */
template <class TConv, class TShape>
size_t get_num_spatial(const TConv* op, const std::vector<TShape>& input_shapes) {
    return get_num_spatial(input_shapes[0], input_shapes[1], filter_non_spatial_dims_count<TConv>());
}

template <class TConv, class TShape>
size_t get_num_spatial(const TConv* op, const std::vector<TShape>& input_shapes, const TShape& out_spatial_shape) {
    auto num_spatial = get_num_spatial(op, input_shapes);

    if (num_spatial == num_spatial_undefined && out_spatial_shape.rank().is_static() && out_spatial_shape.size() > 0) {
        num_spatial = out_spatial_shape.size();
    }

    return num_spatial;
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
template <class TOp, class TShape, class TIter>
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
        if (kernel_dim->is_static()) {
            std::tie(*pad_b, *pad_e) = dim::padding(*data_dim, kernel_dim->get_length(), dilations[i], strides[i]);
        } else {
            *pad_b = 0;
            *pad_e = 0;
        }
    }
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
                    const TShape& out_spatial_shape,
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
        if (data_dim->is_static() && filter_dim->is_static() && out_spatial_shape[i].is_static()) {
            const auto dilated_filter = dim::dilated(*filter_dim, dilations[i]);
            const auto dim_len = static_cast<int64_t>(data_dim->get_length() - 1);
            const auto padding = std::max<int64_t>(
                dim_len * strides[i] + dilated_filter.get_length() - out_spatial_shape[i].get_length() + out_padding[i],
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
          typename std::enable_if<!std::is_base_of<ov::op::util::ConvolutionBackPropBase, TOp>::value>::type* = nullptr>
void append_spatial_shape(const TOp* op, const TShape& data_shape, const TShape& filters_shape, TShape& out_shape) {
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
          typename std::enable_if<std::is_base_of<ov::op::util::ConvolutionBackPropBase, TOp>::value>::type* = nullptr>
void append_spatial_shape(const TOp* op, const TShape& data_shape, const TShape& filters_shape, TShape& out_shape) {
    using namespace ov::util;

    const auto& strides = op->get_strides();
    const auto& dilations = op->get_dilations();
    const auto& output_padding = op->get_output_padding();
    const auto& pads_begin = op->get_pads_begin();
    const auto& pads_end = op->get_pads_end();

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

namespace validate {
template <class TShape>
void data_shape(const ov::op::util::ConvolutionBase* op, const TShape& data_shape) {
    NODE_VALIDATION_CHECK(op,
                          is_rank_compatible_any_of(data_shape.rank(), {3, 4, 5}),
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

inline void common_attributes(const util::ConvolutionBase* op, const size_t num_spatial) {
    auto& strides = op->get_strides();
    auto& dilations = op->get_dilations();
    auto& pads_begin = op->get_pads_begin();
    auto& pads_end = op->get_pads_end();

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

inline void common_attributes(const util::ConvolutionBackPropBase* op, const size_t num_spatial) {
    common_attributes(static_cast<const util::ConvolutionBase*>(op), num_spatial);
    NODE_VALIDATION_CHECK(op,
                          op->get_output_padding().size() == num_spatial,
                          "Output padding should be defined for all and only spatial dimensions.");
}
}  // namespace validate
}  // namespace convolution

namespace util {
/**
 * @brief Resize common attributes to number of spatial dimension.
 *
 * @param op           Pointer to convolution base for forward convolutions.
 * @param num_spatial  Number of spatial dimensions
 */
inline void resize_attributes(ConvolutionBase* op, const size_t num_spatial) {
    if (op->m_strides.empty()) {
        op->m_strides.resize(num_spatial, 1);
    }
    if (op->m_dilations.empty()) {
        op->m_dilations.resize(num_spatial, 1);
    }
    if (op->m_pads_begin.empty()) {
        op->m_pads_begin.resize(num_spatial, 0);
    }
    if (op->m_pads_end.empty()) {
        op->m_pads_end.resize(num_spatial, 0);
    }
}

/**
 * @brief  Apply auto padding for forward convolutions.
 *
 * @tparam TShape       Shape type.
 * @param op            Pointer to forward propagation convolution operator.
 * @param data_shape     Input data shape.
 * @param filters_shape  Input filter shape.
 */
template <class TShape>
void apply_padding(ConvolutionBase* op, const TShape& data_shape, const TShape& filters_shape) {
    if (convolution::is_auto_pad(op) && data_shape.rank().is_static() && filters_shape.rank().is_static()) {
        convolution::apply_auto_pad(op, data_shape, filters_shape, op->m_pads_begin.begin(), op->m_pads_end.begin());
    } else if (op->get_auto_pad() == op::PadType::VALID) {
        std::fill(op->m_pads_begin.begin(), op->m_pads_begin.end(), 0);
        std::fill(op->m_pads_end.begin(), op->m_pads_end.end(), 0);
    }
}

/**
 * @brief Checks if validation attributes is required.
 *
 * @param op  Pointer to convolution base operator.
 * @return True if internal number of spatial dimension not defined otherwise false.
 */
inline bool is_attr_validation_required(const ConvolutionBase* op) {
    return ov::op::convolution::num_spatial_undefined == op->m_num_spatial;
}

/**
 * @brief Resize common attributes to number of spatial dimension.
 *
 * @param op           Pointer to convolution base for forward convolutions.
 * @param num_spatial  Number of spatial dimensions
 */
inline void resize_attributes(ConvolutionBackPropBase* op, const size_t num_spatial) {
    resize_attributes(static_cast<ConvolutionBase*>(op), num_spatial);

    if (op->m_output_padding.empty()) {
        op->m_output_padding.resize(num_spatial, 0);
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
void apply_padding(ConvolutionBackPropBase* op,
                   const TShape& data_shape,
                   const TShape& filters_shape,
                   const TShape& out_spatial_shape) {
    // apply padding if required
    if (convolution::is_auto_pad(op) && data_shape.rank().is_static() && filters_shape.rank().is_static()) {
        convolution::apply_auto_pad(op,
                                    data_shape,
                                    filters_shape,
                                    out_spatial_shape,
                                    op->m_pads_begin.begin(),
                                    op->m_pads_end.begin());
    } else if (op->get_auto_pad() == op::PadType::VALID) {
        std::fill(op->m_pads_begin.begin(), op->m_pads_begin.end(), 0);
        std::fill(op->m_pads_end.begin(), op->m_pads_end.end(), 0);
    }
}

/**
 * @brief  Apply auto padding for back propagation convolutions.
 *
 * When there is no input with output spatial shape.
 *
 * @tparam TShape            Shape type.
 * @param op                 Pointer to back propagation convolution operator.
 * @param data_shape         Input data shape.
 * @param filters_shape      Input filter shape.
 */
template <class TShape>
void apply_padding(ConvolutionBackPropBase* op, const TShape& data_shape, const TShape& filters_shape) {
    if (convolution::is_auto_pad(op) || op->get_auto_pad() == op::PadType::VALID) {
        std::fill(op->m_pads_begin.begin(), op->m_pads_begin.end(), 0);
        std::fill(op->m_pads_end.begin(), op->m_pads_end.end(), 0);
    }
}
}  // namespace util
}  // namespace op
}  // namespace ov
