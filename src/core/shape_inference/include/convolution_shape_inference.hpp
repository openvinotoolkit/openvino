// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "convolution_shape_inference_util.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

/**
 * @brief
 *
 * @tparam ConvType
 * @tparam TShape
 * @param op
 * @param input_shapes
 */
template <class ConvType, class TShape>
void update_and_validate_attributes(ConvType* op, const std::vector<TShape>& input_shapes) {
    const auto num_spatial = convolution::get_num_spatial(op, input_shapes);

    auto& strides = op->get_strides();
    auto& dilations = op->get_dilations();
    auto& pads_begin = op->get_pads_begin();
    auto& pads_end = op->get_pads_end();

    if (strides.empty()) {
        op->m_strides.resize(num_spatial, 1);
    }
    if (dilations.empty()) {
        op->m_dilations.resize(num_spatial, 1);
    }
    if (pads_begin.empty()) {
        op->m_pads_begin.resize(num_spatial, 0);
    }
    if (pads_end.empty()) {
        op->m_pads_end.resize(num_spatial, 0);
    }

    const auto& data_shape = input_shapes[0];
    const auto& filters_shape = input_shapes[1];
    const auto data_rank = data_shape.rank();
    const auto filters_rank = filters_shape.rank();

    // Validation is not required if op has set num_spatial (already done).
    if (op->m_num_spatial == ov::util::dim::inf_bound) {
        NODE_VALIDATION_CHECK(op,
                              is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                              "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                              data_shape);

        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(strides.size()) == num_spatial,
                              "Strides should be defined for all and only spatial dimensions.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(dilations.size()) == num_spatial,
                              "Dilations should be defined for all and only spatial dimensions.");
        NODE_VALIDATION_CHECK(
            op,
            static_cast<int64_t>(pads_begin.size()) == num_spatial && pads_end.size() == pads_begin.size(),
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

    // apply padding if required
    if (convolution::is_auto_pad(op) && data_rank.is_static() && filters_rank.is_static()) {
        convolution::apply_auto_pad(op, input_shapes, op->m_pads_begin.begin(), op->m_pads_end.begin());
    } else if (op->get_auto_pad() == op::PadType::VALID) {
        std::fill(op->m_pads_begin.begin(), op->m_pads_begin.end(), 0);
        std::fill(op->m_pads_end.begin(), op->m_pads_end.end(), 0);
    }
}

template <class ConvType, class TShape>
void update_and_validate_attributes(ConvType* op,
                                    const std::vector<TShape>& input_shapes,
                                    const TShape& out_spatial_shape) {
    const auto num_spatial = convolution::get_num_spatial(op, input_shapes, out_spatial_shape);

    auto& strides = op->get_strides();
    auto& dilations = op->get_dilations();
    auto& pads_begin = op->get_pads_begin();
    auto& pads_end = op->get_pads_end();

    if (strides.empty()) {
        op->m_strides.resize(num_spatial, 1);
    }
    if (dilations.empty()) {
        op->m_dilations.resize(num_spatial, 1);
    }
    if (pads_begin.empty()) {
        op->m_pads_begin.resize(num_spatial, 0);
    }
    if (pads_end.empty()) {
        op->m_pads_end.resize(num_spatial, 0);
    }
    if (op->get_output_padding().empty()) {
        op->m_output_padding.resize(num_spatial, 0);
    }

    const auto& data_shape = input_shapes[0];
    const auto& filters_shape = input_shapes[1];
    const auto data_rank = data_shape.rank();
    const auto filters_rank = filters_shape.rank();

    // Validation is not required if op has set num_spatial (already done).
    if (op->m_num_spatial == ov::util::dim::inf_bound) {
        NODE_VALIDATION_CHECK(op,
                              is_rank_compatible_any_of(data_rank, {3, 4, 5}),
                              "Expected a 3D, 4D or 5D tensor for the input. Got: ",
                              data_shape);

        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(strides.size()) == num_spatial,
                              "Strides should be defined for all and only spatial dimensions.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(dilations.size()) == num_spatial,
                              "Dilations should be defined for all and only spatial dimensions.");
        NODE_VALIDATION_CHECK(
            op,
            static_cast<int64_t>(pads_begin.size()) == num_spatial && pads_end.size() == pads_begin.size(),
            "Pads begin and end should be defined for all and only spatial dimensions.");
        NODE_VALIDATION_CHECK(op,
                              static_cast<int64_t>(op->get_output_padding().size()) == num_spatial,
                              "Output padding should be defined for all and only spatial dimensions.");

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

    // apply padding if required
    if (convolution::is_auto_pad(op) && data_rank.is_static() && filters_rank.is_static()) {
        convolution::apply_auto_pad(op,
                                    input_shapes,
                                    out_spatial_shape,
                                    op->m_pads_begin.begin(),
                                    op->m_pads_end.begin());
    } else if (op->get_auto_pad() == op::PadType::VALID) {
        std::fill(op->m_pads_begin.begin(), op->m_pads_begin.end(), 0);
        std::fill(op->m_pads_end.begin(), op->m_pads_end.end(), 0);
    }
}

template <class TShape>
std::vector<TShape> shape_infer(const Convolution* op,
                                const std::vector<TShape>& input_shapes,
                                const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);
    using namespace ov::util;

    const auto num_spatial = convolution::get_num_spatial(op, input_shapes);

    TShape output_shape;
    if (num_spatial != dim::inf_bound) {
        const auto& data_shape = input_shapes[0];
        const auto& filters_shape = input_shapes[1];
        const auto data_rank = data_shape.rank();
        const auto filters_rank = filters_shape.rank();

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

        update_and_validate_attributes(const_cast<Convolution*>(op), input_shapes);

        output_shape.reserve(convolution::spatial_dim_offset + num_spatial);
        output_shape.emplace_back(data_rank.is_static() ? data_shape[0] : dim::inf_bound);
        output_shape.emplace_back(filters_rank.is_static() ? filters_shape[0] : dim::inf_bound);
        convolution::append_spatial_shape(op, input_shapes, output_shape);
    } else {
        output_shape = PartialShape::dynamic();
    }

    return {output_shape};
}
}  // namespace v1
}  // namespace op
}  // namespace ov
