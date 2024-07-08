// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dimension_util.hpp"
#include "openvino/op/extractimagepatches.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v3 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const ExtractImagePatches* op, const std::vector<T>& input_shapes) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 1);
    using namespace ov::util;
    using TDim = typename T::value_type;

    constexpr size_t num_spatial_dim = 2;
    constexpr size_t input_shape_static_rank = 4;
    constexpr auto is_zero = cmp::Less<size_t>(1);

    const auto& input_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           input_shape.rank().compatible(input_shape_static_rank),
                           "input tensor must be 4D tensor.");

    const auto& sizes = op->get_sizes();
    NODE_VALIDATION_CHECK(op,
                          sizes.size() == num_spatial_dim,
                          "Attribute sizes should be in [size_rows, size_cols] format.");

    const auto& strides = op->get_strides();
    NODE_VALIDATION_CHECK(op,
                          strides.size() == num_spatial_dim,
                          "Attribute strides should be in [stride_rows, stride_cols] format.");
    NODE_VALIDATION_CHECK(op,
                          std::none_of(strides.begin(), strides.end(), is_zero),
                          "Attribute strides should be strictly greater than zeros in values.");

    const auto& rates = op->get_rates();
    NODE_VALIDATION_CHECK(op,
                          rates.size() == num_spatial_dim,
                          "Attribute rates should be in [rate_rows, rate_cols] format.");
    NODE_VALIDATION_CHECK(op,
                          std::none_of(rates.begin(), rates.end(), is_zero),
                          "Attribute rates should be strictly greater than zeros in values.");

    const auto& pad_type = op->get_auto_pad();
    NODE_VALIDATION_CHECK(
        op,
        pad_type == PadType::VALID || pad_type == PadType::SAME_LOWER || pad_type == PadType::SAME_UPPER,
        "Attribute padding should be in either valid or same_lower or same_upper.");

    if (input_shape.rank().is_static()) {
        constexpr auto num_non_spatial_dims = input_shape_static_rank - num_spatial_dim;

        auto out_it = std::copy_n(input_shape.begin(), num_non_spatial_dims, std::back_inserter(output_shape));
        output_shape[1] *=
            std::accumulate(sizes.begin(), sizes.end(), static_cast<size_t>(1), std::multiplies<size_t>());

        auto stride_it = strides.cbegin();

        if (pad_type == PadType::VALID) {
            auto size_it = sizes.cbegin();
            auto rate_it = rates.cbegin();

            for (size_t i = num_non_spatial_dims; i < input_shape.size(); ++i, ++stride_it, ++size_it, ++rate_it) {
                out_it = dim::ceil_div(input_shape[i] - ((*rate_it) * (*size_it - 1)), *stride_it);
            }
        } else {
            std::transform(input_shape.begin() + num_non_spatial_dims,
                           input_shape.end(),
                           stride_it,
                           out_it,
                           &dim::ceil_div<TDim>);
        }
    } else {
        output_shape.resize(4);
    }
    return output_shapes;
}
}  // namespace v3
}  // namespace op
}  // namespace ov
