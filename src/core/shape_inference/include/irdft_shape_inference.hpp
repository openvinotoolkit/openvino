// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "fft_common_validation.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/op/irdft.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v9 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const IRDFT* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);
    auto axes_are_known = static_cast<bool>(axes);

    util::fft_common_validation::shape_validation(op,
                                                  input_shapes,
                                                  axes,
                                                  util::fft_common_validation::FFTKind::ComplexInput);

    if (input_shape.rank().is_dynamic()) {
        output_shape = ov::PartialShape::dynamic();
        return output_shapes;
    }

    const auto input_rank = input_shape.size();

    output_shape = input_shape;
    output_shape.resize(input_rank - 1);

    if (axes_shape.rank().is_dynamic() || !axes_are_known) {
        for (size_t i = 0; i < input_rank - 1; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
        return output_shapes;
    }

    const auto last_axis = axes->back();

    if (input_shapes.size() == 2) {
        output_shape[last_axis] = DimType(2) * (input_shape[last_axis] - DimType(1));
        return output_shapes;
    }

    const auto& signal_size_shape = input_shapes[2];
    auto signal_size = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);

    if (signal_size_shape.rank().is_dynamic() || !signal_size) {
        output_shape[last_axis] = ov::Dimension::dynamic();
        return output_shapes;
    }

    size_t num_of_axes = axes->size();
    for (size_t i = 0; i < num_of_axes; ++i) {
        if ((*signal_size)[i] != -1) {
            output_shape[(*axes)[i]] = DimType((*signal_size)[i]);
        }
    }
    if (signal_size->back() == -1) {
        output_shape[last_axis] = DimType(2) * (input_shape[last_axis] - DimType(1));
    }
    return output_shapes;
}
}  // namespace v9
}  // namespace op
}  // namespace ov
