// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/rdft.hpp>

#include "fft_common_validation.hpp"
#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v9 {
template <typename B>
B get_ouput_dimension_bound(B b) {
    if (b <= 0) {
        return b;
    }
    return b / 2 + 1;
}

template <class DimType>
DimType get_rdft_output_dimension(DimType d) {
    return DimType(get_ouput_dimension_bound(d.get_min_length()), get_ouput_dimension_bound(d.get_max_length()));
}

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const RDFT* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);

    util::fft_common_validation::shape_validation(op,
                                                  input_shapes,
                                                  axes,
                                                  util::fft_common_validation::FFTKind::RealInput);

    if (input_shape.rank().is_dynamic()) {
        output_shape = ov::PartialShape::dynamic();
        return output_shapes;
    }

    output_shape = input_shape;
    output_shape.push_back(DimType(2));

    const auto input_rank = input_shape.size();

    if (axes_shape.rank().is_dynamic() || !axes) {
        for (size_t i = 0; i < input_rank; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
        return output_shapes;
    }

    const auto last_axis = axes->back();

    if (input_shapes.size() == 2) {
        output_shape[last_axis] = get_rdft_output_dimension(input_shape[last_axis]);
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
        const int64_t current_axis = (*axes)[i];
        if ((*signal_size)[i] != -1) {
            output_shape[current_axis] = DimType((*signal_size)[i]);
        }
    }
    output_shape[last_axis] = get_rdft_output_dimension(output_shape[last_axis]);

    return output_shapes;
}
}  // namespace v9
}  // namespace op
}  // namespace ov
