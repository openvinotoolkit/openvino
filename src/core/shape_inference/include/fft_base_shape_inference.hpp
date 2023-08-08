// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/util/fft_base.hpp>

#include "fft_common_validation.hpp"
#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const util::FFTBase* op,
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
                                                  *axes,
                                                  static_cast<bool>(axes),
                                                  util::fft_common_validation::FFTKind::ComplexInput);

    output_shape = input_shape;

    if (input_shape.rank().is_static() && axes_shape.rank().is_static() && input_shapes.size() == 3 && axes) {
        const auto& signal_size_shape = input_shapes[2];
        auto signal_size = get_input_const_data_as<TRShape, int64_t>(op, 2, ta);

        if (signal_size_shape.rank().is_static() && signal_size) {
            size_t num_of_axes = axes->size();
            for (size_t i = 0; i < num_of_axes; ++i) {
                if ((*signal_size)[i] == -1) {
                    continue;
                }
                output_shape[(*axes)[i]] = DimType((*signal_size)[i]);
            }
        } else if (signal_size_shape.rank().is_static()) {
            for (int64_t& axis : *axes) {
                output_shape[axis] = ov::Dimension::dynamic();
            }
        }
    } else if (input_shape.rank().is_static() && (axes_shape.rank().is_dynamic() || !axes)) {
        const auto input_rank = input_shape.size();
        for (size_t i = 0; i < input_rank - 1; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
    }
    return output_shapes;
}
}  // namespace op
}  // namespace ov
