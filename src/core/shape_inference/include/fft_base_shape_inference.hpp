// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "dimension_util.hpp"
#include "fft_common_validation.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/dimension_tracker.hpp"
#include "openvino/op/util/fft_base.hpp"
#include "utils.hpp"

namespace ov {
namespace op {

namespace fft {
/**
 * @brief Set the label of the dimension at the axis in the shape. */
template <class TShape, typename std::enable_if<std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_label(TShape& shape, ov::label_t label, size_t axis) {
    DimensionTracker::set_label(shape[axis], label);
}

/** @brief Shapes other than PartialShape have no labels. */
template <class TShape, typename std::enable_if<!std::is_same<TShape, PartialShape>::value>::type* = nullptr>
void set_pattern_label(TShape& shape, ov::label_t label, size_t axis) {}

}  // namespace fft

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const util::FFTBase* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;
    using namespace ov::util;

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3));

    const auto& input_shape = input_shapes[0];
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];
    auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta);

    util::fft_common_validation::shape_validation(op,
                                                  input_shapes,
                                                  *axes,
                                                  static_cast<bool>(axes),
                                                  util::fft_common_validation::FFTKind::ComplexInput);

    output_shape = input_shape;
    if (input_shapes.size() == 3 && input_shape.rank().is_static()) {
        if (axes) {
            if (const auto output_bounds = get_input_bounds<TRShape, int64_t>(op, 2, ta)) {
                const auto minus_one_bound = std::make_pair(dim::inf_bound, dim::inf_bound);
                auto labels = op->get_input_source_output(2).get_tensor().get_value_label();
                const bool propagate_labels = std::is_same<TRShape, PartialShape>::value && !labels.empty();

                size_t num_of_axes = axes->size();
                for (size_t i = 0; i < num_of_axes; ++i) {
                    if ((*output_bounds)[i] == minus_one_bound) {
                        continue;
                    }
                    output_shape[(*axes)[i]] = DimType((*output_bounds)[i].first, (*output_bounds)[i].second);
                    if (propagate_labels) {
                        fft::set_pattern_label(output_shape, labels[i], (*axes)[i]);
                    }
                }
            } else {
                for (int64_t& axis : *axes) {
                    output_shape[axis] = ov::Dimension::dynamic();
                }
            }
        } else {
            for (size_t i = 0; i < input_shape.size() - 1; ++i) {
                output_shape[i] = ov::Dimension::dynamic();
            }
        }
    }
    return output_shapes;
}
}  // namespace op
}  // namespace ov
