// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <openvino/op/reverse.hpp>

#include "utils.hpp"

namespace ov {
namespace op {

namespace util {

/** \brief Clip if value type T is less than 0, otherwise cast to AxisSet::value_type. */
struct ClipNegative {
    using value_type = typename AxisSet::value_type;

    template <class T>
    constexpr value_type operator()(const T value) const {
        return ov::cmp::lt(value, 0) ? 0 : static_cast<value_type>(value);
    }
};
}  // namespace util

namespace v1 {

/**
 * \brief Reverse shape inference
 *
 * \tparam TShape  Type of shape.
 *
 * \param op             Pointer to Reverse operator.
 * \param input_shapes   Input shapes of Reverse.
 * \param constant_data  Map of constant data. Default empty.
 *
 * \return Vector of output shapes with one shape.
 */
template <class TShape, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> shape_infer(const Reverse* op,
                                 const std::vector<TShape>& input_shapes,
                                 const ITensorAccessor& tensor_accessor = make_tensor_accessor()) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2);

    const auto& data_shape = input_shapes[0];
    const auto& data_rank = data_shape.rank();
    const auto& axes_shape = input_shapes[1];
    const auto& axes_rank = axes_shape.rank();

    NODE_VALIDATION_CHECK(op,
                          axes_rank.compatible(1),
                          "The reversed_axes input must be a 1D tensor (got ",
                          axes_rank,
                          ").");

    if (op->get_mode() == Reverse::Mode::MASK) {
        NODE_VALIDATION_CHECK(
            op,
            data_rank.is_dynamic() || axes_rank.is_dynamic() || axes_shape[0].compatible(data_shape.size()),
            "The number of elements in the reversed_axes tensor (",
            axes_shape[0],
            ") must match the input data tensor rank (",
            data_rank,
            ") in 'mask' mode.");
    } else if (data_rank.is_static()) {
        // mode index & data rank is static
        using TAxis = typename AxisSet::value_type;
        static_assert(std::is_same<TAxis, util::ClipNegative::value_type>(),
                      "AxisSet::value_type != ClipNegative::value_type");

        if (const auto axes =
                get_input_const_data_as<TShape, TAxis, AxisSet>(op, 1, tensor_accessor, util::ClipNegative())) {
            NODE_VALIDATION_CHECK(op,
                                  all_of(axes->begin(), axes->end(), cmp::Less<TAxis>(data_rank.get_length())),
                                  "Some of the provided axes (",
                                  *axes,
                                  ") are out of bounds (input rank: ",
                                  data_rank,
                                  ").");
        }
    }

    return {data_shape};
}
}  // namespace v1
}  // namespace op
}  // namespace ov
