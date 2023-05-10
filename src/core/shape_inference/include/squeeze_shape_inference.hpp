// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/squeeze.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v0 {

/**
 * \brief Do Squeeze shape inference.
 *
 * \tparam T             Type of input/output shapes.
 *
 * \param op             Squeeze operator pointer.
 * \param input_shapes   Squeeze input shapes.
 * \param output_shapes  Output shapes result of squeeze shape inference.
 * \param constant_data  Map of constant data.
 */
template <class T>
void shape_infer(const Squeeze* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    NODE_VALIDATION_CHECK(op, output_shapes.size() == 1);
    const auto number_of_inputs = input_shapes.size();
    OPENVINO_ASSERT(!input_shapes.empty());

    const auto& arg_shape = input_shapes[0];
    const auto& arg_rank = arg_shape.rank();
    auto& output_shape = output_shapes[0];

    std::unique_ptr<std::set<int64_t>> unique_axes;

    if (number_of_inputs == 1) {
        unique_axes.reset(new std::set<int64_t>());
    } else if (number_of_inputs == 2) {
        const auto& axes_shape = input_shapes[1];
        OPENVINO_SUPPRESS_DEPRECATED_START
        NODE_VALIDATION_CHECK(op,
                              axes_shape.is_dynamic() || is_rank_compatible_any_of(axes_shape.rank(), {0, 1}),
                              "Second input (axes) should not be of rank higher than 1. Got: ",
                              axes_shape.rank().get_length());
        OPENVINO_SUPPRESS_DEPRECATED_END

        std::vector<int64_t> axes;
        if (arg_rank.is_static() && axes_shape.is_static() && get_data_as_int64<T>(1, op, axes, constant_data)) {
            OPENVINO_SUPPRESS_DEPRECATED_START
            normalize_axes(op, arg_rank.get_length(), axes);
            OPENVINO_SUPPRESS_DEPRECATED_END
            unique_axes.reset(new std::set<int64_t>(axes.cbegin(), axes.cend()));
        }
    } else {
        // Invalid number of inputs, empty error message for backward compatibility.
        NODE_VALIDATION_CHECK(op, false);
    }

    if (arg_rank.is_static() && (unique_axes != nullptr)) {
        output_shape.resize(0);

        if (unique_axes->empty()) {
            // According to specification, if only first input provided` or axes are empty
            // remove all dimensions equal to 1.
            std::copy_if(arg_shape.cbegin(),
                         arg_shape.cend(),
                         std::back_inserter(output_shape),
                         [](const DimType& dim) {
                             return !dim.compatible(1);
                         });
        } else {
            int64_t idx = 0;
            auto rm_axis_iter = unique_axes->cbegin();
            auto rm_axis_end = unique_axes->cend();

            // Returns true if dimension not squeezable on axis from input axes.
            const auto not_squeezable_at_axis = [&op, &rm_axis_iter, &rm_axis_end, &idx](const DimType& dim) {
                if ((rm_axis_iter != rm_axis_end) && (*rm_axis_iter == idx++)) {
                    NODE_VALIDATION_CHECK(op,
                                          dim.compatible(1),
                                          "provided axis value is invalid. Only axes of size 1 may be removed.");
                    ++rm_axis_iter;
                    return false;
                } else {
                    return true;
                }
            };

            std::copy_if(arg_shape.cbegin(),
                         arg_shape.cend(),
                         std::back_inserter(output_shape),
                         not_squeezable_at_axis);
        }
        // When arg shape has got static rank but shape is dynamic and output shape dimensions is empty (scalar)
        // make dynamic output except the case when arg_shape is 1-D shape with 0 or 1 element then should be scalar.
        if (arg_shape.is_dynamic() && (output_shape.size() == 0) &&
            !(arg_rank.get_length() == 1 && arg_shape[0].get_max_length() <= 1)) {
            output_shape = PartialShape::dynamic();
        }
    } else {
        output_shape = PartialShape::dynamic();
    }
}
}  // namespace v0
}  // namespace op
}  // namespace ov
