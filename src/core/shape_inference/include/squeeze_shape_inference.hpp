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
 * \param ta             Tensor accessor to constant data.
 */
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Squeeze* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;

    const auto number_of_inputs = input_shapes.size();
    OPENVINO_ASSERT(!input_shapes.empty());

    const auto& arg_shape = input_shapes[0];
    const auto& arg_rank = arg_shape.rank();
    auto output_shapes = std::vector<TRShape>(1);
    auto& output_shape = output_shapes[0];

    std::unique_ptr<std::set<int64_t>> unique_axes;

    if (number_of_inputs == 1) {
        unique_axes.reset(new std::set<int64_t>());
    } else if (number_of_inputs == 2) {
        const auto& axes_shape = input_shapes[1];
        NODE_VALIDATION_CHECK(op,
                              axes_shape.is_dynamic() || ov::util::is_rank_compatible_any_of(axes_shape.rank(), {0, 1}),
                              "Second input (axes) should not be of rank higher than 1. Got: ",
                              axes_shape.rank().get_length());

        std::vector<int64_t> axes;
        if (arg_rank.is_static() && axes_shape.is_static()) {
            if (auto axes = get_input_const_data_as<TRShape, int64_t>(op, 1, ta)) {
                // The values of `axes` input are known
                OPENVINO_SUPPRESS_DEPRECATED_START
                normalize_axes(op, arg_rank.get_length(), *axes);
                OPENVINO_SUPPRESS_DEPRECATED_END
                unique_axes.reset(new std::set<int64_t>(axes->cbegin(), axes->cend()));
            } else if (arg_rank.get_length() > 0 && shape_size(axes_shape.to_shape()) == 1) {
                // The `axes` input is a single element tensor which is unique by definition, deducing output rank
                NODE_VALIDATION_CHECK(op,
                                      std::any_of(arg_shape.cbegin(),
                                                  arg_shape.cend(),
                                                  [](const DimType& dim) {
                                                      return dim.compatible(1);
                                                  }),
                                      "Data input shape ",
                                      arg_shape,
                                      " doesn't contain squeezable dimension,"
                                      " but axes input is expected to have one element.");
                output_shape = PartialShape::dynamic(arg_rank.get_length() - 1);
                return output_shapes;
            }
        }
    } else {
        // Invalid number of inputs, empty error message for backward compatibility.
        NODE_VALIDATION_CHECK(op, false);
    }

    if (arg_rank.is_static() && (unique_axes != nullptr)) {
        output_shape.resize(0);
        if (unique_axes->empty()) {
            // if only first input provided or axes are empty remove all dimensions equal to 1.
            if (std::any_of(arg_shape.cbegin(), arg_shape.cend(), [](const DimType& d) {
                    return d.is_dynamic() && d.compatible(1);
                })) {
                // we are unsure if dynamic dimensions would be equal to 1 or not, so we set dynamic output rank
                output_shape = PartialShape::dynamic();
                return output_shapes;
            } else {
                std::copy_if(arg_shape.cbegin(),
                             arg_shape.cend(),
                             std::back_inserter(output_shape),
                             [](const DimType& dim) {
                                 return !dim.compatible(1);
                             });
            }
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
    } else {
        output_shape = PartialShape::dynamic();
    }
    return output_shapes;
}
}  // namespace v0
}  // namespace op
}  // namespace ov
