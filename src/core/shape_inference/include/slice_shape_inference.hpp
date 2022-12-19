// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <openvino/op/slice.hpp>

#include "slice_shape_inference_utils.hpp"

namespace ov {
namespace op {
namespace v8 {

const std::array<char const*, 4> shape_names{"start", "stop", "step", "axes"};

template <class T>
void shape_infer(const Slice* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;

    const auto& num_of_inputs = input_shapes.size();

    NODE_VALIDATION_CHECK(op,
                          num_of_inputs == 4 || num_of_inputs == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          num_of_inputs);
    NODE_VALIDATION_CHECK(op, output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();

    for (size_t i = 1; i < input_shapes.size(); ++i) {
        const auto& shape = input_shapes[i];
        const auto& shape_rank = shape.rank();
        NODE_VALIDATION_CHECK(op,
                              shape_rank.compatible(1),
                              "Slice `",
                              shape_names[i - 1],
                              "` input must be a 1D tensor. Got rank: ",
                              shape_rank);

        if (input_rank.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  shape_rank.is_dynamic() || shape[0].get_min_length() <= input_rank.get_length(),
                                  "Slice `",
                                  shape_names[i - 1],
                                  "` input dim size can't be bigger than `data` rank.");
        }
    }

    const auto& start_shape = input_shapes[1];
    const auto& stop_shape = input_shapes[2];
    const auto& step_shape = input_shapes[3];

    NODE_VALIDATION_CHECK(
        op,
        start_shape.compatible(stop_shape) && start_shape.compatible(step_shape) && stop_shape.compatible(step_shape),
        "Slice `start`, `stop`, `step` inputs must have compatible shapes.");

    // it is not possible to define output shape if input data shape rank is undefined
    // even the lengths of begin, end, or strides are defined
    if (input_rank.is_dynamic()) {
        output_shapes[0] = PartialShape::dynamic();
        return;
    }

    std::vector<int64_t> steps, axes;

    // compute constant values of begin, end, and strides if possible
    auto start = slice::get_input_bounds<T>(op, 1, constant_data);
    auto stop = slice::get_input_bounds<T>(op, 2, constant_data);
    auto got_start = !start.first.empty();
    auto got_stop = !stop.first.empty();
    const auto got_steps = get_data_as_int64<T>(3, op, steps, constant_data);

    bool got_axes;
    if (input_shapes.size() > 4) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[4].compatible(start_shape),
                              "Slice `axes` input must have compatible shape with `start`, `stop`, `step` inputs.");
        got_axes = get_data_as_int64<T>(4, op, axes, constant_data);
        if (got_axes) {
            NODE_VALIDATION_CHECK(op, ov::are_unique(axes), "Slice values in `axes` input must be unique.");
            ov::normalize_axes(op, input_shape.rank().get_length(), axes);
        }
    } else if (got_start) {
        axes.reserve(start.first.size());
        std::generate_n(std::back_inserter(axes), start.first.size(), SeqGen<int64_t>(0));
        got_axes = true;
    } else {
        got_axes = false;
    }

    std::vector<DimType> dims;
    dims.reserve(input_shape.rank().get_length());
    for (size_t dim_idx = 0; dim_idx < static_cast<size_t>(input_shape.rank().get_length()); ++dim_idx) {
        const DimType& input_dim = input_shape[dim_idx];

        const auto axis_it = std::find(axes.begin(), axes.end(), dim_idx);
        if (axis_it != axes.end()) {
            const auto i = std::distance(axes.begin(), axis_it);

            if (got_start && got_stop && got_steps) {
                const auto& step = steps[i];
                NODE_VALIDATION_CHECK(op, step != 0, "Step must be non-zero");

                const auto& start_lb =
                    element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), start.first[i]);
                const auto& start_ub =
                    start.second.empty()
                        ? start_lb
                        : element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), start.second[i]);

                const auto& stop_lb =
                    element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), stop.first[i]);
                const auto& stop_ub =
                    stop.second.empty()
                        ? stop_lb
                        : element::get_value_or_limit_of<int64_t>(op->get_input_element_type(1), stop.second[i]);

                auto lb = slice::get_step_elements(input_dim.get_min_length(), start_ub, stop_lb, step);
                auto ub = slice::get_step_elements(input_dim.get_max_length(), start_lb, stop_ub, step);
                dims.emplace_back(lb, ub);
            } else {
                dims.emplace_back(0, input_dim.get_max_length());
            }

            if (std::is_same<DimType, ov::Dimension>::value && dims.back() == input_dim) {
                // for equal ov::Dimension do merge to get input label (always success)
                DimType::merge(dims.back(), dims.back(), input_dim);
            }
        } else if (got_axes) {
            // dimension not on axes list, no change
            dims.push_back(input_dim);
        } else {
            // axes are unknow so any dimension can be sliced
            dims.emplace_back(0, input_dim.get_max_length());
        }
    }
    output_shapes.front() = T(std::move(dims));
}
}  // namespace v8
}  // namespace op
}  // namespace ov
