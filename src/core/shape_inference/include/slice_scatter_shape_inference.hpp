// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/slice_scatter.hpp"
#include "slice_shape_inference.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v15 {
template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const SliceScatter* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;
    const auto& num_of_inputs = input_shapes.size();

    NODE_VALIDATION_CHECK(op,
                          num_of_inputs == 5 || num_of_inputs == 6,
                          "SliceScatter has to have 5 or 6 inputs. Got: ",
                          num_of_inputs);
    const auto& data_rank = input_shapes[0].rank();
    auto output_shapes = std::vector<TRShape>{input_shapes[0]};
    auto& output_shape = output_shapes.front();
    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           output_shape.merge_rank(input_shapes[1].rank()),
                           "SliceScatter `data` and `updates` need to have compatible rank.");

    NODE_SHAPE_INFER_CHECK(op,
                           input_shapes,
                           output_shape.is_dynamic() || output_shape.rank().get_length() > 0,
                           "SliceScatter `data` and `updates` input can't be a scalar.");
    for (size_t i = 2; i < input_shapes.size(); ++i) {
        const auto& shape = input_shapes[i];
        const auto& shape_rank = shape.rank();
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               shape_rank.compatible(1),
                               "SliceScatter `",
                               slice::shape_names[i - 2],
                               "` input must be a 1D tensor. Got rank: ",
                               shape_rank);

        if (output_shape.rank().is_static()) {
            NODE_SHAPE_INFER_CHECK(
                op,
                input_shapes,
                shape_rank.is_dynamic() || ov::cmp::le(shape[0].get_min_length(), output_shape.rank().get_length()),
                "SliceScatter `",
                slice::shape_names[i - 2],
                "` input dim size can't be bigger than `data` or `updates` rank.");
        }
    }

    const auto& start_shape = input_shapes[2];
    const auto& stop_shape = input_shapes[3];
    const auto& step_shape = input_shapes[4];

    NODE_SHAPE_INFER_CHECK(
        op,
        input_shapes,
        start_shape.compatible(stop_shape) && start_shape.compatible(step_shape) && stop_shape.compatible(step_shape),
        "SliceScatter `start`, `stop`, `step` inputs must have compatible shapes.");
    // compute constant values of begin, end, and strides if possible
    const auto start = get_input_bounds<TRShape, int64_t>(op, 2, ta);
    const auto stop = get_input_bounds<TRShape, int64_t>(op, 3, ta);
    const auto steps = get_input_const_data_as<TRShape, int64_t>(op, 4, ta);
    if (step_shape.is_static() && steps) {
        for (typename DimType::value_type i = 0; i < step_shape[0].get_length(); i++) {
            NODE_SHAPE_INFER_CHECK(op, input_shapes, (*steps)[i] != 0, "SliceScatter step values must be non-zero.");
        }
    }
    slice::AxesMap axes_map;
    if (input_shapes.size() > 5) {
        NODE_SHAPE_INFER_CHECK(
            op,
            input_shapes,
            input_shapes[5].compatible(start_shape),
            "SliceScatter `axes` input must have compatible shape with `start`, `stop`, `step` inputs.");
        auto axes = get_input_const_data_as<TRShape, int64_t>(op, 5, ta);
        if (axes && data_rank.is_static()) {
            ov::util::try_normalize_axes(*axes, data_rank, *op);
            axes_map.add(*axes);
            NODE_SHAPE_INFER_CHECK(op,
                                   input_shapes,
                                   axes_map.is_valid,
                                   "SliceScatter values in `axes` input must be unique.");
        }
    } else if (start) {
        axes_map.generate_n(start->size());
    }
    auto axis_it = axes_map.m.cbegin();

    if (output_shape.rank().is_static()) {
        std::vector<DimType> expected_updates_shape_vector;
        expected_updates_shape_vector.reserve(output_shape.size());
        for (size_t dim_idx = 0; dim_idx < output_shape.size(); ++dim_idx) {
            const DimType& input_dim = output_shape[dim_idx];

            if (axes_map.is_valid && (axis_it != axes_map.m.cend()) && (axis_it->first == dim_idx)) {
                const auto& i = axis_it->second;
                if (start && stop && steps) {
                    const auto& step = (*steps)[i];
                    expected_updates_shape_vector.push_back(slice::make_dim(input_dim, (*start)[i], (*stop)[i], step));
                } else {
                    expected_updates_shape_vector.push_back(DimType{0, input_dim.get_max_length()});
                }
                ++axis_it;
            } else if (axes_map.is_valid) {
                // dimension not on axes list, no change
                expected_updates_shape_vector.push_back(input_dim);
            } else {
                // axes are unknow so any dimension can be sliced
                expected_updates_shape_vector.push_back(DimType{0, input_dim.get_max_length()});
            }
        }
        TRShape expected_updates_shape{expected_updates_shape_vector};
        NODE_SHAPE_INFER_CHECK(op,
                               input_shapes,
                               input_shapes[1].compatible(expected_updates_shape),
                               "SliceScatter updates at index 1 are not compatible with expected slice shape ",
                               expected_updates_shape);
    }

    return output_shapes;
}
}  // namespace v15
}  // namespace op
}  // namespace ov
