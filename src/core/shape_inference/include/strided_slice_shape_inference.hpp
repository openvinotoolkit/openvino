// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>
#include <openvino/op/strided_slice.hpp>

#include "slice_shape_inference_utils.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace v1 {

template <class T>
std::vector<T> shape_infer(const StridedSlice* op,
                           const std::vector<T>& input_shapes,
                           const std::map<size_t, HostTensorPtr>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    static constexpr std::array<char const*, 3> shape_names{"Begin", "End", "Strides"};

    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4));

    const auto& input_shape = input_shapes[0];

    for (size_t i = 1; i < input_shapes.size(); ++i) {
        const auto& shape_rank = input_shapes[i].rank();
        NODE_VALIDATION_CHECK(op,
                              shape_rank.compatible(1),
                              shape_names[i - 1],
                              " input must be 1D (has rank: ",
                              shape_rank,
                              ")");
    }

    const auto& begin_shape = input_shapes[1];
    const auto& end_shape = input_shapes[2];

    // it is not possible to define output shape if input data shape rank is undefined
    // even the lengths of begin, end, or strides are defined
    if (input_shape.rank().is_dynamic()) {
        return {PartialShape::dynamic()};
    }

    auto input_rank = input_shape.size();

    auto number_elements_in_1d = [](const StridedSlice* op, const T& shape_1d) -> int64_t {
        auto rank_1d = shape_1d.rank();
        if (rank_1d.is_static()) {
            NODE_VALIDATION_CHECK(op, rank_1d.get_length() == 1, "Only 1D tensor is allowed.");
            if (shape_1d[0].is_static()) {
                return static_cast<int64_t>(shape_1d[0].get_length());
            }
        }
        return -1;
    };

    // compute constant values of begin, end, and strides if possible
    const auto begin = slice::get_input_bounds<T>(op, 1, constant_data);
    const auto end = slice::get_input_bounds<T>(op, 2, constant_data);

    std::unique_ptr<std::vector<int64_t>> strides;
    if (input_shapes.size() > 3) {
        strides = get_input_const_data_as<T, int64_t>(op, 3, constant_data);
    } else if (begin) {
        // generate default strides
        strides.reset(new std::vector<int64_t>(begin->size(), 1));
    }

    // compute and check a number of axes for which begin, end, and strides are defined
    auto number_axes = number_elements_in_1d(op, begin_shape);
    auto end_number_axes = number_elements_in_1d(op, end_shape);
    if (number_axes != -1 && end_number_axes != -1) {
        NODE_VALIDATION_CHECK(op, number_axes == end_number_axes, "Begin and end need to have same number of values.");
    } else if (end_number_axes != -1) {
        number_axes = end_number_axes;
    }
    auto strides_number_axes = strides ? static_cast<int64_t>(strides->size()) : static_cast<int64_t>(-1);
    if (number_axes != -1 && strides_number_axes != -1) {
        NODE_VALIDATION_CHECK(op,
                              number_axes == strides_number_axes,
                              "Stride needs to have same number of values as begin and end.");
    } else if (strides_number_axes != -1) {
        number_axes = strides_number_axes;
    }

    // if number of axes is undefined we cannot say about output rank
    if (number_axes < 0) {
        return {PartialShape::dynamic()};
    }

    // collect indices of axes by which the shape needs to be changed
    auto convert_mask_to_axis_set = [](const std::vector<int64_t>& mask) {
        AxisSet axis_set;
        for (size_t i = 0; i < mask.size(); ++i) {
            if (mask[i] == 1) {
                axis_set.emplace(i);
            }
        }
        return axis_set;
    };
    AxisSet ellipsis_mask = convert_mask_to_axis_set(op->get_ellipsis_mask());
    NODE_VALIDATION_CHECK(op, ellipsis_mask.size() <= 1, "At most one ellipsis is allowed.");
    AxisSet new_axis_mask = convert_mask_to_axis_set(op->get_new_axis_mask());
    AxisSet begin_mask = convert_mask_to_axis_set(op->get_begin_mask());
    AxisSet end_mask = convert_mask_to_axis_set(op->get_end_mask());
    AxisSet shrink_axis_mask = convert_mask_to_axis_set(op->get_shrink_axis_mask());
    NODE_VALIDATION_CHECK(op,
                          input_rank + new_axis_mask.size() >= static_cast<size_t>(number_axes),
                          "Input rank plus number of new axis has to be at least the size of Lower "
                          "and Upper bounds vector.");

    T out;
    int64_t input_shape_idx = 0;
    for (int64_t axis = 0; axis < number_axes; ++axis) {
        // add all dimensions hidden under the ellipsis mask if ellipsis mask is set
        if (ellipsis_mask.count(axis)) {
            // only one bit in ellipsis mask is allowed
            int num_new_axis_after_ellipses = 0;
            int num_input_axis_before_ellipses = 0;
            for (int64_t i = 0; i < axis; ++i) {
                if (!new_axis_mask.count(i)) {
                    num_input_axis_before_ellipses++;
                }
            }

            if (begin) {
                for (size_t i = axis + 1; i < begin->size(); ++i) {
                    if (new_axis_mask.count(i)) {
                        num_new_axis_after_ellipses++;
                    }
                }
            }

            int64_t num_input_axis_after_ellipses =
                (number_axes - axis - num_new_axis_after_ellipses - 1);  // -1 because it's a position of ellipses
            int64_t num_of_hidden_dims = input_rank - num_input_axis_after_ellipses - num_input_axis_before_ellipses;
            for (int64_t i = 0; i < num_of_hidden_dims; ++i, ++input_shape_idx) {
                out.emplace_back(input_shape[input_shape_idx]);
            }
        } else {
            // add new single dimension if new_axis_mask is set
            if (new_axis_mask.count(axis)) {
                out.emplace_back(1);
            }
            // skip this dimension if shrink_axis_mask is set
            else if (shrink_axis_mask.count(axis)) {
                input_shape_idx++;
            }
            // calculating dimension (begin, end, begin_mask, end_mask, stride)
            else if (begin && end && strides) {
                // set default value for stride or use given value
                const auto& input_dim = input_shape[input_shape_idx];
                auto stride =
                    (strides->size() > static_cast<size_t>(axis)) ? (*strides)[axis] : static_cast<int64_t>(1);
                NODE_VALIDATION_CHECK(op, stride != 0, "Stride must be non-zero");

                constexpr int64_t inf_bound = -1;
                const auto is_reverse_stride = stride < 0;
                const int64_t norm_dim = (input_dim.get_max_length() == inf_bound) ? std::numeric_limits<int64_t>::max()
                                                                                   : input_dim.get_max_length();
                const slice::Bounds default_fstart = std::make_pair<int64_t, int64_t>(0, 0);
                const slice::Bounds default_rstop = std::make_pair(inf_bound - norm_dim, inf_bound - norm_dim);
                const slice::Bounds norm_dim_bounds = std::make_pair(norm_dim, norm_dim);

                const auto& default_start = is_reverse_stride ? norm_dim_bounds : default_fstart;
                const auto& default_stop = is_reverse_stride ? default_rstop : norm_dim_bounds;

                const auto& start = begin_mask.count(axis) ? default_start : (*begin)[axis];
                const auto& stop = end_mask.count(axis) ? default_stop : (*end)[axis];
                auto sliced_dim = slice::make_dim(input_dim, start, stop, stride);

                if (std::is_same<DimType, ov::Dimension>::value && (sliced_dim == input_dim)) {
                    // for equal ov::Dimension do merge to get input label (always success)
                    DimType::merge(sliced_dim, sliced_dim, input_dim);
                }
                out.push_back(std::move(sliced_dim));

                input_shape_idx++;
            } else {
                out.emplace_back(0, input_shape[input_shape_idx].get_max_length());

                input_shape_idx++;
            }
        }
    }

    // get remaining values
    for (; input_shape_idx < input_shape.rank().get_length(); ++input_shape_idx) {
        out.push_back(input_shape[input_shape_idx]);
    }
    return {out};
}

template <class T>
void shape_infer(const StridedSlice* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    output_shapes = shape_infer(op, input_shapes, constant_data);
    NODE_VALIDATION_CHECK(op, output_shapes.size() == 1);
}

}  // namespace v1
}  // namespace op
}  // namespace ov
