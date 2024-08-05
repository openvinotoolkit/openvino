// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <array>

#include "openvino/core/validation_util.hpp"
#include "openvino/op/slice.hpp"
#include "slice_shape_inference_utils.hpp"

namespace ov {
namespace op {

namespace slice {

constexpr std::array<char const*, 4> shape_names{"start", "stop", "step", "axes"};

struct AxesMap {
    bool is_valid{};               //!< Flag indicates current axes map has valid data (unique).
    std::map<size_t, size_t> m{};  //!< Map axis value to index of start, stop order.

    void add(const std::vector<int64_t>& axes) {
        const auto exp_size = std::accumulate(axes.cbegin(), axes.cend(), m.size(), [this](size_t i, int64_t axis) {
            m.emplace(static_cast<size_t>(axis), i);
            return ++i;
        });

        is_valid = exp_size == m.size();
    }

    void generate_n(size_t n) {
        n += m.size();
        for (size_t i = m.size(); i < n; ++i) {
            m.emplace(i, i);
        }
        is_valid = m.size() == n;
    }
};
}  // namespace slice

namespace v8 {

template <class T, class TRShape = result_shape_t<T>>
std::vector<TRShape> shape_infer(const Slice* op,
                                 const std::vector<T>& input_shapes,
                                 const ITensorAccessor& ta = make_tensor_accessor()) {
    using DimType = typename T::value_type;

    const auto& num_of_inputs = input_shapes.size();

    NODE_VALIDATION_CHECK(op,
                          num_of_inputs == 4 || num_of_inputs == 5,
                          "Slice has to have 4 or 5 inputs. Got: ",
                          num_of_inputs);

    const auto& input_shape = input_shapes[0];
    const auto& input_rank = input_shape.rank();

    // it is not possible to define output shape if input data shape rank is undefined
    // even if lengths of begin, end, or strides are defined
    if (input_rank.is_dynamic()) {
        return {PartialShape::dynamic()};
    } else {
        NODE_SHAPE_INFER_CHECK(op, input_shapes, input_rank.get_length() > 0, "Slice `data` input can't be a scalar.");
    }

    for (size_t i = 1; i < input_shapes.size(); ++i) {
        const auto& shape = input_shapes[i];
        const auto& shape_rank = shape.rank();
        NODE_VALIDATION_CHECK(op,
                              shape_rank.compatible(1),
                              "Slice `",
                              slice::shape_names[i - 1],
                              "` input must be a 1D tensor. Got rank: ",
                              shape_rank);

        if (input_rank.is_static()) {
            NODE_VALIDATION_CHECK(
                op,
                shape_rank.is_dynamic() || static_cast<int64_t>(shape[0].get_min_length()) <= input_rank.get_length(),
                "Slice `",
                slice::shape_names[i - 1],
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

    auto output_shapes = std::vector<TRShape>(1);

    // compute constant values of begin, end, and strides if possible
    const auto start = get_input_bounds<TRShape, int64_t>(op, 1, ta);
    const auto stop = get_input_bounds<TRShape, int64_t>(op, 2, ta);
    const auto steps = get_input_const_data_as<TRShape, int64_t>(op, 3, ta);

    slice::AxesMap axes_map;
    if (input_shapes.size() > 4) {
        NODE_VALIDATION_CHECK(op,
                              input_shapes[4].compatible(start_shape),
                              "Slice `axes` input must have compatible shape with `start`, `stop`, `step` inputs.");

        if (auto axes = get_input_const_data_as<TRShape, int64_t>(op, 4, ta)) {
            ov::util::try_normalize_axes(*axes, input_shape.rank(), *op);
            axes_map.add(*axes);
            NODE_VALIDATION_CHECK(op, axes_map.is_valid, "Slice values in `axes` input must be unique.");
        }
    } else if (start) {
        axes_map.generate_n(start->size());
    }

    auto axis_it = axes_map.m.cbegin();

    auto& out = output_shapes.front();
    out.resize(0);
    out.reserve(input_shape.size());
    for (size_t dim_idx = 0; dim_idx < input_shape.size(); ++dim_idx) {
        const DimType& input_dim = input_shape[dim_idx];

        if (axes_map.is_valid && (axis_it != axes_map.m.cend()) && (axis_it->first == dim_idx)) {
            const auto& i = axis_it->second;

            if (start && stop && steps) {
                const auto& step = (*steps)[i];
                NODE_VALIDATION_CHECK(op, step != 0, "Step must be non-zero");
                out.push_back(slice::make_dim(input_dim, (*start)[i], (*stop)[i], step));
            } else {
                out.emplace_back(0, input_dim.get_max_length());
            }

            auto& last_dim = out[out.size() - 1];
            if (std::is_same<DimType, ov::Dimension>::value &&
                (last_dim == input_dim && last_dim != Dimension::dynamic())) {
                // for equal ov::Dimension do merge to get input label (always success)
                DimType::merge(last_dim, last_dim, input_dim);
            }
            ++axis_it;
        } else if (axes_map.is_valid) {
            // dimension not on axes list, no change
            out.push_back(input_dim);
        } else {
            // axes are unknow so any dimension can be sliced
            out.emplace_back(0, input_dim.get_max_length());
        }
    }
    return output_shapes;
}
}  // namespace v8
}  // namespace op
}  // namespace ov
