// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/rdft.hpp>

#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

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

template <class T>
void rdft_shape_infer(const ov::op::v9::RDFT* op,
                      const std::vector<T>& input_shapes,
                      std::vector<T>& output_shapes,
                      const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    using DimType = typename std::iterator_traits<typename T::iterator>::value_type;
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 2 || input_shapes.size() == 3) && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    const auto& axes_shape = input_shapes[1];
    auto& output_shape = output_shapes[0];
    std::vector<int64_t> axes;
    bool axes_are_known = get_data_as_int64<T>(1, op, axes, constant_data);

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.size();
        NODE_VALIDATION_CHECK(op,
                              input_rank >= 1,
                              "The input rank must be greater or equal to 1. Got input rank: ",
                              input_rank);

        if (axes_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  input_rank >= static_cast<int64_t>(axes_shape[0].get_length()),
                                  "The input rank must be greater than or equal to the number of RDFT op axes. Got "
                                  "input rank: ",
                                  input_rank,
                                  ", number of axes: ",
                                  axes_shape[0].get_length());
        }

        // RDFT operation supports for negative axes to transform. More precisely, according to
        // the RDFT operation specification, axes should be integers from -r to (r - 1)
        // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis 'r+ a'.
        if (axes_shape.rank().is_static() && axes_are_known) {
            for (int64_t& axis : axes) {
                if (axis < 0) {
                    axis += input_rank;
                }
            }

            ov::AxisSet axes_set;
            for (const auto& axis : axes) {
                axes_set.insert(static_cast<size_t>(axis));
            }

            NODE_VALIDATION_CHECK(op, axes.size() == axes_set.size(), "RDFT op axes must be unique.");
        }
    }

    NODE_VALIDATION_CHECK(op, axes_shape.rank().compatible(1), "RDFT op axes input must be 1D tensor.");

    if (input_shapes.size() == 3) {
        const auto& signal_size_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              signal_size_shape.rank().compatible(1),
                              "RDFT op signal size input must be 1D tensor. Got signal: ",
                              signal_size_shape);

        if (axes_shape.is_static() && signal_size_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  axes_shape[0].compatible(signal_size_shape[0]),
                                  "Sizes of inputs 'axes' and 'signal_size' must be equal. Got "
                                  "size of 'axes': ",
                                  axes_shape[0],
                                  "size of 'signal_size': ",
                                  signal_size_shape[0]);
        }
    }

    if (input_shape.rank().is_dynamic()) {
        output_shape = ov::PartialShape();
        return;
    }

    output_shape = input_shape;
    output_shape.push_back(DimType(2));

    const auto input_rank = input_shape.size();

    if (axes_shape.rank().is_dynamic() || !axes_are_known) {
        for (int64_t i = 0; i < input_rank; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
        return;
    }

    if (input_shapes.size() == 2) {
        for (int64_t axis : axes) {
            output_shape[axis] = get_rdft_output_dimension(input_shape[axis]);
        }
        return;
    }

    const auto& signal_size_shape = input_shapes[2];
    std::vector<int64_t> signal_size;
    bool status_signal_size = get_data_as_int64<T>(2, op, signal_size, constant_data);

    if (signal_size_shape.rank().is_dynamic() || !status_signal_size) {
        for (int64_t axis : axes) {
            output_shape[axis] = ov::Dimension::dynamic();
        }
        return;
    }

    size_t num_of_axes = axes.size();
    for (size_t i = 0; i < num_of_axes; ++i) {
        const int64_t current_axis = axes[i];
        if (signal_size[i] != -1) {
            output_shape[current_axis] = get_rdft_output_dimension(DimType(signal_size[i]));
        } else {
            output_shape[current_axis] = get_rdft_output_dimension(input_shape[current_axis]);
        }
    }
}
