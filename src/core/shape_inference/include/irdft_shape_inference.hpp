// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/irdft.hpp>

#include "openvino/core/axis_vector.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
template <class T>
void irdft_shape_infer(const ov::op::v9::IRDFT* op,
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
                              input_rank >= 2,
                              "The input rank must be greater or equal to 2. Got input rank: ",
                              input_rank);

        NODE_VALIDATION_CHECK(op,
                              input_shape[input_rank - 1].compatible(2),
                              "The last dimension of input data must be 2. Got: ",
                              input_shape[input_rank - 1]);

        if (axes_shape.is_static()) {
            NODE_VALIDATION_CHECK(op,
                                  input_rank >= static_cast<int64_t>(axes_shape[0].get_length() + 1),
                                  "The input rank must be greater than number of IRDFT op axes. Got "
                                  "input rank: ",
                                  input_rank,
                                  ", number of axes: ",
                                  axes_shape[0].get_length());
        }

        // IRDFT operation supports negative axes to transform. More precisely, according to
        // the IRDFT operation specification, axes should be integers from -(r - 1) to (r - 2)
        // inclusively, where r = rank(data). A negative axis 'a' is interpreted as an axis
        // 'r - 1 + a'. The reason is the following: real input tensor of the shape
        // [n_0, ..., n_{r - 1}, 2] is interpreted as a complex tensor with the shape
        // [n_0, ..., n_{r - 1}].
        if (axes_shape.rank().is_static() && axes_are_known) {
            for (int64_t& axis : axes) {
                if (axis < 0) {
                    axis += input_rank - 1;
                }
            }

            ov::AxisSet axes_set;
            for (const auto& axis : axes) {
                axes_set.insert(static_cast<size_t>(axis));
            }

            NODE_VALIDATION_CHECK(op, axes.size() == axes_set.size(), "IRDFT op axes must be unique.");

            NODE_VALIDATION_CHECK(op,
                                  std::find(axes.begin(), axes.end(), input_rank - 1) == axes.end(),
                                  "IRDFT op axes cannot contain the last axis.");
        }
    }

    NODE_VALIDATION_CHECK(op, axes_shape.rank().compatible(1), "IRDFT op axes input must be 1D tensor.");

    if (input_shapes.size() == 3) {
        const auto& signal_size_shape = input_shapes[2];
        NODE_VALIDATION_CHECK(op,
                              signal_size_shape.rank().compatible(1),
                              "IRDFT op signal size input must be 1D tensor. Got signal: ",
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

    const auto input_rank = input_shape.size();

    output_shape = input_shape;
    output_shape.resize(input_rank - 1);

    if (axes_shape.rank().is_dynamic() || !axes_are_known) {
        for (int64_t i = 0; i < input_rank - 1; ++i) {
            output_shape[i] = ov::Dimension::dynamic();
        }
        return;
    }

    const auto last_axis = axes.back();

    if (input_shapes.size() == 2) {
        output_shape[last_axis] = DimType(2) * (input_shape[last_axis] - DimType(1));
        return;
    }

    const auto& signal_size_shape = input_shapes[2];
    std::vector<int64_t> signal_size;
    bool status_signal_size = get_data_as_int64<T>(2, op, signal_size, constant_data);

    if (signal_size_shape.rank().is_dynamic() || !status_signal_size) {
        output_shape[last_axis] = ov::Dimension::dynamic();
        return;
    }

    size_t num_of_axes = axes.size();
    for (size_t i = 0; i < num_of_axes; ++i) {
        if (signal_size[i] != -1) {
            output_shape[axes[i]] = DimType(signal_size[i]);
        }
    }
    if (signal_size.back() == -1) {
        output_shape[last_axis] = DimType(2) * (input_shape[last_axis] - DimType(1));
    }
}
}  // namespace util
}  // namespace op
}  // namespace ov
