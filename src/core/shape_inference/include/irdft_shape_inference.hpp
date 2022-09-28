// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/irdft.hpp>

#include "openvino/core/axis_vector.hpp"
#include "rfft_common_validation.hpp"
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

    rfft_common_validation::shape_validation(op,
                                             input_shapes,
                                             axes,
                                             axes_are_known,
                                             rfft_common_validation::RFFTKind::Inverse);

    if (input_shape.rank().is_dynamic()) {
        output_shape = ov::PartialShape::dynamic();
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
