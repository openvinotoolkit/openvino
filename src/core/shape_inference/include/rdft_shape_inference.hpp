// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once
#include <openvino/op/rdft.hpp>

#include "openvino/core/axis_vector.hpp"
#include "rfft_common_validation.hpp"
#include "utils.hpp"

namespace ov {
namespace op {
namespace util {
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

    rfft_common_validation::shape_validation(op,
                                             input_shapes,
                                             axes,
                                             axes_are_known,
                                             rfft_common_validation::RFFTKind::Forward);

    if (input_shape.rank().is_dynamic()) {
        output_shape = ov::PartialShape::dynamic();
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

    const auto last_axis = axes.back();

    if (input_shapes.size() == 2) {
        output_shape[last_axis] = get_rdft_output_dimension(input_shape[last_axis]);
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
        const int64_t current_axis = axes[i];
        if (signal_size[i] != -1) {
            output_shape[current_axis] = DimType(signal_size[i]);
        }
    }
    output_shape[last_axis] = get_rdft_output_dimension(output_shape[last_axis]);
}
}  // namespace util
}  // namespace op
}  // namespace ov
