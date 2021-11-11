// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/op/interpolate.hpp>

#include "utils.hpp"

namespace ov {
namespace op {

template <class T>
inline void set_dim(T& dim) {
    OPENVINO_UNREACHABLE("Cannot set Dimension to in-compatible type.");
}

template <>
inline void set_dim<ov::Dimension>(ov::Dimension& dim) {
    dim = ov::Dimension::dynamic();
};

template <class T>
inline void set_low_upper_bound(T& dim, int64_t lower_bound, int64_t upper_bound) {
    OPENVINO_UNREACHABLE("Cannot set Dimension to in-compatible type.");
}

template <>
inline void set_low_upper_bound<ov::Dimension>(ov::Dimension& dim, int64_t lower_bound, int64_t upper_bound) {
    dim = ov::Dimension(lower_bound, upper_bound);
};

namespace v0 {
template <class T>
void shape_infer(const Interpolate* op,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, input_shapes.size() == 2 && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;

    auto attrs = op->get_attrs();

    if (input_shape.rank().is_static()) {
        for (auto axis : attrs.axes) {
            NODE_VALIDATION_CHECK(op, static_cast<int64_t>(axis) < input_shape.rank().get_length());
        }

        std::vector<int64_t> out_shape;
        if (get_data_as_int64<T>(1, op, out_shape, constant_data)) {
            size_t i = 0;
            for (auto axis : attrs.axes) {
                output_shape[axis] = out_shape[i++];
            }
        } else {
            for (auto axis : attrs.axes) {
                set_dim(output_shape[axis]);
            }
        }
    }
}
}  // namespace v0

namespace v4 {
template <typename T>
std::vector<T> correct_pad(const std::vector<T>& p, size_t rank) {
    size_t pad_len = p.size();
    if (pad_len == rank) {
        return p;
    }

    std::vector<T> result;

    if (pad_len > rank) {
        result.insert(result.end(), p.begin(), p.begin() + rank);
    } else {
        result = p;
        result.insert(result.end(), rank - pad_len, T{});
    }

    return result;
}

template <typename T>
void correct_pads_attr(const Interpolate* op,
                  std::vector<size_t>& pads_begin,
                  std::vector<size_t>& pads_end,
                  const std::vector<T>& input_shapes) {
    auto attrs = op->get_attrs();
    auto input_shape = input_shapes[0];
    pads_begin = attrs.pads_begin;
    pads_end = attrs.pads_end;

    if (input_shape.rank().is_dynamic()) {
        return;
    }
    const auto input_rank = input_shape.rank().get_length();

    pads_begin = correct_pad(attrs.pads_begin, input_rank);
    pads_end = correct_pad(attrs.pads_end, input_rank);
}

inline int64_t multiply_bound_and_scale(int64_t bound, float scale) {
    if (bound == -1) {
        return bound;
    }
    return static_cast<int64_t>(static_cast<float>(bound) * scale);
}

template <typename T>
void infer_using_scales(T& output_shape,
                        const std::vector<int64_t>& axes,
                        const std::vector<float>& scales,
                        const T& padded_input_shape) {
    size_t i = 0;
    static constexpr float epsilon = 1.0e-6f;
    for (auto axis : axes) {
        NGRAPH_CHECK(axis < padded_input_shape.rank().get_length());
        const auto& current_dim = padded_input_shape[axis];
        float multiplier = scales[i] + epsilon;
        if (current_dim.is_static()) {
            output_shape[axis] = multiply_bound_and_scale(current_dim.get_length(), multiplier);
        } else {
            int64_t new_lower_bound = multiply_bound_and_scale(current_dim.get_min_length(), multiplier);
            int64_t new_upper_bound = multiply_bound_and_scale(current_dim.get_max_length(), multiplier);
            set_low_upper_bound(output_shape[axis], new_lower_bound, new_upper_bound);
        }
        ++i;
    }
}

template <typename T>
void infer_using_shapes(T& output_shape, const std::vector<int64_t>& axes, const std::vector<int64_t>& sizes) {
    size_t i = 0;
    for (auto axis : axes) {
        NGRAPH_CHECK(axis < output_shape.rank().get_length());
        output_shape[axis] = sizes[i++];
    }
}

template <class T>
void shape_infer(const Interpolate* op,
                 std::vector<size_t>& pads_begin,
                 std::vector<size_t>& pads_end,
                 const std::vector<T>& input_shapes,
                 std::vector<T>& output_shapes,
                 const std::map<size_t, std::shared_ptr<ngraph::runtime::HostTensor>>& constant_data = {}) {
    NODE_VALIDATION_CHECK(op, (input_shapes.size() == 3 || input_shapes.size() == 4) && output_shapes.size() == 1);

    const auto& input_shape = input_shapes[0];
    auto& output_shape = output_shapes[0];
    output_shape = input_shape;

    if (input_shape.rank().is_static()) {
        const auto input_rank = input_shape.rank().get_length();

        // Get axes
        std::vector<int64_t> axes;
        if (input_shapes.size() == 4 && !(get_data_as_int64<T>(3, op, axes, constant_data))) {
            for (size_t i = 0; i < input_rank; i++)
                set_dim(output_shape[i]);
            return;
        } else if (input_shapes.size() == 3) {
            axes.resize(input_rank);
            std::iota(axes.begin(), axes.end(), 0);
        }

        // Get padded input shape
        auto padded_input_shape = input_shape;
        for (int64_t i = 0; i < input_rank; ++i) {
            if (input_shape[i].is_static()) {
                auto new_length = pads_begin[i] + pads_end[i] + input_shape[i].get_length();
                padded_input_shape[i] = new_length;
            }
        }

        output_shape = padded_input_shape;

        auto attrs = op->get_attrs();

        if (attrs.shape_calculation_mode == Interpolate::ShapeCalcMode::SCALES) {
            std::vector<float> scales;
            if (get_data_as_float<T>(2, op, scales, constant_data)) {
                infer_using_scales(output_shape, axes, scales, padded_input_shape);
            } else {
                for (auto axis : axes) {
                    NGRAPH_CHECK(axis < input_rank);
                    set_dim(output_shape[axis]);
                }
            }
        } else {
            std::vector<int64_t> sizes;
            if (get_data_as_int64<T>(1, op, sizes, constant_data)) {
                infer_using_shapes(output_shape, axes, sizes);
            } else {
                for (auto axis : axes) {
                    NGRAPH_CHECK(axis < input_rank);
                    set_dim(output_shape[axis]);
                }
            }
        }
    }
}

}  // namespace v4
}  // namespace op
}  // namespace ov
