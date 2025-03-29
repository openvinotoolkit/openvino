// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/reference/add.hpp"
#include "openvino/reference/divide.hpp"
#include "openvino/reference/multiply.hpp"
#include "openvino/reference/reduce_mean.hpp"
#include "openvino/reference/reduce_sum.hpp"
#include "openvino/reference/sqrt.hpp"
#include "openvino/reference/subtract.hpp"

namespace ov {
namespace reference {
template <typename T>
void mvn(const T* arg,
         T* out,
         const Shape& in_shape,
         const bool normalize_variance,
         const AxisSet& reduction_axes,
         const double eps) {
    auto reduced_shape = util::reduce_keep_dims(in_shape, reduction_axes);
    std::vector<T> tmp_buffer(shape_size(in_shape));
    reduce_mean(arg, tmp_buffer.data(), in_shape, reduction_axes);
    subtract(arg, tmp_buffer.data(), out, in_shape, reduced_shape, op::AutoBroadcastType::NUMPY);

    if (normalize_variance) {
        multiply(out, out, tmp_buffer.data(), shape_size(in_shape));
        std::vector<T> mean_value(shape_size(reduced_shape));
        reduce_mean(tmp_buffer.data(), mean_value.data(), in_shape, reduction_axes);

        add(mean_value.data(),
            std::vector<T>(shape_size(reduced_shape), static_cast<T>(eps)).data(),
            tmp_buffer.data(),
            reduced_shape,
            reduced_shape,
            op::AutoBroadcastType::NUMPY);
        sqrt(tmp_buffer.data(), tmp_buffer.data(), shape_size(reduced_shape));

        divide(out, tmp_buffer.data(), out, in_shape, reduced_shape, op::AutoBroadcastType::NUMPY, true);
    }
}

template <typename T>
void mvn_6(const T* arg,
           T* out,
           const Shape& in_shape,
           AxisSet reduction_axes,
           bool normalize_variance,
           double eps,
           op::MVNEpsMode eps_mode) {
    auto reduced_shape = util::reduce_keep_dims(in_shape, reduction_axes);
    std::vector<T> tmp_buffer(shape_size(in_shape));
    reduce_mean(arg, tmp_buffer.data(), in_shape, reduction_axes);
    subtract(arg, tmp_buffer.data(), out, in_shape, reduced_shape, op::AutoBroadcastType::NUMPY);

    if (normalize_variance) {
        multiply(out, out, tmp_buffer.data(), shape_size(in_shape));
        std::vector<T> mean_value(shape_size(reduced_shape));
        reduce_mean(tmp_buffer.data(), mean_value.data(), in_shape, reduction_axes);

        if (eps_mode == op::MVNEpsMode::INSIDE_SQRT) {
            add(mean_value.data(),
                std::vector<T>(shape_size(reduced_shape), static_cast<T>(eps)).data(),
                tmp_buffer.data(),
                reduced_shape,
                reduced_shape,
                op::AutoBroadcastType::NUMPY);
            sqrt(tmp_buffer.data(), tmp_buffer.data(), shape_size(reduced_shape));
        } else {
            sqrt(mean_value.data(), tmp_buffer.data(), shape_size(reduced_shape));
            add(tmp_buffer.data(),
                std::vector<T>(shape_size(reduced_shape), static_cast<T>(eps)).data(),
                tmp_buffer.data(),
                reduced_shape,
                reduced_shape,
                op::AutoBroadcastType::NUMPY);
        }

        divide(out, tmp_buffer.data(), out, in_shape, reduced_shape, op::AutoBroadcastType::NUMPY, true);
    }
}

template <typename T>
AxisSet mvn_6_reduction_axes(const ov::Tensor& axes_input, size_t rank) {
    T* a = axes_input.data<T>();
    auto v = std::vector<T>(a, a + axes_input.get_shape()[0]);
    std::vector<size_t> axes(v.size(), 0);
    for (size_t i = 0; i < v.size(); i++) {
        if (v[i] < 0) {
            OPENVINO_ASSERT(rank + v[i] >= 0, "Unexpected axis");
            axes[i] = (size_t)(rank + v[i]);
        } else {
            axes[i] = (size_t)(v[i]);
        }
    }
    return AxisSet(axes);
}

}  // namespace reference
}  // namespace ov
