// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/op/normalize_l2.hpp>
#include <ngraph/runtime/reference/sum.hpp>
#include "ngraph/runtime/reference/autobroadcast_binop.hpp"

namespace ngraph
{
    namespace runtime
    {
        namespace reference
        {
            template <typename T>
            void normalize_l2(const T* data,
                              T* out,
                              const Shape& data_shape,
                              const AxisSet& reduction_axes,
                              float eps,
                              op::EpsMode eps_mode)
            {
                AxisSet axes = reduction_axes;
                if (reduction_axes.empty())
                {
                    std::vector<size_t> axes_vec(data_shape.size());
                    std::iota(axes_vec.begin(), axes_vec.end(), 0);
                    axes = AxisSet(axes_vec);
                }
                std::vector<T> sqr_data(shape_size(data_shape));
                for (size_t i = 0; i < shape_size(data_shape); i++)
                {
                    sqr_data[i] = data[i] * data[i];
                }

                Shape reduce_shape = data_shape;
                for (auto axis : axes)
                {
                    reduce_shape[axis] = 1;
                }

                std::vector<T> sum_data(shape_size(reduce_shape));
                sum(sqr_data.data(), sum_data.data(), data_shape, axes, true);
                autobroadcast_binop(data,
                                    sum_data.data(),
                                    out,
                                    data_shape,
                                    reduce_shape,
                                    op::AutoBroadcastSpec(op::AutoBroadcastType::NUMPY),
                                    [&eps, &eps_mode](T x, T y) -> T {
                                        T arg = (eps_mode == op::EpsMode::ADD)
                                                    ? y + eps
                                                    : std::max(y, static_cast<T>(eps));
                                        return x / std::sqrt(arg);
                                    });
            }
        } // namespace reference
    }     // namespace runtime
} // namespace ngraph
