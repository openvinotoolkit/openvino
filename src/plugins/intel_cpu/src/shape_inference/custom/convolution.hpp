// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include "convolution_shape_inference_util.hpp"
#include "cpu_types.h"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/static_dimension.hpp"
#include "utils.hpp"

#pragma once
namespace ov {

template <>
struct result_shape<std::vector<size_t>> {
    using type = std::vector<size_t>;
};

namespace intel_cpu {
namespace node {

template <class TDim>
constexpr auto dilated(const TDim& dim, const TDim dilation) -> TDim {
    return (dim - 1) * dilation + 1;
}

template <class TShape, class U, class V, class TRShape = result_shape_t<TShape>>
std::vector<TRShape> convolution_auto_pad_shape_infer(const std::vector<TShape>& input_shapes,
                                                      const std::vector<U>& strides,
                                                      const std::vector<U>& dilations,
                                                      const std::vector<V>& pads_begin,
                                                      const std::vector<V>& pads_end,
                                                      bool auto_padding,
                                                      bool isGrouped = false) {
    assert(input_shapes.size() >= 2);
    const auto& data_shape = input_shapes[0];
    assert(data_shape.size() >= 3);
    const auto& filters_shape = input_shapes[1];
    assert(filters_shape.size() >= 3);

    const auto data_rank = data_shape.size();
    constexpr int spatial_offset = 2;
    const auto num_spatial = data_rank - spatial_offset;

    // {N, C_OUT, Spatial(1 / 2 / 3)}
    VectorDims output_shape;
    output_shape.reserve(spatial_offset + num_spatial);
    // {N, C_OUT, ...}
    auto N = data_shape[0];
    output_shape.emplace_back(N);
    auto CO = isGrouped ? filters_shape[0] * filters_shape[1] : filters_shape[0];
    output_shape.emplace_back(CO);

    const auto spatial_num = strides.size();

    const auto& d_shape = data_shape;
    auto data_dim_it = d_shape.cend() - spatial_num;

    const auto ceil_div = [](const auto& x, const auto& y) {
        assert(y > 0);
        return (x == 0 ? 0 : (1 + (x - 1) / y));
    };

    if (auto_padding) {
        std::transform(data_dim_it, d_shape.cend(), strides.cbegin(), std::back_inserter(output_shape), ceil_div);
    } else {
        const auto& f_shape = filters_shape;
        auto filters_dim = f_shape.cend() - spatial_num;

        using TDim = typename TShape::value_type;
        for (size_t i = 0; i < spatial_num; ++i, ++data_dim_it, ++filters_dim) {
            TDim dim = *data_dim_it + pads_begin[i] + pads_end[i];
            const TDim filter_dilated = dilated(*filters_dim, dilations[i]);

            dim = (dim - filter_dilated) / strides[i];
            dim += 1;

            if constexpr (std::is_same_v<TDim, ov::intel_cpu::StaticDimension>) {
                output_shape.push_back(dim.get_length());
            } else {
                output_shape.push_back(dim);
            }
        }
    }

    return std::vector<TRShape>{output_shape};
}

}  // namespace node
}  // namespace intel_cpu
}  // namespace ov
