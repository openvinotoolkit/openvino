// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convolution.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <unordered_map>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/except.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/op/util/convolution_base.hpp"
#include "shape_inference/shape_inference_cpu.hpp"
#include "shape_inference/shape_inference_status.hpp"
#include "utils/general_utils.h"

namespace ov::intel_cpu::node {

constexpr auto dilated(const size_t dim, const size_t dilation) -> size_t {
    return (dim - 1) * dilation + 1;
}

VectorDims convolution_shape_infer(const VectorDims& data_shape,
                                   const VectorDims& filters_shape,
                                   const std::vector<size_t>& strides,
                                   const std::vector<size_t>& dilations,
                                   const std::vector<ptrdiff_t>& pads_begin,
                                   const std::vector<ptrdiff_t>& pads_end,
                                   bool auto_padding,
                                   bool isGrouped) {
    OPENVINO_ASSERT(data_shape.size() >= 3, "At least 3D data shape is expected");
    OPENVINO_ASSERT(filters_shape.size() >= 3, "At least 3D filters shape is expected");
    const auto G = isGrouped ? filters_shape[0] : 1;
    const auto data_shape_IC = data_shape[1] / G;
    const auto filter_shape_IC = isGrouped ? filters_shape[2] : filters_shape[1];
    OPENVINO_ASSERT(data_shape_IC == filter_shape_IC, "Input and filter channels must match");

    const auto data_rank = data_shape.size();
    constexpr int spatial_offset = 2;
    const auto num_spatial = data_rank - spatial_offset;

    // {N, C_OUT, Spatial(1 / 2 / 3)}
    VectorDims output_shape;
    output_shape.reserve(spatial_offset + num_spatial);
    // {N, C_OUT, ...}
    const auto N = data_shape[0];
    output_shape.emplace_back(N);
    const auto CO = isGrouped ? filters_shape[0] * filters_shape[1] : filters_shape[0];
    output_shape.emplace_back(CO);

    const auto spatial_num = strides.size();

    auto data_dim_it = data_shape.cend() - spatial_num;

    const auto ceil_div = [](const auto& x, const auto& y) {
        assert(y > 0);
        return (x == 0 ? 0 : (1 + (x - 1) / y));
    };

    if (auto_padding) {
        std::transform(data_dim_it, data_shape.cend(), strides.cbegin(), std::back_inserter(output_shape), ceil_div);
    } else {
        auto filters_dim = filters_shape.cend() - spatial_num;

        for (size_t i = 0; i < spatial_num; ++i, ++data_dim_it, ++filters_dim) {
            auto dim = *data_dim_it + pads_begin[i] + pads_end[i];

            const auto f_dim = *filters_dim;
            const auto filter_dilated = (f_dim - 1) * dilations[i] + 1;

            dim = (dim - filter_dilated) / strides[i];
            dim += 1;

            output_shape.push_back(dim);
        }
    }

    return output_shape;
}

Result ConvolutionShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                                    const std::unordered_map<size_t, MemoryPtr>& /*data_dependency*/) {
    assert(input_shapes.size() >= 2);
    const auto& data_shape = input_shapes[0].get();
    const auto& filters_shape = input_shapes[1].get();

    auto output_shape = convolution_shape_infer(data_shape,
                                                filters_shape,
                                                m_strides,
                                                m_dilations,
                                                m_pads_begin,
                                                m_pads_end,
                                                m_auto_padding,
                                                m_isGrouped);

    return {{output_shape}, ShapeInferStatus::success};
}

ShapeInferPtr ConvolutionShapeInferFactory::makeShapeInfer() const {
    if (const auto convolution = ov::as_type_ptr<const ov::op::util::ConvolutionFwdPropBase>(m_op)) {
        const auto is_grouped = ov::is_type<const ov::op::v1::GroupConvolution>(convolution);
        return std::make_shared<ConvolutionShapeInfer>(
            convolution->get_strides(),
            convolution->get_dilations(),
            convolution->get_pads_begin(),
            convolution->get_pads_end(),
            any_of(convolution->get_auto_pad(), ov::op::PadType::SAME_LOWER, ov::op::PadType::SAME_UPPER),
            is_grouped);
    }

    OPENVINO_THROW("Unexpected operation type in the Convolution shape inference factory");
}

}  // namespace ov::intel_cpu::node
