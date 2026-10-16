// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx {
namespace opset_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    const auto& slope = ov_inputs.at(1);

    // PRelu-1/6 use the legacy (Caffe2-like) broadcasting: a slope of size 1 is shared across channels, i.e. a
    // rank-1 slope is aligned with the channel axis. OpenVINO's v0::PRelu already implements exactly this behavior,
    // so the inputs are forwarded without any reshaping.
    return {std::make_shared<v0::PRelu>(data, slope)};
}

ONNX_OP("PRelu", OPSET_RANGE(1, 6), ai_onnx::opset_1::prelu);
}  // namespace opset_1

namespace opset_7 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    auto slope = ov_inputs.at(1);

    const auto& data_rank = data.get_partial_shape().rank();
    const auto& slope_rank = slope.get_partial_shape().rank();

    // Since PRelu-7 ONNX broadcasts the slope onto the data tensor using unidirectional (numpy-style) broadcasting,
    // i.e. the slope shape is right-aligned with the trailing dimensions of the data. OpenVINO's v0::PRelu instead
    // treats a rank-1 slope as a per-channel parameter and aligns it with the channel axis (axis 1) when its length
    // matches the channel dimension. These interpretations disagree whenever the channel dimension is not the last
    // data dimension, producing results that differ from onnxruntime and the ONNX reference evaluator. To preserve
    // ONNX semantics, a rank-1 slope is explicitly reshaped to [1, ..., 1, C] so it is broadcast against the last
    // data dimension.
    const bool slope_rank_is_static_1d = slope_rank.is_static() && slope_rank.get_length() == 1;
    const bool data_rank_is_static_ge_2 = data_rank.is_static() && data_rank.get_length() >= 2;
    if (slope_rank_is_static_1d && data_rank_is_static_ge_2) {
        std::vector<int64_t> target_shape(data_rank.get_length(), 1);
        target_shape.back() = -1;
        const auto reshape_pattern =
            v0::Constant::create(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        slope = std::make_shared<v1::Reshape>(slope, reshape_pattern, false);
    } else if ((slope_rank.is_dynamic() || slope_rank_is_static_1d) &&
               (data_rank.is_dynamic() || data_rank_is_static_ge_2)) {
        // A rank-1 slope must be right-aligned even if its rank or the data rank is only known at runtime.
        // Explicitly pad its shape with leading ones to avoid v0::PRelu's channel-axis interpretation.
        const auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto data_shape = std::make_shared<v3::ShapeOf>(data, ov::element::i64);
        const auto runtime_data_rank = std::make_shared<v3::ShapeOf>(data_shape, ov::element::i64);
        const auto slope_shape = std::make_shared<v3::ShapeOf>(slope, ov::element::i64);
        const auto runtime_slope_rank = std::make_shared<v3::ShapeOf>(slope_shape, ov::element::i64);
        const auto leading_ones_count = std::make_shared<v1::Subtract>(runtime_data_rank, runtime_slope_rank);
        const auto leading_ones = std::make_shared<v3::Broadcast>(one, leading_ones_count);
        const auto reshape_pattern = std::make_shared<v0::Concat>(ov::OutputVector{leading_ones, slope_shape}, 0);
        slope = std::make_shared<v1::Reshape>(slope, reshape_pattern, false);
    }

    return {std::make_shared<v0::PRelu>(data, slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(7), ai_onnx::opset_7::prelu);
}  // namespace opset_7
}  // namespace ov::frontend::onnx::ai_onnx
