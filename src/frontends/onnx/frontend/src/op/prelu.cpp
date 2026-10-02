// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
using namespace ov::op;

namespace ov::frontend::onnx::ai_onnx::opset_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    auto slope = ov_inputs.at(1);

    const auto& data_rank = data.get_partial_shape().rank();
    const auto& slope_rank = slope.get_partial_shape().rank();

    // ONNX PRelu broadcasts the slope onto the data tensor using unidirectional (numpy-style) broadcasting,
    // i.e. the slope shape is right-aligned with the trailing dimensions of the data. OpenVINO's PRelu instead
    // treats a rank-1 slope as a per-channel parameter and aligns it with the channel axis (axis 1) when its
    // length matches the channel dimension. These interpretations disagree whenever the channel dimension is not
    // the last data dimension, producing results that differ from onnxruntime and the ONNX reference evaluator.
    // To preserve ONNX semantics, a rank-1 slope is explicitly reshaped to [1, ..., 1, C] so it is broadcast
    // against the last data dimension.
    if (slope_rank.is_static() && slope_rank.get_length() == 1 && data_rank.is_static() &&
        data_rank.get_length() >= 2) {
        std::vector<int64_t> target_shape(data_rank.get_length(), 1);
        target_shape.back() = -1;
        const auto reshape_pattern =
            v0::Constant::create(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
        slope = std::make_shared<v1::Reshape>(slope, reshape_pattern, false);
    }

    return {std::make_shared<v0::PRelu>(data, slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(1), ai_onnx::opset_1::prelu);
}  // namespace ov::frontend::onnx::ai_onnx::opset_1
