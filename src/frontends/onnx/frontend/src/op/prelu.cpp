// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/constant.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto data = ov_inputs.at(0);
    auto slope = ov_inputs.at(1);

    const auto& data_pshape = data.get_partial_shape();
    const auto& slope_pshape = slope.get_partial_shape();

    // ONNX PRelu operator expects the slope tensor to be unidirectionally broadcastable to the input data tensor.
    // However, when the slope is a 1D tensor, it typically corresponds to the channel dimension (dim 1) of the input data
    // (assuming format NCHW or similar).
    // OpenVINO's PRelu operation uses standard numpy-style broadcasting (aligning to the last dimension) if the ranks don't match.
    // Therefore, if the slope is rank 1 and the data is rank >= 2, we explicitly reshape the slope to [1, C, 1, ..., 1]
    // to ensure it broadcasts correctly to the channel dimension of the input data.
    if (slope_pshape.rank().is_static() && slope_pshape.rank().get_length() == 1 && data_pshape.rank().is_static() &&
        data_pshape.rank().get_length() >= 2) {
        auto channel_dim = slope_pshape[0];
        if (channel_dim.is_static()) {
            std::vector<int64_t> target_shape(data_pshape.rank().get_length(), 1);
            target_shape[1] = channel_dim.get_length();
            auto reshape_const =
                v0::Constant::create(ov::element::i64, ov::Shape{target_shape.size()}, target_shape);
            slope = std::make_shared<v1::Reshape>(slope, reshape_const, false);
        }
    }

    return {std::make_shared<v0::PRelu>(data, slope)};
}

ONNX_OP("PRelu", OPSET_SINCE(1), ai_onnx::opset_1::prelu);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
