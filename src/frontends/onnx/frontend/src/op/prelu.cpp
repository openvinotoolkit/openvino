// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/prelu.hpp"

#include <numeric>

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/unsqueeze.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    const auto& data = ov_inputs.at(0);
    auto slope = ov_inputs.at(1);
    const auto& data_rank = data.get_partial_shape().rank();
    const auto& slope_rank = slope.get_partial_shape().rank();
    if (data_rank.is_static() && slope_rank.is_static()) {
        const auto rank_diff = data_rank.get_length() - slope_rank.get_length();
        if (rank_diff > 0) {
            std::vector<int64_t> axes(rank_diff);
            std::iota(axes.begin(), axes.end(), 0);
            const auto axes_const =
                std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{axes.size()}, axes);
            slope = std::make_shared<v0::Unsqueeze>(slope, axes_const);
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
