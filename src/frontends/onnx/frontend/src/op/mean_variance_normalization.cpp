// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/mvn.hpp"
#include "utils/common.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector mean_variance_normalization(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    bool across_channels = node.get_attribute_value<std::int64_t>("across_channels", 0);
    bool normalize_variance = node.get_attribute_value<std::int64_t>("normalize_variance", 1);

    return {std::make_shared<v0::MVN>(data, across_channels, normalize_variance)};
}

ONNX_OP("MeanVarianceNormalization", OPSET_RANGE(1, 8), ai_onnx::opset_1::mean_variance_normalization);
}  // namespace opset_1

namespace opset_9 {
ov::OutputVector mean_variance_normalization(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    auto axes = node.get_attribute_value<std::vector<std::int64_t>>("axes", {0, 2, 3});
    auto data_rank = data.get_partial_shape().rank();
    if (data_rank.is_static()) {
        for (auto&& axis : axes) {
            axis = common::normalize_axis(node.get_description(), axis, data_rank);
        }
    }
    auto const_axes = v0::Constant::create(ov::element::i64, ov::Shape{axes.size()}, axes);
    return {std::make_shared<v6::MVN>(data, const_axes, true, 1e-09f, ov::op::MVNEpsMode::OUTSIDE_SQRT)};
}

ONNX_OP("MeanVarianceNormalization", OPSET_SINCE(9), ai_onnx::opset_9::mean_variance_normalization);
}  // namespace opset_9
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
