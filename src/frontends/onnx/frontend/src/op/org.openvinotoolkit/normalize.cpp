// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/normalize_l2.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/util/op_types.hpp"
#include "utils/common.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector normalize(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 2, "Invalid number of inputs");

    auto data = inputs[0];
    float eps = node.get_attribute_value<float>("eps", 0);
    int64_t across_spatial = node.get_attribute_value<int64_t>("across_spatial", 0);
    int64_t channel_shared = node.get_attribute_value<int64_t>("channel_shared", 0);

    std::shared_ptr<ov::Node> weights;
    if (channel_shared) {
        FRONT_END_GENERAL_CHECK(ov::op::util::is_constant(inputs[1].get_node()),
                                "Weights input must be a constant if channel_shared is set to 1");
        const auto& shape = inputs[1].get_partial_shape();
        FRONT_END_GENERAL_CHECK(shape.is_static() && shape.rank().get_length() == 1,
                                "Weights rank must be equal to 1 if channel_shared is set to 1");
        weights = inputs[1].get_node_shared_ptr();
    } else {
        std::vector<int64_t> weights_shape{1};
        const auto& data_shape = inputs[0].get_partial_shape();
        if (data_shape[1].is_static()) {
            weights_shape.push_back(data_shape[1].get_length());
        } else {
            weights_shape.push_back(0);
        }
        for (int64_t i = 2; i < data_shape.rank().get_length(); ++i) {
            weights_shape.push_back(1);
        }
        auto new_shape =
            std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{weights_shape.size()}, weights_shape);
        weights = std::make_shared<v1::Reshape>(inputs[1], new_shape, true);
    }

    std::shared_ptr<ov::Node> axes;
    if (!across_spatial) {
        axes = std::make_shared<v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    } else {
        axes = common::get_monotonic_range_along_node_rank(data, 1);
    }

    return {std::make_shared<v1::Multiply>(std::make_shared<v0::NormalizeL2>(data, axes, eps, ov::op::EpsMode::ADD),
                                           weights)};
}

ONNX_OP("Normalize", OPSET_SINCE(1), org_openvinotoolkit::opset_1::normalize, OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
