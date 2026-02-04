// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/transpose.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector group_norm(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 3,
                            "Invalid number of inputs. Expected 3, actual " + std::to_string(inputs.size()));

    auto data = inputs[0];   // Shape [N, C, ...]
    auto scale = inputs[1];  // Shape [C]
    auto bias = inputs[2];   // Shape [C]

    const auto eps = node.get_attribute_value<float>("epsilon", 1e-05f);
    const auto num_groups = node.get_attribute_value<int64_t>("groups");
    const auto activation = node.get_attribute_value<int64_t>("activation");
    const auto channels_last = node.get_attribute_value<int64_t>("channels_last", 1);

    FRONT_END_GENERAL_CHECK(activation == 0 || activation == 1,
                            "Expected activation to be either 0 or 1, actual value is: " + std::to_string(activation));

    FRONT_END_GENERAL_CHECK(
        channels_last == 0 || channels_last == 1,
        "Expected channels_last to be either 0 or 1, actual value is: " + std::to_string(channels_last));

    if (!scale.get_partial_shape().rank().compatible(1)) {
        scale = std::make_shared<v0::Squeeze>(scale);
    }
    if (!bias.get_partial_shape().rank().compatible(1)) {
        bias = std::make_shared<v0::Squeeze>(bias);
    }

    if (channels_last == 1) {
        // Transpose from NHWC to NCHW
        auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 3, 1, 2});
        data = std::make_shared<v1::Transpose>(data, perm);
    }

    std::shared_ptr<ov::Node> group_norm =
        std::make_shared<v12::GroupNormalization>(data, scale, bias, num_groups, eps);

    if (channels_last == 1) {
        // Transpose back from NCHW to NHWC
        auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 3, 1});
        group_norm = std::make_shared<v1::Transpose>(group_norm, perm);
    }

    if (activation == 1) {
        return {std::make_shared<v4::Swish>(group_norm)};
    }

    return {group_norm};
}
ONNX_OP("GroupNorm", OPSET_SINCE(1), com_microsoft::opset_1::group_norm, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
