// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/org.openvinotoolkit/group_norm.hpp"

#include "onnx_import/core/node.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/squeeze.hpp"

using namespace ov::op;

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector group_norm(const Node& node) {
    auto inputs = node.get_ng_inputs();
    FRONT_END_GENERAL_CHECK(inputs.size() == 3,
                            "Invalid number of inputs. Expected 3, actual " + std::to_string(inputs.size()));

    auto data = inputs[0];
    auto scale = inputs[1];
    auto bias = inputs[2];

    size_t num_groups = static_cast<size_t>(node.get_attribute_value<int64_t>("num_groups"));
    float eps = node.get_attribute_value<float>("eps", 1e-6f);

    if (!scale.get_partial_shape().rank().compatible(1)) {
        scale = std::make_shared<v0::Squeeze>(scale);
    }
    if (!bias.get_partial_shape().rank().compatible(1)) {
        bias = std::make_shared<v0::Squeeze>(bias);
    }

    return {std::make_shared<v12::GroupNormalization>(data, scale, bias, num_groups, eps)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
