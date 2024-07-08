// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/group_normalization.hpp"
#include "openvino/op/squeeze.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector group_norm(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
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

static bool register_multiple_translators(void) {
    ONNX_OP_M("ExperimentalDetectronGroupNorm",
              OPSET_SINCE(1),
              org_openvinotoolkit::opset_1::group_norm,
              OPENVINO_ONNX_DOMAIN);
    ONNX_OP_M("GroupNorm", OPSET_SINCE(1), org_openvinotoolkit::opset_1::group_norm, OPENVINO_ONNX_DOMAIN);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
