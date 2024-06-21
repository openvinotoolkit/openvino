// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/scatter_nd_update.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector scatter_nd(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ov_inputs{node.get_ov_inputs()};
    auto data = ov_inputs.at(0);
    auto indices = ov_inputs.at(1);
    auto updates = ov_inputs.at(2);
    if (node.has_attribute("reduction")) {
        const auto reduction = node.get_attribute_value<std::string>("reduction", "none");
        CHECK_VALID_NODE(node,
                         reduction == "none",
                         "Unsupported value of attribute: `reduction`. Only `none` is supported, got:",
                         reduction);
    }

    return {std::make_shared<v3::ScatterNDUpdate>(data, indices, updates)};
}

static bool registered = register_translator("ScatterND", VersionRange::single_version_for_all_opsets(), scatter_nd);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
