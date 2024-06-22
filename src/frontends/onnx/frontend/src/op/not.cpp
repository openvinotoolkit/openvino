// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "openvino/op/logical_not.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector logical_not(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<ov::op::v1::LogicalNot>(node.get_ov_inputs().at(0))};
}

static bool registered = register_translator("Not", VersionRange::single_version_for_all_opsets(), logical_not);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
