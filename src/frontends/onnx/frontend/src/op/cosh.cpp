// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/cosh.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector cosh(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Cosh>(node.get_ov_inputs().at(0))};
}
static bool registered = register_translator("Cosh", VersionRange::single_version_for_all_opsets(), cosh);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
