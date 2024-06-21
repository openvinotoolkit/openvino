// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector log(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v0::Log>(node.get_ov_inputs().at(0))};
}

static bool registered = register_translator("Log", VersionRange::single_version_for_all_opsets(), log);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
