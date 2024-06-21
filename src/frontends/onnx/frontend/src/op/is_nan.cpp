// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/is_nan.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector is_nan(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v10::IsNaN>(data)};
}

static bool registered = register_translator("IsNaN", VersionRange::single_version_for_all_opsets(), is_nan);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
