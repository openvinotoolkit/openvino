// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/operator_set.hpp"
#include "openvino/op/minimum.hpp"
#include "utils/variadic.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector min(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node, ov::op::AutoBroadcastType::NONE);
}

static bool registered = register_translator("Min", VersionRange{1, 7}, min);
}  // namespace set_1

namespace set_8 {
ov::OutputVector min(const ov::frontend::onnx::Node& node) {
    return variadic::make_ng_variadic_op<ov::op::v1::Minimum>(node);
}

static bool registered = register_translator("Min", VersionRange::since(8), min);
}  // namespace set_8
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
