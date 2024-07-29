// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/non_zero.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector non_zero(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v3::NonZero>(data, ov::element::i64)};
}

ONNX_OP("NonZero", OPSET_SINCE(1), ai_onnx::opset_1::non_zero);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
