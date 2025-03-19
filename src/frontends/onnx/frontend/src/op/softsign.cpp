// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softsign.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector softsign(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v9::SoftSign>(node.get_ov_inputs().at(0))};
}
ONNX_OP("Softsign", OPSET_SINCE(1), ai_onnx::opset_1::softsign);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
