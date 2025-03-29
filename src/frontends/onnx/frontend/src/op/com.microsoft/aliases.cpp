// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
extern ov::OutputVector gather_nd(const ov::frontend::onnx::Node& node);
extern ov::OutputVector trilu(const ov::frontend::onnx::Node& node);
extern ov::OutputVector gelu(const ov::frontend::onnx::Node& node);
}  // namespace opset_1
namespace opset_13 {
extern ov::OutputVector dequantize_linear(const ov::frontend::onnx::Node& node);
extern ov::OutputVector quantize_linear(const ov::frontend::onnx::Node& node);
}  // namespace opset_13
}  // namespace ai_onnx

namespace com_microsoft {
namespace opset_1 {
bool register_multiple_translators(void) {
    ONNX_OP_M("DequantizeLinear", OPSET_SINCE(1), ai_onnx::opset_13::dequantize_linear, MICROSOFT_DOMAIN);
    ONNX_OP_M("GatherND", OPSET_SINCE(1), ai_onnx::opset_1::gather_nd, MICROSOFT_DOMAIN);
    ONNX_OP_M("Gelu", OPSET_SINCE(1), ai_onnx::opset_1::gelu, MICROSOFT_DOMAIN);
    ONNX_OP_M("QuantizeLinear", OPSET_SINCE(1), ai_onnx::opset_13::quantize_linear, MICROSOFT_DOMAIN);
    ONNX_OP_M("Trilu", OPSET_SINCE(1), ai_onnx::opset_1::trilu, MICROSOFT_DOMAIN);
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
