// Copyright (C) 2018-2024 Intel Corporation
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
static bool register_multiple_translators(void) {
    register_translator("DequantizeLinear",
                        VersionRange::single_version_for_all_opsets(),
                        ai_onnx::opset_op:: ::dequantize_linear,
                        "com.microsoft");
    register_translator("GatherND",
                        VersionRange::single_version_for_all_opsets(),
                        ai_onnx::opset_op:: ::gather_nd,
                        "com.microsoft");
    register_translator("Gelu",
                        VersionRange::single_version_for_all_opsets(),
                        ai_onnx::opset_op:: ::gelu,
                        "com.microsoft");
    register_translator("QuantizeLinear",
                        VersionRange::single_version_for_all_opsets(),
                        ai_onnx::opset_op:: ::quantize_linear,
                        "com.microsoft");
    register_translator("Trilu",
                        VersionRange::single_version_for_all_opsets(),
                        ai_onnx::opset_op:: ::trilu,
                        "com.microsoft");
    return true;
}

static bool registered = register_multiple_translators();
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
