// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector identity(const ov::frontend::onnx::Node& node) {
    ov::OutputVector outputs = node.get_ov_inputs();
    for (auto& out : outputs) {
        common::mark_as_optimized_out(out);
    }
    return outputs;
}
ONNX_OP("Identity", OPSET_SINCE(1), ai_onnx::opset_1::identity);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
