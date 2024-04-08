// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "core/node.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
inline ov::OutputVector identity(const ov::frontend::onnx::Node& node) {
    ov::OutputVector outputs = node.get_ov_inputs();
    for (auto& out : outputs) {
        common::mark_as_optimized_out(out);
    }
    return outputs;
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
