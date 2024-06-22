// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "utils/common.hpp"

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector identity(const ov::frontend::onnx::Node& node) {
    ov::OutputVector outputs = node.get_ov_inputs();
    for (auto& out : outputs) {
        common::mark_as_optimized_out(out);
    }
    return outputs;
}
static bool registered = register_translator("Identity", VersionRange::single_version_for_all_opsets(), identity);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
