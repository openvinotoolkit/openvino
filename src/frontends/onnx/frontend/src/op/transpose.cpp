// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "utils/reshape.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector transpose(const ov::frontend::onnx::Node& node) {
    ov::Output<ov::Node> data = node.get_ov_inputs().at(0);

    auto permute_axes = node.get_attribute_value<std::vector<std::size_t>>("perm", {});

    return {(permute_axes.empty()) ? ov::op::util::transpose(data) : ov::op::util::reorder_axes(data, permute_axes)};
}

ONNX_OP("Transpose", OPSET_SINCE(1), ai_onnx::opset_1::transpose);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
