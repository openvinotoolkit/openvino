// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/slice.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector identity(const ov::frontend::onnx::Node& node) {
    // This operator will be optimized out in EliminateSlice pass
    // Need this to avoid data not being copied out when Identity connects from input to result
    // in some cases like:
    //   Input->Identity->Result
    auto input = node.get_ov_inputs().at(0);
    const auto& start = v0::Constant::create(element::i32, {1}, {0});
    const auto& end = v0::Constant::create(element::i32, {1}, {std::numeric_limits<int32_t>::max()});
    const auto& step = v0::Constant::create(element::i32, {1}, {1});
    return {std::make_shared<v8::Slice>(input, start, end, step)};
}
ONNX_OP("Identity", OPSET_SINCE(1), ai_onnx::opset_1::identity);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
