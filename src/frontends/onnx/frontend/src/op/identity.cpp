// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/identity.hpp"

#include "core/operator_set.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
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
    ov::Output<ov::Node> input = node.get_ov_inputs().at(0);
    auto input_shape = input.get_partial_shape();

    ov::Output<ov::Node> input_order;
    ov::Output<ov::Node> output;
    if (input_shape.rank().is_dynamic() || input_shape.rank().get_length() == 0) {
        input_order = v0::Constant::create(element::i64, {0}, {0});
        input = std::make_shared<v1::Transpose>(input, input_order);
    } else {
        std::vector<int64_t> ref_values(input_shape.rank().get_length());
        std::iota(ref_values.begin(), ref_values.end(), 0);
        uint64_t rank_len = input_shape.rank().get_length();
        input_order = v0::Constant::create(element::i64, {rank_len}, ref_values);
    }
    output = std::make_shared<v1::Transpose>(input, input_order);

    return {output};
}
ONNX_OP("Identity", OPSET_SINCE(1), ai_onnx::opset_1::identity);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
