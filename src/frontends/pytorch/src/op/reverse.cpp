// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/frontend/pytorch/op_table.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace ov::op;
using namespace ov::frontend;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reverse(const NodeContext& context) {
    // aten::reverse.list(Tensor[] l) -> Tensor[]
    FRONT_END_OP_CONVERSION_CHECK(context.get_input_size() == 1, "Expected 1 input to aten::reverse");

    auto input_seq = context.get_input(0);  // Tensor[] (a list)

    // OpenVINO does not have direct list reversal — handle list of tensors
    // Convert sequence of tensors to a stack, reverse, and return

    // Convert list to a single tensor via Concat (assuming 1D scalars in our use case)
    auto axis = 0;
    auto list_tensor = context.mark_node(std::make_shared<v0::Concat>(context.get_inputs(), axis));

    // Create indices for reversed order
    auto input_size = context.get_input_size();
    auto size = context.mark_node(op::v0::Constant::create(ov::element::i64, {}, {static_cast<int64_t>(input_size)}));

    auto range = context.mark_node(op::v0::Constant::create(ov::element::i64, {input_size}, {}));
    std::vector<int64_t> indices(input_size);
    std::iota(indices.begin(), indices.end(), 0);
    std::reverse(indices.begin(), indices.end());

    auto reversed_indices = context.mark_node(op::v0::Constant::create(ov::element::i64, {input_size}, indices));

    // Gather elements in reverse
    auto dim = context.mark_node(op::v0::Constant::create(ov::element::i64, {}, {0}));
    auto reversed = context.mark_node(std::make_shared<v8::Gather>(list_tensor, reversed_indices, dim));

    // Done
    return {reversed};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
