// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/packed_sequence.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_pack_padded_sequence(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto seq = context.get_input(0);
    auto lengths = context.get_input(1);
    const auto batch_first = context.const_input<bool>(2);

    const auto order_102 = v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
    if (batch_first)
        seq = context.mark_node(std::make_shared<v1::Transpose>(seq, order_102));
    return context.mark_node(std::make_shared<PackPadded>(seq, lengths))->outputs();
};

OutputVector translate_pad_packed_sequence(const NodeContext& context) {
    // aten::_pad_packed_sequence with schema aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool
    // batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
    num_inputs_check(context, 3, 5);
    auto seq = context.get_input(0);
    auto lengths = context.get_input(1);
    const auto batch_first = context.const_input<bool>(2);
    auto pad_packed = context.mark_node(std::make_shared<PadPacked>(seq, lengths));
    seq = pad_packed->output(0);
    lengths = pad_packed->output(1);
    const auto order_102 = v0::Constant::create(element::i32, Shape{3}, {1, 0, 2});
    if (batch_first)
        seq = context.mark_node(std::make_shared<v1::Transpose>(seq, order_102));
    return {seq, lengths};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
