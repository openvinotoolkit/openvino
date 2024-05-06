// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_derive_index(const NodeContext& context) {
    // aten::__derive_index(int index, int start, int step) -> int
    num_inputs_check(context, 3, 3);
    auto index = context.get_input(0);
    // TODO: figure out why we get segmentation fault if using i64 here
    auto start = get_input_as_i32(context, 1);
    auto step = get_input_as_i32(context, 2);
    auto index_i32 = context.mark_node(std::make_shared<v0::Convert>(index, ov::element::i32));
    auto index_step = context.mark_node(std::make_shared<v1::Multiply>(index_i32, step));
    auto index_res = context.mark_node(std::make_shared<v1::Add>(start, index_step));
    return {context.mark_node(std::make_shared<v1::ConvertLike>(index_res, index))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
