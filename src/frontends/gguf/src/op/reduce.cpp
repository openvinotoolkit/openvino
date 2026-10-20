// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <numeric>
#include <vector>

#include "node_context.hpp"
#include "op_table.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// GGML_OP_MEAN: mean of every row (ggml dim 0, the innermost axis), keeping the reduced axis.
OutputVector translate_mean(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {-1});
    auto res = std::make_shared<ov::op::v1::ReduceMean>(context.get_input(0), axis, true);
    return rename_outputs_with_suffix({res}, context.get_name());
}

// GGML_OP_SUM: sum of all elements, as a tensor of the operand's rank with every extent 1.
OutputVector translate_sum(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    const auto input = context.get_input(0);
    const auto rank = input.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "SUM requires a static operand rank");
    std::vector<int64_t> axes(static_cast<size_t>(rank.get_length()));
    std::iota(axes.begin(), axes.end(), 0);
    auto res = std::make_shared<ov::op::v1::ReduceSum>(input,
                                                       ov::op::v0::Constant::create(ov::element::i64, {axes.size()}, axes),
                                                       true);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
