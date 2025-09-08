// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/topk.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_argmax(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> input = context.get_input(0);
    auto axis_val = context.const_named_param<int64_t>("axes");
    auto axis = context.const_named_param<std::shared_ptr<v0::Constant>>("axes");
    auto dtype = convert_dtype(context.const_named_param<int64_t>("index_dtype"));

    auto k = std::make_shared<v0::Constant>(element::i64, Shape{}, 1);
    auto topk = std::make_shared<v11::TopK>(input,
                                            k,
                                            axis_val,
                                            v11::TopK::Mode::MAX,
                                            v1::TopK::SortType::SORT_VALUES,
                                            dtype,
                                            true);
    auto indices = topk->output(1);

    auto res = std::make_shared<v0::Squeeze>(indices, axis);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov