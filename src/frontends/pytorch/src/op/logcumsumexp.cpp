// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/exp.hpp"
#include "openvino/op/log.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_logcumsumexp(const NodeContext& context) {
    // aten::logcumsumexp(Tensor self, int dim) -> Tensor
    num_inputs_check(context, 2, 2);
    auto input = context.get_input(0);
    auto dim = context.get_input(1);

    auto max = context.mark_node(std::make_shared<v0::ReduceMax>(input,dim))

    // First compute exp(input)
    auto exp = context.mark_node(std::make_shared<v0::Exp>(input-max));
    
    // Then compute cumsum of the exponentials
    auto cumsum = context.mark_node(std::make_shared<v0::CumSum>(exp, dim));
    
    // Finally take log of the result
    auto log = max + context.mark_node(std::make_shared<v0::Log>(cumsum));
    
    return {log};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
