// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp";
#include "openvino/op/ops.hpp";
#include "utils.hpp";

using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_select_n(const NodeContext& context) {
    num_inputs_check(context, 2);
    auto num_inputs = static_cast<int>(context.get_input_size());
    Output<Node> which = context.get_input(0);
    if (which.get_element_type() == element::boolean) {
        which = std::make_shared<v0::Convert>(which, element::i32);  
    }
    OutputVector cases_vector(num_inputs - 1);
    for(int ind = 1; ind < num_inputs; ++ind) {
        cases_vector[ind - 1] = context.get_input(ind);
    }

    Output<Node> cases = std::make_shared<v0::Concat>(cases_vector, 0);
    auto which_shape = which.get_shape();
    std::vector<int64_t> cases_reshape_shape = {num_inputs-1,which_shape[0]};
    std::vector<int64_t> which_reshape_shape = {1,which_shape[0]};
    
    cases = std::make_shared<v1::Reshape>(cases, ov::op::v0::Constant::create(element::i64, Shape{2}, cases_reshape_shape), false);
    which = std::make_shared<v1::Reshape>(which, ov::op::v0::Constant::create(element::i64, Shape{2}, which_reshape_shape), false);
    Output<Node> result = std::make_shared<v6::GatherElements>(cases, which, 0);
    return {result};

};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov