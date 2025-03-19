// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

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
    auto const_axis = ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0});
    OutputVector unsqueezed_cases(num_inputs - 1);
    unsqueezed_cases.reserve(num_inputs - 1);
    for (int ind = 1; ind < num_inputs; ++ind) {
        auto case_input = context.get_input(ind);
        auto unsqueeze = std::make_shared<v0::Unsqueeze>(case_input, const_axis);
        unsqueezed_cases[ind - 1] = unsqueeze;
    }
    Output<Node> cases = std::make_shared<v0::Concat>(unsqueezed_cases, 0);
    which =
        std::make_shared<v0::Unsqueeze>(which,
                                        ov::op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}));
    Output<Node> result = std::make_shared<v6::GatherElements>(cases, which, 0);
    return {result};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov