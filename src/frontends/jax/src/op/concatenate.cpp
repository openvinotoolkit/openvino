// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_concatenate(const NodeContext& context) {
    num_inputs_check(context, 1);
    auto num_inputs = static_cast<int>(context.get_input_size());
    int64_t axis = context.const_named_param<int64_t>("dimension");

    OutputVector inputs(num_inputs);
    for (int ind = 0; ind < num_inputs; ++ind) {
        inputs[ind] = context.get_input(ind);
    }
    Output<Node> res = make_shared<v0::Concat>(inputs, axis);

    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
