// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/jax/node_context.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace jax {
namespace op {

OutputVector translate_erfc(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto x = context.get_input(0);

    // create const one of the same type as x
    auto const_one = create_same_type_const_scalar<int64_t>(x, 1);
    Output<Node> res = make_shared<v0::Erf>(x);
    res = make_shared<v1::Subtract>(const_one, res);
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
