// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/erf.hpp"
#include "openvino/op/subtract.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_erfc(const NodeContext& context) {
    // aten::erf(Tensor self) -> Tensor
    // aten::erf.out(Tensor self, Tensor(!a) out) -> Tensor(!a)
    num_inputs_check(context, 1, 2);
    auto x = context.get_input(0);

    // create 'ones' to use to calculate complementary of Erf output
    auto ones = context.mark_node(make_shared<v0::Constant>(element::f32, Shape{}, 1.0f))->output(0);

    // align data types of input 'x' and ones
    align_eltwise_input_types(context, x, ones);

    // apply Erf to the input tensor 'x'
    auto y = context.mark_node(make_shared<v0::Erf>(x));

    y = context.mark_node(make_shared<v1::Subtract>(ones, y));

    if (!context.input_is_none(1)) {
        context.mutate_input(1, y);
    }
    return {y};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov