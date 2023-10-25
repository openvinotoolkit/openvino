// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/round.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_round(const NodeContext& context) {
    // aten::round(Tensor self) -> Tensor
    // aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // aten::round.int(int a) -> float
    // aten::round.float(float a) -> float
    // aten::round.Scalar(Scalar a) -> Scalar
    num_inputs_check(context, 1, 2);
    auto data = context.get_input(0);
    auto data_rank = data.get_partial_shape().rank();
    auto is_scalar = data_rank.is_static() && data_rank.get_length() == 0;
    auto is_integer = !data.get_element_type().is_dynamic() && data.get_element_type().is_integral();
    if (is_scalar && is_integer) {
        data = context.mark_node(std::make_shared<v0::Convert>(data, element::f32));
    }
    auto res = context.mark_node(std::make_shared<v5::Round>(data, v5::Round::RoundMode::HALF_TO_EVEN));
    if (!context.input_is_none(1)) {
        context.mutate_input(1, res);
    }
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
