// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_logical_and.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_all(const NodeContext& context) {
    // aten::all(Tensor self) -> Tensor
    // aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
    // aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
    // aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
    // aten::all.int(int[] self) -> bool
    // aten::all.float(float[] self) -> bool
    // aten::all.bool(bool[] self) -> bool
    num_inputs_check(context, 1, 4);
    auto input_tensor = context.get_input(0);

    auto num_inputs = context.get_input_size();
    size_t out_id;

    element::Type output_dtype = element::boolean;
    if (input_tensor.get_element_type() == element::u8) {
        output_dtype = element::u8;
    }

    bool keep_dims = false;
    ov::Output<ov::Node> axes;
    if (num_inputs < 3) {
        axes = get_axes_range(context, 0);
        out_id = 1;
    } else {
        const auto dim = context.const_input<int64_t>(1);
        axes = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {dim}));
        if (!context.input_is_none(2)) {
            keep_dims = context.const_input<bool>(2);
        }
        out_id = 3;
    }
    if (input_tensor.get_element_type() != element::boolean) {
        input_tensor = context.mark_node(std::make_shared<v0::Convert>(input_tensor, element::boolean));
    }

    const auto all_nonzero = context.mark_node(std::make_shared<v1::ReduceLogicalAnd>(input_tensor, axes, keep_dims));
    auto result = context.mark_node(std::make_shared<v0::Convert>(all_nonzero, output_dtype));
    if (!context.input_is_none(out_id)) {
        context.mutate_input(out_id, result);
    }
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
