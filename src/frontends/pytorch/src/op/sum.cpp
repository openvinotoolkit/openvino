// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_sum(const NodeContext& context) {
    // aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
    // aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
    // aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out)
    // aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
    // aten::sum.int(int[] self) -> int
    // aten::sum.float(float[] self) -> float
    // aten::sum.bool(bool[] self) -> int
    num_inputs_check(context, 1, 5);
    bool keep_dims = false;
    auto num_inputs = context.get_input_size();
    size_t axis_idx;
    size_t dtype_idx;
    size_t keep_dims_idx;
    size_t out_idx;
    ov::Output<ov::Node> axes;
    if (num_inputs < 4) {
        // move parameters that not in signature out of number of inputs range
        keep_dims_idx = 4;
        axis_idx = 3;
        dtype_idx = 1;
        out_idx = 2;
    } else {
        axis_idx = 1;
        keep_dims_idx = 2;
        dtype_idx = 3;
        out_idx = 4;
    }
    auto data = context.get_input(0);
    auto data_dtype = simplified_type_interpret(context.get_input_type(0));
    // PyTorch sum converts bool and uint8 to i64 for preventing overflow
    if ((data.get_element_type() == element::boolean || data.get_element_type() == element::u8) ||
        (data_dtype.is<element::Type>() &&
         (data_dtype.as<element::Type>() == element::boolean || data_dtype.as<element::Type>() == element::u8))) {
        data = context.mark_node(std::make_shared<ov::op::v0::Convert>(data, element::i64));
    }
    if (context.input_is_none(axis_idx)) {
        axes = get_axes_range(context, 0);
    } else {
        axes = context.get_input(static_cast<int>(axis_idx));
    }
    if (!context.input_is_none(keep_dims_idx)) {
        keep_dims = context.const_input<bool>(keep_dims_idx);
    }

    Output<Node> sum = context.mark_node(std::make_shared<ov::op::v1::ReduceSum>(data, axes, keep_dims));

    if (!context.input_is_none(dtype_idx)) {
        sum = apply_dtype(context, dtype_idx, sum);
    }

    if (!context.input_is_none(out_idx)) {
        context.mutate_input(out_idx, sum);
    }
    return {sum};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov