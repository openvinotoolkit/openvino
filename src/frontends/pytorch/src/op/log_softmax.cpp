// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_log_softmax_common(const NodeContext& context, bool is_fx) {
    /*
    aten::log_softmax(
        Tensor input,
        int64 dim,
        dtype dtype = None
    )
    */
    num_inputs_check(context, 2, 3);
    auto input = context.get_input(0);
    const auto dim = context.const_input<int64_t>(1);

    if (!context.input_is_none(2) && !is_fx) {
        const auto elem_type = input.get_element_type();
        const auto target_dtype_i64 = context.const_input<int64_t>(2);
        const auto target_dtype = convert_dtype(target_dtype_i64);
        if (elem_type != target_dtype) {
            input = context.mark_node(std::make_shared<v0::Convert>(input, target_dtype));
        }
    }

    // Handle scalar (rank-0) input: PyTorch reshapes scalars to 1D before softmax computation.
    // Unsqueeze to rank 1, apply LogSoftmax with axis=0, then squeeze back.
    const auto& input_pshape = input.get_partial_shape();
    if (input_pshape.rank().is_static() && input_pshape.rank().get_length() == 0) {
        auto zero_const = v0::Constant::create(element::i32, Shape{}, {0});
        input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, zero_const));
        auto log_softmax = context.mark_node(std::make_shared<v5::LogSoftmax>(input, 0));
        auto result = context.mark_node(std::make_shared<v0::Squeeze>(log_softmax, zero_const));
        return {result};
    }

    const auto log_softmax = context.mark_node(std::make_shared<v5::LogSoftmax>(input, dim));
    return {log_softmax};
};

OutputVector translate_log_softmax(const NodeContext& context) {
    return translate_log_softmax_common(context, false);
}

OutputVector translate_log_softmax_fx(const NodeContext& context) {
    return translate_log_softmax_common(context, true);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
