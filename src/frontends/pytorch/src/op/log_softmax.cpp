// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/log_softmax.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/convert.hpp"
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
    auto const dim = context.const_input<int64_t>(1);

    if (!context.input_is_none(2) && !is_fx) {
        const auto elem_type = input.get_element_type();
        const auto target_dtype_i64 = context.const_input<int64_t>(2);
        const auto target_dtype = convert_dtype(target_dtype_i64);
        if (elem_type != target_dtype) {
            input = context.mark_node(std::make_shared<v0::Convert>(input, target_dtype));
        }
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
