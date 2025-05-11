// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/softmax.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
OutputVector translate_softmax_common(const NodeContext& context, const bool convert_dtype) {
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto axis = context.const_input<int64_t>(1);
    // Optional input 2 to translate from half(f16) to float
    if (convert_dtype && !context.input_is_none(2)) {
        x = apply_dtype(context, 2, x);
    }
    return {context.mark_node(std::make_shared<v8::Softmax>(x, axis))};
};

OutputVector translate_softmax(const NodeContext& context) {
    return translate_softmax_common(context, true);
}

OutputVector translate_softmax_fx(const NodeContext& context) {
    // _softmax(Tensor self, int dim, bool half_to_float) -> Tensor
    return translate_softmax_common(context, false);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
