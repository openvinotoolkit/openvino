// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/bincount.hpp"
#include "openvino/op/convert.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_bincount(const NodeContext& context) {
    // aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
    num_inputs_check(context, 1, 3);
    auto data = context.get_input(0);

    const auto data_et = data.get_element_type();
    if (data_et != element::i32 && data_et != element::i64) {
        data = context.mark_node(std::make_shared<v0::Convert>(data, element::i64));
    }

    int64_t minlength = 0;
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        minlength = context.const_input<int64_t>(2);
    }

    if (context.get_input_size() == 1 || context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<v17::Bincount>(data, minlength))};
    }

    auto weights = context.get_input(1);
    return {context.mark_node(std::make_shared<v17::Bincount>(data, weights, minlength))};
}

OutputVector translate_bincount_fx(const NodeContext& context) {
    // aten.bincount.default — same signature and semantics as aten::bincount
    return translate_bincount(context);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
