// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/histc.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_histc(const NodeContext& context) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
    num_inputs_check(context, 1, 4);
    auto data = context.get_input(0);

    int64_t bins = 100;
    if (context.get_input_size() > 1 && !context.input_is_none(1)) {
        bins = context.const_input<int64_t>(1);
    }

    double min_val = 0.0;
    if (context.get_input_size() > 2 && !context.input_is_none(2)) {
        min_val = context.const_input<double>(2);
    }

    double max_val = 0.0;
    if (context.get_input_size() > 3 && !context.input_is_none(3)) {
        max_val = context.const_input<double>(3);
    }

    return {context.mark_node(std::make_shared<v17::Histc>(data, bins, min_val, max_val))};
}

OutputVector translate_histc_fx(const NodeContext& context) {
    // aten.histc.default — same signature and semantics as aten::histc
    return translate_histc(context);
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
