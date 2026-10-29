// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/histc.hpp"
#include "utils.hpp"

namespace ov::frontend::pytorch::op {

using namespace ov::op;

OutputVector translate_histc(const NodeContext& context) {
    // aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0); bins/min/max are attributes of v17::Histc
    num_inputs_check(context, 1, 4);
    int64_t bins = 100;
    double min = 0.0;
    double max = 0.0;
    if (!context.input_is_none(1)) {
        bins = context.const_input<int64_t>(1);
    }
    if (!context.input_is_none(2)) {
        min = context.const_input<double>(2);
    }
    if (!context.input_is_none(3)) {
        max = context.const_input<double>(3);
    }
    return {context.mark_node(std::make_shared<v17::Histc>(context.get_input(0), bins, min, max))};
}

}  // namespace ov::frontend::pytorch::op
