// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils_quantize.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_quantize_per_tensor(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    return {context.mark_node(quantize(context, QuantizedPtNodeType::QUANTIZE_PER_TENSOR))};
}

OutputVector translate_dequantize(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    return {context.mark_node(dequantize(context))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
