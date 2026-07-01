// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "node_context.h"
#include "op_table.h"
#include "quant/weights.hpp"
#include "utils.h"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// A GGUF weight surfaced as a node (op type "GGML_OP_WEIGHT"). The decoder provides the raw
// weight bytes, the ggml quant type name and the logical shape; the frontend does all
// dequant / repacking here, so the decoder never builds OV nodes itself.
OutputVector translate_weight(const NodeContext& context) {
    auto data = context.get_attribute<ov::Tensor>("data");
    auto quant_type = context.get_attribute<std::string>("quant_type");
    auto shape = context.get_output_shape().to_shape();

    auto node = make_weight_node(data, quant_type, shape, context.get_name());
    return rename_outputs_with_suffix({node}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
