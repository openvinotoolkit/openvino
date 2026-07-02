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

// A GGUF weight surfaced as a node. A weight is a ggml leaf (op type "GGML_OP_NONE") that the
// decoder marks by exposing a "data" attribute (the raw weight bytes), alongside the ggml quant
// type name and the logical shape. The frontend does all dequant / repacking here, so the
// decoder never builds OV nodes itself. (Model-input leaves are also GGML_OP_NONE, but they are
// resolved to Parameters before the graph walk and never reach this translator.)
OutputVector translate_weight(const NodeContext& context) {
    auto data = context.get_attribute<ov::Tensor>("data");
    FRONT_END_OP_CONVERSION_CHECK(data, "GGML_OP_NONE node has no 'data' attribute; not a weight");
    auto quant_type = context.get_attribute<std::string>("quant_type");
    auto shape = context.get_output_shape().to_shape();

    auto node = make_weight_node(data, quant_type, shape, context.get_name());
    return rename_outputs_with_suffix({node}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
