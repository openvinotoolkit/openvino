// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "helper_ops/gguf_constant.hpp"
#include "node_context.hpp"
#include "op_table.hpp"
#include "quant/weights.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

OutputVector translate_weight(const NodeContext& context) {
    GgufWeight source;
    const auto parsed = context.get_attribute<GgufWeight>("gguf_source", GgufWeight{});
    if (parsed.data) {
        source = parsed;
    } else {
        source.data = context.get_attribute<ov::Tensor>("data");
        FRONT_END_OP_CONVERSION_CHECK(source.data, "GGML_OP_NONE node has no weight payload");
        source.shape = context.get_output_shape().to_shape();
        source.qtype = gguf_type_from_name(context.get_attribute<std::string>("quant_type"));
        source.source_name = context.get_name();
        source.native = false;
    }
    auto node = std::make_shared<GGUFConstant>(source);
    node->set_friendly_name(context.get_name());
    return rename_outputs_with_suffix({node}, context.get_name());
}

}  // namespace ov::frontend::gguf::op
