// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "openvino/op/clamp.hpp"

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace gguf {
namespace op {

// GGML_OP_CLAMP: elementwise clamp to [min, max]. The decoder exposes the bounds as typed
// float attributes ("clamp_min"/"clamp_max"), so the translator never reads ggml op_params.
OutputVector translate_clamp(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    auto input = context.get_input(0);
    float min = context.get_attribute<float>("clamp_min");
    float max = context.get_attribute<float>("clamp_max");

    auto res = std::make_shared<ov::op::v0::Clamp>(input, min, max);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
