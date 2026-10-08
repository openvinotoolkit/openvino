// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "node_context.hpp"
#include "op_table.hpp"
#include "utils.hpp"

namespace ov::frontend::gguf::op {

// L2 normalization over the last dimension: x / max(sqrt(sum(x^2)), eps).
OutputVector translate_l2_norm(const NodeContext& context) {
    num_inputs_check(context, 1, 1);

    return rename_outputs_with_suffix({make_l2_norm(context.get_input(0), context.get_attribute<float>("eps"))},
                                      context.get_name());
}

}  // namespace ov::frontend::gguf::op
