// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/unique_consecutive.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {
using namespace ov::op;

OutputVector translate_unique_consecutive(const NodeContext& context) {
    // aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int dim=None) ->
    // (Tensor output, Tensor inverse_indices, Tensor counts)
    num_inputs_check(context, 1, 4);

    auto input = context.get_input(0);

    bool return_inverse = false;
    if (!context.input_is_none(1)) {
        return_inverse = context.const_input<bool>(1);
    }

    bool return_counts = false;
    if (!context.input_is_none(2)) {
        return_counts = context.const_input<bool>(2);
    }

    int64_t dim = -1;
    bool dim_is_none = context.input_is_none(3);
    if (!dim_is_none) {
        dim = context.const_input<int64_t>(3);
    }

    OutputVector outputs;

    // Choose the axis

    // Compare the neighbors along axis a

    // Build a keep mask of run starts

    // Get run start indices and the values output

    // Compute counts

    // Compute inverse indices
    
    outputs.push_back(unique_consecutive->output(0));
    if (return_inverse) {
        outputs.push_back(unique_consecutive->output(1));
    }
    if (return_counts) {
        outputs.push_back(unique_consecutive->output(return_inverse ? 2 : 1));
    }

    return outputs;
};
}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov