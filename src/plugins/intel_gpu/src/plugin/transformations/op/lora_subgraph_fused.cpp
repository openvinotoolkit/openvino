// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/lora_subgraph_fused.hpp"

namespace ov::intel_gpu::op {

LoraSubgraphFused::LoraSubgraphFused(const ov::Output<Node>& main_input,
                                     const ov::Output<Node>& lora_input,
                                     const OutputVector& states,
                                     bool transposed_states)
    : transposed_states(transposed_states) {
    set_argument(0, main_input);
    set_argument(1, lora_input);

    for (size_t i = 0; i < states.size(); ++i) {
        set_argument(2 + i, states[i]);
    }

    constructor_validate_and_infer_types();
}

std::shared_ptr<ov::Node> LoraSubgraphFused::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);

    ov::OutputVector states(new_args.begin() + 2, new_args.end());

    return std::make_shared<LoraSubgraphFused>(new_args.at(0),
                                               new_args.at(1),
                                               states,
                                               transposed_states);
}

void LoraSubgraphFused::validate_and_infer_types() {
    OPENVINO_ASSERT(get_input_size() == 8 || get_input_size() == 11, "LoraSubgraphFused must have 8 or 11 inputs whereas it has ", get_input_size());

    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

}  // namespace ov::intel_gpu::op
