// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

namespace ov::intel_gpu::op {

class LoraSubgraphFused : public ov::op::Op {
public:
    OPENVINO_OP("LoraSubgraphFused", "gpu_opset");

    LoraSubgraphFused(const ov::Output<Node>& main_input,
                      const ov::Output<Node>& lora_input,
                      const OutputVector& states,
                      bool transposed_states);

    bool is_transposed_states() const { return transposed_states; }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

protected:
    bool transposed_states;
};

}   // namespace ov::intel_gpu::op
