// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"
#include "openvino/op/op.hpp"

namespace ov {
namespace intel_cpu {

class QKVProjectionNode : public ov::op::Op {
public:
    OPENVINO_OP("QKVProjection", "cpu_plugin_opset");

    QKVProjectionNode() = default;

    // args:
    //      0: input
    //      1: gate_proj
    //      2: up_proj
    //      3: down_proj
    QKVProjectionNode(const OutputVector& args) : Op(args) {
        validate_and_infer_types();
    }

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace intel_cpu
}  // namespace ov
