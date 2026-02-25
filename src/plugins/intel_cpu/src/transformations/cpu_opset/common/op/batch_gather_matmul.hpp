// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class BatchGatherMatmul : public ov::op::Op {
public:
    OPENVINO_OP("BatchGatherMatmul", "cpu_plugin_opset");

    BatchGatherMatmul() = default;

    BatchGatherMatmul(const ov::Output<Node>& A,
                      const ov::Output<Node>& B,
                      const ov::Output<Node>& indices,
                      const ov::Output<Node>& bias);

    BatchGatherMatmul(const ov::Output<Node>& A, const ov::Output<Node>& B, const ov::Output<Node>& indices);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;

private:
    // the weights matrix B is expected to have the transposed form [goup, N, K]
    static constexpr bool transp_a = false;
    static constexpr bool transp_b = true;
};

}  // namespace ov::intel_cpu