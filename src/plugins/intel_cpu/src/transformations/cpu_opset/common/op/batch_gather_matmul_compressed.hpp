// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "batch_gather_matmul.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"

namespace ov::intel_cpu {

class BatchGatherMatmulCompressed : public BatchGatherMatmul {
public:
    OPENVINO_OP("BatchGatherMatmulCompressed", "cpu_plugin_opset", BatchGatherMatmul);

    BatchGatherMatmulCompressed() = default;

    BatchGatherMatmulCompressed(const ov::Output<Node>& A,
                                const ov::Output<Node>& B,
                                const ov::Output<Node>& indices,
                                const ov::Output<Node>& bias,
                                const ov::Output<Node>& weight_scales,
                                const ov::Output<Node>& weight_zero_points);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
};

}  // namespace ov::intel_cpu