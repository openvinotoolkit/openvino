// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"
#include "intel_gpu_visibility.hpp"

namespace ov::intel_gpu::op {

class INTEL_GPU_API Placeholder : public ov::op::Op {
public:
    OPENVINO_OP("Placeholder", "gpu_opset");

    Placeholder();

    bool visit_attributes(ov::AttributeVisitor& visitor) override;
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}   // namespace ov::intel_gpu::op
