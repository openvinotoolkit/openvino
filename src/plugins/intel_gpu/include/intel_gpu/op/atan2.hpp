// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/op.hpp"

namespace ov::intel_gpu::op {

// Two-input atan2(y, x). input0 = y, input1 = x. Inserted by FuseAtan2Decomposed
// in place of the frontend's atan2 decomposition; lowered to cldnn::eltwise with
// eltwise_mode::atan2 by ops/atan2.cpp.
class Atan2 : public ov::op::Op {
public:
    OPENVINO_OP("Atan2", "ie_internal_opset");

    Atan2() = default;
    Atan2(const ov::Output<Node>& y, const ov::Output<Node>& x);

    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
};

}  // namespace ov::intel_gpu::op
