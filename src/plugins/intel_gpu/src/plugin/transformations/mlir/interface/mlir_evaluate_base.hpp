// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "openvino/core/node.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::intel_gpu::mlir {

class MLIREvaluateBase {
public:
    virtual bool requires_packed_args() const = 0;
    virtual bool invoke(const ov::TensorVector& inputs,
                        ov::TensorVector& outputs,
                        const ov::EvaluationContext& evaluationContext) = 0;
    virtual bool invoke_packed(std::vector<void*>& args,
                               const ov::EvaluationContext& evaluationContext) = 0;
    virtual ~MLIREvaluateBase() = default;
};

}  // namespace ov::intel_gpu::mlir
