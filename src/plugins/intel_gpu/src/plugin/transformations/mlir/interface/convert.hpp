// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "intel_gpu/runtime/execution_config.hpp"
#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu::mlir {

// 'config' is needed for the per-model ov::intel_gpu::mlir_patterns option
void transformMLIR(const std::shared_ptr<ov::Model>& model,
                   const ExecutionConfig& config,
                   const std::shared_ptr<ov::EvaluationContext>& loweringContext);

}  // namespace ov::intel_gpu::mlir
