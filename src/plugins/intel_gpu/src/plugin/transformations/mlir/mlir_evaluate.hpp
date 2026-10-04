// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "interface/mlir_evaluate_base.hpp"
#include "mlir/IR/BuiltinOps.h"

namespace ov::intel_gpu::mlir {

std::shared_ptr<MLIREvaluateBase> create_mlir_evaluator(::mlir::OwningOpRef<::mlir::ModuleOp> module,
                                                        const std::shared_ptr<ov::EvaluationContext>& loweringContext);

}  // namespace ov::intel_gpu::mlir
