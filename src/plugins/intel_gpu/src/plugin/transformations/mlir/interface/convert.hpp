// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/core/any.hpp"
#include "openvino/core/model.hpp"

namespace ov::intel_gpu::mlir {

void transformMLIR(std::shared_ptr<ov::Model> model,
                   std::shared_ptr<ov::EvaluationContext> loweringContext);

}  // namespace ov::intel_gpu::mlir
