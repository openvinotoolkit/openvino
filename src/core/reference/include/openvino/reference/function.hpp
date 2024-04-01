// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

#include "openvino/core/model.hpp"

namespace ov {
namespace reference {
void function(const std::shared_ptr<Model>& function, const ov::TensorVector& inputs, ov::TensorVector& outputs);
void function(const std::shared_ptr<Model>& function,
              const ov::TensorVector& inputs,
              ov::TensorVector& outputs,
              const EvaluationContext& evaluation_context);
}  // namespace reference
}  // namespace ov
