// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {
void einsum(ov::TensorVector& outputs, const ov::TensorVector& inputs, const std::string& equation);
}  // namespace reference
}  // namespace ov
