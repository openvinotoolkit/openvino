// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <ngraph/opsets/opset7.hpp>

#include "ngraph/shape.hpp"

namespace ov {
namespace reference {
void einsum(ov::TensorVector& outputs, const ov::TensorVector& inputs, const std::string& equation);
}  // namespace reference
}  // namespace ov
