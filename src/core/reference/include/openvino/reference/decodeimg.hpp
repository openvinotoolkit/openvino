// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace reference {

void decodeimg(const Tensor& input, Tensor& out);
}  // namespace reference
}  // namespace ov
