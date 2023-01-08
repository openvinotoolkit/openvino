// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/variadic_split.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::VariadicSplit;
}  // namespace v1
using v1::VariadicSplit;
}  // namespace op
}  // namespace ngraph
