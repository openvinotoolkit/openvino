// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/non_zero.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::NonZero;
}  // namespace v3
using v3::NonZero;
}  // namespace op
}  // namespace ngraph
