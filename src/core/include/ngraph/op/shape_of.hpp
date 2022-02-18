// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/shape_of.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ShapeOf;
}  // namespace v3

namespace v0 {
using ov::op::v0::ShapeOf;
}  // namespace v0
using v0::ShapeOf;
}  // namespace op
}  // namespace ngraph
