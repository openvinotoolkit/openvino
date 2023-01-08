// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/softmax.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Softmax;
}  // namespace v1

namespace v8 {
using ov::op::v8::Softmax;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
