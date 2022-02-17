// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/clamp.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Clamp;
}  // namespace v0
using v0::Clamp;
}  // namespace op
}  // namespace ngraph
