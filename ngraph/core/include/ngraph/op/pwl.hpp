// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/pwl.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Pwl;
}  // namespace v0
using v0::Pwl;
}  // namespace op
}  // namespace ngraph
