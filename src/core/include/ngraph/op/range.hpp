// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/range.hpp"

namespace ngraph {
namespace op {
namespace v4 {
using ov::op::v4::Range;
}  // namespace v4
namespace v0 {
using ov::op::v0::Range;
}  // namespace v0
using v0::Range;
}  // namespace op
}  // namespace ngraph
