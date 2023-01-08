// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

namespace ngraph {
namespace pattern {
namespace op {
using ov::pass::pattern::op::WrapType;
}  // namespace op

using ov::pass::pattern::wrap_type;
}  // namespace pattern
}  // namespace ngraph
