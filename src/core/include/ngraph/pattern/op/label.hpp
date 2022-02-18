// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/label.hpp"

namespace ngraph {
namespace pattern {
namespace op {
using ov::pass::pattern::op::Label;
}  // namespace op

using ov::pass::pattern::any_input;
}  // namespace pattern
}  // namespace ngraph
