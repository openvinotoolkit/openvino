// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/prior_box.hpp"

namespace ngraph {
namespace op {
using PriorBoxAttrs = ov::op::v0::PriorBox::Attributes;
namespace v0 {
using ov::op::v0::PriorBox;
}  // namespace v0
namespace v8 {
using ov::op::v8::PriorBox;
}  // namespace v8
using v0::PriorBox;
}  // namespace op
}  // namespace ngraph
