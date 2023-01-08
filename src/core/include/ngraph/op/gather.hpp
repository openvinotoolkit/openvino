// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/gather_base.hpp"
#include "openvino/op/gather.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Gather;
}  // namespace v1
namespace v7 {
using ov::op::v7::Gather;
}  // namespace v7
namespace v8 {
using ov::op::v8::Gather;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
