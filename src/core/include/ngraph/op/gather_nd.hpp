// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/gather_nd.hpp"

namespace ngraph {
namespace op {
namespace v5 {
using ov::op::v5::GatherND;
}  // namespace v5
namespace v8 {
using ov::op::v8::GatherND;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
