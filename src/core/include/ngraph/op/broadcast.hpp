// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/axis_set.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/broadcast_base.hpp"
#include "openvino/op/broadcast.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::Broadcast;
}  // namespace v3

namespace v1 {
using ov::op::v1::Broadcast;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
