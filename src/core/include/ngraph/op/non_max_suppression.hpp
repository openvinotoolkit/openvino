// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/non_max_suppression.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::NonMaxSuppression;
}  // namespace v1

namespace v3 {
using ov::op::v3::NonMaxSuppression;
}  // namespace v3

namespace v4 {
using ov::op::v4::NonMaxSuppression;
}  // namespace v4

namespace v5 {
using ov::op::v5::NonMaxSuppression;
}  // namespace v5

namespace v9 {
using ov::op::v9::NonMaxSuppression;
}  // namespace v9
}  // namespace op
using ov::operator<<;
}  // namespace ngraph
