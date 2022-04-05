// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/axis_set.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/topk.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::TopK;
}  // namespace v1

namespace v3 {
using ov::op::v3::TopK;
}  // namespace v3
}  // namespace op
}  // namespace ngraph
