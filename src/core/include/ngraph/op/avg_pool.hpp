// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/avg_pool.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::AvgPool;
}  // namespace v1

using v1::AvgPool;
}  // namespace op
}  // namespace ngraph
