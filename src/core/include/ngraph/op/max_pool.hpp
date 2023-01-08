// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>

#include "ngraph/op/util/max_pool_base.hpp"
#include "openvino/op/max_pool.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::MaxPool;
}  // namespace v1

namespace v8 {
using ov::op::v8::MaxPool;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
