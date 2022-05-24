// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/grid_sample.hpp"

namespace ngraph {
namespace op {
namespace v9 {
using ov::op::v9::GridSample;
}  // namespace v0
using v9::GridSample;
}  // namespace op
}  // namespace ngraph
