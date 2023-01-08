// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/region_yolo.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::RegionYolo;
}  // namespace v0
using v0::RegionYolo;
}  // namespace op
}  // namespace ngraph
