// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/roi_align.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ROIAlign;
}  // namespace v3
namespace v9 {
using ov::op::v9::ROIAlign;
}  // namespace v9
using v3::ROIAlign;
}  // namespace op
}  // namespace ngraph
