// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/extractimagepatches.hpp"

namespace ngraph {
namespace op {
namespace v3 {
using ov::op::v3::ExtractImagePatches;
}  // namespace v3
using v3::ExtractImagePatches;
}  // namespace op
}  // namespace ngraph
