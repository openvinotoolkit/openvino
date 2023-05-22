// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")

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

namespace v11 {
using ov::op::v11::TopK;
}  // namespace v11
}  // namespace op
}  // namespace ngraph
