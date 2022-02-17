// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/axis_vector.hpp"
#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Unsqueeze;
}  // namespace v0
using v0::Unsqueeze;
}  // namespace op
}  // namespace ngraph
