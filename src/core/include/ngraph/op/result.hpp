// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/op.hpp"
#include "openvino/op/result.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Result;
}  // namespace v0
using v0::Result;
}  // namespace op
using ResultVector = std::vector<std::shared_ptr<op::Result>>;
}  // namespace ngraph
