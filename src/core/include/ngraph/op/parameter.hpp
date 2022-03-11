// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/parameter.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Parameter;
}  // namespace v0
using v0::Parameter;
}  // namespace op
using ParameterVector = std::vector<std::shared_ptr<op::Parameter>>;
}  // namespace ngraph
