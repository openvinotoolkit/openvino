// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "openvino/op/mvn.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::MVN;
}  // namespace v0
using v0::MVN;

using ov::op::MVNEpsMode;

namespace v6 {
using ov::op::v6::MVN;
}  // namespace v6
}  // namespace op
}  // namespace ngraph
