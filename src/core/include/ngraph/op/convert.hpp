// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "openvino/op/convert.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Convert;
}  // namespace v0
using v0::Convert;
}  // namespace op
}  // namespace ngraph
