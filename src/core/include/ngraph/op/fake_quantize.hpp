// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/fake_quantize.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::FakeQuantize;
}  // namespace v0
using v0::FakeQuantize;
}  // namespace op
}  // namespace ngraph
