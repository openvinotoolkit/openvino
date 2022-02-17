// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/op/util/binary_elementwise_arithmetic.hpp"
#include "openvino/op/floor_mod.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::FloorMod;
}  // namespace v1
using v1::FloorMod;
}  // namespace op
}  // namespace ngraph
