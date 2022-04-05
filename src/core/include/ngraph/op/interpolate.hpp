// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "ngraph/attribute_adapter.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/interpolate.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using InterpolateAttrs = ov::op::v0::Interpolate::Attributes;
using ov::op::v0::Interpolate;
}  // namespace v0
namespace v4 {
using ov::op::v4::Interpolate;
}  // namespace v4
using v0::Interpolate;
using v0::InterpolateAttrs;
}  // namespace op
}  // namespace ngraph
