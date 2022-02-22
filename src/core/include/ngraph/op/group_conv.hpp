// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/convolution.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/group_conv.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::GroupConvolution;
using ov::op::v1::GroupConvolutionBackpropData;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
