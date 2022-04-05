// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/op/util/deformable_convolution_base.hpp"
#include "openvino/op/deformable_convolution.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::DeformableConvolution;
}  // namespace v1

namespace v8 {
using ov::op::v8::DeformableConvolution;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
