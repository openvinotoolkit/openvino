// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/convolution.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::Convolution;
using ov::op::v1::ConvolutionBackpropData;
}  // namespace v1
}  // namespace op
}  // namespace ngraph

#define OPERATION_DEFINED_Convolution             1
#define OPERATION_DEFINED_ConvolutionBackpropData 1
#include "ngraph/opsets/opsets_tbl.hpp"
#undef OPERATION_DEFINED_ConvolutionBackpropData
#undef OPERATION_DEFINED_Convolution
