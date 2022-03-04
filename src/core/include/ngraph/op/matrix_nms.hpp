// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/nms_base.hpp"
#include "openvino/op/matrix_nms.hpp"

namespace ngraph {
namespace op {
namespace v8 {
using ov::op::v8::MatrixNms;
}  // namespace v8
}  // namespace op
using ov::operator<<;
}  // namespace ngraph
