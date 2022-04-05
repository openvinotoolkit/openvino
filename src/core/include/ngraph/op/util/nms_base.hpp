// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/op.hpp"
#include "openvino/op/util/nms_base.hpp"

namespace ngraph {
namespace op {
namespace util {
using ov::op::util::NmsBase;
}  // namespace util
}  // namespace op
using ov::operator<<;
}  // namespace ngraph
