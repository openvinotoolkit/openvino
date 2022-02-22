// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstring>

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/node.hpp"
#include "ngraph/runtime/aligned_buffer.hpp"
#include "ngraph/runtime/host_tensor.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "ngraph/type/element_type.hpp"
#include "ngraph/type/element_type_traits.hpp"
#include "ngraph/util.hpp"
#include "openvino/op/constant.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::Constant;
}  // namespace v0
using v0::Constant;
}  // namespace op
}  // namespace ngraph
