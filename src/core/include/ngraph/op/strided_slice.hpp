// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/strided_slice.hpp"

namespace ngraph {
namespace op {
namespace v1 {
using ov::op::v1::StridedSlice;
}  // namespace v1
}  // namespace op
}  // namespace ngraph
