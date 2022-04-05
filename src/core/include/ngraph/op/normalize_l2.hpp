// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ngraph/node.hpp"
#include "ngraph/op/op.hpp"
#include "ngraph/op/util/attr_types.hpp"
#include "openvino/op/normalize_l2.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::NormalizeL2;
}  // namespace v0
}  // namespace op
}  // namespace ngraph
