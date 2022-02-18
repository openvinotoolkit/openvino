// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/multi_subgraph_base.hpp"
#include "openvino/op/if.hpp"

namespace ngraph {
namespace op {
namespace v8 {
using ov::op::v8::If;
}  // namespace v8
}  // namespace op
}  // namespace ngraph
