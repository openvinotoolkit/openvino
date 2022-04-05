// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/factory_adapter.hpp"
#include "ngraph/function.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/tensor_iterator.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "openvino/op/loop.hpp"

namespace ngraph {
namespace op {
namespace v5 {
using ov::op::v5::Loop;
}  // namespace v5
}  // namespace op
}  // namespace ngraph
