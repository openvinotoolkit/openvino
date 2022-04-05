// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>

#include "ngraph/function.hpp"
#include "ngraph/op/parameter.hpp"
#include "ngraph/op/util/sub_graph_base.hpp"
#include "openvino/op/tensor_iterator.hpp"

namespace ngraph {
namespace op {
namespace v0 {
using ov::op::v0::TensorIterator;
}  // namespace v0
using v0::TensorIterator;
}  // namespace op
}  // namespace ngraph
