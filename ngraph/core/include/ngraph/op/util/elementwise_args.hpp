// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/op/util/attr_types.hpp"
#include "ngraph/partial_shape.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {

class Node;

namespace op {
namespace util {
std::tuple<element::Type, PartialShape> validate_and_infer_elementwise_args(
    Node* node,
    const op::AutoBroadcastSpec& autob = op::AutoBroadcastSpec());
}
}  // namespace op
}  // namespace ngraph
