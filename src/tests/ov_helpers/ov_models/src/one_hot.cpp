// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/one_hot.hpp"

#include <memory>

#include "ov_models/builders.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeOneHot(const ov::Output<Node>& indices,
                                     const element::Type& depth_type,
                                     const int64_t& depth_val,
                                     const element::Type& set_type,
                                     const float& on_val,
                                     const float& off_val,
                                     const int64_t& axis) {
    auto depth_const = std::make_shared<ov::op::v0::Constant>(depth_type, ov::Shape{}, depth_val);
    auto on_value_const = std::make_shared<ov::op::v0::Constant>(set_type, ov::Shape{}, on_val);
    auto off_value_const = std::make_shared<ov::op::v0::Constant>(set_type, ov::Shape{}, off_val);
    return std::make_shared<ov::op::v1::OneHot>(indices, depth_const, on_value_const, off_value_const, axis);
}
}  // namespace builder
}  // namespace ngraph
