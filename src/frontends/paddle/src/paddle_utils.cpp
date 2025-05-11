// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paddle_utils.hpp"

std::shared_ptr<ov::Node> ov::frontend::paddle::reorder_axes(const ov::Output<ov::Node>& value,
                                                             std::vector<size_t> axes_order) {
    const auto axes_order_const =
        std::make_shared<opset6::Constant>(element::i64,
                                           Shape{axes_order.size()},
                                           std::vector<int64_t>(axes_order.begin(), axes_order.end()));
    return std::make_shared<opset6::Transpose>(value, axes_order_const);
}
