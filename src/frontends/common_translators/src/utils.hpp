// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/frontend/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"

namespace ov {
namespace frontend {
namespace common_translators {

void num_inputs_check(const NodeContext& context, size_t min_inputs, size_t max_inputs, bool allow_complex = false);

template <typename T>
ov::Output<ov::Node> create_same_type_const_scalar(const ov::Output<ov::Node>& same_type_output, const T& value) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), ov::Shape{}, value);
    } else {
        ov::Output<ov::Node> const_res =
            std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), ov::Shape{}, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

template <typename T>
ov::Output<ov::Node> create_same_type_const(const ov::Output<ov::Node>& same_type_output,
                                            const std::vector<T>& value,
                                            const ov::Shape& shape) {
    if (same_type_output.get_element_type().is_static()) {
        return std::make_shared<ov::op::v0::Constant>(same_type_output.get_element_type(), shape, value);
    } else {
        ov::Output<ov::Node> const_res = std::make_shared<ov::op::v0::Constant>(ov::element::from<T>(), shape, value);
        const_res = std::make_shared<ov::op::v1::ConvertLike>(const_res, same_type_output);
        return const_res;
    }
}

}  // namespace common_translators
}  // namespace frontend
}  // namespace ov
