// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node_vector.hpp"

#include "openvino/core/node_output.hpp"
#include "openvino/core/type.hpp"
#include "openvino/op/result.hpp"

ov::OutputVector ov::as_output_vector(const ov::NodeVector& args) {
    ov::OutputVector output_vector;
    for (const auto& arg : args) {
        output_vector.push_back(arg);
    }
    return output_vector;
}

ov::NodeVector ov::as_node_vector(const ov::OutputVector& values) {
    NodeVector node_vector;
    for (auto& value : values) {
        node_vector.emplace_back(value.get_node_shared_ptr());
    }
    return node_vector;
}

ov::ResultVector ov::as_result_vector(const OutputVector& values) {
    ResultVector result;
    for (const auto& value : values) {
        std::shared_ptr<Node> node = value.get_node_shared_ptr();
        result.push_back(ov::is_type<ov::op::v0::Result>(node) ? ov::as_type_ptr<ov::op::v0::Result>(node)
                                                               : std::make_shared<ov::op::v0::Result>(value, true));
    }
    return result;
}
