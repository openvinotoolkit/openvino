// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/unsqueeze.hpp"

#include <memory>

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "ngraph/shape.hpp"
#include "ngraph/validation_util.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector unsqueeze(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    auto axes_node = node.get_attribute_as_constant<std::vector<std::int64_t>>("axes", {});
    return {std::make_shared<default_opset::Unsqueeze>(data, axes_node)};
}

}  // namespace set_1

namespace set_13 {
OutputVector unsqueeze(const Node& node) {
    auto inputs = node.get_ng_inputs();
    // when onnx format is < 13 but ai.onnx > 13
    // which results in using unsqueeze-13 but
    // the node itself has format of unsqueeze-8
    if (inputs.size() == 1) {
        auto data = node.get_ng_inputs().at(0);
        auto axes_node = node.get_attribute_as_constant<std::vector<std::int64_t>>("axes", {});
        return {std::make_shared<default_opset::Unsqueeze>(data, axes_node)};
    }
    // set 13
    return {std::make_shared<default_opset::Unsqueeze>(inputs.at(0), inputs.at(1))};
}

}  // namespace set_13
}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
