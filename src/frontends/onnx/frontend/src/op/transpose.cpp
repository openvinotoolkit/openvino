// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/transpose.hpp"

#include <memory>
#include <vector>

#include "ngraph/builder/reshape.hpp"
#include "ngraph/node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector transpose(const Node& node) {
    Output<ngraph::Node> data = node.get_ng_inputs().at(0);

    auto permute_axes = node.get_attribute_value<std::vector<std::size_t>>("perm", {});

    return {(permute_axes.empty()) ? ngraph::builder::opset1::transpose(data)
                                   : ngraph::builder::opset1::reorder_axes(data, permute_axes)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
