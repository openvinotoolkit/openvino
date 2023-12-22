// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/depth_to_space.hpp"

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector depth_to_space(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    const auto& shape = data.get_partial_shape();
    NGRAPH_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4, "Input must be 4-dimensional");

    const auto mode = node.get_attribute_value<std::string>("mode", "DCR");
    default_opset::DepthToSpace::DepthToSpaceMode ngraph_mode;
    if (mode == "DCR")
        ngraph_mode = default_opset::DepthToSpace::DepthToSpaceMode::BLOCKS_FIRST;
    else if (mode == "CRD")
        ngraph_mode = default_opset::DepthToSpace::DepthToSpaceMode::DEPTH_FIRST;
    else
        NGRAPH_CHECK(false, "only 'DCR' and 'CRD' modes are supported");

    const auto block_size = node.get_attribute_value<std::int64_t>("blocksize");
    return OutputVector{std::make_shared<default_opset::DepthToSpace>(data, ngraph_mode, block_size)};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
