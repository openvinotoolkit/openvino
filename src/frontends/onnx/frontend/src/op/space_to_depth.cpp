// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/space_to_depth.hpp"

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector space_to_depth(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    const auto& shape = data.get_partial_shape();
    NGRAPH_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4, "Input must be 4-dimensional");
    std::size_t block_size = node.get_attribute_value<std::int64_t>("blocksize");
    const auto mode = default_opset::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    return OutputVector{std::make_shared<default_opset::SpaceToDepth>(data, mode, block_size)};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
