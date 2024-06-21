// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/space_to_depth.hpp"

#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector space_to_depth(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    const auto& shape = data.get_partial_shape();
    FRONT_END_GENERAL_CHECK(shape.rank().is_static() && shape.rank().get_length() == 4, "Input must be 4-dimensional");
    std::size_t block_size = node.get_attribute_value<std::int64_t>("blocksize");
    const auto mode = v0::SpaceToDepth::SpaceToDepthMode::BLOCKS_FIRST;
    return {std::make_shared<v0::SpaceToDepth>(data, mode, block_size)};
}
static bool registered =
    register_translator("SpaceToDepth", VersionRange::single_version_for_all_opsets(), space_to_depth);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
