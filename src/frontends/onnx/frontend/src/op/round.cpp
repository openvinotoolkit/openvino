// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Disabled in CMakeList
// Update to higher opset required

#include "op/round.hpp"

#include "openvino/op/round.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector round(const Node& node) {
    return {std::make_shared<v5::Round>(node.get_ng_inputs().at(0), v5::Round::RoundMode::HALF_TO_EVEN)};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
