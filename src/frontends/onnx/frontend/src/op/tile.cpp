// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/tile.hpp"

#include <memory>

#include "default_opset.hpp"
#include "onnx_import/core/node.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector tile(const Node& node) {
    auto input = node.get_ng_inputs().at(0);
    auto repeats = node.get_ng_inputs().at(1);

    // Workaround for backends which require repeats to be i64.
    // Remove the following line when no longer needed.
    repeats = std::make_shared<default_opset::Convert>(repeats, element::i64);

    return {std::make_shared<default_opset::Tile>(input, repeats)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
