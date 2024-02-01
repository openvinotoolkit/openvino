// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/tile.hpp"

#include "onnx_import/core/node.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/tile.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector tile(const Node& node) {
    auto input = node.get_ng_inputs().at(0);
    auto repeats = node.get_ng_inputs().at(1);

    // Workaround for backends which require repeats to be i64.
    // Remove the following line when no longer needed.
    repeats = std::make_shared<v0::Convert>(repeats, ov::element::i64);

    return {std::make_shared<v0::Tile>(input, repeats)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
