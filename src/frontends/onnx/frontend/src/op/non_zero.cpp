// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/non_zero.hpp"

#include <memory>

#include "default_opset.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector non_zero(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    return {std::make_shared<default_opset::NonZero>(data, element::i64)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
