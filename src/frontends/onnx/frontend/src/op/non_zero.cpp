// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/non_zero.hpp"

#include "openvino/op/non_zero.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
ov::OutputVector non_zero(const Node& node) {
    auto data = node.get_ng_inputs().at(0);
    return {std::make_shared<v3::NonZero>(data, ov::element::i64)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
