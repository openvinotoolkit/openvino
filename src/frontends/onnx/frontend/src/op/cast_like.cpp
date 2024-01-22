// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cast_like.hpp"

#include "openvino/op/convert_like.hpp"

using namespace ov::op;

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector cast_like(const Node& node) {
    auto inputs = node.get_ng_inputs();
    return {std::make_shared<v1::ConvertLike>(inputs.at(0), inputs.at(1))};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
