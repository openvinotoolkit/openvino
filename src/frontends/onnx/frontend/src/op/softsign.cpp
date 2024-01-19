// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softsign.hpp"

#include <memory>
#include <vector>

#include "default_opset.hpp"
#include "ngraph/shape.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector softsign(const Node& node) {
    return {std::make_shared<ov::op::v9::SoftSign>(node.get_ng_inputs().at(0))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
