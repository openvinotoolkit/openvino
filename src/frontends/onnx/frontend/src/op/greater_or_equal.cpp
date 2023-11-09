// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/greater_or_equal.hpp"

#include <memory>

#include "default_opset.hpp"
#include "utils/common.hpp"
#define _USE_MATH_DEFINES
#include <math.h>

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector greater_or_equal(const Node& node) {
    const auto A = node.get_ng_inputs().at(0);
    const auto B = node.get_ng_inputs().at(1);

    const auto C = std::make_shared<default_opset::GreaterEqual>(A, B);

    return {C};
}
}  // namespace set_1

namespace set_12 {
OutputVector greater_or_equal(const Node& node) {
    const auto A = node.get_ng_inputs().at(0);
    const auto B = node.get_ng_inputs().at(1);

    const auto C = std::make_shared<default_opset::GreaterEqual>(A, B);

    return {C};
}
}  // namespace set_12
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
