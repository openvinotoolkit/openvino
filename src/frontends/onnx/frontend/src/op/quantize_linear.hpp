// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "core/node.hpp"
#include "openvino/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
std::shared_ptr<ov::Node> make_fake_quantize(const ov::Output<ov::Node>& y_scale,
                                             const ov::Output<ov::Node>& y_zero_point,
                                             const ov::Output<ov::Node>& data);
}
namespace set_1 {
ov::OutputVector quantize_linear(const Node& node);

}  // namespace set_1

namespace set_13 {

ov::OutputVector quantize_linear(const Node& node);

}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
