// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "ngraph/node.hpp"
#include "onnx_import/core/node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace detail {
std::shared_ptr<ngraph::Node> make_fake_quantize(const Output<ngraph::Node>& y_scale,
                                                 const Output<ngraph::Node>& y_zero_point,
                                                 const Output<ngraph::Node>& data);
}
namespace set_1 {
OutputVector quantize_linear(const Node& node);

}  // namespace set_1

namespace set_13 {

OutputVector quantize_linear(const Node& node);

}  // namespace set_13

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
