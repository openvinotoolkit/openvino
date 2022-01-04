// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/range.hpp"

#include <memory>

#include "default_opset.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector range(const Node& node) {
    const Output<ov::Node> start{node.get_ng_inputs().at(0)};
    const Output<ov::Node> stop{node.get_ng_inputs().at(1)};
    const Output<ov::Node> step{node.get_ng_inputs().at(2)};
    return {std::make_shared<default_opset::Range>(start, stop, step, start.get_element_type())};
}
}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ov
