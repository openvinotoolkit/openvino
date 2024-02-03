// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/prelu.hpp"

#include "openvino/op/prelu.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector prelu(const ov::frontend::onnx::Node& node) {
    ov::OutputVector ng_inputs{node.get_ng_inputs()};
    const auto& data = ng_inputs.at(0);
    const auto& slope = ng_inputs.at(1);
    return {std::make_shared<v0::PRelu>(data, slope)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
