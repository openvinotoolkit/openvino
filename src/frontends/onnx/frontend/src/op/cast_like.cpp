// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/cast_like.hpp"

#include "openvino/op/convert_like.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {

ov::OutputVector cast_like(const ov::frontend::onnx::Node& node) {
    auto inputs = node.get_ov_inputs();
    return {std::make_shared<v1::ConvertLike>(inputs.at(0), inputs.at(1))};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
