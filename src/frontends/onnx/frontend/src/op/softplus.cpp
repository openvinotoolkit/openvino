// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softplus.hpp"

#include "openvino/op/softplus.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector softplus(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);
    return {std::make_shared<v4::SoftPlus>(data)};
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
