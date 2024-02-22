// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/mish.hpp"

#include "openvino/op/mish.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector mish(const ov::frontend::onnx::Node& node) {
    const auto data = node.get_ov_inputs().at(0);

    return {std::make_shared<v4::Mish>(data)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
