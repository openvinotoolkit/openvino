// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/softsign.hpp"

#include "openvino/op/softsign.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector softsign(const ov::frontend::onnx::Node& node) {
    return {std::make_shared<v9::SoftSign>(node.get_ov_inputs().at(0))};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
