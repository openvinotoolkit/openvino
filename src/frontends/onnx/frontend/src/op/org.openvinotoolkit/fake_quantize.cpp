// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/fake_quantize.hpp"

#include <memory>

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace org_openvinotoolkit {
namespace opset_1 {
ov::OutputVector fake_quantize(const ov::frontend::onnx::Node& node) {
    const auto inputs = node.get_ov_inputs();
    const auto X = inputs.at(0);
    const auto input_low = inputs.at(1);
    const auto input_high = inputs.at(2);
    const auto output_low = inputs.at(3);
    const auto output_high = inputs.at(4);

    const auto levels = node.get_attribute_value<std::size_t>("levels");

    return {std::make_shared<v0::FakeQuantize>(X, input_low, input_high, output_low, output_high, levels)};
}

ONNX_OP("FakeQuantize", OPSET_SINCE(1), org_openvinotoolkit::opset_1::fake_quantize, OPENVINO_ONNX_DOMAIN);
}  // namespace opset_1
}  // namespace org_openvinotoolkit
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
