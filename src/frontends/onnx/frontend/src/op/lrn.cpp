// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector lrn(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 1e-4);
    double beta = node.get_attribute_value<double>("beta", 0.75);
    double bias = node.get_attribute_value<double>("bias", 1);
    size_t size = node.get_attribute_value<size_t>("size");

    return {std::make_shared<v0::LRN>(data, alpha, beta, bias, size)};
}

ONNX_OP("LRN", OPSET_SINCE(1), ai_onnx::opset_1::lrn);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
