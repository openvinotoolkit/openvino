// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include "core/operator_set.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector lrn(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 1e-4);
    double beta = node.get_attribute_value<double>("beta", 0.75);
    double bias = node.get_attribute_value<double>("bias", 1);
    size_t size = node.get_attribute_value<size_t>("size");

    return {std::make_shared<v0::LRN>(data, alpha, beta, bias, size)};
}

static bool registered = register_translator("LRN", VersionRange::single_version_for_all_opsets(), lrn);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
