// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/prelu.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace op {
namespace set_1 {
ov::OutputVector leaky_relu(const ov::frontend::onnx::Node& node) {
    auto data = node.get_ov_inputs().at(0);
    double alpha = node.get_attribute_value<double>("alpha", 0.01);

    std::shared_ptr<ov::Node> alpha_node = v0::Constant::create(data.get_element_type(), ov::Shape{1}, {alpha});
    return {std::make_shared<v0::PRelu>(data, alpha_node)};
}

static bool registered = register_translator("LeakyRelu", VersionRange::single_version_for_all_opsets(), leaky_relu);
}  // namespace set_1
}  // namespace op
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
