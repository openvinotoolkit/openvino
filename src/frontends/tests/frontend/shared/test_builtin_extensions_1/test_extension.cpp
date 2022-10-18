// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_extension.hpp"

#include <openvino/core/core.hpp>

#include "openvino/opsets/opset8.hpp"

bool TestExtension1::transform(const std::shared_ptr<ov::Model>& function, const std::string& config) const {
    function->set_friendly_name("TestFunction");
    return true;
}

TestExtension1::TestExtension1() : ov::frontend::JsonTransformationExtension("buildin_extensions_1::TestExtension1") {}

ov::OutputVector CustomTranslatorCommon_1(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

std::map<std::string, ov::OutputVector> CustomTranslatorCommon_2(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}

ov::OutputVector CustomTranslatorTensorflow(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

ov::OutputVector CustomTranslatorONNX(const ov::frontend::NodeContext& node) {
    return ov::OutputVector();
}

ov::OutputVector ReluToSwishTranslatorONNX(const ov::frontend::NodeContext& node) {
    auto swish = std::make_shared<ov::opset8::Swish>(node.get_input(0));
    return {swish};
}

std::map<std::string, ov::OutputVector> CustomTranslatorPaddle(const ov::frontend::NodeContext& node) {
    return std::map<std::string, ov::OutputVector>();
}
