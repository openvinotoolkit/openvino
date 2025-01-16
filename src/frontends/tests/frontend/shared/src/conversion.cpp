// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/extension/conversion.hpp"

#include "common_test_utils/file_utils.hpp"
#include "conversion_extension.hpp"
#include "openvino/frontend/extension/decoder_transformation.hpp"
#include "openvino/op/util/framework_node.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndConversionExtensionTest::getTestCaseName(
    const testing::TestParamInfo<ConversionExtensionFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndConversionExtensionTest::SetUp() {
    initParamTest();
}

void FrontEndConversionExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

inline std::string get_lib_path(const std::string& lib_name) {
    return ov::util::make_plugin_library_name<char>(ov::test::utils::getExecutableDirectory(),
                                                    lib_name + OV_BUILD_POSTFIX);
}

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndConversionExtensionTest, TestConversionExtension) {
    auto frontend = m_param.m_frontend;
    bool invoked = false;
    if (m_param.m_frontEndName == "paddle") {
        frontend->add_extension(std::make_shared<ConversionExtension>(
            m_param.m_translatorName,
            [&](const NodeContext& node) -> std::map<std::string, ov::OutputVector> {
                auto relu = std::make_shared<ov::opset8::Relu>(node.get_input("X"));
                invoked = true;
                return {{"Out", {relu}}};
            }));
    } else if (m_param.m_frontEndName == "tf") {
        frontend->add_extension(
            std::make_shared<ConversionExtension>(m_param.m_translatorName,
                                                  [&](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                                                      invoked = true;
                                                      auto ng_input = node.get_input(0);
                                                      auto res = std::make_shared<ov::opset8::Relu>(ng_input);
                                                      return {res};
                                                  }));
    } else if (m_param.m_frontEndName == "tflite") {
        frontend->add_extension(
            std::make_shared<ConversionExtension>(m_param.m_translatorName,
                                                  [&](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                                                      invoked = true;
                                                      auto input = node.get_input(0);
                                                      auto res = std::make_shared<ov::opset8::Sigmoid>(input);
                                                      return {res};
                                                  }));
    } else if (m_param.m_frontEndName == "onnx") {
        frontend->add_extension(
            std::make_shared<ConversionExtension>(m_param.m_translatorName,
                                                  [&](const ov::frontend::NodeContext& node) -> ov::OutputVector {
                                                      invoked = true;
                                                      auto a = node.get_input(0);
                                                      auto b = node.get_input(1);
                                                      auto res = std::make_shared<ov::opset8::Add>(a, b);
                                                      return {res};
                                                  }));
    }
    std::shared_ptr<InputModel> input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(m_param.m_modelName));
    ASSERT_NE(input_model, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = frontend->convert(input_model));
    ASSERT_NE(model, nullptr);
    EXPECT_EQ(invoked, true);
}

TEST_P(FrontEndConversionExtensionTest, TestConversionExtensionViaSO) {
    auto frontend = m_param.m_frontend;
    const auto& lib_path = get_lib_path("test_builtin_extensions");
    frontend->add_extension(lib_path);
    std::shared_ptr<InputModel> input_model;
    OV_ASSERT_NO_THROW(input_model = frontend->load(m_param.m_modelName));
    ASSERT_NE(input_model, nullptr);
    std::shared_ptr<ov::Model> model;
    OV_ASSERT_NO_THROW(model = frontend->convert(input_model));
    ASSERT_NE(model, nullptr);
}
