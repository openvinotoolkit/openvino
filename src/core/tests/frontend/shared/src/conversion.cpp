// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/frontend/extension/conversion.hpp>
#include <openvino/frontend/extension/decoder_transformation.hpp>
#include <openvino/op/util/framework_node.hpp>

#include "conversion_extension.hpp"
#include "utils.hpp"

using namespace ov::frontend;

std::string FrontEndConversionExtensionTest::getTestCaseName(
    const testing::TestParamInfo<ConversionExtensionFEParam>& obj) {
    std::string res = obj.param.m_frontEndName + "_" + obj.param.m_modelName;
    return FrontEndTestUtils::fileToTestName(res);
}

void FrontEndConversionExtensionTest::SetUp() {
    FrontEndTestUtils::setupTestEnv();
    initParamTest();
}

void FrontEndConversionExtensionTest::initParamTest() {
    m_param = GetParam();
    m_param.m_modelName = FrontEndTestUtils::make_model_path(m_param.m_modelsPath + m_param.m_modelName);
}

template <class T>
class ConversionExtensionMock : public ov::frontend::ConversionExtension<T> {
public:
    ConversionExtensionMock(const std::string& op_type, const ov::frontend::CreatorFunction<T>& converter)
        : ov::frontend::ConversionExtension<T>(op_type, converter) {}

    ~ConversionExtensionMock() override = default;
};

///////////////////////////////////////////////////////////////////

TEST_P(FrontEndConversionExtensionTest, TestConversionExtensionMock) {
    auto frontend = m_param.m_frontend;
    if (m_param.m_frontEndName == "paddle") {
        frontend->add_extension(std::make_shared<ConversionExtensionMock<std::map<std::string, ov::OutputVector>>>(
            m_param.m_translatorName,
            [&](const NodeContext<std::map<std::string, ov::OutputVector>>& node)
                -> std::map<std::string, ov::OutputVector> {
                return {};
            }));
    } else {
        frontend->add_extension(std::make_shared<ConversionExtensionMock<ov::OutputVector>>(
            m_param.m_translatorName,
            [&](const ov::frontend::NodeContext<ov::OutputVector>& node) -> ov::OutputVector {
                return {};
            }));
    }
    std::shared_ptr<InputModel> input_model;
    ASSERT_NO_THROW(input_model = frontend->load(m_param.m_modelName));
    ASSERT_NE(input_model, nullptr);
    std::shared_ptr<ov::Model> model;
    ASSERT_NO_THROW(model = frontend->convert(input_model));
    ASSERT_NE(model, nullptr);
}
